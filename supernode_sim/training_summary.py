"""Training report accounting on completed traces, independent of event scheduling."""
from collections import Counter, defaultdict
from heapq import heappop, heappush
from math import fsum


def _partition(title, unit, amounts):
    total = fsum(amounts.values())
    return {'title': title, 'unit': unit, 'total': total, 'segments': [
        {'name': name, 'value': value, 'percent': 100 * value / total if total else None}
        for name, value in sorted(amounts.items(), key=lambda item: (-item[1], item[0]))]}


def _credit(ledger, active, elapsed):
    count = sum(active.values())
    for name, multiplicity in active.items():
        ledger[name] += elapsed * multiplicity / count


class _RankTimeline:
    """Sweep start-ordered intervals, retaining only overlapping interval ends."""
    def __init__(self):
        self.ends = []
        self.active = {kind: Counter() for kind in ('compute', 'transfer', 'other')}
        self.time = 0.0
        self.first_start = None
        self.last_end = 0.0
        self.states = dict.fromkeys(('compute_only', 'compute_and_transfer', 'exposed_transfer', 'other_local', 'idle'), 0.0)

    def _accrue(self, end, compute_ops, exposed_ops):
        elapsed = end - self.time
        if not elapsed:
            return
        compute, transfers = self.active['compute'], self.active['transfer']
        if compute:
            state = 'compute_and_transfer' if transfers else 'compute_only'
            _credit(compute_ops, compute, elapsed)
        elif transfers:
            state = 'exposed_transfer'
            _credit(exposed_ops, transfers, elapsed)
        else:
            state = 'other_local' if self.active['other'] else 'idle'
        self.states[state] += elapsed
        self.time = end

    def advance(self, to, compute_ops, exposed_ops):
        while self.ends and self.ends[0][0] <= to:
            end, kind, name = heappop(self.ends)
            self._accrue(end, compute_ops, exposed_ops)
            active = self.active[kind]
            active[name] -= 1
            if not active[name]:
                del active[name]
        self._accrue(to, compute_ops, exposed_ops)

    def start(self, end, kind, name):
        if self.first_start is None:
            self.first_start = self.time
        self.last_end = max(self.last_end, end)
        self.active[kind][name] += 1
        heappush(self.ends, (end, kind, name))


def rank_activity(graph, simulation, rank_count):
    ranks = {f'npu{i}': _RankTimeline() for i in range(rank_count)}
    events = {event.id: event for event in graph.events}
    compute_ops, exposed_ops = defaultdict(float), defaultdict(float)
    for row in simulation['trace']:
        start, end = row['start_s'], row['end_s']
        if end <= start:
            continue
        event = events[row['id']]
        meta = event.metadata
        if meta.get('type') == 'transfer':
            owners = {meta['src'], meta['dst']}.intersection(ranks)
            kind = 'other' if meta['src'] == meta['dst'] else 'transfer'
            op = meta.get('collective') or meta.get('operation') or meta.get('category') or 'point_to_point'
            direction = meta.get('semantic') or meta.get('operation')
            if op == 'all_to_all_v' and direction in ('dispatch', 'combine'):
                op = f'{op} ({direction})'
            name = ' / '.join(filter(None, (meta.get('traffic_class'), op)))
        else:
            resources = [key for key, amount in event.work.items() if amount > 0]
            owners = {key.partition('.')[0] for key in (*resources, *event.slots)}.intersection(ranks)
            kind = ('compute' if any(slot.endswith('.compute_stream') for slot in event.slots)
                    or any(graph.resources[key].kind == 'compute' for key in resources) else 'other')
            name = meta.get('kernel') or meta.get('operation') or 'unclassified'
        for owner in owners:
            rank = ranks[owner]
            rank.advance(start, compute_ops, exposed_ops)
            rank.start(end, kind, name)
    totals, idle = Counter(), Counter()
    for rank in ranks.values():
        rank.advance(simulation['duration_s'], compute_ops, exposed_ops)
        totals.update(rank.states)
        if rank.first_start is None:
            idle['inactive_rank'] += simulation['duration_s']
        else:
            tail = simulation['duration_s'] - rank.last_end
            idle['before_first_activity'] += rank.first_start
            idle['after_last_activity'] += tail
            idle['between_activities'] += max(0.0, rank.states['idle'] - rank.first_start - tail)
    return totals, compute_ops, exposed_ops, idle, {key: rank.states for key, rank in ranks.items()}


def build_training_summary(spec, graph, result):
    sim = result.get('simulation', {})
    summary = {'schema_version': 1, 'model': spec.model.name, 'config_sha256': result['config_sha256'],
               'npu_count': spec.hardware.npu_count, 'pp': spec.execution.pp, 'status': 'unavailable'}
    if sim.get('status') != 'feasible' or sim['duration_s'] <= 0:
        summary['reason'] = '需要完整且耗时大于零的训练仿真轨迹。'
        return summary
    totals, compute_ops, exposed_ops, idle, by_rank = rank_activity(graph, sim, spec.hardware.npu_count)
    pools = [p for p in result['memory_breakdown']['pools'] if p['pool_kind'] == 'device_hbm']
    peak = max(pools, key=lambda p: p['peak_bytes'])
    hbm = result['metrics']['training_hbm']
    step = result['metrics']['training_latency_breakdown'].get('step')
    groups = {'hbm': _partition('峰值卡 HBM 构成', 'byte', peak['state_bytes_at_peak'])}
    if step:
        groups['step_latency'] = _partition('平均训练步 · 关键路径时延', 's', step['components_s'])
    groups.update(
        rank_time=_partition('全体 NPU · 时间分配', 'rank·s', totals),
        communication=_partition('通信时间 · 暴露与掩盖', 'rank·s', {
            'exposed_transfer': totals['exposed_transfer'], 'compute_and_transfer': totals['compute_and_transfer']}),
        compute_ops=_partition('计算时段 · 按计算算子', 'rank·s', compute_ops),
        exposed_ops=_partition('暴露通信时段 · 按通信算子', 'rank·s', exposed_ops),
        idle=_partition('空泡 · 按每卡活动位置', 'rank·s', idle))
    accounting = result['metrics'].get('training_accounting', {})
    if 'moe_routing' in accounting or 'reference_mfu' in accounting:
        summary['training_compute'] = {key: accounting[key] for key in
            ('model_mfu', 'executed_compute_utilization', 'npu_flops_by_role', 'moe_routing', 'reference_mfu')
            if key in accounting}
    ep = result['metrics']['training_communication']['by_class'].get('ep', {})
    if 'routed_payload_bytes_by_phase' in ep:
        summary['moe_wire'] = {
            'off_rank_sent_bytes_by_phase': ep['routed_payload_bytes_by_phase'],
            'steps': spec.workload.training_steps, 'devices': spec.hardware.npu_count,
            'basis': 'Cluster sender payload over the simulated run. Each off-rank transfer counts once, including supplied metadata.'}
    return {**summary, 'status': 'complete', 'duration_s': sim['duration_s'], 'groups': groups,
            'rank_activity_s': by_rank,
            'peak_hbm': {'pool': peak['pool'], 'time_s': peak['peak_time_s'], 'peak_bytes': peak['peak_bytes'],
                         'physical_bytes': hbm['physical_bytes_per_npu'], 'usable_bytes': peak['capacity_bytes'],
                         'reserved_bytes': hbm['reserved_bytes_per_npu']},
            'definitions': [
                '显存取占用峰值最高的单卡在同一时刻的实际分配；分母为已用 HBM，另列物理容量和预留。',
                '步时延为已完成训练步的平均关键路径墙钟时间。设备时间分母为配置 NPU 数乘整个运行时长。',
                '通信覆盖不同端点间有 NPU 参与的传输，含 SU、SO 和主机卸载。通信与计算在同卡重叠的部分计为掩盖，启动时间随事件计入。',
                '暴露通信是同卡未与计算重叠的传输时间。其对端到端步时延的影响还取决于关键路径。',
                '空泡是该卡没有计算、端点传输或其他本地资源活动的时段，含流水填充排空、依赖等待和 CPU 尾部。此口径不能单独识别纯 PP 空泡。',
                '计算算子含占用计算流的 kernel；独立访存计入其他本地活动。同类事件并发时等分区间，各算子占比以其所属时段为分母。',
            ]}

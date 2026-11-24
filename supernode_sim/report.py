"""Portable Chinese reports, CSV tables, and Chrome/Perfetto trace export."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import re
from typing import Iterable

from .operator_costs import COST_LABELS


AUTO_DETAIL_THRESHOLD = 50_000
COMPRESS_TABLE_THRESHOLD = 50_000

LATENCY_COMPONENT_LABELS = {
    'compute_s': 'kernel 服务（含计算 / 访存 / 启动）',
    'communication_node_s': '通信 · node 内',
    'communication_scaleup_s': '通信 · scale-up',
    'communication_scaleout_s': '通信 · scale-out',
    'communication_other_s': '通信 · 其他路径',
    'synchronization_wait_s': '显式同步等待',
    'scheduling_wait_s': '调度槽位等待',
    'memory_wait_s': '内存准入等待',
    'resource_contention_s': '资源争用',
    'other_s': '其他',
}


def _latency_candidates(result: dict):
    """Visit numerical results without copying their potentially large traces."""
    if result.get('mode') in ('sweep', 'feedback'):
        for index, candidate in enumerate(result.get('candidates', [])):
            yield str(candidate.get('id', index)), candidate.get('result', candidate)
    elif result.get('mode') == 'size-su':
        for name, search in result.get('searches', {}).items():
            for index, candidate in enumerate(search.get('candidates', [])):
                yield f'{name}/{index}', candidate['result']
    else:
        yield 'run', result


def _latency_rows(result: dict):
    for candidate_id, candidate in _latency_candidates(result):
        metrics = candidate.get('metrics', {})
        breakdown = metrics.get('latency_breakdown') or {}
        fallback_reason = ('not_applicable_to_training' if
                           candidate.get('config', {}).get('workload', {}).get('scenario') == 'training'
                           else 'latency_breakdown_unavailable')
        for metric in ('ttft', 'tpot'):
            record = breakdown.get(metric)
            row = {'candidate_id': candidate_id, 'status': candidate.get('status'),
                   'metric': metric, 'alias': 'tpop' if metric == 'tpot' else '',
                   'available': record is not None, 'method': breakdown.get('method'),
                   'unavailable_reason': (breakdown.get(metric + '_unavailable_reason') or
                                          breakdown.get('unavailable_reason') or fallback_reason) if record is None else None}
            if record is not None:
                row.update(total_s=record.get('total_s'), replica=record.get('replica'), sequence=record.get('sequence'))
                row.update({key: record.get('components_s', {}).get(key) for key in LATENCY_COMPONENT_LABELS})
            yield row


def write_latency_csv(result: dict, directory: Path):
    """One additive latency record per metric and candidate; unavailable cells stay empty."""
    _csv(directory / 'latency_breakdown.csv', [
        'candidate_id', 'status', 'metric', 'alias', 'available', 'total_s',
        *LATENCY_COMPONENT_LABELS, 'replica', 'sequence', 'method', 'unavailable_reason',
    ], _latency_rows(result))
    write_operator_csv(result, directory)


OPERATOR_METRICS = {
    'ttft': 'TTFT', 'tpot': 'TPOT（TPOP）', 'step': '平均训练步',
    'first_step': '首个训练步', 'steady_step': '稳态平均训练步（第 2 步起）',
}


def operator_records(metrics: dict, scenario: str | None = None):
    training = scenario == 'training' or 'training_latency_breakdown' in metrics
    source = metrics.get('training_latency_breakdown' if training else 'latency_breakdown') or {}
    for metric in (('step', 'first_step', 'steady_step') if training else ('ttft', 'tpot')):
        record = source.get(metric)
        breakdown = record.get('operator_breakdown') if record is not None else None
        reason = (source.get(metric + '_unavailable_reason') or source.get('unavailable_reason')
                  or 'operator_attribution_unavailable') if breakdown is None else None
        yield metric, record, breakdown, reason


def _operator_rows(result: dict):
    for candidate_id, candidate in _latency_candidates(result):
        scenario = candidate.get('config', {}).get('workload', {}).get('scenario')
        for metric, record, breakdown, reason in operator_records(candidate.get('metrics', {}), scenario):
            base = {'candidate_id': candidate_id, 'status': candidate.get('status'),
                    'metric': metric, 'alias': 'tpop' if metric == 'tpot' else '',
                    'available': breakdown is not None, 'unavailable_reason': reason}
            if breakdown is None:
                yield base
                continue
            base.update(window_total_s=breakdown['total_s'], method=breakdown['method'],
                        replica=record.get('replica'), sequence=record.get('sequence'),
                        interval_count=record.get('interval_count'))
            # A defined zero-time window has no operator shares (0/0).
            if not breakdown['operators']:
                yield base
            for row in breakdown['operators']:
                yield {**base, **{key: row[key] for key in
                                  ('phase', 'component', 'operator', 'total_s', 'share_pct')},
                       **row['components_s'],
                       **{key.removesuffix('_s') + '_pct': value
                          for key, value in row['component_shares_pct'].items()}}


def write_operator_csv(result: dict, directory: Path):
    _csv(directory / 'operator_breakdown.csv', [
        'candidate_id', 'status', 'metric', 'alias', 'available', 'unavailable_reason',
        'window_total_s', 'phase', 'component', 'operator', 'total_s', 'share_pct',
        *COST_LABELS, *(key.removesuffix('_s') + '_pct' for key in COST_LABELS),
        'replica', 'sequence', 'interval_count', 'method',
    ], _operator_rows(result))


def operator_markdown(metrics: dict, scenario: str | None = None) -> list[str]:
    records = list(operator_records(metrics, scenario))
    lines = ['', '### 算子耗时占比与内部开销', '',
             '算子占比 = 该算子关键路径归因耗时 / 对应窗口总时延；算子内占比 = 该分项耗时 / 该算子归因耗时。'
             '按阶段、组件、实际算子名称汇总并按耗时排序，不累加并发 rank；时间为 ms，占比为 %。',
             '计算与访存在 roofline 模型中重叠，服务时间归到限制资源（同值按资源 ID 稳定择一），'
             '并非独立计算时间与访存时间之和；访存与内存准入等待分列。'
             '链路启动归通信，kernel 启动单列；整 kernel 校准服务无法细分时保留“校准服务（未细分）”。',
             '通信按实际通信算子单列，不猜测归属哪个计算算子；融合算子保留融合边界，'
             '已展开的 kernel_traces 使用子算子名称。只列对该窗口关键路径有时间贡献的算子，缺席不代表未执行。']
    available = [(metric, record, costs) for metric, record, costs, _ in records if costs is not None]
    if available:
        lines += ['', '#### 窗口总开销', '', '| 分项 | ' + ' | '.join(OPERATOR_METRICS[key] + ' / ms' for key, _, _ in available) + ' |',
                  '|---|' + '---:|' * len(available)]
        for key, label in [('total_s', '**总时延**'), *COST_LABELS.items()]:
            values = [costs['total_s'] if key == 'total_s' else costs['components_s'][key] for _, _, costs in available]
            lines.append('| ' + label + ' | ' + ' | '.join(_number(value * 1000) for value in values) + ' |')
    groups = [('compute_s',), ('communication_s',), ('memory_access_s',), ('launch_s',),
              ('synchronization_wait_s', 'scheduling_wait_s', 'memory_wait_s'),
              ('resource_contention_s',), ('unattributed_service_s', 'other_s')]
    for metric, record, costs, reason in records:
        lines += ['', '#### ' + OPERATOR_METRICS[metric], '']
        if costs is None:
            lines += ['—：没有可用算子归因（' + _safe_cell(reason) + '），不补零。']
            continue
        if record.get('replica') is not None:
            lines += [f"请求：DP {_safe_cell(record['replica'])} / sequence {_safe_cell(record['sequence'])}。", '']
        lines += ['| 阶段 | 组件 | 算子 | 耗时 / ms | 占窗口 / % | 计算 / % | 通信 / % | 访存 / % | 启动 / % | 等待 / % | 争用 / % | 未细分及其他 / % |',
                  '|---|---|---|' + '---:|' * 9]
        for row in costs['operators']:
            shares = row['component_shares_pct']
            percentages = [sum(shares[key] for key in group) if row['total_s'] > 0 else None for group in groups]
            cells = [_safe_cell(row[key]) for key in ('phase', 'component', 'operator')]
            cells += [_number(row['total_s'] * 1000), _number(row['share_pct']), *map(_number, percentages)]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += [f"| **合计** | | | {_number(costs['total_s'] * 1000)} | {'100' if costs['total_s'] > 0 else '—'} | | | | | | | |"]
    lines += ['', '各行占窗口比例加和为 100%；每行内部各项比例加和为 100%（显示舍入可能产生尾差）。'
              '零耗时分母的占比未定义，用“—”表示。训练使用训练步时间，不定义 TTFT/TPOT；稳态均值需要至少两步。'
              '所有算子的分项秒值与比例见 operator_breakdown.csv；JSON 保留 operator_breakdown。']
    return lines


def latency_markdown(metrics: dict, *, heading: str = '## TTFT / TPOT（TPOP）时延拆解',
                     scenario: str | None = None) -> list[str]:
    if scenario == 'training' or 'training_latency_breakdown' in metrics:
        return ['', '## 训练步时延拆解', '', '训练使用平均步、首步和第 2 步起的稳态平均步时间；TTFT / TPOT 不适用。'] + operator_markdown(metrics, 'training')
    breakdown = metrics.get('latency_breakdown') or {}
    records = [breakdown.get(key) for key in ('ttft', 'tpot')]
    lines = ['', heading, '', '单位：毫秒（ms）；JSON / latency_breakdown.csv 使用秒（s）。', '',
             '| 分项 | TTFT / ms | TPOT（TPOP）/ ms |', '|---|---:|---:|']
    for key, label in [('total_s', '**总时延**'), *LATENCY_COMPONENT_LABELS.items()]:
        values = [None if record is None else
                  record.get('total_s') if key == 'total_s' else record.get('components_s', {}).get(key)
                  for record in records]
        lines.append('| ' + label + ' | ' + ' | '.join(_number(value * 1000 if value is not None else None) for value in values) + ' |')
    if any(record is not None for record in records):
        lines += ['', '各分项沿决定该指标的请求关键依赖路径归因，可相加得到总时延；不累加所有 rank 或并发事件的耗时。TTFT 与最大平均 TPOT 可能来自不同请求。',
                  'kernel 服务项包含计算、访存与启动；下方算子表进一步细分。通信服务时间按实际路径的限制资源归因，启动延迟按各跳链路域归因。node 内表示同主机 SU 或 host 搬运，scale-up 表示跨主机 SU 资源，scale-out 表示 SO 资源；其他项保留内存带宽或未分类路径，不按流量比例摊分时延。',
                  '依赖同步造成的延迟已沿上游关键路径归因，不再重复记作等待；显式同步等待、调度槽位等待、内存准入等待与共享资源争用分别列出。',
                  '跨 token 边界的事件按时间窗覆盖比例分配服务、争用或准入等待，未采样更细的事件内进度。']
        selected = [f"{label}：DP {_number(record.get('replica'))} / sequence {_number(record.get('sequence'))}"
                    for label, record in zip(('TTFT', 'TPOT（TPOP）'), records) if record is not None]
        lines += ['', '对应请求：' + '；'.join(selected) + '。']
    if any(record is None for record in records):
        reason = breakdown.get('unavailable_reason')
        lines += ['', '“—”表示指标未定义或没有可用拆解（例如仅估算、训练、执行未完成、未执行 Prefill 或不足两个输出 token），不表示零时延。' + (f' 引擎说明：{_safe_cell(reason)}。' if reason else '')]
    return lines + operator_markdown(metrics, scenario)


def _candidate_latency_markdown(result: dict) -> list[str]:
    lines = ['', '## 候选 TTFT / TPOT（TPOP）', '',
             '| 候选 | TTFT / ms | TPOT（TPOP）/ ms |', '|---|---:|---:|']
    for candidate_id, candidate in _latency_candidates(result):
        metrics = candidate.get('metrics', {})
        values = [metrics.get(key) for key in ('ttft_s', 'tpot_s')]
        lines.append(f'| {_safe_cell(candidate_id)} | ' + ' | '.join(_number(value * 1000 if value is not None else None) for value in values) + ' |')
    lines += ['', '每个候选的计算、node 内 / scale-up / scale-out 通信、显式同步、调度槽位等待、内存准入等待与资源争用拆解见 latency_breakdown.csv 和 result.json。CSV 每行对应一个候选的一项聚合指标，单位为秒；空值表示不可用。',
              '逐候选的算子耗时、占窗口比例及算子内部计算 / 通信 / 访存 / 启动 / 等待 / 争用比例见 operator_breakdown.csv；比例单位为 %，空值表示不可用。',
              '分项沿关键依赖路径归因，可加和为对应总时延；并发工作不重复累加，依赖同步的上游时间不再重复记入等待。']
    return lines


def _number(value):
    if value is None:
        return '—'
    if isinstance(value, (int, float)):
        return f'{value:,.6g}'
    return str(value)


_STATE_LABELS = {
    'weights': '权重', 'gradients': '梯度', 'master_weights': '主权重',
    'adam_momentum': 'Adam 一阶动量', 'adam_variance': 'Adam 二阶矩',
    'momentum': '动量', 'variance': '二阶矩',
    'muon_momentum': 'Muon 动量', 'activation': '激活', 'kv_cache': 'KV',
    'recurrent_state': '循环状态', 'quantization_metadata': '量化元数据',
    'buffers': '缓冲', 'runtime': '运行时预留', 'optimizer_combined': '未拆分优化器状态',
    'weight_materialization': '临时完整权重', 'extra_weight_copy': '额外权重副本',
    'attnres_residual': 'AttnRes 残差', 'convolution_state': '卷积历史', 'page_table': '页表',
    'saved_recurrent_states': '训练循环状态', 'gathered_weight': '展开权重',
    'unsharded_gradient_buffer': '完整梯度缓冲', 'muon_workspace': 'Muon 工作区',
}


def _safe_cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _display_pool_label(group: dict) -> str:
    pools = sorted(group.get('pools', []), key=lambda name: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)])
    if not pools:
        return group['pool_label']
    if len(pools) <= 4:
        return ', '.join(pools)
    return f'{pools[0]} … {pools[-1]} ({len(pools)} pools)'


def _memory_markdown(breakdown: dict, tensor_artifact: str = 'tensors.csv') -> list[str]:
    mode = breakdown['mode']
    basis = {'simulated': '仿真实际并发峰值', 'observed_partial': '中断前已观测并发峰值',
             'resident_only': '初始固定驻留，未计算执行峰值'}[mode]
    lines = ['', '## 存储分解：组件 × 状态', '', f'口径：**{basis}**。表格使用十进制 GB（1 GB = 10⁹ byte）；CSV 和 JSON 使用 byte。']
    inventory = breakdown.get('component_inventory') or {}
    if inventory.get('components'):
        lines += ['', '### 全模型逻辑参数（分片和复制之前）', '',
                  '结构清单按唯一参数所有权计数；共享权重别名不重复计入。精确形状、参数量、层号、分片标记和别名见 model_parameters.csv。',
                  '', '| 组件 | 总参数 / B | 每 token 活跃参数 / B | 独立参数张量数 | 层数 |',
                  '|---|---:|---:|---:|---:|']
        for component, values in inventory['components'].items():
            lines.append(f"| {_safe_cell(component)} | {_number(values.get('params', 0) / 1e9)} | "
                         f"{_number(values.get('active_params', 0) / 1e9)} | {_number(values.get('tensor_count'))} | "
                         f"{len(values.get('layers', []))} |")
    logical = breakdown.get('logical_component_state_bytes', {})
    if logical:
        states = sorted({state for values in logical.values() for state in values})
        lines += ['', '### 全模型逻辑状态容量（分片和复制之前）', '',
                  '以下状态由编译器按具体参数和优化器归属给出；没有从物理副本或合并状态反推 Adam/Muon。单位：GB。',
                  '', '| 组件 | ' + ' | '.join(_safe_cell(_STATE_LABELS.get(s, s)) for s in states) + ' | 合计 |',
                  '|---|' + '---:|' * (len(states) + 1)]
        for component, sizes in sorted(logical.items()):
            values = [sizes.get(state, 0) for state in states]
            lines.append('| ' + _safe_cell(component) + ' | ' + ' | '.join(_number(v / 1e9) for v in values) +
                         ' | ' + _number(sum(values) / 1e9) + ' |')
        totals = [sum(sizes.get(state, 0) for sizes in logical.values()) for state in states]
        lines.append('| **总计** | ' + ' | '.join(_number(v / 1e9) for v in totals) +
                     ' | **' + _number(sum(totals) / 1e9) + '** |')
    lines += ['', '### 物理内存池容量与峰值',
             '', '容量和分解均为每个内存池的数值；相同容量与分解的内存池合并展示，完整设备明细保留在 CSV。',
             '', '| 池组 | 等价内存池 | 数量 | 单池容量 / GB | 单池固定驻留 / GB | 单池峰值或驻留 / GB |',
             '|---|---|---:|---:|---:|---:|']
    all_groups = breakdown['equivalent_pool_groups']
    groups = all_groups
    if len(groups) > 16:
        # Bound human-facing output even if every device has a different peak.
        # Exact individual rows remain available in the CSV/JSON artifacts.
        priority = lambda g: (g['peak_bytes_per_pool'] / (g['capacity_bytes_per_pool'] or 1),
                              g['peak_bytes_per_pool'], g['pool_count'])
        ranked = sorted(groups, key=priority, reverse=True)
        selected, kinds = [], set()
        for group in ranked:
            if group['pool_kind'] not in kinds:
                selected.append(group)
                kinds.add(group['pool_kind'])
        for group in ranked:
            if group not in selected and len(selected) < 16:
                selected.append(group)
        groups = selected[:16]
        # Place the note before the existing group table, not inside its rows.
        table = lines[-2:]
        lines = lines[:-2]
        lines += [f"共有 {len(all_groups)} 种等价池组。此处展示容量压力最高的最多 16 组，并覆盖各内存类型；全部逐池结果见 memory_peaks.csv。", '',
                  '| 内存类型 | 内存池数量 | 单池峰值或驻留范围 / GB |', '|---|---:|---:|']
        for kind in sorted({g['pool_kind'] for g in all_groups}):
            same_kind = [g for g in all_groups if g['pool_kind'] == kind]
            values = [g['peak_bytes_per_pool'] / 1e9 for g in same_kind]
            lines.append(f"| {kind} | {sum(g['pool_count'] for g in same_kind)} | {_number(min(values))}–{_number(max(values))} |")
        lines += [''] + table
    for group in groups:
        capacity = group['capacity_bytes_per_pool']
        lines.append(f"| {group['group_id']} | {_safe_cell(_display_pool_label(group))} | {group['pool_count']} | "
                     f"{_number(capacity / 1e9 if capacity is not None else None)} | "
                     f"{_number(group['resident_bytes_per_pool'] / 1e9)} | {_number(group['peak_bytes_per_pool'] / 1e9)} |")
    if mode == 'observed_partial':
        lines += ['', '执行未完成：已观测峰值仅覆盖成功申请的内存，不能用来证明完整任务能运行。初始驻留和失败诊断仍须单独检查。']
    elif mode == 'resident_only':
        lines += ['', f'没有有序执行内存日志，以下组件分解仅代表初始驻留。解析乐观峰值没有附会组件构成；临时张量的计划生命周期见 {tensor_artifact}。']
    lines += ['', '### 每池同一峰值时刻的构成', '', '每组中各组件和状态可以相加得到上表的单池总值。不同池的峰值时刻可能不同。']
    for group in groups:
        rows = group['composition_at_peak_per_pool']
        states = sorted({row['state'] for row in rows})
        components = sorted({row['component'] for row in rows})
        cells = {(row['component'], row['state']): row['bytes'] for row in rows}
        if not rows:
            lines += ['', f"{group['group_id']}：没有已驻留或已观测分配。"]
            continue
        lines += ['', f"**{group['group_id']} · {_safe_cell(_display_pool_label(group))} · 单池 GB**", '',
                  '| 组件 | ' + ' | '.join(_safe_cell(_STATE_LABELS.get(s, s)) for s in states) + ' | 合计 |',
                  '|---|' + '---:|' * (len(states) + 1)]
        for component in components:
            values = [cells.get((component, state), 0) for state in states]
            lines.append('| ' + _safe_cell(component) + ' | ' + ' | '.join(_number(v / 1e9) for v in values) +
                         ' | ' + _number(sum(values) / 1e9) + ' |')
        totals = [sum(cells.get((component, state), 0) for component in components) for state in states]
        lines.append('| **总计** | ' + ' | '.join(_number(v / 1e9) for v in totals) +
                     ' | **' + _number(sum(totals) / 1e9) + '** |')
    if mode != 'resident_only':
        lines += ['', '### 各组件独立峰值（不可相加）', '',
                  '下表每项取该组件在整段已观测执行中的最大值，各项可能出现在不同时间；相加不能得到设备峰值。',
                  '', '| 池组 | 组件 | 单池组件独立峰值 / GB |', '|---|---|---:|']
        for group in groups:
            for component, amount in group['component_independent_peak_bytes_per_pool'].items():
                lines.append(f"| {group['group_id']} | {_safe_cell(component)} | {_number(amount / 1e9)} |")
        cluster = breakdown['cluster']
        lines += ['', f"所有内存池同一时刻的合计峰值为 {_number(cluster['peak_bytes'] / 1e9)} GB，"
                  f"发生于 {_number(cluster['peak_time_s'])} s。该数值按全局时间线重放得到，未相加各池独立峰值；不同内存类型的容量不能互相替代。"]
    lines += ['', f'张量、层号、形状、精度、组件来源与申请/释放时刻见 {tensor_artifact}；组件状态交叉表见 component_memory.csv，状态汇总见 state_memory.csv，逐池峰值见 memory_peaks.csv。',
              '', '缺失组件归属保留为 `Unassigned`；通用缓冲和运行时归为 `System`。已知优化器按其状态语义拆分；无法确定语义的合并状态才保持未拆分，不推定 Adam/Muon 的具体张量。']
    return lines


def training_task_markdown(result: dict) -> list[str]:
    task = result.get('model_metadata', {}).get('training_task')
    if not task:
        return []
    metrics = result.get('metrics', {}).get('training_task', {})
    labels = {'pretrain': '预训练', 'sft': '监督微调 SFT', 'dpo': '偏好优化 DPO', 'rl': '后训练 RL'}
    lines = ['', '## 训练任务与阶段', '', f"任务：{labels[task['kind']]}；调度：`{task['schedule']}`。",
             '', '处理 token 含 padding 和重复更新；有效 token 排除 padding；监督 token 仅计 loss 位置。',
             'RL 的 step 是一次 rollout→更新迭代，optimizer_steps 单独计数；生成、打分、critic 不混入 actor 训练吞吐。',
             '', '| 工作量 | 数值 |', '|---|---:|']
    for key, label in (('processed_tokens', 'Actor 处理 token'), ('effective_tokens', 'Actor 有效 token'),
                       ('supervised_tokens', '监督 token'), ('preference_pairs', '偏好对'),
                       ('optimizer_steps', 'Actor 优化步'), ('critic_optimizer_steps', 'Critic 优化步'),
                       ('rollout_generated_tokens', 'Rollout 生成 token'), ('rollout_sequences', 'Rollout 序列数')):
        if key in task['counts']:
            lines.append(f"| {label} | {_number(task['counts'][key])} |")
    if metrics:
        lines += ['', '| 阶段 | 累计时间 / s |', '|---|---:|']
        lines += [f'| {_safe_cell(stage)} | {_number(seconds)} |' for stage, seconds in metrics['stage_times_s'].items()]
        lines += ['', '阶段串行执行，可相加；逐次记录见 training_task_stages.csv。MFU 仅计可训练 actor/critic 的模型前反向，排除 rollout、冻结打分和 loss/优势算子。']
        if metrics.get('budget_projection'):
            projection = metrics['budget_projection']
            lines += ['', f"数据预算推算：{projection['optimizer_steps']} 个优化步、{_number(projection['duration_s'])} s；按当前平均步时外推并向上取整，不包含数据加载或收敛预测。"]
    else:
        lines += ['', '没有完整可行执行，阶段时间和吞吐不可用；上表仅为计划工作量。']
    lines += ['', '各角色的逻辑状态容量见 model_metadata.training_task.role_logical_memory_bytes；张量 metadata.training_role 区分 actor/reference/reward/critic。']
    return lines


def markdown_report(result: dict) -> str:
    if result.get('mode') == 'feedback':
        from .feedback_report import feedback_markdown
        return feedback_markdown(result)
    lines = ['# 超节点资源建模报告', '', f"状态：**{result['status']}**；模式：`{result['mode']}`。", '']
    export = result.get('artifact_export', {})
    tensor_artifact = export.get('tensors', {}).get('artifact') or 'tensors.csv'
    if export.get('compacted'):
        lines += ['大图使用精简导出，数值仿真仍包含全部事件。所有容量、通信、性能、组件状态和并发峰值汇总均保持完整。', '',
                  '| 明细 | 原始行数 | 导出行数 | 未导出行数 | 文件/处理 |', '|---|---:|---:|---:|---|']
        for label, key in (('事件时间线', 'events'), ('物理张量', 'tensors'), ('原始申请/释放日志', 'memory_log')):
            detail = export[key]
            lines.append(f"| {label} | {detail['total_rows']} | {detail['exported_rows']} | {detail['omitted_rows']} | {_safe_cell(detail.get('artifact') or '保留精确峰值汇总')} |")
        lines += ['', f'events.csv / timeline.json 保留模型 kernel、token/step 完成标记和模型张量申请事件，省略 collective 搬运与缓冲控制事件。{tensor_artifact} 省略临时 `buffers` 明细；这些缓冲仍完整计入组件状态和并发峰值。',
                  '', '需要全部原始事件、张量和有序内存日志时，用相同配置在 estimate/simulate 命令添加 `--full-detail` 重新导出。原始日志未写入其他隐藏文件。', '']
        if tensor_artifact.endswith('.gz'):
            lines += ['大型张量明细使用 gzip 无损压缩；解压后是完整 CSV，列名和导出行数保持不变。', '']
    if result['mode'] in ('sweep', 'size-su'):
        lines += ['结果限定于显式候选集合和给定调度策略，完整配置与失败原因见 JSON。', '', '| 候选 | 状态 | SU 卡数 | 总卡数 | 执行时间 / s |', '|---|---|---:|---:|---:|']
        if result['mode'] == 'sweep':
            rows = [(str(i), r['result']) for i, r in enumerate(result['candidates'])]
        else:
            rows = []
            for mode, search in result['searches'].items():
                lines += [f"| {mode} 搜索 | {search['status']} | {_number(search.get('minimum_su_size'))} | — | — |"]
                rows.extend((f'{mode}/{i}', row['result']) for i, row in enumerate(search.get('candidates', [])))
        for name, row in rows:
            hardware = row.get('config', {}).get('hardware', {})
            lines.append(f"| {name} | {row['status']} | {_number(hardware.get('su_size'))} | {_number(hardware.get('npu_count'))} | {_number(row.get('metrics', {}).get('duration_s'))} |")
        for name, search in result.get('searches', {}).items():
            fastest = search.get('highest_throughput')
            if fastest:
                index = fastest['candidate_index']
                row = search['candidates'][index]
                layout = row['parallelism']
                lines += ['', f"{name} 训练吞吐最高的可行候选为 {index}，"
                          f"{_number(fastest['tokens_s'])} tokens/s。"
                          f"TP/PP/EP/DP/FSDP = {layout['tp']}/{layout['pp']}/{layout['ep']}/{layout['dp']}/{layout['fsdp']}，"
                          f"DP 域大小 {layout.get('dp_domain_size') or '平坦归约'}，rank 排布为 `{row['placement']}`。"]
        lines += _candidate_latency_markdown(result)
    else:
        config = result['config']
        lines += [f"模型：`{config['model']['name']}`；场景：`{config['workload']['scenario']}`；权重：`{config['execution']['precision']['weight']}`。"]
        hardware = config.get('hardware', {})
        if hardware:
            capacity = hardware.get('npu', {}).get('hbm_bytes')
            lines += ['', f"硬件：`{hardware.get('name', '用户硬件配置')}`；NPU 数量：{_number(hardware.get('npu_count'))}；"
                      f"每颗 HBM：{_number(capacity / 1e9 if capacity is not None else None)} GB。"]
        fidelity = result.get('model_metadata', {}).get('fidelity', {})
        if fidelity.get('parameter_capacity') == 'shape_derived':
            lines += ['', '容量按张量形状推导；激活采用解析模型，耗时由解析模型及已匹配计时记录约束，尚未通过整机实测校准。`feasible` 表示在本模型假设下满足约束，不代表真实整机部署验证。']
        evidence = fidelity.get('kernel_timing_evidence', {})
        if any(evidence.values()):
            lines += ['', 'Kernel 计时记录实际命中：' + '；'.join(
                f'{label} {evidence.get(kind, 0)}' for kind, label in (
                    ('measurement', '声明为硬件实测'), ('simulator', '仿真器预测'),
                    ('synthetic', '合成数据'), ('unspecified', '未注明类型'))) + '。',
                '此处按事件计数；证据类型来自输入声明，不表示独立验证。仿真器预测与合成数据不计入硬件实测覆盖。']
        if config['workload']['scenario'] == 'training' and fidelity.get('distributed_muon') == 'not_implemented':
            lines += ['', '**Muon 训练限制：尚未实现分布式 Muon 正交化。当前训练步时间是本地更新与通信的代理模型，不能作为已验证的生产训练性能。**']
        budget = result.get('model_metadata', {}).get('training_hbm')
        if budget:
            lines += ['', f"训练每卡 HBM 准入上限为 {_number(budget['usable_bytes_per_npu'] / 1e9)} GB"
                      f"（物理容量的 {_number(budget['usable_fraction'] * 100)}%），"
                      f"预留 {_number(budget['reserved_bytes_per_npu'] / 1e9)} GB。"]
        communication = result.get('metrics', {}).get('training_communication')
        if communication and communication['by_class']:
            lines += ['', f"卡间通信活跃时间合计 {_number(communication['network_rank_s'])} rank·s，"
                      f"其中 {_number(communication['overlap_with_compute_rank_s'])} rank·s 与同卡计算重叠。",
                      '', '| 训练流量 | 权重 | 传输 payload / GB | 通信槽位等待 / event·s |', '|---|---:|---:|---:|']
            weights = result['model_metadata'].get('training_qos', {}).get('weights', {})
            for name, item in sorted(communication['by_class'].items()):
                lines.append(f"| {name} | {weights.get(name, 1)} | {_number(item['payload_bytes'] / 1e9)} | {_number(item['slot_wait_event_s'])} |")
        plan = result.get('model_metadata', {}).get('feature_plan', {})
        selected = [row for row in plan.get('features', []) if row.get('selected')]
        if selected:
            lines += ['', '## 已选择的特性', '', '| 特性 | 建模方式 | 适用边界 |', '|---|---|---|']
            for row in selected:
                lines.append(f"| {_safe_cell(row['name'])} | {_safe_cell(row['modeling'])} | {_safe_cell(row.get('notes', ''))} |")
            lines += ['', '完整选择与实际 kernel 匹配覆盖率见 result.json 的 model_metadata.feature_plan。支持特性表示存在资源/时序模型，不表示已验证提速或精度保持。']
        speculative = result.get('model_metadata', {}).get('speculative_decode')
        if speculative:
            lines += ['', f"投机解码轮次：{speculative['round_count']}；每序列实际输出 token：{speculative['committed_tokens_per_sequence']}；"
                      f"验证 token：{speculative['verified_tokens_per_sequence']}。接受数来自 `{speculative['acceptance_mode']}`。",
                      '', '同一轮提交的 token 共用模型完成时间，零轮内间隔不代表客户端观测的逐 token 延迟；该路径并非 DSpark 的算法/质量复现。']
        fusion = result.get('model_metadata', {}).get('megakernel')
        if fusion and fusion.get('enabled'):
            lines += ['', f"Megakernel 图变换：融合 {fusion['matched_groups']} 组、{fusion['matched_events']} 个原 kernel 事件；"
                      f"匹配覆盖率 {_number(fusion['coverage_fraction'] * 100)}%；计划减少 HBM 访问 {_number(fusion['saved_hbm_bytes'])} byte。",
                      '', '融合只作用于安全的局部串行链。原分配统一提前、释放延后，属于保守内存生命周期，可能高估峰值；不代表自动消除了中间张量容量。']
            if not fusion['matched_groups']:
                lines += ['', '**没有合法融合匹配，未应用 megakernel 提速。**']
            if fusion.get('skip_reasons'):
                lines += ['', '跳过原因：`' + _safe_cell(json.dumps(fusion['skip_reasons'], ensure_ascii=False)) + '`。']
            if fusion.get('member_profiles_not_applied'):
                lines += ['', '成员 kernel profile 未继承到融合事件；需核对融合组 profile：`' +
                          _safe_cell(', '.join(fusion['member_profiles_not_applied'])) + '`。']
        pd = result.get('model_metadata', {}).get('prefill_decode_disaggregation')
        if pd:
            lines += ['', f"PD 分离：P 池 `{pd['prefill_ranks']}`，D 池 `{pd['decode_ranks']}`；部署 {pd['model_copies']} 份模型。"
                      f"状态交接计划 payload 为 {_number(pd['state_payload_bytes'])} byte，首 token ID 为 {_number(pd['token_id_payload_bytes'])} byte。",
                      '', 'TTFT 截止 P 池完成首 token；交接计入第一个 decode 间隔。两池相同 TP/EP，完整状态通过实际拓扑传输；不包含异构池重分片、在线排队或客户端转发。逻辑参数表仍描述单模型，两池实际驻留见物理内存明细。']
        lines += ['', '## 性能与约束', '', '| 指标 | 数值 |', '|---|---:|']
        labels = {'duration_s': '总执行时间 / s', 'step_time_s': '平均训练步时间 / s', 'ttft_s': '执行 TTFT / s', 'tpot_s': '批内最大平均 TPOT（TPOP）/ s', 'throughput_tokens_s': '聚合吞吐 / tokens/s', 'input_throughput_tokens_s': '输入吞吐 / tokens/s', 'throughput_per_replica_tokens_s': '每副本吞吐 / tokens/s'}
        labels.update(pd_prefill_time_s='PD prefill 执行时间 / s', pd_handoff_time_s='PD 状态交接时间 / s',
                      pd_first_decode_gap_s='PD 首个 decode 间隔（含交接）/ s', pd_decode_time_s='PD 交接后 decode 时间 / s')
        for key, label in labels.items():
            if key in result.get('metrics', {}):
                lines.append(f"| {label} | {_number(result['metrics'][key])} |")
        lines.append(f"| 理想执行时间必要下界 / s | {_number(result['estimate']['duration_lower_bound_s'])} |")
        for check in result.get('slo', {}).get('checks', []):
            lines.append(f"| SLO {check['metric']} | {'满足' if check['passed'] else '不满足'}；目标 {_number(check['target'])} |")
        if result['mode'] == 'estimate':
            lines += ['', '解析结果提供必要下界，尚未证明资源争用下的 SLO 可行性。']
        accounting = result.get('metrics', {}).get('training_accounting')
        lines += training_task_markdown(result)
        if accounting:
            lines += ['', '## 训练计算量与利用率口径', '',
                      '| 项目 | 数值 |', '|---|---:|']
            for key, label in (
                ('model_forward_backward_flops', '模型前向与反向 FLOPs'),
                ('model_recompute_flops', '模型重计算 FLOPs'),
                ('executed_npu_flops', 'NPU 全部执行 FLOPs'),
                ('model_mfu', 'model_mfu（前向＋反向代理，比例）'),
                ('executed_compute_utilization', '全部 NPU 工作的峰值归一比例'),
            ):
                lines.append(f"| {label} | {_number(accounting.get(key))} |")
            lines += ['', 'model_mfu 排除重计算、优化器和转换/打包；各 dtype 分别除以裸峰值，再按完整卡数与执行时间归一。',
                      '这是当前模型的计算量代理，不能不核对分子就与其他工具或论文的同名 MFU 比较。CPU 工作单独统计；详细阶段、角色与 dtype 账本见 metrics.training_accounting。']
        lines += latency_markdown(result.get('metrics', {}), scenario=result.get('config', {}).get('workload', {}).get('scenario'))
        logical = result.get('model_metadata', {}).get('logical_memory_bytes', {})
        if logical:
            lines += ['', '## 全局逻辑容量', '', '单位为十进制 TB；实际分片、复制和生命周期见设备明细。', '', '| 组成 | TB |', '|---|---:|']
            for key, value in logical.items():
                if isinstance(value, (int, float)):
                    lines.append(f'| {key} | {_number(value / 1e12)} |')
        if result.get('memory_breakdown'):
            lines += _memory_markdown(result['memory_breakdown'], tensor_artifact)
            lines += ['', '## 通信']
        else:
            peaks = result.get('simulation', {}).get('memory_peaks_bytes', result['estimate']['resident_memory_bytes'])
            lines += ['', '## 存储与通信', '', '| 内存池 | 峰值或固定驻留 / GB | 配置容量 / GB |', '|---|---:|---:|']
            for pool, value in peaks.items():
                capacity = result.get('memory_capacities_bytes', {}).get(pool)
                lines.append(f'| {pool} | {_number(value / 1e9)} | {_number(capacity / 1e9 if capacity is not None else None)} |')
        lines += ['', '物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。', '', '| 网络域 | 物理传输 / GB |', '|---|---:|']
        for domain, amount in result['estimate']['communication']['physical_bytes_by_domain'].items():
            lines.append(f'| {domain} | {_number(amount / 1e9)} |')
        lines += ['', '| 应用语义 | 传输 payload / GB |', '|---|---:|']
        for category, amount in result['estimate']['communication']['payload_by_semantic_bytes'].items():
            lines.append(f'| {category} | {_number(amount / 1e9)} |')
        lines += ['', '应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。']
        if result['requirements']['target_window_s'] is not None:
            lines += ['', f"按吞吐或训练步 SLO 得到的整任务时间窗口：{_number(result['requirements']['target_window_s'])} s。各资源平均速率必要下界见 resources.csv；还需满足局部依赖与争用约束。"]
        resource_times = result['estimate']['resource_time_lower_bounds_s']
        lines += ['', '## 主要资源下界', '', '| 资源 | 独占服务时间下界 / s |', '|---|---:|']
        for key, value in sorted(resource_times.items(), key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f'| {key} | {_number(value)} |')
        lines += ['', '缓存容量和 MTE 通道数需要 kernel/硬件校准；当前输出工作集与搬运量下界。']
        diagnostics = result.get('simulation', {}).get('diagnostics', [])
        if diagnostics:
            lines += ['', '## 未满足约束', '', '```json', json.dumps(diagnostics, ensure_ascii=False, indent=2), '```']
        lines += ['', '## 参数来源', '']
        for key, source in result['sources'].items():
            lines.append(f"- `{key}`：{source['status']}；{source['description']}" + (f"；{source.get('uri')}" if source.get('uri') else ''))
    lines += ['', '## 假设与复现', '']
    for assumption in result.get('assumptions', []):
        lines.append(f'- {assumption}')
    lines += ['- 固定负载执行模型，不含请求到达排队或生产尾延迟。', '- FP8 scale、转换缓冲与额外副本按配置单独计入。', '- 未经实测校准的合成配置用于架构比较。']
    if result.get('config_sha256'):
        lines += [f"- 配置 SHA256：`{result['config_sha256']}`。"]
    return '\n'.join(lines) + '\n'


def _csv(path: Path, fields: list[str], rows: Iterable[dict]):
    stream = (gzip.open(path, 'wt', newline='', encoding='utf-8-sig', compresslevel=6)
              if path.suffix == '.gz' else path.open('w', newline='', encoding='utf-8-sig'))
    with stream as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _prepare_export(result: dict, full_detail: bool):
    """Make a shallow artifact view; leave all API-owned arrays unchanged."""
    trace = result.get('simulation', {}).get('trace', [])
    tensors = result.get('memory_breakdown', {}).get('tensors', [])
    memory_log = result.get('simulation', {}).get('memory_log', [])
    record_stream = not isinstance(trace, (dict, str, bytes)) and hasattr(trace, '__iter__') and hasattr(trace, '__len__')
    if not record_stream or not isinstance(tensors, list):
        raise ValueError('write_report expects the full run() result, not a previously filtered artifact reference')
    tensor_details = result.get('memory_breakdown', {}).get('tensor_detail_export', {})
    original_tensor_count = tensor_details.get('total_rows', len(tensors))
    prefiltered_count = tensor_details.get('omitted_rows', 0)
    if tensor_details and tensor_details.get('retained_rows') != len(tensors):
        raise ValueError('Tensor detail counts disagree with the available allocation rows')
    omitted_trace_fields = result.get('simulation', {}).get('trace_payload_fields_omitted', [])
    if full_detail and omitted_trace_fields:
        raise ValueError('Full event payloads were not retained; rerun with retain_event_payload=True before requesting --full-detail export')
    if full_detail and prefiltered_count:
        raise ValueError('Full tensor details were not materialized; rerun with full detail enabled (include_ephemeral_details=True) before requesting --full-detail export')
    large = max(len(trace), original_tensor_count, len(memory_log)) > AUTO_DETAIL_THRESHOLD
    compact = (large and not full_detail) or bool(prefiltered_count)
    keep_tensor = lambda row: not compact or row.get('state') != 'buffers' or bool(row.get('is_initial'))
    model_allocators = ({row['allocation_event'] for row in tensors
                         if keep_tensor(row) and row.get('allocation_event')} if compact else set())
    markers = set()

    def collect_markers(value):
        if isinstance(value, str):
            markers.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect_markers(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_markers(item)

    metadata = result.get('model_metadata', {})
    for key in ('sequence_token_completion_events', 'token_completion_events', 'step_end_events', 'prefill_end_events'):
        collect_markers(metadata.get(key, []))

    def keep_event(row):
        if not compact:
            return True
        meta = row.get('metadata', {})
        return (bool(meta.get('kernel')) or row['id'] in markers or row['id'] in model_allocators or
                meta.get('phase') in ('token_complete', 'step_complete') or
                meta.get('operation') in ('token_complete', 'step_complete'))

    if not large and not prefiltered_count:
        return result, trace, tensors, keep_event, keep_tensor
    # Count retained events while writing both event artifacts. A checkpoint
    # can otherwise be decompressed three times merely to export one report.
    event_count = 0 if compact else len(trace)
    tensor_count = sum(keep_tensor(row) for row in tensors) if compact else len(tensors)
    tensor_artifact = ('tensors.csv.gz' if compact and tensor_count > COMPRESS_TABLE_THRESHOLD else 'tensors.csv')
    manifest = {
        'compacted': compact, 'detail_mode': 'compact' if compact else 'full',
        'requested_full_detail': full_detail, 'auto_detail_threshold_rows': AUTO_DETAIL_THRESHOLD,
        'trace_payload_fields_omitted': list(omitted_trace_fields),
        'reason': ('Ephemeral buffer details were omitted during memory report construction; numerical replay remains complete.'
                   if prefiltered_count else 'Raw trace, physical allocation, or memory log row count exceeds the automatic detail threshold.'),
        'events': {'total_rows': len(trace), 'exported_rows': event_count,
                   'omitted_rows': len(trace) - event_count, 'artifact': 'events.csv' if 'simulation' in result else None,
                   'timeline_artifact': 'timeline.json' if 'simulation' in result else None},
        'tensors': {'total_rows': original_tensor_count, 'available_rows': len(tensors),
                    'prefiltered_rows': prefiltered_count, 'exported_rows': tensor_count,
                    'omitted_rows': original_tensor_count - tensor_count,
                    'artifact': tensor_artifact if 'memory_breakdown' in result else None,
                    'compression': 'gzip' if tensor_artifact.endswith('.gz') else None},
        'memory_log': {'total_rows': len(memory_log), 'exported_rows': 0 if compact else len(memory_log),
                       'omitted_rows': len(memory_log) if compact else 0,
                       'representation': 'omitted_raw_log' if compact else 'inline_result_json',
                       'summary_artifacts': ['memory_peaks.csv', 'component_memory.csv', 'state_memory.csv']},
        'preserved': 'All numerical metrics, communication/resource totals, logical parameters/states, and exact concurrent pool/cluster peak compositions are unchanged.',
        'event_filter': 'Model kernels, token/step completion markers, and allocating events of retained model tensors.',
        'tensor_filter': 'Only ephemeral state=buffers rows are omitted; initial buffers and every other state remain.',
        'reproduce_raw_detail': 'Rerun estimate/simulate with the same resolved configuration and --full-detail.',
    }
    exported = dict(result)
    exported['artifact_export'] = manifest
    if compact:
        if 'simulation' in result:
            simulation = dict(result['simulation'])
            simulation['trace'] = {'representation': 'filtered_artifact', **manifest['events']}
            simulation['memory_log'] = dict(manifest['memory_log'])
            exported['simulation'] = simulation
        if 'memory_breakdown' in result:
            breakdown = dict(result['memory_breakdown'])
            breakdown['tensors'] = {'representation': 'filtered_artifact', **manifest['tensors']}
            exported['memory_breakdown'] = breakdown
    return exported, trace, tensors, keep_event, keep_tensor


def _write_event_artifacts(directory: Path, rows: Iterable[dict]) -> int:
    """Write CSV and Perfetto in one pass, including for gzip-backed records."""
    count = 0
    with (directory / 'timeline.json').open('w', encoding='utf-8') as handle, \
            (directory / 'events.csv').open('w', newline='', encoding='utf-8-sig') as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=['id', 'start_s', 'end_s', 'duration_s', 'metadata'])
        writer.writeheader()
        handle.write('{"traceEvents":[')
        for row in rows:
            metadata = row.get('metadata', {})
            duration = row['end_s'] - row['start_s']
            writer.writerow({'id': row['id'], 'start_s': row['start_s'], 'end_s': row['end_s'],
                             'duration_s': duration,
                             'metadata': json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(',', ':'))})
            if count:
                handle.write(',')
            count += 1
            event = {'name': row['id'], 'ph': 'X', 'ts': row['start_s'] * 1e6,
                     'dur': duration * 1e6,
                     'pid': 1, 'tid': str(row.get('slots') or metadata.get('src') or 'events'),
                     'args': metadata}
            json.dump(event, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('],"displayTimeUnit":"ms"}')
    return count


class _RecordArray(list):
    """Let json.dump iterate a sized record stream without copying it to a list.

    The standard encoder's streaming path handles list subclasses through their
    public iteration and length methods. No JSON encoder internals are replaced.
    """
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)


def _write_result_json(path: Path, exported: dict):
    simulation = exported.get('simulation')
    if simulation:
        simulation = dict(simulation)
        for key in ('trace', 'memory_log'):
            records = simulation.get(key)
            if records is not None and not isinstance(records, (dict, list, tuple)):
                simulation[key] = _RecordArray(records)
        exported = {**exported, 'simulation': simulation}
    # Indentation dominates large repetitive pool/tensor tables. Keep small
    # reports human-readable and large reports losslessly compact.
    compact = bool(exported.get('artifact_export'))
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(exported, handle, ensure_ascii=False, allow_nan=False,
                      indent=None if compact else 2,
                      separators=(',', ':') if compact else None)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(result: dict, directory: str | Path, *, full_detail: bool = False) -> Path:
    """Stream report artifacts; automatically filter only large raw detail tables.

    Numerical summaries always preserve the complete simulation. ``full_detail``
    restores raw arrays and all CSV/timeline rows; the in-memory API result is
    never modified by either export mode.
    """
    if result.get('mode') == 'feedback':
        from .feedback_report import write_feedback_report
        return write_feedback_report(result, directory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exported, trace, tensors, keep_event, keep_tensor = _prepare_export(result, full_detail)
    if 'config' in result:
        snapshot = dict(result['config'])
        # The normalized snapshot already includes the calibration values.
        snapshot['calibration'] = None
        with (directory / 'config.resolved.json').open('w', encoding='utf-8') as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if 'estimate' in result:
        resources = result.get('simulation', {}).get('resources', {})
        _csv(directory / 'resources.csv', ['resource', 'unit', 'work_unit', 'capacity', 'work', 'utilization', 'busy_s', 'average_rate_at_lower_bound', 'required_average_rate_lower_bound'],
             ({'resource': key, **item, **resources.get(key, {})} for key, item in result.get('requirements', {}).get('per_resource', {}).items()))
        peaks = result.get('simulation', {}).get('memory_peaks_bytes', {})
        budget = result.get('model_metadata', {}).get('training_hbm', {})
        _csv(directory / 'memory.csv', ['pool', 'capacity_bytes', 'physical_capacity_bytes', 'reserved_bytes', 'resident_bytes', 'optimistic_peak_bytes', 'simulated_peak_bytes'], (
            {'pool': pool, 'resident_bytes': result['estimate']['resident_memory_bytes'].get(pool, 0),
             'capacity_bytes': result['memory_capacities_bytes'].get(pool),
             'physical_capacity_bytes': (budget.get('physical_bytes_per_npu') if pool.endswith('.hbm') else None) or result['memory_capacities_bytes'].get(pool),
             'reserved_bytes': budget.get('reserved_bytes_per_npu', 0) if pool.endswith('.hbm') else 0,
             'optimistic_peak_bytes': result['estimate']['optimistic_schedule_memory_peak_bytes'].get(pool, 0),
             'simulated_peak_bytes': peaks.get(pool)}
            for pool in sorted(set(result['estimate']['resident_memory_bytes']) | set(result['estimate']['optimistic_schedule_memory_peak_bytes']) | set(peaks))))
    if 'simulation' in result:
        event_count = _write_event_artifacts(directory, (row for row in trace if keep_event(row)))
        manifest = exported.get('artifact_export')
        if manifest:
            manifest['events'].update(exported_rows=event_count, omitted_rows=len(trace) - event_count)
            if manifest['compacted']:
                exported['simulation']['trace'].update(manifest['events'])
    if result.get('memory_breakdown'):
        breakdown = result['memory_breakdown']
        _csv(directory / 'component_memory.csv', [
            'pool', 'pool_kind', 'component', 'state', 'resident_bytes', 'bytes_at_pool_peak',
            'independent_component_state_peak_bytes', 'independent_component_peak_bytes',
            'pool_peak_bytes', 'pool_peak_time_s', 'peak_basis'], breakdown['component_state_rows'])
        _csv(directory / 'state_memory.csv', [
            'pool', 'pool_kind', 'state', 'resident_bytes', 'bytes_at_pool_peak',
            'independent_state_peak_bytes', 'pool_peak_bytes', 'pool_peak_time_s', 'peak_basis'], breakdown['state_rows'])
        tensor_artifact = exported.get('artifact_export', {}).get('tensors', {}).get('artifact') or 'tensors.csv'
        _csv(directory / tensor_artifact, [
            'allocation_id', 'pool', 'pool_kind', 'component', 'layer', 'tensor_name', 'tensor_members', 'state',
            'category', 'dtype', 'shape', 'bytes', 'is_initial', 'allocation_event', 'release_events',
            'observed', 'allocated_at_s', 'released_at_s', 'attribution', 'metadata'], (
                {**row, 'shape': json.dumps(row['shape']), 'tensor_members': json.dumps(row.get('tensor_members', [])),
                 'release_events': json.dumps(row['release_events']),
                 'metadata': json.dumps(row['metadata'], ensure_ascii=False)} for row in tensors if keep_tensor(row)))
        _csv(directory / 'memory_peaks.csv', [
            'pool', 'pool_kind', 'capacity_bytes', 'resident_bytes', 'peak_bytes', 'peak_basis', 'peak_time_s',
            'optimistic_peak_bytes', 'resident_exceeds_capacity', 'observed_peak_exceeds_capacity',
            'composition_at_peak', 'component_independent_peak_bytes', 'state_independent_peak_bytes'], (
                {**row, 'composition_at_peak': json.dumps(row['composition_at_peak'], ensure_ascii=False),
                 'component_independent_peak_bytes': json.dumps(row['component_independent_peak_bytes'], ensure_ascii=False),
                 'state_independent_peak_bytes': json.dumps(row['state_independent_peak_bytes'], ensure_ascii=False)}
                for row in breakdown['pools']))
        inventory_tensors = (breakdown.get('component_inventory') or {}).get('tensors', [])
        if inventory_tensors:
            preferred = ['name', 'component', 'layer', 'shape', 'elements', 'accounted_elements', 'active_elements',
                         'role', 'tp_shard', 'ep_shard', 'tp_max_shards', 'active_fraction', 'alias_of', 'formula']
            actual = {key for row in inventory_tensors for key in row}
            fields = [key for key in preferred if key in actual] + sorted(actual - set(preferred))
            _csv(directory / 'model_parameters.csv', fields, (
                {key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                 for key, value in row.items()} for row in inventory_tensors))
    if result.get('model_metadata', {}).get('training_task'):
        _csv(directory / 'training_task_stages.csv', ['iteration', 'stage', 'role', 'duration_s', 'start_event', 'end_event'],
             result.get('metrics', {}).get('training_task', {}).get('stages', []))
    write_latency_csv(result, directory)
    _write_result_json(directory / 'result.json', exported)
    (directory / 'report.md').write_text(markdown_report(exported), encoding='utf-8')
    return directory.resolve()

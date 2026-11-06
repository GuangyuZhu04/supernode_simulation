"""Adapt exact structured streaming to web artifacts without inventing traces."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .analysis import evaluate_targets
from .cache import BoundedCache
from .config import Config
from .host_memory import DEFAULT_LIMIT_BYTES, MIB
from .report import _csv, latency_markdown, write_latency_csv
from .streaming import run_streaming


DETAIL_NOTE = ('本次采用流式分块执行，保留完整性能、通信、逐池内存峰值和算子归因；'
               '未保留逐事件轨迹、逐张量生命周期和组件峰值构成。'
               '分块时间范围见 streaming_chunks.csv，不将分块伪装成单算子事件。')


def should_stream(config: Config, mode: str, memory_limit: int = DEFAULT_LIMIT_BYTES) -> bool:
    """Select retention strategy from workload size and host budget, not model name.

    This conservative graph-storage heuristic is NOT a simulated resource or
    latency estimate. A hard process budget remains necessary for both paths.
    """
    execution = config.execution
    features = execution.features
    if (mode != 'simulate' or config.workload.scenario != 'inference'
            or config.model.architecture is None or execution.dp != 1 or execution.pp != 1
            or features.megakernel.enabled or features.prefill_decode_disaggregation.enabled):
        return False
    rank_layers = config.hardware.npu_count * config.model.layers * max(config.workload.output_lengths)
    events_per_rank_layer = 24 + 4 * execution.tp + 2 * execution.ep
    return rank_layers * events_per_rank_layer * 2048 > memory_limit // 3


def run_for_web(config: Config, *, memory_limit: int = DEFAULT_LIMIT_BYTES, progress=None) -> dict:
    # Grow caches with the host budget instead of pinning powerful hosts to the
    # small-machine default. Entries allocate on demand and evict by byte budget.
    cache_bytes = max(MIB, memory_limit // 10)
    cache = BoundedCache(max_bytes=cache_bytes, max_entries=max(16, cache_bytes // (256 * 1024)))
    result = run_streaming(config, progress=progress, cache=cache)
    snapshot = config.model_dump()
    snapshot['calibration'] = None
    simulation = result['simulation']
    feasible = simulation['status'] == 'feasible'
    result.update(mode='simulate', schema_version='1.0', engine_version='0.1.0', config=snapshot,
        config_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        sources=snapshot['sources'],
        units={'memory': 'byte', 'bandwidth': 'byte/s', 'time': 'second', 'compute': 'FLOP'},
        host_memory={'limit_bytes': memory_limit, 'timing_cache_budget_bytes': cache_bytes},
        slo=evaluate_targets(config, result['metrics'], feasible))
    if feasible and not result['slo']['satisfied']:
        result['status'] = 'slo_failed'
    traffic = {'transport_payload_bytes': 0.0, 'classification_note':
               'Completed streaming chunks only; semantic and collective views must not be added.'}
    for key in ('physical_bytes_by_domain', 'payload_by_semantic_bytes', 'payload_by_collective_bytes'):
        traffic[key] = defaultdict(float)
    phases, elapsed = [], 0.0
    for phase, values in result['phase_metrics'].items():
        source = values['communication']
        traffic['transport_payload_bytes'] += source['transport_payload_bytes']
        for key in ('physical_bytes_by_domain', 'payload_by_semantic_bytes', 'payload_by_collective_bytes'):
            for name, amount in source[key].items():
                traffic[key][name] += amount
        phases.append({'phase': phase, 'start_s': elapsed, 'end_s': elapsed + values['duration_s'],
                       'event_count': values['completed_events']})
        elapsed += values['duration_s']
    # No full analytical DAG was built. Do not label actual streaming duration
    # or attempted chunks as a whole-workload lower bound / planned event count.
    result['estimate'] = {'duration_lower_bound_s': None, 'event_count': None,
                          'communication': traffic, 'memory_note': DETAIL_NOTE}
    result['communication_basis'] = 'completed_streaming_chunks'
    result['execution_summary'] = {'basis': 'completed_chunks', 'complete': feasible,
        'planned_events': None, 'completed_events': simulation['completed_events'],
        'trace_events': 0, 'phases': phases, 'grouped_phases': False}
    basis = 'simulated' if feasible else 'observed_partial'
    pools = []
    for pool, capacity in result['memory_capacities_bytes'].items():
        resident = result['resident_memory_bytes'].get(pool, 0)
        peak = simulation['memory_peaks_bytes'].get(pool, 0)
        pools.append({'pool': pool, 'pool_kind': pool.rsplit('.', 1)[-1], 'capacity_bytes': capacity,
                      'resident_bytes': resident, 'peak_bytes': peak, 'peak_basis': basis,
                      'peak_time_s': None, 'optimistic_peak_bytes': None, 'composition_at_peak': [],
                      'resident_exceeds_capacity': resident > capacity * (1 + 1e-12),
                      'observed_peak_exceeds_capacity': peak > capacity * (1 + 1e-12)})
    result['memory_breakdown'] = {'mode': basis, 'pools': pools, 'notes': [DETAIL_NOTE],
        'logical_component_state_bytes': result['model_metadata'].get('logical_component_state_bytes', {})}
    result['requirements'] = {'per_resource': simulation['resources']}
    result['artifact_export'] = {'mode': 'streaming', 'events_retained': False,
                                'tensor_lifetimes_retained': False, 'note': DETAIL_NOTE}
    return result


def write_streaming_report(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (('result.json', result), ('config.resolved.json', result['config'])):
        with (directory / name).open('w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    _csv(directory / 'resources.csv', ['resource', 'capacity', 'unit', 'kind', 'work', 'busy_s', 'utilization'],
         ({'resource': key, **values} for key, values in result['simulation']['resources'].items()))
    _csv(directory / 'memory_pools.csv', ['pool', 'pool_kind', 'capacity_bytes', 'resident_bytes',
         'peak_bytes', 'peak_basis', 'resident_exceeds_capacity', 'observed_peak_exceeds_capacity'],
         result['memory_breakdown']['pools'])
    _csv(directory / 'streaming_chunks.csv', ['id', 'phase', 'token_index', 'component', 'start_s',
         'end_s', 'duration_s', 'status', 'event_count', 'peak_hbm_bytes',
         'timing_cache_hit', 'component_macro_hit'], result['chunks'])
    write_latency_csv(result, directory)
    metrics = result['metrics']
    lines = ['# 流式仿真报告', '', f"状态：{result['status']}", '', DETAIL_NOTE, '',
             f"执行时间：{metrics.get('duration_s')} s", f"TTFT：{metrics.get('ttft_s')} s",
             f"TPOT：{metrics.get('tpot_s')} s", '',
             '通信仅汇总已完成分块；中断分块中的部分传输未计入，未执行阶段不会推算或补零。',
             '逐池峰值时刻可能不同，不可相加当作同时峰值；主机内存预算独立于模拟硬件容量。', '',
             '完整数值与输入见 result.json / config.resolved.json；资源、内存峰值及分块分别见对应 CSV。', '',
             '## 诊断', '', json.dumps(result['simulation']['diagnostics'], ensure_ascii=False), '',
             '## 假设', '', *('- ' + value for value in result['assumptions'])]
    lines.extend(latency_markdown(metrics))
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

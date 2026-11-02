"""Isolated numerical worker for the local web interface (no HTTP state)."""
from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path

from .analysis import run
from .config import Config
from .report import write_report
from .serving import ServingSpec, simulate_serving
from .serving_report import write_serving_report


MODEL_METADATA_KEYS = ('operator_trace', 'kernel_calibration', 'collective_calibration', 'fidelity', 'feature_plan')


def _phase(phase: str, message: str) -> None:
    print(json.dumps({'phase': phase, 'message': message}, ensure_ascii=False), flush=True)


def _bounded(value, limit: int = 40):
    """Keep optional explanatory metadata from dominating a UI response."""
    if isinstance(value, dict):
        return {str(k): _bounded(v, limit) for k, v in list(value.items())[:limit]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v, limit) for v in value[:limit]]
    if isinstance(value, str):
        return value[:2000]
    return value


def execution_summary(result: dict) -> dict:
    """Aggregate the full observed trace before sampling rows for display.

    Phase windows can overlap; event durations are summed service spans, not
    elapsed execution time. Events without phase metadata remain unclassified.
    """
    simulation = result.get('simulation', {})
    trace = simulation.get('trace', [])
    phases = {}
    overflow = None
    phase_limit = 40
    for event in trace:
        phase = str(event.get('metadata', {}).get('phase') or 'unclassified')
        if phase not in phases and len(phases) >= phase_limit - 1:
            if overflow is None:
                overflow = {'phase': 'other_phases', 'grouped': True, 'event_count': 0,
                            'start_s': None, 'end_s': None, 'event_duration_sum_s': 0.0}
            row = overflow
        else:
            if phase not in phases:
                phases[phase] = {'phase': phase[:2000], 'event_count': 0,
                    'start_s': None, 'end_s': None, 'event_duration_sum_s': 0.0}
            row = phases[phase]
        row['event_count'] += 1
        start, end = event.get('start_s'), event.get('end_s')
        if start is not None:
            row['start_s'] = start if row['start_s'] is None else min(row['start_s'], start)
        if end is not None:
            row['end_s'] = end if row['end_s'] is None else max(row['end_s'], end)
        row['event_duration_sum_s'] += event.get('duration_s') or 0.0
    rows = list(phases.values()) + ([overflow] if overflow is not None else [])
    rows.sort(key=lambda row: (row['start_s'] is None, row['start_s'] or 0, row['phase']))
    estimate_only = result.get('mode') == 'estimate'
    return {
        'basis': 'estimate_only' if estimate_only else 'completed_trace' if simulation else 'unavailable',
        'complete': None if estimate_only or not simulation else simulation.get('status') == 'feasible',
        'planned_events': result.get('estimate', {}).get('event_count'),
        'completed_events': simulation.get('completed_events') if simulation else None,
        'trace_events': len(trace), 'phases': rows,
        'phase_limit': phase_limit, 'grouped_phases': overflow is not None,
    }


def compact_result(result: dict) -> dict:
    """A bounded display view; all exact tables remain in report artifacts."""
    estimate = result.get('estimate', {})
    simulation = result.get('simulation', {})
    memory = result.get('memory_breakdown', {})
    pools = memory.get('pools', [])
    selected_pools = sorted(pools, key=lambda p: p['peak_bytes'] / (p.get('capacity_bytes') or 1), reverse=True)[:240]
    pool_keys = ('pool', 'pool_kind', 'capacity_bytes', 'resident_bytes', 'peak_bytes', 'peak_basis',
                 'peak_time_s', 'optimistic_peak_bytes', 'composition_at_peak', 'resident_exceeds_capacity',
                 'observed_peak_exceeds_capacity')
    resources = [{'resource': name, **values, **simulation.get('resources', {}).get(name, {})}
                 for name, values in result.get('requirements', {}).get('per_resource', {}).items()]
    resources.sort(key=lambda r: (r.get('utilization', 0), r.get('average_rate_at_lower_bound', 0) / (r.get('capacity') or 1)), reverse=True)
    trace = simulation.get('trace', [])
    indices = (range(len(trace)) if len(trace) <= 300 else
               sorted({round(i * (len(trace) - 1) / 299) for i in range(300)}))
    metadata_keys = ('kind', 'category', 'operation', 'phase', 'layer', 'rank', 'semantic', 'collective',
                     'payload_bytes', 'physical_bytes_by_domain', 'component', 'replica', 'token',
                     'token_index', 'step', 'microbatch', 'src', 'dst', 'kernel', 'direction',
                     'round_index', 'pd_role', 'trace_id', 'original_kernel', 'trace_parent_event',
                     'operation_index', 'matmul', 'flops', 'workload_shape', 'operator_trace_completion')
    events = []
    for i in indices:
        row = trace[i]
        events.append({**{k: row.get(k) for k in ('id', 'start_s', 'end_s', 'duration_s')},
                       'slots': row.get('slots', [])[:16],
                       'metadata': _bounded({k: v for k, v in row.get('metadata', {}).items() if k in metadata_keys})})
    model_metadata = {key: _bounded(result['model_metadata'][key]) for key in MODEL_METADATA_KEYS
                      if key in result.get('model_metadata', {})}
    training = result.get('metrics', {}).get('training_accounting')
    metrics = {k: v for k, v in result.get('metrics', {}).items()
               if k == 'latency_breakdown' or not isinstance(v, (list, dict))}
    if training is not None:
        metrics['training_accounting'] = _bounded(training)
    return {
        **{k: result.get(k) for k in ('status', 'mode', 'config', 'config_sha256', 'units', 'slo', 'engine_version')},
        'metrics': metrics, 'model_metadata': model_metadata,
        'estimate': {k: estimate.get(k) for k in ('duration_lower_bound_s', 'event_count', 'communication', 'memory_note')},
        'communication': estimate.get('communication', {}),
        'memory': {'mode': memory.get('mode'), 'pools': [{k: p.get(k) for k in pool_keys} for p in selected_pools],
                   'pool_count': len(pools), 'sampled': len(pools) > len(selected_pools),
                   'logical_component_state_bytes': memory.get('logical_component_state_bytes', {}),
                   'cluster': memory.get('cluster', {}), 'notes': memory.get('notes', [])},
        'resources': resources[:160], 'resource_count': len(resources),
        'execution_summary': execution_summary(result),
        'trace': {'events': events, 'total_events': len(trace), 'sampled': len(trace) > len(events)},
        'simulation': {'duration_s': simulation.get('duration_s'), 'completed_events': simulation.get('completed_events')},
        'diagnostics': _bounded(simulation.get('diagnostics', [])),
        'sources': result.get('sources', {}), 'assumptions': result.get('assumptions', [])[:80],
        'display_limits': {'pools': 240, 'resources': 160, 'events': 300, 'metadata_items': 40,
                           'training_accounting_truncated': training is not None and metrics['training_accounting'] != training,
                           'model_metadata_truncated': {key: value != result['model_metadata'][key]
                                                        for key, value in model_metadata.items()},
                           'note': 'Display tables may be sampled; download report artifacts for complete numerical details.'},
    }


def compact_serving_result(result: dict) -> dict:
    """Bound display rows without recomputing population metrics from samples."""
    requests, batches, executions = [], [], []
    for source in result['requests'][:200]:
        row = _bounded({key: value for key, value in source.items()
                        if key not in ('token_times_s', 'token_times_relative_s')})
        for key in ('token_times_s', 'token_times_relative_s'):
            row[key] = source[key][:128]
        row['token_times_count'] = len(source['token_times_relative_s'])
        row['token_times_truncated'] = row['token_times_count'] > 128
        requests.append(row)
    for source in result['batches'][:200]:
        row = _bounded({key: value for key, value in source.items() if key != 'request_ids'})
        row['request_ids'] = [str(value)[:2000] for value in source['request_ids'][:128]]
        row['request_count'] = len(source['request_ids'])
        row['request_ids_truncated'] = row['request_count'] > 128
        row['memory_pools_count'] = len(source.get('memory_peaks_bytes', {}))
        row['memory_pools_truncated'] = row['memory_pools_count'] > 40
        batches.append(row)
    for source in result['executions'][:40]:
        metadata = {key: _bounded(source['model_metadata'][key]) for key in MODEL_METADATA_KEYS
                    if key in source.get('model_metadata', {})}
        row = _bounded({key: value for key, value in source.items()
                        if key not in ('model_metadata', 'input_lengths', 'output_lengths')})
        row['model_metadata'] = metadata
        row['model_metadata_truncated'] = {key: value != source['model_metadata'][key] for key, value in metadata.items()}
        for key in ('input_lengths', 'output_lengths'):
            row[key] = source[key][:128]
        row['batch_size'] = len(source['input_lengths'])
        row['lengths_truncated'] = row['batch_size'] > 128
        executions.append(row)
    return {
        'mode': 'serve', 'engine_mode': result['mode'], 'status': result['status'],
        'config': result['config'],
        'serving': {key: value for key, value in result['serving'].items() if key != 'requests'} |
                   {'request_count': len(result['requests'])},
        **{key: result[key] for key in ('metrics', 'throughput', 'goodput', 'hardware',
                                       'clock_origin_s', 'time_window_s', 'time_window_relative_s', 'limitations')},
        'requests': requests, 'batches': batches, 'executions': executions,
        'provenance': _bounded(result['provenance']),
        'display_limits': {'requests': 200, 'batches': 200, 'executions': 40, 'tokens_per_request': 128,
            'request_ids_per_batch': 128, 'metadata_items': 40,
            'request_count': len(result['requests']), 'requests_truncated': len(result['requests']) > 200,
            'batch_count': len(result['batches']), 'batches_truncated': len(result['batches']) > 200,
            'execution_count': len(result['executions']), 'executions_truncated': len(result['executions']) > 40,
            'note': 'Tables show bounded prefixes only. Metrics and quantiles use the complete request population. Relative timestamps are authoritative; download result.json for all rows and token times.'},
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[1] not in ('simulate', 'estimate', 'serve'):
        raise ValueError('worker requires an output directory and simulate/estimate/serve mode')
    directory, mode = Path(args[0]), args[1]
    # Match the CLI's performance policy for acyclic, ID-linked event graphs.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        _phase('validating', '正在校验已保存的输入配置')
        config = Config.model_validate_json((directory / 'input.config.json').read_text(encoding='utf-8'))
        if config.calibration is not None:
            raise ValueError('Web jobs require resolved configuration without calibration paths')
        if mode == 'serve':
            serving = ServingSpec.model_validate_json((directory / 'input.serving.json').read_text(encoding='utf-8'))
            _phase('serving', '正在按请求到达与 FCFS 成批策略调用原仿真引擎')
            result = simulate_serving(config, serving)
            _phase('exporting', '服务仿真完成，正在导出请求、批次与 token trace')
            write_serving_report(result, directory)
            compact = compact_serving_result(result)
        else:
            _phase('simulating' if mode == 'simulate' else 'estimating',
                   '正在构建执行图、计算资源下界并运行离散事件仿真' if mode == 'simulate' else '正在构建执行图并计算解析下界')
            result = run(config, mode, full_detail=False)
            _phase('exporting', '数值计算完成，正在导出报告和完整结果表')
            write_report(result, directory, full_detail=False)
            compact = compact_result(result)
        _phase('summarizing', '正在生成网页可视化摘要')
        temporary = directory / 'ui.result.json.tmp'
        temporary.write_text(json.dumps(compact, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(directory / 'ui.result.json')
        print(json.dumps({'phase': 'completed', 'message': '报告与可视化已生成', 'result_status': result['status']}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'phase': 'failed', 'message': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False), flush=True)
        traceback.print_exc()
        return 1
    finally:
        if gc_was_enabled:
            gc.enable()


if __name__ == '__main__':
    raise SystemExit(main())

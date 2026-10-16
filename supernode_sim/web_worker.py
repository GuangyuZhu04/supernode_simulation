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
                     'payload_bytes', 'physical_bytes_by_domain', 'component', 'replica', 'token')
    events = []
    for i in indices:
        row = trace[i]
        events.append({**{k: row.get(k) for k in ('id', 'start_s', 'end_s', 'duration_s')},
                       'slots': row.get('slots', [])[:16],
                       'metadata': _bounded({k: v for k, v in row.get('metadata', {}).items() if k in metadata_keys})})
    return {
        **{k: result.get(k) for k in ('status', 'mode', 'config', 'config_sha256', 'units', 'slo', 'engine_version')},
        'metrics': {k: v for k, v in result.get('metrics', {}).items()
                    if k == 'latency_breakdown' or not isinstance(v, (list, dict))},
        'estimate': {k: estimate.get(k) for k in ('duration_lower_bound_s', 'event_count', 'communication', 'memory_note')},
        'communication': estimate.get('communication', {}),
        'memory': {'mode': memory.get('mode'), 'pools': [{k: p.get(k) for k in pool_keys} for p in selected_pools],
                   'pool_count': len(pools), 'sampled': len(pools) > len(selected_pools),
                   'logical_component_state_bytes': memory.get('logical_component_state_bytes', {}),
                   'cluster': memory.get('cluster', {}), 'notes': memory.get('notes', [])},
        'resources': resources[:160], 'resource_count': len(resources),
        'trace': {'events': events, 'total_events': len(trace), 'sampled': len(trace) > len(events)},
        'simulation': {'duration_s': simulation.get('duration_s'), 'completed_events': simulation.get('completed_events')},
        'diagnostics': _bounded(simulation.get('diagnostics', [])),
        'sources': result.get('sources', {}), 'assumptions': result.get('assumptions', [])[:80],
        'display_limits': {'pools': 240, 'resources': 160, 'events': 300, 'metadata_items': 40,
                           'note': 'Display tables may be sampled; download report artifacts for complete numerical details.'},
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[1] not in ('simulate', 'estimate'):
        raise ValueError('worker requires an output directory and simulate/estimate mode')
    directory, mode = Path(args[0]), args[1]
    # Match the CLI's performance policy for acyclic, ID-linked event graphs.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        _phase('validating', '正在校验已保存的输入配置')
        config = Config.model_validate_json((directory / 'input.config.json').read_text(encoding='utf-8'))
        if config.calibration is not None:
            raise ValueError('Web jobs require resolved configuration without calibration paths')
        _phase('simulating' if mode == 'simulate' else 'estimating',
               '正在构建执行图、计算资源下界并运行离散事件仿真' if mode == 'simulate' else '正在构建执行图并计算解析下界')
        result = run(config, mode, full_detail=False)
        _phase('exporting', '数值计算完成，正在导出报告和完整结果表')
        write_report(result, directory, full_detail=False)
        _phase('summarizing', '正在生成网页可视化摘要')
        temporary = directory / 'ui.result.json.tmp'
        temporary.write_text(json.dumps(compact_result(result), ensure_ascii=False, allow_nan=False), encoding='utf-8')
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

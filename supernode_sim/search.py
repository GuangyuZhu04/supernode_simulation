"""Finite, auditable candidate searches; no claim of global optimality."""
from __future__ import annotations

import copy
import itertools
import math

from .analysis import run
from .config import Config
from .parallel import ordered_map, worker_count


def set_field(data: dict, path: str, value) -> None:
    parts = path.split('.')
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f'Unknown configuration path: {path}')
        node = node[part]
    if parts[-1] not in node:
        raise ValueError(f'Unknown configuration path: {path}')
    node[parts[-1]] = value
    if path in ('workload.input_length', 'workload.batch_size'):
        data['workload']['input_lengths'] = []
    if path in ('workload.output_length', 'workload.batch_size'):
        data['workload']['output_lengths'] = []


def compact_result(result: dict) -> dict:
    compact = {key: value for key, value in result.items() if key not in ('simulation', 'requirements')} | {
        'diagnostics': result.get('simulation', {}).get('diagnostics', []),
        'memory_peaks_bytes': result.get('simulation', {}).get('memory_peaks_bytes', {}),
    }
    if compact.get('memory_breakdown'):
        breakdown = dict(compact['memory_breakdown'])
        tensors = breakdown.pop('tensors', [])
        breakdown['tensor_allocation_count'] = len(tensors)
        breakdown['detail_note'] = 'Candidate summaries retain all numerical component/peak totals. Run this candidate config separately to export physical tensor lifetimes.'
        compact['memory_breakdown'] = breakdown
    return compact


def _sweep_candidate(arguments) -> dict:
    base, changes, mode = arguments
    candidate = copy.deepcopy(base)
    try:
        for path, value in changes.items():
            set_field(candidate, path, value)
        result = compact_result(run(Config.model_validate(candidate), mode, full_detail=False))
    except (ValueError, KeyError) as exc:
        result = {'status': 'invalid_config', 'diagnostics': [str(exc)], 'config': candidate}
    return {'parameters': changes, 'result': result}


def sweep(config: Config, axes: dict[str, list], mode: str = 'simulate', *, workers: int = 1) -> dict:
    if not axes or any(not isinstance(v, list) or not v for v in axes.values()):
        raise ValueError('sweep axes must map dotted config paths to nonempty lists')
    count = math.prod(len(v) for v in axes.values())
    if count > config.targets.max_candidates:
        raise ValueError(f'{count} combinations exceed max_candidates={config.targets.max_candidates}')
    base = config.model_dump()
    arguments = ((base, dict(zip(axes, values)), mode) for values in itertools.product(*axes.values()))
    rows = list(ordered_map(_sweep_candidate, arguments, workers=workers, count=count))
    return {'schema_version': '1.0', 'mode': 'sweep', 'status': 'completed', 'axes': axes, 'candidates': rows,
            'assumptions': ['Each combination is independently validated. Invalid combinations are retained with diagnostics.']}


def _payload_summary(result: dict) -> dict:
    return {key: result[key] for key in ('status', 'config', 'metrics', 'slo') if key in result} | {
        'diagnostics': result.get('simulation', {}).get('diagnostics', []),
        'duration_lower_bound_s': result.get('estimate', {}).get('duration_lower_bound_s'),
        'memory_peaks_bytes': result.get('simulation', {}).get('memory_peaks_bytes', {}),
    }


def _su_candidate(arguments) -> dict:
    base, npu_count, su_size, layout = arguments
    data = copy.deepcopy(base)
    data['hardware'].update(npu_count=npu_count, su_size=su_size)
    data['execution'].update(layout)
    row = {'npu_count': npu_count, 'su_size': su_size, 'domains': npu_count // su_size, 'parallelism': layout}
    try:
        spec = Config.model_validate(data)
        estimate = run(spec, 'estimate', full_detail=False)
        residents = estimate['estimate']['resident_memory_bytes']
        capacities = {f'npu{i}.hbm': spec.hardware.npu.hbm_bytes for i in range(npu_count)}
        capacities.update({f'cpu{i}.dram': spec.hardware.cpu.dram_bytes for i in range(math.ceil(npu_count / spec.hardware.npus_per_host))})
        capacities.update({'remote0.memory': spec.hardware.remote.capacity_bytes, 'ssd0.memory': spec.hardware.ssd.capacity_bytes})
        violations = [{'pool': pool, 'required': amount, 'capacity': capacities[pool]} for pool, amount in residents.items() if pool in capacities and amount > capacities[pool]]
        if violations:
            row['result'] = {'status': 'infeasible', 'pruned': 'resident_memory_lower_bound', 'diagnostics': violations, 'config': data}
        else:
            deadline = estimate['requirements']['target_window_s']
            lower = estimate['estimate']['duration_lower_bound_s']
            if deadline is not None and lower > deadline:
                row['result'] = {'status': 'slo_failed', 'pruned': 'performance_lower_bound',
                                 'diagnostics': [{'required_window_s': deadline, 'duration_lower_bound_s': lower}], 'config': data}
            else:
                row['result'] = _payload_summary(run(spec, 'simulate', full_detail=False))
    except (ValueError, KeyError) as exc:
        row['result'] = {'status': 'invalid_config', 'diagnostics': [str(exc)], 'config': data}
    return row


def size_su(config: Config, mode: str = 'both', *, workers: int = 1) -> dict:
    if mode not in ('single', 'multi', 'both'):
        raise ValueError('SU mode must be single, multi or both')
    worker_count(workers, 0)  # Validate even for empty/missing-input searches.
    base_config = config.model_dump()
    target = config.targets
    pd_enabled = config.execution.features.prefill_decode_disaggregation.enabled
    deployment_copies = 2 if pd_enabled else 1
    modes = ['single', 'multi'] if mode == 'both' else [mode]
    result = {'schema_version': '1.0', 'mode': 'size-su', 'status': 'completed', 'searches': {},
              'assumptions': ['Minimum is over the complete declared candidate set and fixed scheduler only.',
                              ('PD search retains DP=PP=1 and equal TP/EP layouts in both pools; supply parallel_candidates to vary the per-pool layout. Physical pool mappings remain explicit and are revalidated.' if pd_enabled else
                               'Without parallel_candidates, TP/PP/EP/FSDP are retained and DP is derived from candidate device count.'),
                              'batch_size is per DP replica; changing DP changes aggregate workload.']}
    total_evaluated = 0
    for current_mode in modes:
        if current_mode == 'multi' and target.total_npus is None and target.max_total_npus is None:
            result['searches'][current_mode] = {'status': 'missing_input', 'missing_fields': ['targets.total_npus or targets.max_total_npus']}
            continue
        counts = sorted(set(target.candidate_npu_counts or [target.total_npus or config.hardware.npu_count]))
        if current_mode == 'multi':
            counts = [n for n in counts if (target.total_npus is None or n == target.total_npus) and (target.max_total_npus is None or n <= target.max_total_npus)]
        candidates = []
        for npu_count in counts:
            sizes = [npu_count] if current_mode == 'single' else sorted(set(target.candidate_su_sizes or [config.hardware.su_size]))
            for su_size in sizes:
                if npu_count % su_size or su_size > npu_count:
                    continue
                layouts = [p.model_dump() for p in target.parallel_candidates]
                if not layouts:
                    e = config.execution
                    base = deployment_copies * e.tp * e.pp * e.ep
                    layouts = [{'tp': e.tp, 'pp': e.pp, 'ep': e.ep, 'dp': 1 if pd_enabled else npu_count // base,
                                'fsdp': e.fsdp, 'rank_order': e.rank_order if npu_count == config.hardware.npu_count else []}] if npu_count % base == 0 else []
                for layout in layouts:
                    if deployment_copies * layout['tp'] * layout['pp'] * layout['ep'] * layout['dp'] == npu_count:
                        candidates.append((npu_count, su_size, layout))
        if total_evaluated + len(candidates) > target.max_candidates:
            raise ValueError('SU candidate set exceeds max_candidates; narrow candidates or raise the explicit limit')
        total_evaluated += len(candidates)
        arguments = ((base_config, npu_count, su_size, layout) for npu_count, su_size, layout in candidates)
        rows = list(ordered_map(_su_candidate, arguments, workers=workers, count=len(candidates)))
        feasible = [r for r in rows if r['result']['status'] == 'feasible']
        order = lambda r: (r['su_size'], r['npu_count'], r['result'].get('metrics', {}).get('duration_s', float('inf')))
        feasible.sort(key=order)
        minimum = feasible[0] if feasible else None
        result['searches'][current_mode] = {
            'status': 'feasible' if minimum else ('no_candidates' if not rows else 'no_feasible_candidate'),
            'minimum_su_size': minimum['su_size'] if minimum else None,
            'best': minimum, 'candidates': sorted(rows, key=order), 'candidate_count': len(rows),
            'slo_specified': any(getattr(target, k) is not None for k in ('throughput_tokens_s', 'ttft_s', 'tpot_s', 'step_time_s')),
        }
    if any(v['status'] == 'missing_input' for v in result['searches'].values()):
        result['status'] = 'missing_input'
    return result

"""Finite, auditable candidate searches; no claim of global optimality."""
from __future__ import annotations

import copy
import itertools
import math

from .analysis import compile_graph, graph_estimate, run, run_compiled, workload_target_window
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
    base, npu_count, su_size, layout, placement = arguments
    data = copy.deepcopy(base)
    data['hardware'].update(npu_count=npu_count, su_size=su_size)
    data['execution'].update(layout)
    row = {'npu_count': npu_count, 'su_size': su_size, 'domains': npu_count // su_size, 'parallelism': layout}
    try:
        if placement:
            row['placement'] = placement
        batch = data['targets'].get('training_global_batch_size')
        if batch is not None:
            per_micro = data['workload']['batch_size'] * layout['dp']
            if batch % per_micro:
                raise ValueError('training_global_batch_size must be divisible by batch_size * candidate DP')
            data['workload']['gradient_accumulation'] = batch // per_micro
        spec = Config.model_validate(data)
        graph = compile_graph(spec)
        residents = {}
        for alloc in graph.initial_allocations:
            residents[alloc.pool] = residents.get(alloc.pool, 0.0) + alloc.size_bytes
        capacities = graph.memory_caps
        if placement:
            row['dp_domains'] = graph.metadata['dp_domains']
            row['training_hbm'] = graph.metadata['training_hbm']
            row['samples_per_accumulation'] = spec.workload.batch_size * layout['dp'] * spec.workload.gradient_accumulation
        violations = [{'pool': pool, 'required': amount, 'capacity': capacities[pool]} for pool, amount in residents.items()
                      if amount > capacities[pool] * (1 + 1e-12)]
        if violations:
            row['result'] = {'status': 'infeasible', 'pruned': 'resident_memory_lower_bound', 'diagnostics': violations, 'config': data}
        else:
            estimate = graph_estimate(graph)
            deadline = workload_target_window(spec, graph)
            lower = estimate['duration_lower_bound_s']
            if deadline is not None and lower > deadline:
                row['result'] = {'status': 'slo_failed', 'pruned': 'performance_lower_bound',
                                 'diagnostics': [{'required_window_s': deadline, 'duration_lower_bound_s': lower}], 'config': data}
            else:
                row['result'] = _payload_summary(run_compiled(spec, graph, estimate, 'simulate', full_detail=False))
    except (ValueError, KeyError) as exc:
        row['result'] = {'status': 'invalid_config', 'diagnostics': [str(exc)], 'config': data}
    return row


def _training_placements(config, layout, su_size):
    """Search a flat baseline and SU-sized domains; honor explicit sizes and pinning."""
    from .training_schedule import domain_rank_order
    dp, fsdp = layout['dp'], layout['fsdp']
    mp = layout['tp'] * layout['ep'] * layout['pp']
    identity = list(range(mp * dp))
    pinned = layout['rank_order'] or (config.execution.rank_order if len(config.execution.rank_order) == len(identity) else [])
    sizes = config.targets.candidate_dp_domain_sizes
    if layout.get('dp_domain_size') is not None:
        sizes = [layout['dp_domain_size']]
    elif not sizes and config.execution.dp_domain_size is not None:
        sizes = [config.execution.dp_domain_size]
    automatic = not sizes
    if automatic:
        local = {math.gcd(dp, su_size)}
        if su_size % mp == 0:
            local.add(math.gcd(dp, su_size // mp))
        sizes = [None, *sorted(size for size in local if size > 1 and size % fsdp == 0)]
    seen = set()
    for size in dict.fromkeys(sizes):
        orders = [('explicit' if pinned else 'model_parallel_first', pinned)]
        if not pinned and size and dp % size == 0 and size % fsdp == 0:
            orders.append(('dp_domain_first', domain_rank_order(layout, size)))
        for label, order in orders:
            # In a one-domain or one-replica-per-domain mesh the replica
            # collective is flat. Avoid duplicate automatic simulations.
            effective_size = None if automatic and size in (dp, fsdp) else size
            key = effective_size, tuple(order or identity)
            if key not in seen:
                seen.add(key)
                yield {**layout, 'rank_order': order, 'dp_domain_size': size}, label


def size_su(config: Config, mode: str = 'both', *, workers: int = 1) -> dict:
    if mode not in ('single', 'multi', 'both'):
        raise ValueError('SU mode must be single, multi or both')
    worker_count(workers, 0)  # Validate even for empty/missing-input searches.
    base_config = config.model_dump()
    target = config.targets
    training = config.workload.scenario == 'training'
    pd_enabled = config.execution.features.prefill_decode_disaggregation.enabled
    deployment_copies = 2 if pd_enabled else 1
    modes = ['single', 'multi'] if mode == 'both' else [mode]
    result = {'schema_version': '1.0', 'mode': 'size-su', 'status': 'completed', 'searches': {},
              'assumptions': ['Minimum is over the complete declared candidate set and fixed scheduler only.',
                              ('PD search retains DP=PP=1 and equal TP/EP layouts in both pools; supply parallel_candidates to vary the per-pool layout. Physical pool mappings remain explicit and are revalidated.' if pd_enabled else
                               'Without parallel_candidates, TP/PP/EP/FSDP are retained and DP is derived from candidate device count.'),
                              'batch_size is per DP replica; changing DP changes aggregate workload.']}
    if training:
        result['assumptions'].append(
            'Training tests declared or SU-derived DP domain sizes and physical placements. highest_throughput ranks feasible candidates by aggregate actor tokens/s.')
        if target.training_global_batch_size:
            result['assumptions'][-2] = 'Training holds global batch fixed by adjusting gradient accumulation for candidate DP.'
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
                        variants = _training_placements(config, layout, su_size) if training else [(layout, None)]
                        for variant, placement in variants:
                            candidates.append((npu_count, su_size, variant, placement))
                            if total_evaluated + len(candidates) > target.max_candidates:
                                raise ValueError('SU candidate set exceeds max_candidates; narrow candidates or raise the explicit limit')
        total_evaluated += len(candidates)
        arguments = ((base_config, *candidate) for candidate in candidates)
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
        if training:
            search = result['searches'][current_mode]
            feasible_indices = [i for i, row in enumerate(search['candidates']) if row['result']['status'] == 'feasible']
            fastest = max(feasible_indices, key=lambda i: search['candidates'][i]['result']['metrics']['throughput_tokens_s'],
                          default=None)
            search['highest_throughput'] = None if fastest is None else {
                'candidate_index': fastest, 'tokens_s': search['candidates'][fastest]['result']['metrics']['throughput_tokens_s']}
    if any(v['status'] == 'missing_input' for v in result['searches'].values()):
        result['status'] = 'missing_input'
    return result

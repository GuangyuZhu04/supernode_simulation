"""Shared graph compilation, analytical bounds, and execution-level metrics."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from .config import Config
from .core import Allocation, Graph
from .engine import simulate as simulate_graph
from .topology import Topology
from .calibration import apply_kernel_profiles
from .components import inventory_summary
from .memory_breakdown import build_memory_breakdown


def compile_graph(config: Config | dict) -> Graph:
    spec = config if isinstance(config, Config) else Config.model_validate(config)
    data = spec.model_dump()
    topology = Topology(data['hardware'], data['execution']['precision'])
    if spec.model.architecture is not None:
        from .structured import build_structured
        graph = build_structured(data, topology)
    elif spec.workload.scenario == 'training':
        from .training import build_training
        graph = build_training(data, topology)
    else:
        from .inference import build_inference
        graph = build_inference(data, topology)
    if spec.model.architecture is None:
        graph.metadata['component_inventory'] = inventory_summary(data['model'])
        # Legacy training explicitly implements Adam, so its two same-size
        # states can be split truthfully without fabricating model components.
        initial = []
        for allocation in graph.initial_allocations:
            if allocation.category == 'optimizer_states':
                for state in ('optimizer_momentum', 'optimizer_variance'):
                    initial.append(Allocation(allocation.id + '.' + state, allocation.pool, allocation.size_bytes / 2,
                        state, allocation.dtype, metadata={**allocation.metadata, 'component': 'Unassigned', 'state': state}))
            else:
                initial.append(allocation)
        graph.initial_allocations = initial
    apply_kernel_profiles(graph, data)
    assumptions = graph.metadata.setdefault('assumptions', [])
    assumptions.extend(graph.metadata.get('approximations', []))
    assumptions.extend(spec.assumptions)
    if spec.execution.precision.communication in ('fp8', 'int8'):
        assumptions.append('FP8/INT8 activation communication assumes shared static quantization scales; dynamic per-message scale exchange is not modeled. Weight/KV scale storage and offload metadata are counted separately.')
    graph.metadata['parameterization'] = ('Shape-derived component tensors with analytical execution kernels; no measured allocator/kernel trace.'
        if spec.model.architecture else 'Declared parameter budgets with coarse layer kernels; component attribution unavailable without model.architecture.')
    graph.metadata['fidelity'] = {
        'parameter_capacity': 'shape_derived' if spec.model.architecture else 'declared_budget',
        'activation_capacity': 'analytical_kernel_layout',
        'hardware_performance': 'uncalibrated' if not spec.hardware.kernel_profiles else 'partially_kernel_calibrated',
        'distributed_muon': 'not_implemented' if spec.execution.optimizer.kind == 'muon' else 'not_applicable',
    }
    return graph


def graph_estimate(graph: Graph) -> dict:
    """Necessary lower bounds and an explicitly optimistic memory schedule.

    Memory from earliest starts ignores contention and slots; it is NOT a
    universal capacity lower bound and is never used for rejecting candidates.
    """
    events = {event.id: event for event in graph.events}
    if len(events) != len(graph.events):
        raise ValueError('duplicate event IDs')
    for resource in graph.resources.values():
        if not math.isfinite(resource.capacity) or resource.capacity <= 0:
            raise ValueError(f'Invalid resource capacity: {resource.id}')
    for pool, capacity in graph.memory_caps.items():
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f'Invalid memory capacity: {pool}')
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {}
    work: dict[str, float] = defaultdict(float)
    initial: dict[str, float] = defaultdict(float)
    categories: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    allocs = {}
    for allocation in graph.initial_allocations:
        if allocation.id in allocs or allocation.pool not in graph.memory_caps or not math.isfinite(allocation.size_bytes) or allocation.size_bytes < 0:
            raise ValueError(f'Invalid initial allocation: {allocation.id}')
        initial[allocation.pool] += allocation.size_bytes
        categories[allocation.pool][allocation.category] += allocation.size_bytes
        allocs[allocation.id] = allocation
    for event in graph.events:
        if not math.isfinite(event.startup_s) or event.startup_s < 0:
            raise ValueError(f'Invalid startup latency: {event.id}')
        indegree[event.id] = len(set(event.deps))
        for dep in set(event.deps):
            if dep not in events:
                raise ValueError(f'{event.id}: unknown dependency {dep}')
            children[dep].append(event.id)
        for key, amount in event.work.items():
            if key not in graph.resources or not math.isfinite(amount) or amount < 0:
                raise ValueError(f'{event.id}: invalid work resource {key}')
            work[key] += amount
        for allocation in event.allocs:
            if allocation.pool not in graph.memory_caps or not math.isfinite(allocation.size_bytes) or allocation.size_bytes < 0:
                raise ValueError(f'Invalid event allocation: {allocation.id}')
            if allocation.id in allocs:
                raise ValueError(f'duplicate allocation ID {allocation.id}')
            allocs[allocation.id] = allocation
    ready = [key for key, value in indegree.items() if value == 0]
    heapq.heapify(ready)
    ends = {}
    while ready:
        eid = heapq.heappop(ready)
        event = events[eid]
        start = max((ends[d] for d in event.deps), default=0.0)
        duration = event.startup_s + max((v / graph.resources[r].capacity for r, v in event.work.items()), default=0.0)
        ends[eid] = start + duration
        for child in children.pop(eid, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ends) != len(events):
        raise ValueError('cyclic execution dependency graph')
    del events, children, indegree, ready
    changes = []
    for event in graph.events:
        # Recompute starts from dependency ends instead of keeping a second
        # full event timing table alive while constructing memory changes.
        start = max((ends[dep] for dep in event.deps), default=0.0) if event.allocs else 0.0
        for allocation in event.allocs:
            changes.append((start, 0, allocation.id, allocation.pool, allocation.size_bytes))
        for aid in event.releases:
            if aid not in allocs:
                raise ValueError(f'{event.id}: release of unknown allocation {aid}')
            a = allocs[aid]
            changes.append((ends[event.id], 1, aid, a.pool, -a.size_bytes))
    del allocs
    current, optimistic_peak = dict(initial), dict(initial)
    changes.sort()
    for _, _, _, pool, delta in changes:
        current[pool] = current.get(pool, 0) + delta
        optimistic_peak[pool] = max(optimistic_peak.get(pool, 0), current[pool])
    del changes
    resource_bounds = {r: v / graph.resources[r].capacity for r, v in work.items()}
    critical = max(ends.values(), default=0.0)
    logical_payload = 0.0
    physical = defaultdict(float)
    semantic = defaultdict(float)
    collective = defaultdict(float)
    for event in graph.events:
        metadata = event.metadata
        payload = metadata.get('payload_bytes', 0.0)
        logical_payload += payload
        if payload:
            semantic[str(metadata.get('semantic', metadata.get('operation', metadata.get('category', metadata.get('kind', 'transfer')))))] += payload
            collective[str(metadata.get('collective', 'point_to_point'))] += payload
        for domain, value in metadata.get('physical_bytes_by_domain', {}).items():
            physical[domain] += value
    return {
        'duration_lower_bound_s': max(critical, max(resource_bounds.values(), default=0.0)),
        'dependency_path_lower_bound_s': critical,
        'resource_time_lower_bounds_s': dict(resource_bounds),
        'resource_work': dict(work),
        'resident_memory_bytes': dict(initial),
        'resident_memory_categories_bytes': {k: dict(v) for k, v in categories.items()},
        'optimistic_schedule_memory_peak_bytes': optimistic_peak,
        'optimistic_event_completion_s': ends,
        'communication': {'transport_payload_bytes': logical_payload,
                          'physical_bytes_by_domain': dict(physical),
                          'payload_by_semantic_bytes': dict(semantic),
                          'payload_by_collective_bytes': dict(collective),
                          'classification_note': 'Semantic and collective breakdowns are two views of the same payload; do not add them.'},
        'event_count': len(graph.events),
        'memory_note': 'Resident memory is a necessary bound; optimistic schedule peaks ignore contention/slots and are not feasibility proofs.',
    }


def execution_metrics(config: Config, graph: Graph, simulation: dict) -> dict:
    if simulation['status'] != 'feasible':
        return {}
    duration = simulation['duration_s']
    meta, w, e = graph.metadata, config.workload, config.execution
    metrics = {'duration_s': duration}
    if w.scenario == 'training':
        tokens = w.batch_size * w.sequence_length * w.gradient_accumulation * e.dp * w.training_steps
        metrics.update(training_tokens=tokens, step_time_s=duration / w.training_steps,
                       throughput_tokens_s=tokens / duration if duration else 0,
                       throughput_per_replica_tokens_s=tokens / e.dp / duration if duration else 0)
        step_ids = meta.get('step_end_events', [])
        needed = set(step_ids)
        times = {row['id']: row['end_s'] for row in simulation['trace'] if row['id'] in needed}
        step_ends = [times[eid] for eid in step_ids]
        step_times = [b - a for a, b in zip([0.0] + step_ends, step_ends)]
        metrics['step_times_s'] = step_times
        metrics['first_step_time_s'] = step_times[0] if step_times else duration
        metrics['steady_step_time_s'] = sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else None
    else:
        completion = meta.get('sequence_token_completion_events', {})
        needed = {eid for sequences in completion.values() for ids in sequences.values() for eid in ids}
        times = {row['id']: row['end_s'] for row in simulation['trace'] if row['id'] in needed}
        requests = []
        for replica, sequences in completion.items():
            for seq, ids in sequences.items():
                token_times = [times[eid] for eid in ids]
                intervals = [b - a for a, b in zip(token_times, token_times[1:])]
                requests.append({'replica': replica, 'sequence': seq, 'ttft_s': token_times[0],
                                 'tpot_s': sum(intervals) / len(intervals) if intervals else None,
                                 'itl_s': intervals, 'token_completion_s': token_times})
        if not requests:
            raise ValueError('Inference compiler did not supply sequence token completion metadata')
        tpots = [r['tpot_s'] for r in requests if r['tpot_s'] is not None]
        input_tokens = sum(w.input_lengths) * e.dp
        output_tokens = sum(w.output_lengths) * e.dp
        metrics.update(ttft_s=max(r['ttft_s'] for r in requests),
                       tpot_s=max(tpots) if tpots else None,
                       input_tokens=input_tokens, output_tokens=output_tokens,
                       input_throughput_tokens_s=input_tokens / duration if duration else 0,
                       throughput_tokens_s=output_tokens / duration if duration else 0,
                       throughput_per_replica_tokens_s=output_tokens / e.dp / duration if duration else 0,
                       requests=requests, slo_aggregation='Maximum per-sequence TTFT and mean TPOT in a fixed batch; no arrival queue.')
    return metrics


def evaluate_targets(config: Config, metrics: dict, feasible: bool) -> dict:
    checks = []
    for field in ('throughput_tokens_s', 'ttft_s', 'tpot_s', 'step_time_s'):
        target = getattr(config.targets, field)
        if target is None:
            continue
        value = metrics.get(field)
        passed = value is not None and (value >= target if field == 'throughput_tokens_s' else value <= target)
        checks.append({'metric': field, 'target': target, 'actual': value, 'passed': bool(passed)})
    return {'specified': bool(checks), 'satisfied': feasible and all(x['passed'] for x in checks), 'checks': checks}


def device_requirements(graph: Graph, npu_count: int, resource_work: dict) -> dict:
    """Accumulate device footprints in one event pass, independent of rank count."""
    devices = {f'npu{i}': {'hbm_access_bytes': resource_work.get(f'npu{i}.hbm', 0),
                          'max_operator_working_set_bytes': 0,
                          'largest_transfer_payload_bytes': 0} for i in range(npu_count)}
    for event in graph.events:
        for resource in event.work:
            node, _, kind = resource.partition('.')
            if node in devices and kind.startswith('compute'):
                devices[node]['max_operator_working_set_bytes'] = max(
                    devices[node]['max_operator_working_set_bytes'], event.metadata.get('working_set_bytes', 0))
        for node in (event.metadata.get('src'), event.metadata.get('dst')):
            if node in devices:
                devices[node]['largest_transfer_payload_bytes'] = max(
                    devices[node]['largest_transfer_payload_bytes'], event.metadata.get('payload_bytes', 0))
    return devices


def run(config: Config | dict, mode: str = 'simulate', *, full_detail: bool = True) -> dict[str, Any]:
    if mode not in ('estimate', 'simulate'):
        raise ValueError('mode must be estimate or simulate')
    spec = config if isinstance(config, Config) else Config.model_validate(config)
    graph = compile_graph(spec)
    estimate = graph_estimate(graph)
    snapshot = spec.model_dump()
    snapshot['calibration'] = None  # Values and provenance are already resolved.
    result = {'schema_version': '1.0', 'engine_version': '0.1.0', 'mode': mode,
              'status': 'estimated', 'config': snapshot,
              'config_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
              'units': {'memory': 'byte', 'bandwidth': 'byte/s', 'time': 'second', 'compute': 'FLOP'},
              'sources': snapshot['sources'], 'assumptions': graph.metadata.get('assumptions', []),
              'memory_capacities_bytes': dict(graph.memory_caps),
              'model_metadata': graph.metadata, 'estimate': estimate,
              'metrics': {}, 'slo': {'specified': False, 'satisfied': None, 'checks': []}}
    # Avoid duplicating the entire DAG's analytical completion table in reports.
    result['estimate'].pop('optimistic_event_completion_s', None)
    duration = estimate['duration_lower_bound_s']
    windows = []
    if spec.workload.scenario == 'training' and spec.targets.step_time_s:
        windows.append(spec.targets.step_time_s * spec.workload.training_steps)
    if spec.targets.throughput_tokens_s:
        tokens = (spec.workload.batch_size * spec.workload.sequence_length * spec.workload.gradient_accumulation * spec.execution.dp * spec.workload.training_steps
                  if spec.workload.scenario == 'training' else sum(spec.workload.output_lengths) * spec.execution.dp)
        windows.append(tokens / spec.targets.throughput_tokens_s)
    target_window = min(windows) if windows else None
    devices = device_requirements(graph, spec.hardware.npu_count, estimate['resource_work'])
    result['requirements'] = {
        'per_resource': {key: {'work': amount, 'unit': graph.resources[key].unit,
                              'capacity': graph.resources[key].capacity,
                              'work_unit': 'FLOP' if graph.resources[key].unit == 'FLOP/s' else ('byte' if graph.resources[key].unit == 'byte/s' else graph.resources[key].unit.split('/')[0]),
                              'average_rate_at_lower_bound': amount / duration if duration else 0,
                              'required_average_rate_lower_bound': amount / target_window if target_window else None}
                         for key, amount in estimate['resource_work'].items()},
        'target_window_s': target_window,
        'window_note': 'Whole-workload step/throughput deadline only; TTFT/TPOT have different subgraph windows. Average rates are necessary, not sufficient.',
        'cache_mte': {'level': 'Operator footprints and access/transfer demand; not L1/L2/L3 capacity or MTE queue/channel predictions.',
                      'per_device': devices}}
    if mode == 'simulate':
        sim = simulate_graph(graph, retain_event_payload=full_detail or len(graph.events) <= 50_000)
        result['simulation'] = sim
        result['metrics'] = execution_metrics(spec, graph, sim)
        result['slo'] = evaluate_targets(spec, result['metrics'], sim['status'] == 'feasible')
        result['status'] = sim['status'] if sim['status'] != 'feasible' or result['slo']['satisfied'] else 'slo_failed'
    allocation_count = len(graph.initial_allocations) + sum(len(event.allocs) for event in graph.events)
    result['memory_breakdown'] = build_memory_breakdown(graph, result.get('simulation'), estimate,
        include_ephemeral_details=full_detail or allocation_count <= 50_000)
    return result

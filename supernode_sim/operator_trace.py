"""Refine scoped component events into explicit serial operator work.

This is an execution trace input, not a timing fit: each suboperator retains
the original compute/HBM resources and can use the ordinary kernel profiles.
No parallel suboperator schedule, tensor lifetimes, or measured time is inferred.
"""
from __future__ import annotations

import math
import re

from .calibration import _scope_mismatch
from .core import Allocation, Event, Graph, Resource


POLICY = (
    'Exact aggregate work, phase, dtype, hardware and explicit metadata guards; '
    'a supplied runtime context must also match. Serial operations conserve FLOPs '
    'and may add explicitly declared HBM traffic, startup and local workspace. '
    'Original allocations span the entire trace; workspace is private to one operation. '
    'Working-set reports retain the parent component proxy plus local workspace, not an inferred suboperator footprint. '
    'Trace coverage is execution-detail coverage, not measured timing accuracy. '
    'Explicit resource service demands share capacity per physical NPU: demands on the same resource sum, '
    'independent resources and the supplied dependency path form concurrent lower bounds. '
    'These constraints do not add model FLOPs, HBM bytes, startup, or measured evidence. '
    'Suboperators use existing kernel_profiles; parent profiles are not inherited.'
)


def operator_trace_summary() -> dict:
    return {'enabled': True, 'eligible_events': 0, 'matched_events': 0,
            'unmatched_events': 0, 'generated_operations': 0, 'coverage_fraction': 0.0,
            'additional_memory_bytes': 0.0, 'additional_startup_s': 0.0,
            'resource_modeled_operations': 0, 'dependency_modeled_operations': 0,
            'matched_trace_ids': [], 'unmatched_kernels': [], 'skip_reasons': {},
            'by_phase': {}, 'policy': POLICY}


def _coverage(summary: dict) -> None:
    summary['unmatched_events'] = summary['eligible_events'] - summary['matched_events']
    summary['coverage_fraction'] = (summary['matched_events'] / summary['eligible_events']
                                    if summary['eligible_events'] else 0.0)


def merge_operator_trace_summaries(target: dict, source: dict) -> None:
    """Count represented parent events, including streaming macro replay."""
    for key in ('eligible_events', 'matched_events', 'generated_operations',
                'additional_memory_bytes', 'additional_startup_s',
                'resource_modeled_operations', 'dependency_modeled_operations'):
        target[key] = target.get(key, 0) + source.get(key, 0)
    for key in ('matched_trace_ids', 'unmatched_kernels'):
        target[key] = sorted(set(target[key]) | set(source[key]))
    for reason, count in source['skip_reasons'].items():
        target['skip_reasons'][reason] = target['skip_reasons'].get(reason, 0) + count
    for phase, counts in source['by_phase'].items():
        combined = target['by_phase'].setdefault(phase, {'eligible_events': 0, 'matched_events': 0})
        for key in ('eligible_events', 'matched_events'):
            combined[key] += counts[key]
        _coverage(combined)
    _coverage(target)


def _supported(event: Event, compute: str) -> str | None:
    rank = compute.split('.')[0]
    if compute not in (rank + '.compute', rank + '.compute.fp32'):
        return 'already_calibrated_or_unsupported_compute'
    if set(event.work) - {compute, rank + '.hbm'}:
        return 'unsupported_resource'
    if event.metadata.get('collective') or event.metadata.get('payload_bytes'):
        return 'communication_boundary'
    if event.metadata.get('calibration') or event.metadata.get('megakernel'):
        return 'already_calibrated_or_fused'
    if event.slots != (rank + '.compute_stream',):
        return 'unsupported_execution_slot'
    if any(allocation.pool != rank + '.hbm' for allocation in event.allocs):
        return 'nonlocal_allocation'
    return None


def _resource_demands(graph: Graph, rank: str, eid: str, operation: dict) -> dict[str, float]:
    """Use normalized seconds without mislabeling local traffic as HBM work."""
    work = {}
    values = {}
    for demand in operation.get('resource_demands', []):
        key = f"{rank}.kernel_resource.{demand['resource']}"
        kind = 'microarchitecture_' + demand['kind']
        resource = graph.resources.get(key)
        if resource is not None and (resource.kind != kind or resource.capacity != 1.0
                                     or resource.unit != 'second/second'):
            raise ValueError(f'Conflicting shared kernel resource definition: {key}')
        graph.resources.setdefault(key, Resource(key, 1.0, 'second/second', kind))
        values.setdefault(key, []).append(demand['service_s'])
    for key, amounts in values.items():
        try:
            total = math.fsum(amounts)
        except OverflowError as exc:
            raise ValueError(f'Kernel resource demand must be finite: {key}') from exc
        if not math.isfinite(total):
            raise ValueError(f'Kernel resource demand must be finite: {key}')
        work[key] = total
    if operation.get('dependency_floor_s', 0):
        key = f'kernel_dependency:{eid}'
        if key in graph.resources:
            raise ValueError(f'Kernel dependency generated duplicate resource ID: {key}')
        graph.resources[key] = Resource(key, 1.0, 'second/second', 'microarchitecture_dependency')
        work[key] = operation['dependency_floor_s']
    return work


def apply_kernel_traces(graph: Graph, config: dict) -> None:
    traces = config['hardware'].get('kernel_traces', {})
    if not traces:
        return
    summary = operator_trace_summary()
    original_ids = {event.id for event in graph.events}
    allocation_ids = {allocation.id for allocation in graph.initial_allocations}
    allocation_ids.update(allocation.id for event in graph.events for allocation in event.allocs)
    expanded = []
    for event in graph.events:
        kernel = event.metadata.get('kernel')
        if not kernel:
            expanded.append(event)
            continue
        compute_keys = [key for key in event.work if re.match(r'^npu\d+\.compute(?:\.|$)', key)]
        if len(compute_keys) != 1:
            expanded.append(event)
            if kernel in traces:
                summary['skip_reasons']['missing_or_multiple_compute_resources'] = (
                    summary['skip_reasons'].get('missing_or_multiple_compute_resources', 0) + 1)
                summary['unmatched_kernels'].append(kernel)
            continue
        summary['eligible_events'] += 1
        phase = str(event.metadata.get('phase') or 'unspecified')
        counts = summary['by_phase'].setdefault(phase, {'eligible_events': 0, 'matched_events': 0})
        counts['eligible_events'] += 1

        def skip(reason):
            expanded.append(event)
            summary['unmatched_kernels'].append(kernel)
            summary['skip_reasons'][reason] = summary['skip_reasons'].get(reason, 0) + 1

        if kernel not in traces:
            skip('no_trace_for_kernel')
            continue
        compute = compute_keys[0]
        rank = compute.split('.')[0]
        dtype = 'fp32' if compute.endswith('.fp32') else config['execution']['precision']['compute']
        memory = event.work.get(rank + '.hbm', 0.0)
        candidates = [trace for trace in traces[kernel]
                      if trace['phase'] == phase and trace['dtype'] == dtype
                      and math.isclose(trace['flops'], event.work[compute], rel_tol=1e-9, abs_tol=1e-6)
                      and math.isclose(trace['memory_bytes'], memory, rel_tol=1e-9, abs_tol=1e-6)]
        if not candidates:
            skip('work_phase_or_dtype_mismatch')
            continue
        scoped = [(trace, _scope_mismatch(trace, config['hardware'], event.metadata)) for trace in candidates]
        matching = [trace for trace, reason in scoped if reason is None]
        if not matching:
            expanded.append(event)
            summary['unmatched_kernels'].append(kernel)
            for reason in sorted({reason for _, reason in scoped}):
                summary['skip_reasons'][reason] = summary['skip_reasons'].get(reason, 0) + 1
            continue
        if len(matching) > 1:
            raise ValueError(f'Ambiguous kernel trace for {event.id}')
        reason = _supported(event, compute)
        if reason:
            skip(reason)
            continue
        trace = matching[0]
        dependencies = event.deps
        for index, operation in enumerate(trace['operations']):
            eid = f'{event.id}.operator_trace.{index}'
            if eid in original_ids:
                raise ValueError(f'Kernel trace generated duplicate event ID: {eid}')
            original_ids.add(eid)
            allocations = event.allocs if index == 0 else ()
            releases = ()
            if operation.get('workspace_bytes', 0):
                aid = eid + '.workspace'
                if aid in allocation_ids:
                    raise ValueError(f'Kernel trace generated duplicate allocation ID: {aid}')
                allocation_ids.add(aid)
                allocations += (Allocation(aid, rank + '.hbm', operation['workspace_bytes'],
                    'operator_workspace', None, metadata={
                        'component': event.metadata.get('component', 'Unassigned'),
                        'layer': event.metadata.get('layer'), 'state': 'operator_workspace',
                        'trace_id': trace['id'], 'kernel': operation['kernel']}),)
                releases = (aid,)
            metadata = {key: value for key, value in event.metadata.items()
                        if key not in ('working_set_bytes', 'calibration', 'matmul', 'implementation')}
            metadata.update(kernel=operation['kernel'], flops=operation['flops'],
                original_kernel=kernel, trace_id=trace['id'], trace_source=trace['source'],
                trace_parent_event=event.id, operation_index=index,
                operator_workspace_bytes=operation.get('workspace_bytes', 0))
            # HBM accesses are not a footprint. Preserve the existing component
            # proxy so refinement cannot erase it from device requirements.
            if 'working_set_bytes' in event.metadata:
                metadata['working_set_bytes'] = (event.metadata['working_set_bytes']
                                                + operation.get('workspace_bytes', 0))
                metadata['working_set_scope'] = 'parent_component_proxy_plus_local_workspace'
                metadata['parent_working_set_bytes'] = event.metadata['working_set_bytes']
            if operation.get('matmul') is not None:
                metadata['matmul'] = operation['matmul']
            if operation.get('implementation'):
                metadata['implementation'] = operation['implementation']
            work = {compute: operation['flops'], rank + '.hbm': operation['memory_bytes']}
            work.update(_resource_demands(graph, rank, eid, operation))
            if operation.get('resource_demands'):
                metadata['resource_demands'] = operation['resource_demands']
                summary['resource_modeled_operations'] += 1
            if operation.get('dependency_floor_s', 0):
                metadata['dependency_floor_s'] = operation['dependency_floor_s']
                summary['dependency_modeled_operations'] += 1
            expanded.append(Event(eid, dependencies,
                work,
                startup_s=(event.startup_s if index == 0 else 0) + operation.get('startup_s', 0),
                slots=event.slots, allocs=allocations, releases=releases, metadata=metadata))
            dependencies = (eid,)
        # Keep the original event identity for consumers, timing windows and
        # references held by pipeline/speculative/streaming compiler metadata.
        metadata = {key: value for key, value in event.metadata.items()
                    if key not in ('kernel', 'flops', 'working_set_bytes', 'calibration')}
        metadata.update(trace_id=trace['id'], trace_source=trace['source'],
                        original_kernel=kernel, operator_trace_completion=True)
        expanded.append(Event(event.id, dependencies, releases=event.releases, metadata=metadata))
        summary['matched_events'] += 1
        counts['matched_events'] += 1
        summary['generated_operations'] += len(trace['operations'])
        summary['additional_memory_bytes'] += max(0.0, math.fsum(op['memory_bytes'] for op in trace['operations']) - memory)
        summary['additional_startup_s'] += math.fsum(op.get('startup_s', 0) for op in trace['operations'])
        summary['matched_trace_ids'].append(trace['id'])
    for key in ('matched_trace_ids', 'unmatched_kernels'):
        summary[key] = sorted(set(summary[key]))
    _coverage(summary)
    for counts in summary['by_phase'].values():
        _coverage(counts)
    graph.events = expanded
    graph.metadata['operator_trace'] = summary

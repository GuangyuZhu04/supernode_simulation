"""Exact-work measured kernel timing, preserving shared memory contention."""
from __future__ import annotations

import math

from .core import Graph, Resource


CALIBRATION_POLICY = (
    'Exact FLOPs/bytes with optional dtype/phase filters; this is not shape or runtime identity validation. '
    'duration_s excludes startup; an explicit startup_s replaces event startup. '
    'Physical contention remains active. Coverage counts modeled kernel events, including cached replay, '
    'not elapsed time; kernel matches do not validate end-to-end serving performance.'
)


def calibration_summary() -> dict:
    return {'eligible_events': 0, 'matched_events': 0, 'unmatched_events': 0,
            'coverage_fraction': 0.0, 'unmatched_kernels': [], 'unprofiled_kernels': [],
            'by_phase': {}, 'policy': CALIBRATION_POLICY}


def _coverage(counts: dict) -> None:
    counts['unmatched_events'] = counts['eligible_events'] - counts['matched_events']
    counts['coverage_fraction'] = (counts['matched_events'] / counts['eligible_events']
                                   if counts['eligible_events'] else 0.0)


def merge_calibration_summaries(target: dict, source: dict) -> None:
    """Accumulate represented events, including chunks replayed from a macro."""
    for key in ('eligible_events', 'matched_events'):
        target[key] += source[key]
    for key in ('unmatched_kernels', 'unprofiled_kernels'):
        target[key] = sorted(set(target[key]) | set(source[key]))
    for phase, counts in source['by_phase'].items():
        combined = target['by_phase'].setdefault(phase, {'eligible_events': 0, 'matched_events': 0})
        for key in ('eligible_events', 'matched_events'):
            combined[key] += counts[key]
        _coverage(combined)
    _coverage(target)


def apply_kernel_profiles(graph: Graph, config: dict) -> None:
    profiles = config['hardware'].get('kernel_profiles', {})
    summary = calibration_summary()
    unmatched = set()
    unprofiled = set()
    for event in graph.events:
        kernel = event.metadata.get('kernel')
        if not kernel:
            continue
        compute_keys = [key for key in event.work if key.startswith('npu') and '.compute' in key]
        if len(compute_keys) != 1:
            if kernel in profiles:
                unmatched.add(kernel)
            continue
        summary['eligible_events'] += 1
        phase = str(event.metadata.get('phase') or 'unspecified')
        phase_counts = summary['by_phase'].setdefault(phase, {'eligible_events': 0, 'matched_events': 0})
        phase_counts['eligible_events'] += 1
        if kernel not in profiles:
            unprofiled.add(kernel)
            continue
        key = compute_keys[0]
        node = key.split('.')[0]
        dtype = 'fp32' if key.endswith('.fp32') else config['execution']['precision']['compute']
        flops, memory = event.work[key], event.work.get(f'{node}.hbm', 0)
        fitting = [p for p in profiles[kernel] if math.isclose(p['flops'], flops, rel_tol=1e-9, abs_tol=1e-6)
                   and math.isclose(p['memory_bytes'], memory, rel_tol=1e-9, abs_tol=1e-6)
                   and (p.get('dtype') is None or p['dtype'] == dtype)
                   and (p.get('phase') is None or p['phase'] == event.metadata.get('phase'))]
        if not fitting:
            unmatched.add(kernel)
            continue
        if len(fitting) != 1:
            raise ValueError(f'Ambiguous kernel calibration for {event.id}')
        profile = fitting[0]
        measured = profile['duration_s']
        peak = config['hardware']['npu']['flops'][dtype]
        physical_floor = max(flops / peak, memory / graph.resources[f'{node}.hbm'].capacity)
        if measured < physical_floor * (1 - 1e-9):
            raise ValueError(f'Kernel calibration {kernel} is faster than configured physical compute/memory limits')
        calibrated_key = f'{node}.compute.calibrated.{dtype}'
        graph.resources[calibrated_key] = Resource(calibrated_key, peak, 'FLOP/s', 'compute')
        event.work.pop(key)
        event.work[calibrated_key] = flops
        limiter = f'kernel_limit:{event.id}'
        graph.resources[limiter] = Resource(limiter, 1.0, 'second/second', 'kernel_timing')
        event.work[limiter] = measured
        startup = profile.get('startup_s')
        if startup is not None:
            event.startup_s = startup
        event.metadata['calibration'] = {'kernel': kernel, 'service_time_s': measured, 'match': 'exact_work',
            'startup_s': event.startup_s, 'startup_calibrated': startup is not None}
        summary['matched_events'] += 1
        phase_counts['matched_events'] += 1
    summary['unmatched_kernels'] = sorted(unmatched)
    summary['unprofiled_kernels'] = sorted(unprofiled)
    _coverage(summary)
    for counts in summary['by_phase'].values():
        _coverage(counts)
    graph.metadata['kernel_calibration'] = summary

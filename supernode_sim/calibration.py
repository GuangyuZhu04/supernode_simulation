"""Exact-work measured kernel timing, preserving shared memory contention."""
from __future__ import annotations

import math

from .core import Graph, Resource


def apply_kernel_profiles(graph: Graph, config: dict) -> None:
    profiles = config['hardware'].get('kernel_profiles', {})
    if not profiles:
        return
    matched = 0
    unmatched = set()
    for event in graph.events:
        kernel = event.metadata.get('kernel')
        if kernel not in profiles:
            continue
        compute_keys = [key for key in event.work if key.startswith('npu') and '.compute' in key]
        if len(compute_keys) != 1:
            unmatched.add(kernel)
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
        measured = fitting[0]['duration_s']
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
        event.metadata['calibration'] = {'kernel': kernel, 'service_time_s': measured, 'match': 'exact_work'}
        matched += 1
    graph.metadata['kernel_calibration'] = {'matched_events': matched, 'unmatched_kernels': sorted(unmatched),
                                           'policy': 'Exact FLOPs/bytes/dtype/phase; physical contention remains active; startup is separate.'}

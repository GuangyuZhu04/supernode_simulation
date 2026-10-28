"""Exact-work measured kernel timing, preserving shared memory contention."""
from __future__ import annotations

import math
import json
import hashlib

from .core import Graph, Resource


CALIBRATION_POLICY = (
    'Exact FLOPs/bytes with optional dtype/phase filters and explicit hardware/context/metadata guards. '
    'Unspecified tensor shape or runtime identity is not inferred; legacy profiles are unguarded. '
    'duration_s excludes startup; an explicit startup_s replaces event startup. '
    'Physical contention remains active. Coverage counts modeled kernel events, including cached replay, '
    'not elapsed time; kernel matches do not validate end-to-end serving performance.'
)


def model_shape_signature(model: dict) -> str:
    """Identity of declared structure, not checkpoint contents or kernel code."""
    shape = {key: value for key, value in model.items()
             if key not in {'name', 'reported_total_params', 'reported_active_params'}}
    encoded = json.dumps(shape, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return 'model-shape-v1:sha256:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def calibration_summary() -> dict:
    return {'eligible_events': 0, 'matched_events': 0, 'unmatched_events': 0,
            'guarded_matched_events': 0, 'unguarded_matched_events': 0, 'scope_mismatched_events': 0,
            'coverage_fraction': 0.0, 'unmatched_kernels': [], 'unprofiled_kernels': [],
            'scope_mismatch_reasons': {}, 'by_phase': {}, 'policy': CALIBRATION_POLICY}


def _coverage(counts: dict) -> None:
    counts['unmatched_events'] = counts['eligible_events'] - counts['matched_events']
    counts['coverage_fraction'] = (counts['matched_events'] / counts['eligible_events']
                                   if counts['eligible_events'] else 0.0)


def merge_calibration_summaries(target: dict, source: dict) -> None:
    """Accumulate represented events, including chunks replayed from a macro."""
    count_keys = ('eligible_events', 'matched_events', 'guarded_matched_events',
                  'unguarded_matched_events', 'scope_mismatched_events')
    for key in count_keys:
        target[key] = target.get(key, 0) + source.get(key, 0)
    for key in ('unmatched_kernels', 'unprofiled_kernels'):
        target[key] = sorted(set(target[key]) | set(source[key]))
    for phase, counts in source['by_phase'].items():
        combined = target['by_phase'].setdefault(phase, {'eligible_events': 0, 'matched_events': 0})
        for key in count_keys:
            combined[key] = combined.get(key, 0) + counts.get(key, 0)
        _coverage(combined)
    for reason, count in source.get('scope_mismatch_reasons', {}).items():
        target.setdefault('scope_mismatch_reasons', {})[reason] = target['scope_mismatch_reasons'].get(reason, 0) + count
    _coverage(target)


def _guarded(profile: dict) -> bool:
    return bool(profile.get('hardware_name') is not None or profile.get('calibration_context')
                or profile.get('metadata_match'))


def _same_declaration(left, right) -> bool:
    # bool and integer IDs must not match merely because Python treats True == 1.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _scope_mismatch(profile: dict, hardware: dict, metadata: dict) -> str | None:
    if profile.get('hardware_name') is not None and profile['hardware_name'] != hardware.get('name'):
        return 'hardware_mismatch'
    if profile.get('calibration_context') and not _same_declaration(
            profile['calibration_context'], hardware.get('calibration_context', {})):
        return 'environment_mismatch'
    for key, value in profile.get('metadata_match', {}).items():
        if key not in metadata or not _same_declaration(value, metadata[key]):
            return 'metadata_mismatch'
    return None


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
        phase_counts = summary['by_phase'].setdefault(phase, {
            'eligible_events': 0, 'matched_events': 0, 'guarded_matched_events': 0,
            'unguarded_matched_events': 0, 'scope_mismatched_events': 0})
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
        guarded = [profile for profile in fitting if _guarded(profile)]
        if guarded:
            # A scoped record must not silently fall back to an unguarded record
            # for the same work when the software/platform/metadata differs.
            scope_results = [(profile, _scope_mismatch(profile, config['hardware'], event.metadata))
                             for profile in guarded]
            fitting = [profile for profile, reason in scope_results if reason is None]
            if not fitting:
                unmatched.add(kernel)
                summary['scope_mismatched_events'] += 1
                phase_counts['scope_mismatched_events'] += 1
                for reason in sorted({reason for _, reason in scope_results}):
                    summary['scope_mismatch_reasons'][reason] = summary['scope_mismatch_reasons'].get(reason, 0) + 1
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
            'startup_s': event.startup_s, 'startup_calibrated': startup is not None,
            'applicability_guard': 'guarded' if _guarded(profile) else 'unguarded_legacy',
            'guard_fields': [field for field in ('hardware_name', 'calibration_context', 'metadata_match') if profile.get(field)],
            'source': profile.get('source')}
        summary['matched_events'] += 1
        phase_counts['matched_events'] += 1
        guard_key = 'guarded_matched_events' if _guarded(profile) else 'unguarded_matched_events'
        summary[guard_key] += 1
        phase_counts[guard_key] += 1
    summary['unmatched_kernels'] = sorted(unmatched)
    summary['unprofiled_kernels'] = sorted(unprofiled)
    _coverage(summary)
    for counts in summary['by_phase'].values():
        _coverage(counts)
    graph.metadata['kernel_calibration'] = summary

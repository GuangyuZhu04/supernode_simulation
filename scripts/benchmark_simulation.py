"""Measure host execution time and compare simulation physics across revisions.

Run from any directory. --implementation-root can point at a saved package tree;
configuration and axes paths are resolved relative to the caller's directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def physics(result: dict) -> dict:
    """Keep numerical outputs, including failures, without retaining raw traces."""
    answer = {key: result[key] for key in ('status', 'metrics', 'slo', 'diagnostics', 'parameters')
              if key in result}
    estimate = result.get('estimate', {})
    answer['estimate'] = {key: estimate[key] for key in (
        'duration_lower_bound_s', 'resource_work', 'resident_memory_bytes',
        'optimistic_schedule_memory_peak_bytes', 'communication', 'event_count') if key in estimate}
    simulation = result.get('simulation', result)
    answer['simulation'] = {key: simulation[key] for key in (
        'status', 'duration_s', 'memory_peaks_bytes', 'memory_final_bytes',
        'memory_categories_at_peak_bytes', 'resource_work', 'resource_busy_s',
        'completed_events', 'total_events', 'diagnostics')
        if key in simulation}
    metadata = result.get('model_metadata', {})
    answer['calibration'] = {key: metadata[key] for key in (
        'kernel_calibration', 'collective_calibration', 'fidelity') if key in metadata}
    if 'candidates' in result:
        answer['candidates'] = [
            {'parameters': row.get('parameters'), 'result': physics(row.get('result', row))}
            for row in result['candidates']]
    return answer


def difference(left, right, path='result') -> str | None:
    if isinstance(left, bool) or isinstance(right, bool):
        return None if type(left) is type(right) and left == right else f'{path}: {left!r} != {right!r}'
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return f'{path}: different keys'
        for key in left:
            mismatch = difference(left[key], right[key], f'{path}.{key}')
            if mismatch:
                return mismatch
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f'{path}: different lengths'
        for index, (a, b) in enumerate(zip(left, right)):
            mismatch = difference(a, b, f'{path}[{index}]')
            if mismatch:
                return mismatch
        return None
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
            return None
    elif left == right:
        return None
    return f'{path}: {left!r} != {right!r}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--mode', choices=('simulate', 'estimate', 'streaming', 'sweep'), default='simulate')
    parser.add_argument('--axes', type=Path, help='Required for sweep; same dotted-path mapping as snmodel sweep')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--implementation-root', type=Path, default=ROOT)
    parser.add_argument('--compare', type=Path, help='Earlier benchmark JSON with the same config, mode and axes')
    parser.add_argument('--output', '-o', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.workers < 0 or (args.mode != 'sweep' and args.workers != 1):
        parser.error('--workers is a nonnegative sweep-only option')
    if (args.mode == 'sweep') != bool(args.axes):
        parser.error('--axes is required only for sweep')
    implementation = args.implementation_root.resolve()
    if not (implementation / 'supernode_sim' / '__init__.py').is_file():
        parser.error('--implementation-root must contain the supernode_sim package')
    sys.path.insert(0, str(implementation))
    from supernode_sim.analysis import run
    from supernode_sim.config import load_config, read_data
    from supernode_sim.search import sweep
    from supernode_sim.streaming import run_streaming

    config = load_config(args.config)
    axes = read_data(args.axes) if args.axes else None
    supports_workers = 'workers' in inspect.signature(sweep).parameters
    if args.workers != 1 and not supports_workers:
        parser.error('This implementation does not support parallel sweep')
    snapshot = config.model_dump()
    snapshot['calibration'] = None
    scenario = {'config': snapshot, 'mode': args.mode, 'axes': axes}
    identity = hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()
    previous = json.loads(args.compare.read_text(encoding='utf-8')) if args.compare else None
    if previous and previous['scenario_sha256'] != identity:
        parser.error('--compare must use the same resolved config, mode and axes')

    durations, expected = [], None
    enabled = gc.isenabled()
    gc.disable()  # Match the numerical CLI; restore the calling interpreter's state.
    try:
        for repeat in range(args.repeats):
            gc.collect()
            start = time.perf_counter()
            if args.mode == 'streaming':
                result = run_streaming(config)
            elif args.mode == 'sweep':
                options = {'workers': args.workers} if supports_workers else {}
                result = sweep(config, axes, **options)
            else:
                result = run(config, args.mode, full_detail=False)
            elapsed = time.perf_counter() - start
            measured = physics(result)
            del result
            if expected is not None and (mismatch := difference(expected, measured)):
                raise ValueError(f'Non-reproducible repeat: {mismatch}')
            expected = measured
            durations.append(elapsed)
            print(f'{args.mode} repeat {repeat + 1}/{args.repeats}: {elapsed:.6f} s', flush=True)
    finally:
        if enabled:
            gc.enable()
    report = {
        'scenario_sha256': identity, 'config_path': str(args.config.resolve()),
        'mode': args.mode, 'workers': args.workers, 'implementation_root': str(implementation),
        'python': sys.version, 'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
        'wall_seconds': durations, 'median_seconds': statistics.median(durations),
        'timing_scope': 'Engine API including graph compilation and numerical summaries; sweep includes worker startup. Config loading and artifact export excluded. Fresh streaming cache per repeat.',
        'numerical_summary': expected,
    }
    mismatch = None
    if previous:
        mismatch = difference(previous['numerical_summary'], expected)
        report['comparison'] = {'baseline': str(args.compare.resolve()), 'equivalent': mismatch is None,
                                'first_difference': mismatch,
                                'speedup': previous['median_seconds'] / report['median_seconds'] if mismatch is None else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('numerical_summary', 'python', 'timing_scope')}, ensure_ascii=False))
    return 1 if mismatch else 0


if __name__ == '__main__':
    raise SystemExit(main())

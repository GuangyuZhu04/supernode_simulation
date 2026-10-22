"""Rerun the complete Kimi study with routed FP4 / other FP8 weights."""
from __future__ import annotations

import argparse
import copy
import hashlib
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_kimi_10t_950dt_dspark import (
    Config, code_hashes, parameter_inventory, read, run_cases, save, sha, table)

BASELINE = ROOT / 'outputs/kimi-k3-10t-950dt-dspark'
OUTPUT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark-fp4'

PRECISION_NOTE = (
    'Routed-expert weights only: FP4. All other target weights and compute, '
    'activation, communication, KV and weight_transfer remain FP8; recurrent FP32. '
    'Retain weight_scale_block=128, scale_bytes=4 and kv_scale_block=128. '
    'This is a storage/conversion abstraction, not a validated MXFP4 checkpoint. '
    'Native weight_conversion reads packed weights/scales, writes an FP8 buffer, '
    'charges scalar conversion work and buffer residency, then the kernel reads '
    'that FP8 buffer. No fused dequantization or FP4 compute speedup is assumed. '
    'DSpark draft precision, costs, residency and acceptance assumptions are unchanged.')


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def numerical_snapshot(baseline):
    files = list((baseline / 'configs').glob('*.json'))
    for pattern in ('*/case.json', '*/full/result.json', '*/full/config.resolved.json',
                    '*/warm-decode/result.json', '*/warm-decode/config.resolved.json'):
        files.extend((baseline / 'cases').glob(pattern))
    files.extend(baseline / name for name in ('manifest.json', 'summary.json', 'summary.csv',
        'audit.json', 'report.html', 'report.md', 'reproducibility.json'))
    return {str(path.relative_to(baseline)): file_sha(path) for path in sorted(files)}


def numerical_config(raw):
    return {key: value for key, value in raw.items() if key not in ('sources', 'assumptions')}


def prepare(directory, baseline=BASELINE):
    baseline, directory = baseline.resolve(), directory.resolve()
    assert directory != baseline
    assert ROOT.resolve() in directory.parents
    manifest = copy.deepcopy(read(baseline / 'manifest.json'))
    assert len(manifest['cases']) == 392
    current_code = code_hashes()
    protection = directory / 'fp8-protection.json'
    if protection.exists():
        assert read(protection)['files'] == numerical_snapshot(baseline), 'FP8 baseline changed'
    else:
        save(protection, {'baseline': str(baseline), 'files': numerical_snapshot(baseline)})
    inventory = []
    for item in manifest['cases']:
        old = read(baseline / item['config'])
        index = read(baseline / 'cases' / item['name'] / 'case.json')
        assert index['fingerprint'] == sha({'config': old, 'source': current_code}), item['name']
        assert old['execution']['precision']['weight'] == 'fp8'
        assert old['execution']['precision']['routed_expert_weight'] is None
        raw = copy.deepcopy(old)
        raw['execution']['precision']['routed_expert_weight'] = 'fp4'
        raw['sources']['execution.precision'] = {'status': 'user', 'description': PRECISION_NOTE,
                                                'uri': str(baseline / item['config'])}
        raw['assumptions'] = [s for s in raw['assumptions'] if not s.startswith('Prior F_kv precision retained;')]
        raw['assumptions'].append(PRECISION_NOTE)
        config = Config.model_validate(raw)
        resolved = config.model_dump()
        compared = copy.deepcopy(resolved)
        compared['execution']['precision']['routed_expert_weight'] = None
        assert numerical_config(compared) == numerical_config(old), item['name']
        assert sha(config.hardware.model_dump()) == item['hardware_sha256']
        assert sha(config.model.model_dump()) == item['model_sha256']
        item['fp8_case'] = item['name']
        item['fp8_config_sha256'] = item['config_sha256']
        item['config_sha256'] = sha(resolved)
        item['precision_variant'] = 'routed_fp4_other_fp8'
        destination = directory / item['config']
        if destination.exists():
            assert read(destination) == resolved, f'Different config already exists: {destination}'
        else:
            save(destination, resolved)
        if not inventory:
            for tensor in parameter_inventory(config.model.model_dump()):
                inventory.append({'tensor': tensor.name, 'component': tensor.component,
                    'layer': tensor.layer, 'ep_shard': tensor.ep_shard,
                    'logical_parameters': tensor.accounted_elements,
                    'fp8_dtype': 'fp8', 'mixed_dtype': 'fp4' if tensor.ep_shard else 'fp8',
                    'fp8_logical_bytes': tensor.accounted_elements,
                    'mixed_logical_bytes': tensor.accounted_elements * (0.5 if tensor.ep_shard else 1)})
    routed = sum(t['logical_parameters'] for t in inventory if t['ep_shard'])
    total = sum(t['logical_parameters'] for t in inventory)
    before = sum(t['fp8_logical_bytes'] for t in inventory)
    after = sum(t['mixed_logical_bytes'] for t in inventory)
    assert isclose(before - after, routed * 0.5, abs_tol=0.1)
    manifest['fp8_baseline_directory'] = str(baseline)
    manifest['precision_variant'] = 'routed_fp4_other_fp8'
    manifest['precision_note'] = PRECISION_NOTE
    manifest['source_sha256'][str((baseline / 'manifest.json').relative_to(ROOT))] = file_sha(baseline / 'manifest.json')
    manifest['counts'] = {'main_workloads': 180, 'main_runs': 360,
        'topology_sensitivity_runs': 12, 'additional_sensitivity_runs': 20, 'total_runs': 392}
    save(directory / 'manifest.json', manifest)
    for name in ('model.design.config.json', 'dspark-assumptions.json'):
        save(directory / name, read(baseline / name))
    assumptions = {'description': PRECISION_NOTE, 'total_parameters': total,
        'routed_parameters': routed, 'other_parameters': total - routed,
        'fp8_logical_weight_bytes_excluding_scales': before,
        'mixed_logical_weight_bytes_excluding_scales': after,
        'logical_weight_saving_bytes': before - after,
        'weight_scale_block': 128, 'scale_bytes': 4,
        'weight_scales_and_kv_metadata_unchanged': True,
        'numeric_config_delta': {'execution.precision.routed_expert_weight': [None, 'fp4']},
        'all_392_config_pairs_validated': True, 'baseline_source_fingerprints_match': True}
    save(directory / 'precision-assumptions.json', assumptions)
    table(directory / 'precision_tensor_inventory.csv', inventory)
    print({'stage': 'fp4_prepared', 'cases': len(manifest['cases']), **assumptions}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.baseline)
    if args.run:
        run_cases(args.output, set(args.case or []))

"""Audit the completed 20-point study's inputs, provenance and arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'outputs/scaling-10case-20261113')
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['cases']) != 20:
        raise AssertionError('Expected two architecture templates times ten cases')
    expected_models = {'2.8T': (2.8e12, 104e9, 512, 16, 128),
                       '10T': (1e13, 250e9, 2048, 32, 512),
                       '100T': (1e14, 1e12, 8192, 64, 2048)}
    rows, hashes, results = [], [], {}
    for item in manifest['cases']:
        result_path = directory / item['result_path']
        result = json.loads(result_path.read_text(encoding='utf-8'))
        results[item['id']] = result
        config = result['config']
        actual_input = json.loads((directory / item['config_path']).read_text(encoding='utf-8'))
        if config != actual_input or result['config_sha256'] != digest(config):
            raise AssertionError(f'{item["id"]}: input/config digest mismatch')
        model, workload = config['model'], config['workload']
        expected = expected_models[item['model_label']]
        observed = (model['total_params'], model['active_params'], model['num_experts'], model['top_k'], config['execution']['ep'])
        if observed != expected:
            raise AssertionError(f'{item["id"]}: user model budget changed')
        if workload['batch_size'] != 32 or workload['input_length'] != 1048576 or workload['output_length'] != 2:
            raise AssertionError(f'{item["id"]}: workload changed')
        provenance = result['provenance']
        current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in provenance['source_sha256']}
        if current != provenance['source_sha256']:
            raise AssertionError(f'{item["id"]}: recorded source differs from frozen files')
        hashes.append(provenance['source_sha256'])
        attention = json.loads((directory / item['attention_profile_path']).read_text(encoding='utf-8'))
        if provenance['attention_profile_sha256'] != digest(attention):
            raise AssertionError(f'{item["id"]}: Attention sidecar identity mismatch')
        if item['afd_path']:
            afd = json.loads((directory / item['afd_path']).read_text(encoding='utf-8'))
            if afd != result['study']['afd'] or afd['attention_count'] != 6144 or afd['ffn_count'] != 6144:
                raise AssertionError('AFD sidecar mismatch')
        tpot = result['metrics']['tpot_s']
        record = result['metrics']['latency_breakdown']['tpot']
        if tpot is not None:
            if not math.isclose(sum(record['components_s'].values()), tpot, rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError('Latency components do not sum')
            operator_sum = sum(op['total_s'] for op in record['operator_breakdown']['operators'])
            if not math.isclose(operator_sum, tpot, rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError('Operator components do not sum')
            expected_status = 'feasible' if tpot <= config['targets']['tpot_s'] else 'slo_failed'
            if result['status'] != expected_status:
                raise AssertionError('SLO status contradicts simulated token time')
            peaks = result['simulation']['memory_peaks_bytes']
            caps = result['memory_capacities_bytes']
            if any(peak > caps[pool] + 1 for pool, peak in peaks.items()):
                raise AssertionError('Feasible execution exceeded a memory capacity')
        if result['metrics']['ttft_s'] is not None:
            raise AssertionError('Decode-only study cannot invent TTFT')
        rows.append({'case': item['id'], 'status': result['status'], 'tpot_s': tpot,
                     'input_source_and_sidecar_identity': 'passed', 'arithmetic_and_capacity': 'passed'})
    if any(value != hashes[0] for value in hashes):
        raise AssertionError('Production matrix mixed numeric source versions')
    duplicate_checks = []
    for architecture in ('kimi-k3', 'deepseek-v41'):
        for hardware in ('960DT', '970DT'):
            left_id, right_id = f'{architecture}-model-{hardware}-10T', f'{architecture}-hardware-{hardware}-10T'
            left, right = results[left_id], results[right_id]
            if left['metrics']['tpot_s'] != right['metrics']['tpot_s']:
                raise AssertionError(f'Duplicate physical case differs: {left_id}/{right_id}')
            duplicate_checks.append({'left': left_id, 'right': right_id, 'tpot_identity': 'passed'})
    payload = {'status': 'passed', 'case_count': 20,
               'scope': 'Inputs, frozen numeric source identity, sidecars, latency sums, SLO labels, memory capacity and repeated physical cases; not hardware validation',
               'source_sha256': hashes[0], 'cases': rows, 'duplicate_physical_cases': duplicate_checks}
    (directory / 'audit.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'cases': len(rows), 'duplicate_pairs': len(duplicate_checks), 'output': str(directory / 'audit.json')}))


if __name__ == '__main__':
    main()

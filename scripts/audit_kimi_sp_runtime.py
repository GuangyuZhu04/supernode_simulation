"""Validate frozen SP runtime provenance without importing current engine code.

Run in a fresh subprocess. The recovered snapshot is explicitly first on the
import path. Case artifacts and their original headers are strictly read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    workspace = Path(__file__).resolve().parents[1]
    frozen = directory / 'runtime-frozen'
    proof = read(frozen / 'manifest.json')
    worker_proof = read(directory / 'audit/worker-provenance.json')
    manifest = read(directory / 'manifest.json')
    expected_order = ['core', 'config', 'analysis', 'structured', 'components', 'topology', 'engine', 'streaming', 'calibration']
    assert proof['module_order'] == expected_order, 'unexpected frozen fingerprint recipe'
    assert [item['module'] for item in proof['modules']] == expected_order
    module_bytes = []
    for item in proof['modules'] + proof['full_package_modules']:
        path = frozen / item['path']
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == item['sha256'], f'frozen source hash mismatch: {path}'
        assert len(content) == item['bytes'], f'frozen source size mismatch: {path}'
    for item in proof['modules']:
        module_bytes.append((frozen / item['path']).read_bytes())
    digest = hashlib.sha256(b''.join(module_bytes)).hexdigest()
    assert digest == proof['snapshot_sha256']
    assert proof['runtime_snapshot_id'] == 'sha256:' + digest
    snapshot_path = (frozen / proof['config_normalizer']['package_path']).resolve()
    sys.path.insert(0, str(snapshot_path))
    from supernode_sim.config import load_config
    import supernode_sim.config as frozen_config
    assert Path(frozen_config.__file__).resolve().is_relative_to(snapshot_path), 'configuration loader escaped frozen package'
    current_bytes = [(workspace / 'supernode_sim' / (module + '.py')).read_bytes() for module in expected_order]
    normalized = {}
    for case in manifest['cases']:
        config_bytes = json.dumps(load_config(directory / case['config']).model_dump(), sort_keys=True).encode()
        runtime_fp = hashlib.sha256(config_bytes)
        current_fp = hashlib.sha256(config_bytes)
        for content in module_bytes:
            runtime_fp.update(content)
        for content in current_bytes:
            current_fp.update(content)
        normalized[case['name']] = {'runtime_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                                    'runtime_snapshot_fingerprint': runtime_fp.hexdigest(),
                                    'current_disk_fingerprint': current_fp.hexdigest()}
    anchors = []
    for anchor in proof['anchor_cases']:
        artifact = read(directory / 'cases' / (anchor['case'] + '.json'))
        expected = normalized[anchor['case']]
        match = (expected['runtime_snapshot_fingerprint'] == artifact['fingerprint'] == anchor['recorded_fingerprint'] == anchor['recovered_fingerprint']
                 and expected['runtime_config_sha256'] == artifact['config_sha256'] == anchor['config_sha256'])
        assert match, f"frozen runtime anchor failed: {anchor['case']}"
        anchors.append({'case': anchor['case'], 'verified': match})
    assert len(anchors) == 3
    assert worker_proof['continuing_original_workers_observed']
    assert worker_proof['eager_numerical_imports_before_first_case']
    assert not worker_proof['reload_or_dynamic_source_execution_found']
    first_change = datetime.fromisoformat(worker_proof['first_external_numerical_source_write_Asia_Shanghai'])
    workers, logs = {}, {}
    for worker in worker_proof['workers']:
        assert datetime.fromisoformat(worker['created_Asia_Shanghai']) < first_change
        shard = worker['shard_index']
        workers[shard] = worker
        records = []
        for line in (directory / worker['log']).read_text(encoding='utf-8-sig').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # An actively written last line may be incomplete.
        logs[shard] = records
    cases, issues = {}, []
    for index, case in enumerate(manifest['cases']):
        name = case['name']
        path = directory / 'cases' / (name + '.json')
        if not path.exists():
            continue
        content = path.read_bytes()
        artifact = json.loads(content)
        shard = index % len(workers)
        worker = workers[shard]
        expected = normalized[name]
        started = [row for row in logs[shard] if row.get('case') == name and row.get('stage') == 'case_started']
        completed = [row for row in logs[shard] if row.get('case') == name and row.get('stage') == 'case_complete']
        conflicts = [other for other, log in logs.items() if other != shard and any(row.get('case') == name and row.get('stage') == 'case_complete' for row in log)]
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone(timedelta(hours=8)))
        config_match = artifact.get('config_sha256') == expected['runtime_config_sha256']
        metrics_match = bool(completed) and completed[-1].get('metrics') == artifact.get('all', {}).get('metrics')
        producer_match = bool(started and completed and not conflicts and metrics_match and mtime >= datetime.fromisoformat(worker['created_Asia_Shanghai']))
        verified = config_match and producer_match
        if not verified:
            issues.append(f'{name}: frozen config/worker attribution not yet verified')
        cases[name] = {**expected, 'observed_disk_fingerprint': artifact.get('fingerprint'),
                       'recorded_config_sha256': artifact.get('config_sha256'),
                       'runtime_config_sha256_matches': config_match,
                       'observed_disk_matches_runtime_snapshot': artifact.get('fingerprint') == expected['runtime_snapshot_fingerprint'],
                       'observed_disk_matches_current_disk': artifact.get('fingerprint') == expected['current_disk_fingerprint'],
                       'runtime_snapshot_id': proof['runtime_snapshot_id'], 'runtime_provenance_verified': verified,
                       'producer_shard': shard, 'producer_pid': worker['pid'], 'producer_log': worker['log'],
                       'producer_started_Asia_Shanghai': worker['created_Asia_Shanghai'],
                       'producer_log_started_count': len(started), 'producer_log_completed_count': len(completed),
                       'producer_log_metrics_match': metrics_match, 'producer_conflicting_shards': conflicts,
                       'artifact_sha256': hashlib.sha256(content).hexdigest(),
                       'artifact_mtime_Asia_Shanghai': mtime.isoformat(timespec='seconds')}
    guards = {str((directory / 'manifest.json').relative_to(directory)): hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
              'hardware.profile.json': hashlib.sha256((directory / 'hardware.profile.json').read_bytes()).hexdigest(),
              'runtime-frozen/manifest.json': hashlib.sha256((frozen / 'manifest.json').read_bytes()).hexdigest(),
              'audit/worker-provenance.json': hashlib.sha256((directory / 'audit/worker-provenance.json').read_bytes()).hexdigest()}
    guards.update({case['config']: hashlib.sha256((directory / case['config']).read_bytes()).hexdigest() for case in manifest['cases']})
    result = {'runtime_snapshot_id': proof['runtime_snapshot_id'], 'frozen_package': str(snapshot_path),
              'verified_at_Asia_Shanghai': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
              'input_file_sha256': guards,
              'module_hashes_verified': True, 'normalizer_import_verified': True,
              'anchor_proofs': anchors, 'worker_continuity_evidence': worker_proof,
              'observed_disk_fingerprints_drifting_from_runtime': sum(not row['observed_disk_matches_runtime_snapshot'] for row in cases.values()),
              'verified_case_count': sum(row['runtime_provenance_verified'] for row in cases.values()),
              'case_count': len(cases), 'expected_case_count': len(manifest['cases']), 'issues': issues, 'cases': cases}
    print(json.dumps(result))


if __name__ == '__main__':
    main()

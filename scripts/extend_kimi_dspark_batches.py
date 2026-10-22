"""Add global batch16/64 pairs; retain the existing batch32 and every result."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kimi_10t_950dt_dspark import DEFAULT_OUTPUT, read, save, sha, dspark_budget
from supernode_sim.config import Config

directory = DEFAULT_OUTPUT
manifest = read(directory / 'manifest.json')
before = sha(manifest)
existing = {item['name'] for item in manifest['cases']}
templates = [item for item in manifest['cases'] if item['group'] == 'main' and item['batch'] == 32]
added = []
for batch in (16, 64):
    for template in templates:
        item = copy.deepcopy(template)
        item['name'] = template['name'].replace('-b32-', f'-b{batch}-')
        if item['name'] in existing:
            continue
        item['batch'] = batch
        item['comparison_case'] = item['name'].removesuffix('-baseline')
        item['config'] = 'configs/' + item['name'] + '.json'
        raw = read(directory / template['config'])
        raw['workload']['batch_size'] = batch
        raw['workload']['input_lengths'] = [item['length']] * batch
        raw['workload']['output_lengths'] = [item['output_length']] * batch
        raw['sources']['workload'] = {'status': 'user', 'description':
            'User requested adding batch16/32/64. Existing batch32 is retained; this adds batch16/64 across the same main matrix.'}
        if item['variant'] == 'dspark':
            raw['execution']['features']['speculative_decode'] = dspark_budget(raw, 4, 1)
        config = Config.model_validate(raw)
        item['config_sha256'] = sha(config.model_dump())
        assert item['model_sha256'] == sha(config.model.model_dump())
        assert item['hardware_sha256'] == sha(config.hardware.model_dump())
        save(directory / item['config'], config.model_dump())
        added.append(item)
        existing.add(item['name'])
manifest['cases'].extend(added)
manifest['batches'] = [1, 16, 32, 64, 256]
manifest['counts'].update(main_workloads=120, added_batch_workloads=48,
    added_batch_paired_runs=96, total_runs=len(manifest['cases']))
manifest['batch_extension'] = {'requested_batches': [16, 32, 64], 'reused_batch': 32,
    'added_batches': [16, 64], 'scope': 'same 3 card counts x 2 TP layouts x 4 contexts; paired DSpark on/off',
    'hardware_reference': 'Corresponding reference batch32 configuration; hardware/topology are unchanged.'}
save(directory / 'manifest.json', manifest)
save(directory / 'batch-extension.json', {'previous_manifest_sha256': before,
    'new_manifest_sha256': sha(manifest), 'added_cases': [item['name'] for item in added],
    'existing_results_overwritten': False})
print({'added_runs': len(added), 'total_runs': len(manifest['cases'])})

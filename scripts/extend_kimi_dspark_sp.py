"""Add SP16/TP16 with EP=N/16, preserving and reusing the SP8 runs."""
import copy
import hashlib
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kimi_10t_950dt_dspark import DEFAULT_OUTPUT, read, save, sha, code_hashes, dspark_budget
from supernode_sim.config import Config

OUT = DEFAULT_OUTPUT
manifest = read(OUT / 'manifest.json')
archive = OUT / 'archive/pre-sp16'
archive.mkdir(parents=True, exist_ok=True)
if not (archive / 'manifest.json').exists():
    # Preserve the prior aggregate views and every numerical/configuration hash.
    protected = {}
    for item in manifest['cases']:
        dest = OUT / 'cases' / item['name']
        paths = [OUT / item['config'], dest / 'case.json', dest / 'full/result.json',
                 dest / 'full/config.resolved.json']
        if (dest / 'warm-decode/result.json').exists():
            paths += [dest / 'warm-decode/result.json', dest / 'warm-decode/config.resolved.json']
        for path in paths:
            with path.open('rb') as handle:
                protected[str(path.relative_to(OUT))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    original_files = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.html', '.md', '.png', '.jpg', '.txt')]
    for path in original_files:
        shutil.copy2(path, archive / path.name)
    save(archive / 'numerical-protection.json', protected)
    html = (OUT / 'report.html').read_text(encoding='utf-8')
    for path in original_files:
        html = html.replace(f'href="{path.name}"', f'href="archive/pre-sp16/{path.name}"')
    html = html.replace('<body>', '<body><div style="padding:12px;background:#fff0d9">SP16扩展前的258组结果快照。<a href="report.html">查看最新汇总</a></div>', 1)
    (OUT / 'report.before-sp16.html').write_text(html, encoding='utf-8')

templates = [item for item in manifest['cases'] if item['tp'] == 8]
existing = {item['name'] for item in manifest['cases']}
hashes = code_hashes()
added, reused = [], []
for template in templates:
    source = read(OUT / template['config'])
    assert source['execution']['sequence_parallel'] is True
    index = read(OUT / 'cases' / template['name'] / 'case.json')
    assert index['fingerprint'] == sha({'config': source, 'source': hashes})
    reused.append(template['name'])
    item = copy.deepcopy(template)
    item['tp'], item['ep'] = 16, item['cards'] // 16
    convert = lambda name: re.sub(r'-tp8-ep\d+-', f"-tp16-ep{item['ep']}-", name, count=1)
    item['name'] = convert(template['name'])
    if item['name'] in existing:
        continue
    item['sp_parent_case'] = template['name']
    item['comparison_case'] = convert(template.get('comparison_case', template['reference_case']))
    item['config'] = 'configs/' + item['name'] + '.json'
    raw = copy.deepcopy(source)
    raw['execution'].update(tp=16, ep=item['ep'], sequence_parallel=True)
    raw['sources']['execution.parallelism'] = {'status': 'user', 'description':
        'SP16 shares TP16 ranks; EP=total NPUs/16, DP=PP=1. Physical hardware, topology and request batch are unchanged.'}
    raw['assumptions'].append('SP16/TP16 reuses the physical topology; TP16 spans two 8-NPU hosts. MLA compressed KV is TP-replicated; KDA heads are TP-sharded. Draft scalar budgets are recomputed with ceil(B/EP).')
    if item['variant'] == 'dspark':
        raw['execution']['features']['speculative_decode'] = dspark_budget(raw, item['accepted_length'], item['cost_multiplier'])
    config = Config.model_validate(raw)
    assert sha(config.model.model_dump()) == item['model_sha256']
    assert sha(config.hardware.model_dump()) == item['hardware_sha256']
    assert config.workload.model_dump() == source['workload']
    assert config.execution.precision.model_dump() == source['execution']['precision']
    item['config_sha256'] = sha(config.model_dump())
    save(OUT / item['config'], config.model_dump())
    added.append(item)
    existing.add(item['name'])
manifest['cases'].extend(added)
manifest['counts'].update(main_workloads=180, sp16_added_runs=134,
    sp8_reused_runs=134, additional_sensitivity_runs=20,
    topology_sensitivity_runs=12, main_runs=360, total_runs=len(manifest['cases']))
manifest['sp_extension'] = {'requested_sp_degrees': [8, 16],
    'contract': 'SP shares TP ranks, TP=SP; TP*EP*PP*DP=N, DP=PP=1',
    'mapping': [{'cards': n, 'sp8_tp': 8, 'sp8_ep': n//8, 'sp16_tp': 16, 'sp16_ep': n//16} for n in (64, 128, 256)],
    'scope': 'All five batches and four contexts; retain matching topology/algorithm and DSpark sensitivity pairs.',
    'preserved_legacy_tp1_runs': 124}
save(OUT / 'manifest.json', manifest)
save(OUT / 'sp-extension.json', {'added_cases': [i['name'] for i in added], 'reused_sp8_cases': reused,
    'added_count': len(added), 'archive': 'archive/pre-sp16',
    'numerical_protection': 'archive/pre-sp16/numerical-protection.json',
    'hardware_model_precision_workload_unchanged': True, 'validated': True})
print({'added': len(added), 'reused_sp8': len(reused), 'total': len(manifest['cases']), 'mapping': manifest['sp_extension']['mapping']})

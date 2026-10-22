"""Check final artifacts, frozen replay and provenance without rerunning the matrix."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/kimi-k3-10t-950dt-dspark'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_kimi_10t_950dt_dspark import read, save, sha

manifest, summary, audit = (read(OUT / name) for name in ('manifest.json', 'summary.json', 'audit.json'))
assert summary['aggregate']['complete'] and audit['passed']
expected = len(manifest['cases'])
assert summary['aggregate']['completed_runs'] == audit['audited_cases'] == expected
required = ('result.json', 'config.resolved.json', 'report.md', 'component_timeline.csv',
            'latency_breakdown.csv', 'memory_peaks.csv', 'model_parameters.csv', 'state_memory.csv', 'resources.csv')
scopes = 0
for item in manifest['cases']:
    config = read(OUT / item['config'])
    assert config['hardware']['npu_count'] == item['cards']
    assert config['hardware']['su_size'] == item['su_size']
    assert config['execution']['tp'] == item['tp'] and config['execution']['ep'] == item['ep']
    assert config['workload']['batch_size'] == item['batch']
    assert set(config['workload']['input_lengths']) == {item['length']}
    assert set(config['workload']['output_lengths']) == {item['output_length']}
    dest = OUT / 'cases' / item['name']
    index = read(dest / 'case.json')
    expected_scopes = ['full'] + (['warm-decode'] if index['full_status'] != 'feasible' else [])
    assert (dest / 'warm-decode').exists() == ('warm-decode' in expected_scopes)
    for scope in expected_scopes:
        assert all((dest / scope / name).is_file() for name in required)
        assert sha(read(dest / scope / 'config.resolved.json')) == item['config_sha256']
        scopes += 1

case = 'n128-tp16-ep8-b32-128K' if 'sp_extension' in manifest else 'n128-tp8-ep16-b16-128K'
original = read(OUT / 'cases' / case / 'full/result.json')
replayed = read(OUT / 'replayed' / case / 'all/result.json')
assert original['status'] == replayed['status'] == 'feasible'
metric_deltas = {}
for key in ('duration_s', 'ttft_s', 'tpot_s', 'input_tokens', 'output_tokens'):
    x, y = original['metrics'][key], replayed['metrics'][key]
    assert math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10), key
    metric_deltas[key] = abs(x - y)
resource_deltas = {}
for key in ('resource_work', 'resource_busy_s', 'memory_peaks_bytes', 'memory_final_bytes'):
    old, new = original['simulation'][key], replayed['simulation'][key]
    maximum = 0.0
    for resource in old.keys() | new.keys():
        x, y = old.get(resource, 0), new.get(resource, 0)
        assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6), (key, resource)
        maximum = max(maximum, abs(x - y))
    resource_deltas[key] = maximum

repro = read(OUT / 'reproducibility.json')
for rel, digest in repro['source_sha256'].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((OUT / 'runtime' / rel).read_bytes()).hexdigest() == digest
for rel, digest in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
protection = OUT / 'archive/pre-sp16/numerical-protection.json'
preserved = 0
if protection.exists():
    for rel, digest in read(protection).items():
        with (OUT / rel).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, rel
        preserved += 1
    assert len(summary['sp_comparisons']) == 134
    assert '50 passed' in (OUT / 'sp-regression-tests.log').read_text(encoding='utf-8-sig')

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href' and v and not v.startswith('#'))

links = Links()
links.feed((OUT / 'report.html').read_text(encoding='utf-8'))
for target in links.links:
    path = (OUT / target).resolve()
    assert path.is_relative_to(OUT.resolve()) and path.is_file(), target
log = (OUT / 'regression-tests.log').read_text(encoding='utf-8-sig')
assert '678 passed' in log and '19 subtests passed' in log
verification = {'passed': True, 'completed_runs': expected, 'case_scopes_verified': scopes,
    'pre_sp16_numerical_files_preserved': preserved, 'sp_pairs_verified': len(summary.get('sp_comparisons', [])),
    'sp_parallel_regression_tests_passed': 50 if protection.exists() else None,
    'required_files_per_scope': list(required), 'resolved_configs_match': True,
    'hardware_model_workload_metadata_match': True, 'source_and_snapshot_hashes_match': True,
    'original_model_and_reference_manifest_unchanged': True,
    'frozen_replay': {'case': case, 'matched': True, 'metric_absolute_deltas': metric_deltas,
                      'resource_maximum_absolute_deltas': resource_deltas},
    'report_static_links_verified': len(links.links), 'engine_regression_tests_passed': 678,
    'engine_subtests_passed': 19,
    'browser': {'initial_report_visual_check': True, 'filter_and_failure_scope_check': True,
        'final_live_check': 'unavailable: browser inventory became empty',
        'configuration_screenshot': 'frontend-proof.jpg'},
    'static_batch_chart': 'batch_16_32_64_comparison.png',
    'sp_chart': 'sp8_sp16_comparison.png' if protection.exists() else None}
save(OUT / 'verification.json', verification)
repro.update(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    working_tree_modifications=True,
    dependencies={k: importlib.metadata.version(k) for k in ('pydantic', 'PyYAML', 'Pillow', 'pytest')},
    completed_runs=expected, batches=[1, 16, 32, 64, 256], sp_degrees=[8, 16],
    commands=['python scripts/run_kimi_10t_950dt_dspark.py --prepare',
              'python scripts/extend_kimi_dspark_batches.py',
              'python scripts/run_kimi_10t_950dt_dspark.py --run',
              'python scripts/extend_kimi_dspark_sp.py',
              'python scripts/run_kimi_10t_950dt_dspark.py --run',
              'python scripts/summarize_kimi_10t_950dt_dspark.py',
              'python scripts/render_kimi_dspark_batch_plot.py',
              'python scripts/render_kimi_sp_comparison.py'],
    standalone_replay=f'python reproduce.py --case {case}',
    verification='verification.json', runtime_snapshot='runtime/')
save(OUT / 'reproducibility.json', repro)
print(json.dumps({'passed': True, 'runs': expected, 'scopes': scopes,
    'frozen_replay_metric_deltas': metric_deltas}, ensure_ascii=False))

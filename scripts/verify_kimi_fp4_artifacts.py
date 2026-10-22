"""Freeze and verify the mixed-FP4 study, including a real frozen-code replay."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import importlib.metadata
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_kimi_10t_950dt_dspark import read, save, sha, code_hashes, Config, parameter_inventory
from run_kimi_10t_950dt_dspark_fp4 import OUTPUT, file_sha, numerical_snapshot

SCRIPTS = ('run_kimi_10t_950dt_dspark.py', 'summarize_kimi_10t_950dt_dspark.py',
    'run_kimi_10t_950dt_dspark_fp4.py', 'summarize_kimi_10t_950dt_dspark_fp4.py',
    'render_kimi_fp4_comparison.py', 'verify_kimi_fp4_artifacts.py',
    'kimi_10t_950dt_dspark_report.html', 'kimi_10t_950dt_dspark_fp4_report.html')
REPLAY_CASE = 'n128-tp16-ep8-b32-128K'


def freeze(directory):
    baseline = Path(read(directory / 'manifest.json')['fp8_baseline_directory'])
    sources = list((ROOT / 'supernode_sim').glob('*.py'))
    sources += [ROOT / 'scripts' / name for name in SCRIPTS]
    sources += [ROOT / 'pyproject.toml']
    fingerprints = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = directory / 'runtime' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        fingerprints[str(relative)] = file_sha(destination)
    shutil.copy2(baseline / 'reproduce.py', directory / 'reproduce.py')
    save(directory / 'runtime-manifest.json', {'files': fingerprints,
        'reproduce_sha256': file_sha(directory / 'reproduce.py'),
        'scope': 'Frozen numerical engine, runners, reports and project dependency declaration'})
    print({'stage': 'frozen', 'files': len(fingerprints)}, flush=True)


def verify(directory):
    manifest, summary, audit = (read(directory / name) for name in ('manifest.json', 'summary.json', 'audit.json'))
    assert summary['aggregate']['complete'] and audit['passed']
    assert read(directory / 'precision-audit.json')['passed']
    expected = len(manifest['cases'])
    assert expected == summary['aggregate']['completed_runs'] == audit['audited_cases'] == 392
    assert len(summary['precision_comparison']['rows']) == 392
    required = ('result.json', 'config.resolved.json', 'report.md', 'component_timeline.csv',
                'latency_breakdown.csv', 'memory_peaks.csv', 'model_parameters.csv', 'state_memory.csv', 'resources.csv')
    scopes, tensor_rows, routed_rows = 0, 0, 0
    for item in manifest['cases']:
        config = Config.model_validate(read(directory / item['config']))
        raw = config.model_dump()
        assert raw['hardware']['npu_count'] == item['cards']
        assert config.execution.tp == item['tp'] and config.execution.ep == item['ep']
        assert config.workload.batch_size == item['batch']
        assert set(config.workload.input_lengths) == {item['length']}
        assert set(config.workload.output_lengths) == {item['output_length']}
        assert config.execution.precision.weight == 'fp8'
        assert config.execution.precision.routed_expert_weight == 'fp4'
        inventory = {t.name: t for t in parameter_inventory(raw['model'])}
        dest = directory / 'cases' / item['name']
        index = read(dest / 'case.json')
        expected_scopes = ['full'] + (['warm-decode'] if index['full_status'] != 'feasible' else [])
        assert (dest / 'warm-decode').exists() == ('warm-decode' in expected_scopes)
        for scope in expected_scopes:
            assert all((dest / scope / name).is_file() for name in required)
            assert sha(read(dest / scope / 'config.resolved.json')) == item['config_sha256']
            scopes += 1
            seen = set()
            with (dest / scope / 'model_parameters.csv').open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    tensor = inventory[row['name']]
                    assert row['name'] not in seen
                    seen.add(row['name'])
                    count = tensor.local_elements(config.execution.tp, config.execution.ep)
                    dtype = 'fp4' if tensor.ep_shard else 'fp8'
                    weight = count * (0.5 if tensor.ep_shard else 1)
                    scale = math.ceil(count / 128) * 4 if count else 0
                    assert row['weight_dtype'] == dtype
                    assert float(row['weight_bytes_per_rank']) == weight
                    assert float(row['scale_bytes_per_rank']) == scale
                    assert float(row['weight_physical_bytes']) == weight * item['cards']
                    tensor_rows += 1
                    routed_rows += bool(tensor.ep_shard)
            assert seen == inventory.keys()
    baseline = Path(manifest['fp8_baseline_directory'])
    protection = read(directory / 'fp8-protection.json')
    assert protection['files'] == numerical_snapshot(baseline), 'FP8 baseline has changed'
    protected = len(protection['files'])
    frozen = read(directory / 'runtime-manifest.json')
    for relative, digest in frozen['files'].items():
        assert file_sha(ROOT / relative) == file_sha(directory / 'runtime' / relative) == digest
    assert file_sha(directory / 'reproduce.py') == frozen['reproduce_sha256']
    repro = read(directory / 'reproducibility.json')
    assert repro['source_sha256'] == code_hashes()
    for relative, digest in manifest['source_sha256'].items():
        assert file_sha(ROOT / relative) == digest
    original = read(directory / 'cases' / REPLAY_CASE / 'full/result.json')
    replayed = read(directory / 'replayed' / REPLAY_CASE / 'all/result.json')
    assert original['status'] == replayed['status'] == 'feasible'
    metric_deltas, resource_deltas = {}, {}
    for key in ('duration_s', 'ttft_s', 'tpot_s', 'input_tokens', 'output_tokens'):
        x, y = original['metrics'][key], replayed['metrics'][key]
        assert math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10), key
        metric_deltas[key] = abs(x - y)
    for key in ('resource_work', 'resource_busy_s', 'memory_peaks_bytes', 'memory_final_bytes'):
        old, new = original['simulation'][key], replayed['simulation'][key]
        maximum = 0
        for resource in old.keys() | new.keys():
            x, y = old.get(resource, 0), new.get(resource, 0)
            assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6), (key, resource)
            maximum = max(maximum, abs(x - y))
        resource_deltas[key] = maximum
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.links.extend(v for k, v in attrs if k == 'href' and v and not v.startswith('#'))
    links = Links()
    links.feed((directory / 'report.html').read_text(encoding='utf-8'))
    # verification.json is the output of this verification itself.
    for target in links.links:
        path = (directory / target).resolve()
        assert path.is_relative_to(directory.resolve()), target
        assert path.is_file() or target == 'verification.json', target
    assert '16 passed' in (directory / 'regression-tests.log').read_text(encoding='utf-8-sig')
    verification = {'passed': True, 'completed_runs': expected, 'case_scopes_verified': scopes,
        'exported_tensor_rows_checked': tensor_rows, 'routed_fp4_tensor_rows_checked': routed_rows,
        'all_other_weights_fp8_checked': True, 'unchanged_scale_metadata_checked': True,
        'fp8_baseline_files_preserved': protected, 'sp_pairs_verified': len(summary['sp_comparisons']),
        'numeric_config_delta': 'Only execution.precision.routed_expert_weight: null -> fp4',
        'resolved_configs_and_fingerprints_match': True, 'frozen_source_hashes_match': True,
        'required_files_per_scope': list(required), 'report_static_links_verified': len(links.links),
        'existing_component_and_precision_tests_passed': 16,
        'frozen_replay': {'case': REPLAY_CASE, 'matched': True,
            'metric_absolute_deltas': metric_deltas, 'resource_maximum_absolute_deltas': resource_deltas},
        'browser_verification': read(directory / 'report-ui-check.json') if (directory / 'report-ui-check.json').exists() else None,
        'chart_visual_verification': read(directory / 'chart-visual-check.json') if (directory / 'chart-visual-check.json').exists() else None}
    save(directory / 'verification.json', verification)
    repro.update(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dependencies={k: importlib.metadata.version(k) for k in ('pydantic', 'PyYAML', 'Pillow', 'pytest')},
        completed_runs=expected, precision_variant='routed_fp4_other_fp8',
        fp8_baseline=str(baseline), runtime_snapshot='runtime/', verification='verification.json',
        command='python scripts/run_kimi_10t_950dt_dspark_fp4.py --prepare --run',
        commands=['python scripts/run_kimi_10t_950dt_dspark_fp4.py --prepare --run',
                  'python scripts/summarize_kimi_10t_950dt_dspark_fp4.py',
                  'python scripts/render_kimi_fp4_comparison.py',
                  'python scripts/verify_kimi_fp4_artifacts.py --freeze',
                  f'python outputs/kimi-k3-10t-950dt-dspark-fp4/reproduce.py --case {REPLAY_CASE}',
                  'python scripts/verify_kimi_fp4_artifacts.py --check'],
        standalone_replay=f'python reproduce.py --case {REPLAY_CASE}')
    save(directory / 'reproducibility.json', repro)
    progress = read(directory / 'progress.json')
    progress.update(status='complete', completed_runs=expected, total_runs=expected,
                    finalized_at=datetime.now(timezone.utc).isoformat(), verification='verification.json')
    save(directory / 'progress.json', progress)
    print(verification, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output)
    if args.check:
        verify(args.output)

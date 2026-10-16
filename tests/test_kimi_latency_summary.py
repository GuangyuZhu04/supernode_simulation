"""Matrix summaries export saved attribution without fabricating archive data."""
import copy
import csv
import importlib
import json
from pathlib import Path
import sys

import pytest

from scripts import summarize_kimi_10t_inference as base_summary
from supernode_sim.report import LATENCY_COMPONENT_LABELS
from supernode_sim.streaming import run_streaming
from tests.test_feedback import inference_configuration
from tests.test_report_export import artifact_directory


@pytest.fixture(scope='module')
def execution():
    config = inference_configuration()
    return config, run_streaming(config), run_streaming(config, phase='decode')


def case(name='new-case'):
    return {'name': name, 'top_k': 2, 'context': 'tiny', 'input_length': 8,
            'batch_size': 2, 'tp': 1, 'sp': 1, 'ep': 1, 'config': 'configs/case.json'}


def assert_breakdown(row, record):
    assert row['available']
    assert row['total_s'] == record['total_s']
    assert sum(row[key] for key in LATENCY_COMPONENT_LABELS) == pytest.approx(record['total_s'], rel=1e-9, abs=1e-20)
    assert {key: row[key] for key in LATENCY_COMPONENT_LABELS} == record['components_s']


def test_case_phase_attribution_uses_saved_full_and_standalone_records(execution):
    _, full, standalone = execution
    payload = {'all': full, 'standalone_decode': standalone, 'fingerprint': 'recorded-engine', 'config_sha256': 'recorded-config'}
    before = copy.deepcopy(payload)
    rows = base_summary.case_latency_rows(case(), payload)
    by_metric = {(row['phase'], row['metric']): row for row in rows}
    assert len(rows) == 6
    for metric in ('ttft', 'tpot'):
        assert_breakdown(by_metric['all', metric], full['metrics']['latency_breakdown'][metric])
    assert_breakdown(by_metric['prefill', 'ttft'], full['metrics']['latency_breakdown']['ttft'])
    assert_breakdown(by_metric['standalone_decode', 'tpot'], standalone['metrics']['latency_breakdown']['tpot'])
    for phase, metric in [('prefill', 'tpot'), ('standalone_decode', 'ttft')]:
        row = by_metric[phase, metric]
        assert row['available'] is False
        assert row['total_s'] is None
        assert row['unavailable_reason']
    assert by_metric['standalone_decode', 'ttft']['unavailable_reason'] == standalone['metrics']['latency_breakdown']['ttft_unavailable_reason']
    assert {row['recorded_fingerprint'] for row in rows} == {'recorded-engine'}
    assert {row['artifact'] for row in rows} == {'cases/new-case.json'}
    assert payload == before


@pytest.mark.parametrize('payload', [None, {'all': {'status': 'feasible', 'metrics': {'ttft_s': 10, 'tpot_s': 2},
                                                    'phase_metrics': {'prefill': {'status': 'feasible', 'duration_s': 10}}},
                                         'fingerprint': 'archive'}])
def test_missing_saved_breakdowns_remain_unavailable(payload):
    rows = base_summary.case_latency_rows(case('archived'), payload)
    assert len(rows) == 6
    assert all(not row['available'] and row['total_s'] is None for row in rows)
    assert all(row['unavailable_reason'] for row in rows)
    assert all(row[key] is None for row in rows for key in LATENCY_COMPONENT_LABELS)


def write_fixture(directory, config, result):
    planned_phase = {key: 0 for key in (
        'total_network_payload_bytes', 'same_su_payload_bytes', 'cross_su_payload_bytes',
        'dispatch_network_bytes', 'combine_network_bytes', 'tp_all_reduce_payload_bytes',
        'tp_all_gather_payload_bytes', 'tp_reduce_scatter_payload_bytes', 'physical_su_hopbytes',
        'physical_so_hopbytes', 'valid_input_rows', 'moe_network_payload_bytes',
        'moe_semantic_assignments_with_local_bytes', 'moe_local_no_network_bytes')}
    entry = case()
    files = {
        'manifest.json': {'cases': [entry]},
        'audit/traffic-formulas.json': {'matrix': [{**entry, 'prefill': planned_phase, 'one_decode_step': planned_phase}],
                                      'cache_capacity': {'rows': []}},
        'hardware.profile.json': {'hardware': config.model_dump()['hardware']},
        'configs/case.json': config.model_dump(),
        'cases/new-case.json': {'all': result, 'standalone_decode': None, 'fingerprint': 'old-observation',
                               'config_sha256': 'old-config'},
    }
    for filename, payload in files.items():
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')


@pytest.mark.parametrize('summarizer', ['base', 'sp'])
def test_matrix_cli_exports_new_csv_and_preserves_raw_case(monkeypatch, execution, summarizer):
    config, result, _ = execution
    if summarizer == 'sp':
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
        module = importlib.import_module('summarize_kimi_sp_inference')
        # The tiny fixture has no historical frozen workers. Retain the
        # unverified state; the real provenance implementation is unchanged.
        monkeypatch.setattr(module, 'runtime_provenance', lambda directory: {
            'cases': {}, 'runtime_snapshot_id': 'fixture-unverified', 'verified_case_count': 0,
            'observed_disk_fingerprints_drifting_from_runtime': 0})
        monkeypatch.setattr(module, 'archive_baseline', lambda directory: {})
        monkeypatch.setattr(module, 'comparisons', lambda rows, baseline: [])
        monkeypatch.setattr(module, 'md_report', lambda data, comparison: module.LATENCY_NOTE)
    else:
        module = base_summary
        # The fixed 48-case narrative is outside the numerical CSV contract.
        monkeypatch.setattr(module, 'make_report', lambda *args: module.LATENCY_NOTE)
    with artifact_directory() as directory:
        write_fixture(directory, config, result)
        path = directory / 'cases/new-case.json'
        original = path.read_bytes()
        monkeypatch.setattr(sys, 'argv', ['summary', str(directory)])
        module.main()
        assert path.read_bytes() == original
        with (directory / 'latency_breakdown.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 6
        assert {row['case'] for row in rows} == {'new-case'}
        full = [row for row in rows if row['phase'] == 'all']
        assert {row['metric'] for row in full} == {'ttft', 'tpot'}
        for row in full:
            record = result['metrics']['latency_breakdown'][row['metric']]
            assert float(row['total_s']) == record['total_s']
            assert {key: float(row[key]) for key in LATENCY_COMPONENT_LABELS} == record['components_s']
        absent = [row for row in rows if row['phase'] == 'standalone_decode']
        assert all(row['available'] == 'False' and row['total_s'] == '' for row in absent)
        assert all(row['recorded_fingerprint'] == 'old-observation' for row in rows)
        assert 'href="latency_breakdown.csv"' in (directory / 'report.html').read_text(encoding='utf-8')
        with (directory / 'summary.csv').open(encoding='utf-8-sig', newline='') as handle:
            summary = next(csv.DictReader(handle))
        assert summary['tpop_s'] == summary['tpot_s']


def test_current_fingerprint_matches_case_producer(execution):
    from scripts.run_kimi_10t_inference import fingerprint

    config, _, _ = execution
    with artifact_directory() as directory:
        path = directory / 'config.json'
        path.write_text(json.dumps(config.model_dump()), encoding='utf-8')
        assert base_summary.current_fingerprint(path) == fingerprint(config)

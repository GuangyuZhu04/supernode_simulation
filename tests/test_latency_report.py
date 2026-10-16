"""Latency artifacts preserve the numerical attribution, including missing metrics."""
import csv
import json

import pytest

from supernode_sim.analysis import run
from supernode_sim.feedback import feedback
from supernode_sim.report import LATENCY_COMPONENT_LABELS, write_report
from tests.test_feedback import inference_configuration, single_search
from tests.test_report_export import artifact_directory


def read_rows(directory):
    with (directory / 'latency_breakdown.csv').open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def assert_record(row, record):
    assert row['available'] == 'True'
    assert float(row['total_s']) == record['total_s']
    assert float(row['total_s']) == pytest.approx(sum(float(row[key]) for key in LATENCY_COMPONENT_LABELS), rel=1e-9, abs=1e-20)
    for key in LATENCY_COMPONENT_LABELS:
        assert float(row[key]) == record['components_s'][key]
    assert row['replica'] == str(record['replica'])
    assert row['sequence'] == str(record['sequence'])


def test_simulated_latency_csv_matches_json_and_survives_compact_export(monkeypatch):
    result = run(inference_configuration())
    monkeypatch.setattr('supernode_sim.report.AUTO_DETAIL_THRESHOLD', 1)
    with artifact_directory() as directory:
        write_report(result, directory)
        rows = read_rows(directory)
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert persisted['artifact_export']['compacted']
        assert len(rows) == 2
        assert [row['metric'] for row in rows] == ['ttft', 'tpot']
        assert rows[1]['alias'] == 'tpop'
        assert persisted['metrics']['tpop_s'] == persisted['metrics']['tpot_s']
        for row in rows:
            original = result['metrics']['latency_breakdown'][row['metric']]
            assert_record(row, original)
            assert persisted['metrics']['latency_breakdown'][row['metric']] == original


@pytest.mark.parametrize('case', ['estimate', 'training', 'infeasible', 'one_token'])
def test_undefined_latency_is_exported_empty_not_zero(case):
    patch = {'workload': {'scenario': 'training'}} if case == 'training' else {}
    if case == 'infeasible':
        patch = {'hardware': {'npu': {'hbm_bytes': 1}}}
    elif case == 'one_token':
        patch = {'workload': {'output_length': 1}}
    result = run(inference_configuration(patch), mode='estimate' if case == 'estimate' else 'simulate')
    with artifact_directory() as directory:
        write_report(result, directory)
        rows = read_rows(directory)
        missing = rows if case != 'one_token' else [rows[1]]
        for row in missing:
            assert row['available'] == 'False'
            assert row['unavailable_reason']
            assert row['total_s'] == ''
            assert all(row[key] == '' for key in LATENCY_COMPONENT_LABELS)
        if case == 'one_token':
            assert_record(rows[0], result['metrics']['latency_breakdown']['ttft'])
            assert rows[1]['unavailable_reason'] == result['metrics']['latency_breakdown']['tpot_unavailable_reason']


@pytest.mark.parametrize('mode', ['sweep', 'size-su'])
def test_search_export_keeps_candidate_ids_and_individual_breakdowns(mode):
    results = [run(inference_configuration({'workload': {'input_length': length}})) for length in (4, 8)]
    candidates = [{'result': item} for item in results]
    if mode == 'sweep':
        aggregate = {'mode': mode, 'status': 'completed', 'candidates': candidates}
        ids = ['0', '1']
    else:
        aggregate = {'mode': mode, 'status': 'completed', 'searches': {
            'fixed': {'status': 'completed', 'minimum_su_size': 1, 'candidates': candidates}}}
        ids = ['fixed/0', 'fixed/1']
    with artifact_directory() as directory:
        write_report(aggregate, directory)
        rows = read_rows(directory)
        assert len(rows) == 4
        for index, expected in enumerate(results):
            for row in rows[2 * index:2 * index + 2]:
                assert row['candidate_id'] == ids[index]
                assert_record(row, expected['metrics']['latency_breakdown'][row['metric']])


def test_feedback_exports_all_candidates_and_tpop_alias():
    result = feedback(inference_configuration(), single_search(prefill_lengths=[4, 8]))
    assert result['best'] is not None
    with artifact_directory() as directory:
        write_report(result, directory)
        rows = read_rows(directory)
        assert len(rows) == 2 * len(result['candidates'])
        for candidate in result['candidates']:
            for row in [row for row in rows if row['candidate_id'] == str(candidate['id'])]:
                assert_record(row, candidate['metrics']['latency_breakdown'][row['metric']])
        with (directory / 'candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
            candidates = list(csv.DictReader(handle))
        for row in candidates:
            assert row['tpot_s'] == row['tpop_s']

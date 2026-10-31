"""Serving artifacts preserve request clocks and the existing trace contract."""
import importlib.util
import json
from pathlib import Path

import pytest

from supernode_sim.cli import main
from supernode_sim.config import load_config, read_data
from supernode_sim.serving import simulate_serving
from supernode_sim.serving_report import token_trace, write_serving_report

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('serving_trace_analyzer',
    ROOT / 'skills/supernode-modeling/scripts/analyze_inference_trace.py')
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def test_real_engine_serving_metrics_roundtrip_through_token_trace():
    config = load_config(ROOT / 'examples/serving_synthetic_model.yaml')
    requests = read_data(ROOT / 'examples/serving_synthetic_requests.yaml')
    result = simulate_serving(config, requests)
    observed = analyzer.analyze_trace(token_trace(result))
    assert observed['counts']['completed_in_window_requests'] == len(requests['requests'])
    assert observed['goodput']['passed_requests'] == result['goodput']['passed_requests']
    assert observed['metrics']['admission_wait']['mean_s'] > 0
    for row, source in zip(observed['requests'], result['requests']):
        assert row['id'] == source['id']
        for key in ('admission_wait_s', 'server_ttft_s', 'tpot_s', 'e2e_s'):
            assert row[key] == pytest.approx(source[key]) if source[key] is not None else row[key] is None
        assert row['ttft_s'] == pytest.approx(source['client_ttft_s'])


def test_serve_cli_writes_resolved_inputs_and_consistent_csv(tmp_path):
    output = tmp_path / 'serving'
    code = main(['serve', str(ROOT / 'examples/serving_synthetic_model.yaml'),
        '--serving', str(ROOT / 'examples/serving_synthetic_requests.yaml'), '-o', str(output)])
    assert code == 0
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert result['hardware']['total_npus'] == 1
    assert (output / 'requests.csv').exists()
    assert json.loads((output / 'serving.resolved.json').read_text(encoding='utf-8')) == result['serving']
    assert json.loads((output / 'config.resolved.json').read_text(encoding='utf-8')) == result['config']


def test_serving_schema_is_separate_and_slo_failure_sets_cli_exit_code(tmp_path):
    schema = tmp_path / 'schema.json'
    assert main(['serving-schema', '-o', str(schema)]) == 0
    assert 'requests' in json.loads(schema.read_text())['properties']
    spec = read_data(ROOT / 'examples/serving_synthetic_requests.yaml')
    spec['slo']['ttft_s'] = 0
    path = tmp_path / 'slo.json'
    path.write_text(json.dumps(spec))
    assert main(['serve', str(ROOT / 'examples/serving_synthetic_model.yaml'),
        '--serving', str(path), '-o', str(tmp_path / 'failed-slo')]) == 3
    spec['slo'].pop('tpot_s')
    result = simulate_serving(load_config(ROOT / 'examples/serving_synthetic_model.yaml'), spec)
    exported = token_trace(result)
    assert 'slo' not in exported
    assert exported['metadata']['slo_exported'] is False
    assert exported['metadata']['serving_slo']['ttft_s'] == 0


def test_epoch_origin_trace_export_retains_small_latencies():
    config = load_config(ROOT / 'examples/serving_synthetic_model.yaml')
    request = {'requests': [{'id': 'epoch', 'arrival_s': 1_700_000_000.0,
                           'input_tokens': 1, 'output_tokens': 2}]}
    result = simulate_serving(config, request)
    exported = token_trace(result)
    assert exported['metadata']['clock_origin_s'] == 1_700_000_000.0
    assert exported['requests'][0]['arrival_s'] == 0
    observed = analyzer.analyze_trace(exported)
    assert observed['requests'][0]['ttft_s'] == pytest.approx(result['requests'][0]['client_ttft_s'])
    assert observed['requests'][0]['tpot_s'] == pytest.approx(result['requests'][0]['tpot_s'])


def test_zero_duration_failure_does_not_invent_a_measurement_window(tmp_path):
    config = load_config(ROOT / 'examples/serving_synthetic_model.yaml').model_dump()
    config['hardware']['npu']['hbm_bytes'] = 1
    result = simulate_serving(config, {'requests': [
        {'id': 'oom', 'arrival_s': 0, 'input_tokens': 1, 'output_tokens': 1}]})
    assert result['status'] == 'completed_with_failures'
    assert result['time_window_s']['duration'] == 0
    assert token_trace(result) is None
    write_serving_report(result, tmp_path)
    assert json.loads((tmp_path / 'token-trace.json').read_text()) is None

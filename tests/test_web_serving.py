"""Serve API uses the real worker, immutable snapshots and bounded UI views."""
from copy import deepcopy
import json
import threading

import pytest

from supernode_sim.config import load_config
from supernode_sim.serving import simulate_serving
from supernode_sim.web_worker import compact_result, compact_serving_result
from tests.test_web import ROOT, WebClient, client, demo, finish, submit
from tests.test_serving import oracle, request


def paired(client):
    return client.json('GET', '/api/serving-presets/serving_synthetic')


def test_serving_catalog_schema_and_validation_return_paired_resolved_inputs(client):
    catalog = client.json('GET', '/api/serving-presets')
    assert catalog['presets'][0]['id'] == 'serving_synthetic'
    assert catalog['limits'] == {'request_count': 10000, 'max_batch_size': 1024,
                                 'instance_count': 128, 'body_bytes': 2097152}
    schema = client.json('GET', '/api/serving-schema')
    assert schema['x-web-limits'] == catalog['limits']
    assert 'requests' in schema['required']
    inputs = paired(client)
    result = client.json('POST', '/api/validate', {'mode': 'serve', **inputs})
    assert result['mode'] == 'serve'
    assert result['serving'] == inputs['serving']
    assert result['config'] == inputs['config']
    assert result['summary']['request_count'] == len(inputs['serving']['requests'])
    assert result['summary']['total_npus'] == inputs['config']['hardware']['npu_count'] * inputs['serving']['instance_count']
    client.json('GET', '/api/serving-presets/absent', status=404)


def test_real_serving_worker_preserves_metrics_relative_clock_and_snapshots(client):
    inputs = paired(client)
    # Epoch-sized input verifies that the web does not rebuild time from rounded absolute displays.
    for row in inputs['serving']['requests']:
        row['arrival_s'] += 1_700_000_000.0
    expected = simulate_serving(inputs['config'], inputs['serving'])
    original = deepcopy(inputs)
    queued = client.json('POST', '/api/jobs', {'mode': 'serve', **inputs}, status=202)
    inputs['serving']['requests'][0]['id'] = 'later-editor-change'
    job = finish(client, queued)
    assert job['mode'] == 'serve'
    assert job['result_status'] == expected['status'] == 'completed'
    assert job['serving_summary']['request_count'] == 5
    assert client.json('GET', job['config_url'])['config'] == original['config']
    assert client.json('GET', job['serving_url'])['serving'] == original['serving']
    summary = client.json('GET', f"/api/jobs/{job['id']}/result")
    assert summary['mode'] == 'serve' and summary['engine_mode'] == 'serving'
    assert summary['metrics'] == expected['metrics']
    for key in ('clock_origin_s', 'time_window_relative_s', 'throughput', 'goodput', 'hardware'):
        assert summary[key] == expected[key]
    for actual, full in zip(summary['requests'], expected['requests']):
        for key in ('client_ttft_s', 'admission_relative_s', 'token_times_relative_s'):
            assert actual[key] == full[key]
    assert 'communication' not in summary and 'latency_breakdown' not in summary['metrics']
    assert summary['executions'][0]['model_metadata']['fidelity'] == expected['executions'][0]['model_metadata']['fidelity']
    artifacts = {entry['name']: entry['url'] for entry in job['artifacts']}
    assert {'input.serving.json', 'serving.resolved.json', 'requests.csv', 'token-trace.json', 'report.md', 'result.json'} <= artifacts.keys()
    assert client.json('GET', artifacts['result.json'])['metrics'] == expected['metrics']
    trace = client.json('GET', artifacts['token-trace.json'])
    assert trace['metadata']['clock_origin_s'] == expected['clock_origin_s']
    assert trace['requests'][0]['token_times_s'] == expected['requests'][0]['token_times_relative_s']


def test_failed_serving_capacity_is_completed_with_failures_not_worker_error(client):
    inputs = paired(client)
    inputs['config']['hardware']['npu']['hbm_bytes'] = 1
    job = finish(client, client.json('POST', '/api/jobs', {'mode': 'serve', **inputs}, status=202))
    assert job['status'] == 'completed'
    assert job['result_status'] == 'completed_with_failures'
    result = client.json('GET', f"/api/jobs/{job['id']}/result")
    assert result['metrics']['failed_requests'] == 5
    assert result['metrics']['output_tokens'] == 0
    assert result['batches'][0]['diagnostics'][0]['code'] == 'initial_memory_exceeded'
    finish(client, submit(client, demo(), 'estimate'))


def test_serving_history_restores_both_input_snapshots_after_restart(client):
    from supernode_sim.web import create_server

    inputs = paired(client)
    job = finish(client, client.json('POST', '/api/jobs', {'mode': 'serve', **inputs}, status=202))
    output = client.server.manager.output
    client.server.shutdown()
    client.server.server_close()
    restarted = create_server('127.0.0.1', 0, output)
    thread = threading.Thread(target=restarted.serve_forever, daemon=True)
    thread.start()
    try:
        fresh = WebClient(restarted)
        history = fresh.json('GET', '/api/jobs')['jobs']
        assert history[0]['id'] == job['id'] and history[0]['mode'] == 'serve'
        assert fresh.json('GET', history[0]['serving_url'])['serving'] == inputs['serving']
        assert fresh.json('GET', history[0]['config_url'])['config'] == inputs['config']
        assert fresh.json('GET', f"/api/jobs/{job['id']}/result")['status'] == 'completed'
    finally:
        restarted.shutdown()
        restarted.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_serving_can_be_cancelled_and_input_snapshot_remains_available(client):
    inputs = paired(client)
    inputs['serving']['requests'] = [request('long', inputs=4096, outputs=10000)]
    job = client.json('POST', '/api/jobs', {'mode': 'serve', **inputs}, status=202)
    client.json('POST', '/api/jobs', {'config': demo(), 'mode': 'estimate'}, status=409)
    cancelled = client.json('POST', f"/api/jobs/{job['id']}/cancel", {})
    assert cancelled['status'] == 'cancelled'
    assert client.json('GET', cancelled['serving_url'])['serving']['requests'][0]['output_tokens'] == 10000
    client.json('GET', f"/api/jobs/{job['id']}/result", status=409)
    next_job = finish(client, submit(client, demo(), 'estimate'))
    client.json('GET', f"/api/jobs/{next_job['id']}/serving", status=404)


@pytest.mark.parametrize('patch,error', [
    ({'instance_count': 129}, '128'), ({'max_batch_size': 1025}, '1024'),
    ({'requests': [request('same')] * 10001}, '10000'),
])
def test_web_serving_caps_reject_without_rewriting_input_or_creating_jobs(client, patch, error):
    inputs = paired(client)
    inputs['serving'].update(patch)
    for endpoint in ('/api/validate', '/api/jobs'):
        result = client.json('POST', endpoint, {'mode': 'serve', **inputs}, status=400)
        assert error in result['error'] and 'CLI' in result['error']
    assert client.json('GET', '/api/jobs')['jobs'] == []


@pytest.mark.parametrize('patch', [
    {'requests': []}, {'requests': [request('a'), request('a')]},
    {'requests': [request('a', arrival=-1)]}, {'requests': [request('a', inputs=True)]},
    {'instance_count': '2'}, {'max_batch_size': True}, {'unknown': 1},
])
def test_web_serving_uses_strict_engine_schema(client, patch):
    inputs = paired(client)
    inputs['serving'].update(patch)
    client.json('POST', '/api/validate', {'mode': 'serve', **inputs}, status=400)


def test_serving_base_compatibility_and_no_silent_ignored_serving_input(client):
    inputs = paired(client)
    client.json('POST', '/api/validate', {'mode': 'serve', 'config': inputs['config']}, status=400)
    client.json('POST', '/api/validate', {'mode': 'simulate', **inputs}, status=400)
    client.json('POST', '/api/validate', {'mode': 'serve', 'config': demo('training'), 'serving': inputs['serving']}, status=400)
    config = inputs['config']
    config['execution']['dp'] = 2
    config['hardware'].update(npu_count=2, su_size=2)
    error = client.json('POST', '/api/validate', {'mode': 'serve', **inputs}, status=400)
    assert 'DP=1' in error['error']
    for name in ('synthetic-dspark', 'synthetic-pd'):
        config = load_config(ROOT / 'examples' / 'features' / f'{name}.yaml').model_dump()
        error = client.json('POST', '/api/validate', {'mode': 'serve', 'config': config, 'serving': inputs['serving']}, status=400)
        assert 'speculative_decode' in error['error']


def test_serving_retains_http_body_size_limit(client):
    client.json('POST', '/api/jobs', raw=b'{}', headers={
        'Content-Type': 'application/json', 'Content-Length': str(2 * 1024 * 1024 + 1)}, status=413)


def test_serving_summary_bounds_rows_tokens_and_keeps_full_population_metrics(monkeypatch):
    oracle(monkeypatch)
    config = load_config(ROOT / 'examples' / 'serving_synthetic_model.yaml')
    full = simulate_serving(config, {'requests': [request(str(i), outputs=129) for i in range(241)]})
    ui = compact_serving_result(full)
    assert ui['metrics'] == full['metrics']
    assert ui['metrics']['distributions']['client_ttft_s']['sample_count'] == 241
    assert len(ui['requests']) == len(ui['batches']) == 200
    assert ui['display_limits']['requests_truncated'] is ui['display_limits']['batches_truncated'] is True
    assert len(ui['requests'][0]['token_times_relative_s']) == 128
    assert ui['requests'][0]['token_times_count'] == 129
    assert ui['requests'][0]['token_times_truncated'] is True
    full = simulate_serving(config, {'requests': [request(str(i), inputs=i + 1) for i in range(45)]})
    ui = compact_serving_result(full)
    assert len(ui['executions']) == 40
    assert ui['display_limits']['execution_count'] == 45
    assert ui['display_limits']['executions_truncated'] is True
    json.dumps(ui, allow_nan=False)


def test_ordinary_summary_preserves_training_and_operator_evidence_and_marks_truncation():
    metadata = {'operator_trace': {'matched_events': 1}, 'kernel_calibration': {'matched_events': 2},
                'collective_calibration': {'matched_collectives': 3}, 'fidelity': {'hardware_performance': 'uncalibrated'}}
    event = {'id': 'part', 'start_s': 0, 'end_s': 1, 'duration_s': 1,
             'metadata': {'kernel': 'dense.up', 'original_kernel': 'dense', 'trace_id': 'test',
                          'trace_parent_event': 'parent', 'operation_index': 0,
                          'matmul': {'m': 1, 'n': 2, 'k': 3}, 'flops': 12}}
    training = {'model_mfu': .2, 'active_npu_ranks': list(range(41))}
    result = compact_result({'mode': 'simulate', 'metrics': {'training_accounting': training},
                             'model_metadata': metadata, 'simulation': {'trace': [event]}})
    assert result['model_metadata'] == metadata
    assert result['trace']['events'][0]['metadata'] == event['metadata']
    assert result['metrics']['training_accounting']['model_mfu'] == .2
    assert len(result['metrics']['training_accounting']['active_npu_ranks']) == 40
    assert result['display_limits']['training_accounting_truncated'] is True


def test_real_training_worker_exposes_suboperators_profiles_and_mfu(client):
    from supernode_sim.analysis import compile_graph, run

    config = load_config(ROOT / 'examples' / 'features' / 'synthetic-training.yaml').model_dump()
    event = next(event for event in compile_graph(config).events
                 if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'backward')
    compute = next(key for key in event.work if '.compute' in key)
    rank = compute.split('.')[0]
    flops, memory = event.work[compute], event.work[rank + '.hbm']
    config['hardware']['kernel_traces'] = {'dense': [{
        'id': 'web-training-trace', 'source': 'Synthetic web regression; not a measurement.',
        'hardware_name': config['hardware']['name'], 'phase': 'backward',
        'dtype': config['execution']['precision']['compute'], 'flops': flops, 'memory_bytes': memory,
        'metadata_match': {key: event.metadata[key] for key in ('model_shape_signature', 'workload_shape')},
        'operations': [{'kernel': name, 'flops': flops / 2, 'memory_bytes': memory / 2}
                       for name in ('dense.part_a', 'dense.part_b')],
    }]}
    config['hardware']['kernel_profiles'] = {'dense.part_a': [{
        'flops': flops / 2, 'memory_bytes': memory / 2, 'duration_s': .01,
        'phase': 'backward', 'dtype': config['execution']['precision']['compute'],
        'metadata_match': {'trace_id': 'web-training-trace', 'workload_shape': event.metadata['workload_shape']},
        'source': 'Synthetic slow service for HTTP regression.',
    }]}
    expected = run(config)
    # JSON.parse + JSON.stringify in the full editor serializes 8.0 as 8.
    # The Any-valued shape selectors must still match the compiler's floats.
    config = json.loads(json.dumps(config), parse_float=lambda raw:
        int(float(raw)) if float(raw).is_integer() else float(raw))
    assert type(config['hardware']['kernel_traces']['dense'][0]['metadata_match']['workload_shape']['operator_tokens']) is int
    job = finish(client, submit(client, config))
    actual = client.json('GET', f"/api/jobs/{job['id']}/result")
    assert actual['metrics']['training_accounting'] == expected['metrics']['training_accounting']
    for key in ('operator_trace', 'kernel_calibration', 'collective_calibration', 'fidelity'):
        assert actual['model_metadata'][key] == expected['model_metadata'][key]
    assert actual['model_metadata']['operator_trace']['matched_events'] > 0
    assert actual['model_metadata']['kernel_calibration']['matched_events'] > 0
    parts = [event for event in actual['trace']['events']
             if event['metadata'].get('trace_id') == 'web-training-trace'
             and 'operation_index' in event['metadata']]
    assert parts
    assert all(event['metadata']['trace_parent_event'] and event['metadata']['original_kernel'] == 'dense'
               and event['metadata']['flops'] == flops / 2 for event in parts)

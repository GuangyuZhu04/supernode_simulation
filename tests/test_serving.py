"""FCFS timing arithmetic plus integration with actual fixed-batch execution."""
from copy import deepcopy
import json

import pytest

from supernode_sim.analysis import run
from supernode_sim.config import Config
from supernode_sim.serving import ServingSpec, simulate_serving
from tests.test_structured import configuration


def base():
    return configuration({'workload': {'scenario': 'inference'}})


def request(name, arrival=0, inputs=8, outputs=2):
    return dict(id=name, arrival_s=arrival, input_tokens=inputs, output_tokens=outputs)


def oracle(monkeypatch, duration=4, *, fail_batch_sizes=()):
    calls = []

    def execute(config):
        calls.append(config.model_dump())
        failed = config.workload.batch_size in fail_batch_sizes
        rows = []
        for index, count in enumerate(config.workload.output_lengths):
            times = ([1.0] if count == 1 else [1 + (duration - 1) * token / (count - 1) for token in range(count)])
            rows.append(dict(replica='0', sequence=str(index), token_completion_s=times))
        return {'simulation': {'status': 'infeasible' if failed else 'feasible', 'duration_s': duration,
                               'diagnostics': [{'code': 'synthetic_oom'}] if failed else [],
                               'memory_peaks_bytes': {'npu0.hbm': 42}},
                'metrics': {'requests': rows}, 'model_metadata': {'fidelity': {'hardware_performance': 'uncalibrated'}},
                'sources': {'test': {'status': 'assumed', 'description': 'Hand-computed unit test oracle.'}}}

    monkeypatch.setattr('supernode_sim.serving._execute_batch', execute)
    return calls


def test_fcfs_queue_and_overhead_are_counted_once_per_batch(monkeypatch):
    calls = oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('a'), request('b', 1)], 'scheduler_overhead_s': .5})
    a, b = result['requests']
    assert [a['admission_s'], b['admission_s']] == [0, 4.5]
    assert [a['model_start_s'], b['model_start_s']] == [.5, 5]
    assert a['token_times_s'] == [1.5, 4.5]
    assert b['token_times_s'] == [6, 9]
    assert b['admission_wait_s'] == 3.5
    assert b['server_ttft_s'] == 1.5
    assert b['model_ttft_s'] == 1
    assert b['client_ttft_s'] == 5
    assert b['e2e_s'] == 8
    assert b['tpot_s'] == 3
    assert result['time_window_s'] == {'start': 0, 'end': 9, 'duration': 9}
    assert result['throughput']['output_tokens_per_s'] == pytest.approx(4 / 9)
    assert len(calls) == 1  # Only exact geometry is reused, not queue timestamps.
    assert result['provenance']['execution_cache']['hits'] == 1


def test_batch_wait_fills_early_and_never_admits_a_future_request(monkeypatch):
    oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('a'), request('b', 2), request('c', 9)],
                                      'max_batch_size': 2, 'max_batch_wait_s': 5, 'scheduler_overhead_s': .5})
    first, second = result['batches']
    assert first['request_ids'] == ['a', 'b']
    assert first['admission_s'] == 2
    assert first['model_start_s'] == 2.5
    assert first['end_s'] == 6.5
    assert first['batch_wait_s'] == 2
    assert second['request_ids'] == ['c']
    assert second['admission_s'] == 14
    assert all(row['admission_s'] >= row['arrival_s'] for row in result['requests'])


def test_expired_batch_wait_dispatches_immediately_when_instance_becomes_free(monkeypatch):
    oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('a'), request('b'), request('c', 1)],
                                      'max_batch_size': 2, 'max_batch_wait_s': 2})
    assert [batch['admission_s'] for batch in result['batches']] == [0, 4]
    assert result['batches'][1]['batch_wait_s'] == 0


def test_stable_arrival_ties_and_idle_instance_ties(monkeypatch):
    oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('late', 10), request('b'), request('a'), request('c')],
                                      'instance_count': 2})
    assert [batch['request_ids'][0] for batch in result['batches']] == ['b', 'a', 'c', 'late']
    assert [batch['instance_id'] for batch in result['batches']] == [0, 1, 0, 0]
    assert [batch['admission_s'] for batch in result['batches']] == [0, 0, 4, 10]
    assert [row['id'] for row in result['requests']] == ['late', 'b', 'a', 'c']
    assert result['hardware']['total_npus'] == 2 * base().hardware.npu_count


def test_joint_slo_goodput_and_empirical_request_quantiles(monkeypatch):
    oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('a'), request('b'), request('single', outputs=1)],
                                      'slo': {'ttft_s': 5, 'tpot_s': 3}})
    assert [row['client_ttft_s'] for row in result['requests']] == [1, 5, 9]
    assert [row['slo_passed'] for row in result['requests']] == [True, True, False]
    assert result['requests'][2]['tpot_s'] is None
    assert result['requests'][2]['slo_checks'][1]['status'] == 'not_applicable'
    assert result['goodput']['passed_requests'] == 2
    assert result['goodput']['joint_requests_per_s'] == pytest.approx(2 / 12)
    assert result['goodput']['tpot_not_applicable_requests'] == 1
    distribution = result['metrics']['distributions']['client_ttft_s']
    assert distribution['sample_count'] == 3
    assert distribution['p50_s'] == 5
    assert distribution['p90_s'] == pytest.approx(8.2)
    assert distribution['p99_s'] == pytest.approx(8.92)
    assert result['metrics']['distributions']['tpot_s']['sample_count'] == 2


def test_failed_batch_is_not_shrunk_and_does_not_leak_success_tokens(monkeypatch):
    calls = oracle(monkeypatch, fail_batch_sizes={2})
    result = simulate_serving(base(), {'requests': [request('a'), request('b'), request('c')],
                                      'max_batch_size': 2, 'scheduler_overhead_s': .25,
                                      'slo': {'ttft_s': 20}})
    assert [call['workload']['batch_size'] for call in calls] == [2, 1]
    assert result['status'] == 'completed_with_failures'
    assert result['metrics']['failed_requests'] == 2
    assert result['requests'][2]['admission_s'] == 4.25
    assert result['batches'][0]['diagnostics'] == [{'code': 'synthetic_oom'}]
    for row in result['requests'][:2]:
        assert row['token_times_s'] == []
        assert row['output_tokens'] == 0
        assert row['requested_output_tokens'] == 2
        assert row['client_ttft_s'] is row['tpot_s'] is row['e2e_s'] is None
        assert row['finish_s'] == row['failure_elapsed_s'] == 4.25
        assert row['slo_passed'] is False
    assert result['metrics']['output_tokens'] == 2
    assert result['throughput']['output_tokens_per_s'] == pytest.approx(2 / 8.5)
    assert result['goodput']['passed_requests'] == 1


def test_cache_is_exact_ordered_geometry_and_local_to_one_call(monkeypatch):
    calls = oracle(monkeypatch)
    requests = [request('a', inputs=2, outputs=1), request('b', inputs=3),
                request('c', 10, inputs=3), request('d', 10, inputs=2, outputs=1),
                request('e', 20, inputs=2, outputs=1), request('f', 20, inputs=3)]
    settings = {'requests': requests, 'max_batch_size': 2}
    result = simulate_serving(base(), settings)
    assert [batch['cache_hit'] for batch in result['batches']] == [False, False, True]
    assert len(calls) == 2
    simulate_serving(base(), settings)
    assert len(calls) == 4


def test_base_targets_are_retained_as_provenance_and_not_used_as_serving_slos(monkeypatch):
    calls = oracle(monkeypatch)
    config = configuration({'workload': {'scenario': 'inference'}, 'targets': {'tpot_s': .001}})
    original = config.model_dump()
    result = simulate_serving(config, {'requests': [request('single', outputs=1)]})
    assert result['config']['targets']['tpot_s'] == .001
    assert calls[0]['targets']['tpot_s'] is None
    assert result['goodput']['specified'] is False
    assert result['goodput']['joint_requests_per_s'] is None
    assert config.model_dump() == original


def test_zero_arrivals_full_batch_reproduces_actual_engine_token_times():
    config = configuration({'workload': {'scenario': 'inference', 'batch_size': 2,
                                        'input_lengths': [8, 16], 'output_lengths': [3, 1]}})
    fixed = run(config, full_detail=False)
    served = simulate_serving(config, {'requests': [request('long', inputs=8, outputs=3),
                                                   request('short', inputs=16, outputs=1)], 'max_batch_size': 2})
    assert served['status'] == 'completed'
    assert served['time_window_s']['duration'] == pytest.approx(fixed['simulation']['duration_s'])
    tokens = {int(row['sequence']): row['token_completion_s'] for row in fixed['metrics']['requests']}
    for index, row in enumerate(served['requests']):
        assert row['token_times_s'] == pytest.approx(tokens[index])
        assert row['client_ttft_s'] == pytest.approx(tokens[index][0])
    assert served['requests'][1]['tpot_s'] is None
    assert served['batches'][0]['memory_peaks_bytes'] == fixed['simulation']['memory_peaks_bytes']
    assert served['executions'][0]['model_metadata']['fidelity'] == fixed['model_metadata']['fidelity']
    json.dumps(served, allow_nan=False)


def test_real_memory_failure_counts_no_tokens_or_infinite_rates():
    config = configuration({'workload': {'scenario': 'inference'}, 'hardware': {'npu': {'hbm_bytes': 1}}})
    result = simulate_serving(config, {'requests': [request('oom')]})
    assert result['status'] == 'completed_with_failures'
    assert result['requests'][0]['token_times_s'] == []
    assert result['batches'][0]['diagnostics'][0]['code'] == 'initial_memory_exceeded'
    assert result['time_window_s']['duration'] == 0
    assert result['throughput']['output_tokens_per_s'] is None
    assert result['metrics']['distributions']['client_ttft_s']['sample_count'] == 0
    json.dumps(result, allow_nan=False)


def test_real_batch_memory_failure_releases_instance_for_next_independent_request():
    config = configuration({'workload': {'scenario': 'inference', 'batch_size': 1,
                                        'input_length': 128, 'output_length': 2},
                            'hardware': {'npu': {'hbm_bytes': 50000}}})
    result = simulate_serving(config, {'requests': [request(str(index), inputs=128) for index in range(3)],
                                      'max_batch_size': 2})
    assert [batch['status'] for batch in result['batches']] == ['failed', 'completed']
    assert result['batches'][0]['diagnostics'][0]['code'] == 'memory_admission_deadlock'
    assert result['batches'][0]['duration_s'] > 0
    assert result['batches'][1]['admission_s'] == result['batches'][0]['end_s']
    assert [row['output_tokens'] for row in result['requests']] == [0, 0, 2]
    assert result['metrics']['completed_requests'] == 1


def test_real_two_instances_execute_concurrently_on_distinct_hardware_copies():
    config = configuration({'workload': {'scenario': 'inference', 'batch_size': 1, 'output_length': 2}})
    single = run(config, full_detail=False)
    result = simulate_serving(config, {'requests': [request('a'), request('b')], 'instance_count': 2})
    assert [batch['instance_id'] for batch in result['batches']] == [0, 1]
    assert [batch['admission_s'] for batch in result['batches']] == [0, 0]
    assert result['time_window_s']['duration'] == pytest.approx(single['simulation']['duration_s'])
    assert result['throughput']['output_tokens_per_s'] == pytest.approx(2 * single['metrics']['throughput_tokens_s'])
    assert result['hardware']['total_npus'] == 2


def test_large_clock_origin_preserves_real_latencies_throughput_and_slo():
    def scenario(origin):
        return simulate_serving(base(), {'requests': [request(str(i), origin) for i in range(3)],
                                          'slo': {'ttft_s': 1e-10}})
    relative, shifted = scenario(0.0), scenario(1_700_000_000.0)
    assert shifted['clock_origin_s'] == 1_700_000_000.0
    assert shifted['time_window_s']['duration'] == relative['time_window_s']['duration'] > 0
    assert shifted['time_window_relative_s'] == relative['time_window_relative_s']
    assert shifted['throughput'] == relative['throughput']
    assert shifted['goodput'] == relative['goodput']
    assert shifted['metrics'] == relative['metrics']
    for first, second in zip(relative['requests'], shifted['requests']):
        assert second['token_times_relative_s'] == first['token_times_s']
        assert second['client_ttft_s'] == first['client_ttft_s'] > 0
        assert second['slo_passed'] is False
        assert second['client_ttft_s'] == pytest.approx(second['admission_wait_s'] + second['server_ttft_s'])
    # Absolute displays lose nanoseconds at an epoch-sized origin; these are
    # deliberately excluded from all metric calculations and trace export.
    assert shifted['requests'][0]['first_token_s'] == shifted['requests'][0]['arrival_s']
    assert shifted['requests'][0]['first_token_relative_s'] > 0


def test_large_clock_origin_preserves_future_arrival_batching_and_instance_selection(monkeypatch):
    oracle(monkeypatch)
    def scenario(origin):
        return simulate_serving(base(), {'requests': [
            request('a', origin), request('b', origin + .5),
            request('c', origin + .5), request('d', origin + 1)],
            'instance_count': 2, 'max_batch_size': 2, 'max_batch_wait_s': .75,
            'scheduler_overhead_s': .125})
    before, after = scenario(0.0), scenario(1_700_000_000.0)
    assert after['metrics'] == before['metrics']
    assert after['throughput'] == before['throughput']
    for first, second in zip(before['batches'], after['batches']):
        for key in ('request_ids', 'instance_id', 'admission_relative_s', 'model_start_relative_s', 'end_relative_s'):
            assert second[key] == first[key]


def test_large_clock_origin_preserves_failed_batch_elapsed_time(monkeypatch):
    oracle(monkeypatch, duration=4, fail_batch_sizes={2})
    origin = 1_700_000_000.0
    result = simulate_serving(base(), {'requests': [request(str(i), origin) for i in range(3)],
                                      'max_batch_size': 2, 'scheduler_overhead_s': .25})
    assert result['requests'][0]['failure_elapsed_s'] == 4.25
    assert result['requests'][0]['finish_relative_s'] == 4.25
    assert result['requests'][0]['finish_s'] == origin + 4.25
    assert result['requests'][2]['admission_relative_s'] == 4.25
    assert result['time_window_s']['duration'] == 8.5


@pytest.mark.parametrize('patch', [
    {'requests': []}, {'requests': [request('a'), request('a')]},
    {'requests': [request(' ')]}, {'requests': [request('a', arrival=-1)]},
    {'requests': [request('a', arrival=float('nan'))]}, {'requests': [request('a', arrival=float('inf'))]},
    {'requests': [request('a', arrival=True)]}, {'requests': [request('a', arrival='1')]},
    {'requests': [request('a', inputs=0)]}, {'requests': [request('a', inputs=True)]},
    {'requests': [request('a', outputs=1.5)]}, {'requests': [request('a') | {'unknown': 1}]},
    {'instance_count': 0}, {'instance_count': True}, {'max_batch_size': '2'},
    {'max_batch_wait_s': -1}, {'scheduler_overhead_s': float('inf')},
    {'scheduler_overhead_s': True}, {'slo': {'ttft_s': -1}}, {'slo': {'tpot_s': '1'}}, {'unknown': True},
])
def test_strict_serving_input_validation(patch):
    with pytest.raises(ValueError):
        ServingSpec.model_validate({'requests': [request('a')]} | patch)


def test_slo_zero_thresholds_are_valid_and_have_explicit_results(monkeypatch):
    oracle(monkeypatch)
    result = simulate_serving(base(), {'requests': [request('a')], 'slo': {'ttft_s': 0, 'tpot_s': 0}})
    assert result['goodput']['specified'] is True
    assert result['goodput']['passed_requests'] == 0


def test_unsupported_training_dp_speculative_and_disaggregation_are_rejected():
    with pytest.raises(ValueError, match='inference'):
        simulate_serving(configuration(), {'requests': [request('a')]})
    dp = configuration({'workload': {'scenario': 'inference'},
                        'hardware': {'npu_count': 2, 'su_size': 1}, 'execution': {'dp': 2}})
    with pytest.raises(ValueError, match='DP=1'):
        simulate_serving(dp, {'requests': [request('a')]})
    # These validated example files carry the full feature-specific contracts.
    from supernode_sim.config import load_config
    for filename in ('examples/features/synthetic-dspark.yaml', 'examples/features/synthetic-pd.yaml'):
        with pytest.raises(ValueError, match='does not implement'):
            simulate_serving(load_config(filename), {'requests': [request('a')]})


@pytest.mark.parametrize('corruption', ['status', 'duration', 'rows', 'replica', 'sequence', 'count', 'negative', 'order', 'outside'])
def test_executor_result_is_validated_before_success_is_counted(monkeypatch, corruption):
    result = {'simulation': {'status': 'feasible', 'duration_s': 4},
              'metrics': {'requests': [dict(replica='0', sequence='0', token_completion_s=[1, 4])]}}
    if corruption == 'status':
        result['simulation']['status'] = 'error'
    elif corruption == 'duration':
        result['simulation']['duration_s'] = float('inf')
    elif corruption == 'rows':
        result['metrics']['requests'] = []
    else:
        row = result['metrics']['requests'][0]
        if corruption == 'replica':
            row['replica'] = 1
        elif corruption == 'sequence':
            row['sequence'] = 2
        elif corruption == 'count':
            row['token_completion_s'] = [1]
        elif corruption == 'negative':
            row['token_completion_s'] = [-1, 4]
        elif corruption == 'order':
            row['token_completion_s'] = [2, 1]
        elif corruption == 'outside':
            row['token_completion_s'] = [1, 5]
    monkeypatch.setattr('supernode_sim.serving._execute_batch', lambda config: deepcopy(result))
    with pytest.raises(ValueError):
        simulate_serving(base(), {'requests': [request('a')]})

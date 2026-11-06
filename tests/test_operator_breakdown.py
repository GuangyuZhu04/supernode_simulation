"""Numerical anchors for operator shares, overlapping costs and exported windows."""
import csv
import json
import math

import pytest

from supernode_sim.analysis import run
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.feedback import feedback
from supernode_sim.latency_breakdown import mean_breakdowns, window_breakdown
from supernode_sim.operator_costs import COST_LABELS
from supernode_sim.report import write_report
from supernode_sim.streaming import run_streaming
from tests.test_feedback import inference_configuration, single_search
from tests.test_report_export import artifact_directory
from tests.test_streaming_latency import inference_config
from tests.test_training import configuration as training_config


def assert_costs(record):
    costs = record['operator_breakdown']
    assert costs['total_s'] == pytest.approx(record['total_s'])
    assert sum(costs['components_s'].values()) == pytest.approx(record['total_s'])
    assert sum(row['total_s'] for row in costs['operators']) == pytest.approx(record['total_s'])
    if record['total_s'] > 0:
        assert sum(row['share_pct'] for row in costs['operators']) == pytest.approx(100)
        assert sum(costs['component_shares_pct'].values()) == pytest.approx(100)
    else:
        assert all(value is None for value in costs['component_shares_pct'].values())
    for row in costs['operators']:
        assert all(value >= 0 and math.isfinite(value) for value in row['components_s'].values())
        assert sum(row['components_s'].values()) == pytest.approx(row['total_s'])
        assert row['share_pct'] == pytest.approx(100 * row['total_s'] / costs['total_s'])
        assert sum(row['component_shares_pct'].values()) == pytest.approx(100)
        for key, value in row['components_s'].items():
            assert row['component_shares_pct'][key] == pytest.approx(100 * value / row['total_s'])
    json.dumps(costs, allow_nan=False)


def mixed_graph():
    identity = {'phase': 'prefill', 'component': 'Attention', 'kernel': 'matmul'}
    return Graph(resources={
        'npu0.compute': Resource('npu0.compute', 10, 'FLOP/s', 'compute'),
        'npu0.hbm': Resource('npu0.hbm', 10, 'byte/s'),
        'npu0.SU.tx': Resource('npu0.SU.tx', 10, 'byte/s'),
        'npu1.compute': Resource('npu1.compute', 10, 'FLOP/s', 'compute'),
    }, events=[
        Event('memory', work={'npu0.compute': 20, 'npu0.hbm': 40}, startup_s=1, metadata=identity),
        Event('compute', deps=('memory',), work={'npu0.compute': 40, 'npu0.hbm': 10}, startup_s=.5, metadata=identity),
        Event('communication', deps=('compute',), work={'npu0.SU.tx': 20, 'npu0.hbm': 10}, startup_s=.5,
              metadata={'type': 'transfer', 'phase': 'prefill', 'semantic': 'all_reduce'}),
        Event('parallel', work={'npu1.compute': 10}, metadata={'kernel': 'parallel_op'}),
        Event('done', deps=('communication', 'parallel')),
    ])


def test_overlap_launch_and_parallel_work_have_independent_numerical_anchors():
    graph = mixed_graph()
    record = window_breakdown(graph, simulate(graph), 'done')
    assert_costs(record)
    costs = record['operator_breakdown']
    assert record['total_s'] == 12
    assert costs['components_s'] == {**dict.fromkeys(COST_LABELS, 0), 'compute_s': 4,
                                    'memory_access_s': 4, 'launch_s': 1.5, 'communication_s': 2.5}
    assert [row['operator'] for row in costs['operators']] == ['matmul', 'all_reduce']
    matmul = costs['operators'][0]
    assert matmul['total_s'] == 9.5
    assert matmul['share_pct'] == pytest.approx(100 * 9.5 / 12)
    assert matmul['component_shares_pct']['memory_access_s'] == pytest.approx(100 * 4 / 9.5)
    assert record['components_s']['compute_s'] == 9.5  # Legacy combined service is preserved.


def test_clipped_windows_only_charge_the_overlapping_part_of_launch_and_service():
    graph = mixed_graph()
    sim = simulate(graph)
    record = window_breakdown(graph, sim, 'memory', .5)
    assert_costs(record)
    costs = record['operator_breakdown']['components_s']
    assert record['total_s'] == 4.5
    assert costs['launch_s'] == .5
    assert costs['memory_access_s'] == 4
    active_only = window_breakdown(graph, sim, 'memory', 2)
    assert_costs(active_only)
    assert active_only['operator_breakdown']['components_s']['memory_access_s'] == 3
    assert active_only['operator_breakdown']['components_s']['launch_s'] == 0


def test_whole_kernel_profile_is_not_falsely_split_into_compute_and_memory():
    graph = mixed_graph()
    graph.events = [graph.events[0]]
    graph.resources['profile'] = Resource('profile', 1, 'second/second', 'kernel_timing')
    graph.events[0].work['profile'] = 10
    record = window_breakdown(graph, simulate(graph), 'memory')
    assert_costs(record)
    costs = record['operator_breakdown']['components_s']
    assert costs['unattributed_service_s'] == 10
    assert costs['launch_s'] == 1
    assert costs['compute_s'] == costs['memory_access_s'] == 0


def test_mean_recomputes_shares_from_durations_instead_of_averaging_percentages():
    graph = mixed_graph()
    graph.events = [Event('a', work={'npu0.compute': 20}, metadata={'kernel': 'same'}),
                    Event('b', deps=('a',), work={'npu0.hbm': 80}, metadata={'kernel': 'same'})]
    sim = simulate(graph)
    record = mean_breakdowns([window_breakdown(graph, sim, 'a'), window_breakdown(graph, sim, 'b', 2)])
    assert_costs(record)
    assert record['total_s'] == 5
    assert record['operator_breakdown']['operators'][0]['component_shares_pct']['memory_access_s'] == 80
    zero = window_breakdown(graph, sim, 'b', 10)
    assert_costs(zero)
    assert zero['operator_breakdown']['operators'] == []


@pytest.mark.parametrize('block', ['slot', 'contention'])
def test_waiting_and_contention_remain_in_the_operator_denominator(block):
    slots = ('shared',) if block == 'slot' else ()
    graph = mixed_graph()
    graph.events = [Event(key, work={'npu0.compute': 10}, slots=slots, metadata={'kernel': key}) for key in ('a', 'b')]
    record = window_breakdown(graph, simulate(graph), 'b')
    assert_costs(record)
    key = 'scheduling_wait_s' if block == 'slot' else 'resource_contention_s'
    row = record['operator_breakdown']['operators'][0]
    assert row['component_shares_pct']['compute_s'] == 50
    assert row['component_shares_pct'][key] == 50


def test_transfer_memory_bottleneck_and_collective_timer_keep_their_boundaries():
    graph = mixed_graph()
    transfer = graph.events[2]
    transfer.deps = ()
    transfer.work['npu0.hbm'] = 40
    graph.events = [transfer, Event('timer', deps=(transfer.id,), startup_s=5,
                                  metadata={'type': 'collective_timing', 'collective': 'all_reduce'})]
    record = window_breakdown(graph, simulate(graph), 'timer')
    assert_costs(record)
    costs = record['operator_breakdown']['components_s']
    assert costs['memory_access_s'] == 4
    assert costs['communication_s'] == 5.5
    assert costs['launch_s'] == 0


@pytest.mark.parametrize('steps', [1, 3])
def test_training_windows_match_scalar_metrics_and_average_actual_steps(steps):
    result = run(training_config({'workload': {'training_steps': steps}}))
    metrics = result['metrics']
    breakdown = metrics['training_latency_breakdown']
    for metric in ('step', 'first_step', 'steady_step'):
        record = breakdown[metric]
        if record is None:
            assert steps == 1 and metric == 'steady_step'
            continue
        assert_costs(record)
        assert record['total_s'] == pytest.approx(metrics[metric + '_time_s'])
    phases = {row['phase'] for row in breakdown['step']['operator_breakdown']['operators']}
    assert {'forward', 'backward'} <= phases
    assert 'ttft_s' not in metrics


@pytest.mark.parametrize('patch', [{}, {'model': {'layers': 4, 'architecture': {
    'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA'], 'ffn_by_layer': ['Dense', 'MoE', 'Dense', 'MoE']}}},
                                  {'workload': {'input_lengths': [8, 16], 'output_lengths': [4, 2]}}])
def test_full_streaming_and_reused_cache_preserve_operator_costs(patch):
    config = inference_config(patch)
    expected = run(config)['metrics']['latency_breakdown']
    cache = {}
    actual = run_streaming(config, cache=cache)['metrics']['latency_breakdown']
    repeated = run_streaming(config, cache=cache)['metrics']['latency_breakdown']
    assert repeated == actual
    for metric in ('ttft', 'tpot'):
        assert_costs(actual[metric])
        left, right = actual[metric]['operator_breakdown'], expected[metric]['operator_breakdown']
        assert left['components_s'] == pytest.approx(right['components_s'], abs=1e-12)
        by_identity = lambda costs: {(row['phase'], row['component'], row['operator']): row for row in costs['operators']}
        lrows, rrows = by_identity(left), by_identity(right)
        assert lrows.keys() == rrows.keys()
        for identity in lrows:
            assert lrows[identity]['components_s'] == pytest.approx(rrows[identity]['components_s'], abs=1e-12)


@pytest.mark.parametrize('scenario', ['inference', 'training'])
def test_compact_csv_and_json_preserve_every_operator_and_both_denominators(monkeypatch, scenario):
    result = run(inference_configuration() if scenario == 'inference' else training_config({'workload': {'training_steps': 2}}))
    monkeypatch.setattr('supernode_sim.report.AUTO_DETAIL_THRESHOLD', 1)
    with artifact_directory() as directory:
        write_report(result, directory)
        with (directory / 'operator_breakdown.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert persisted['metrics'] == result['metrics']
        source = result['metrics']['latency_breakdown' if scenario == 'inference' else 'training_latency_breakdown']
        metrics = ('ttft', 'tpot') if scenario == 'inference' else ('step', 'first_step', 'steady_step')
        for metric in metrics:
            costs = source[metric]['operator_breakdown']
            exported = [row for row in rows if row['metric'] == metric]
            assert len(exported) == len(costs['operators'])
            assert sum(float(row['share_pct']) for row in exported) == pytest.approx(100)
            for actual, expected in zip(exported, costs['operators']):
                assert float(actual['total_s']) == expected['total_s']
                assert float(actual['window_total_s']) == costs['total_s']
                assert float(actual['share_pct']) == expected['share_pct']
                for key in COST_LABELS:
                    assert float(actual[key]) == expected['components_s'][key]
                    assert float(actual[key.removesuffix('_s') + '_pct']) == expected['component_shares_pct'][key]


@pytest.mark.parametrize('case', ['estimate', 'failed', 'one_token', 'one_step'])
def test_unavailable_operator_rows_have_no_fabricated_zero_percentages(case):
    config = training_config() if case == 'one_step' else inference_configuration().model_dump()
    if case == 'failed':
        config['hardware']['npu']['hbm_bytes'] = 1
    if case == 'one_token':
        config['workload']['output_lengths'] = [1] * config['workload']['batch_size']
        config['workload']['output_length'] = 1
    result = run(config, mode='estimate' if case == 'estimate' else 'simulate')
    with artifact_directory() as directory:
        write_report(result, directory)
        with (directory / 'operator_breakdown.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        unavailable = [row for row in rows if row['available'] == 'False']
        assert unavailable
        for row in unavailable:
            assert row['unavailable_reason']
            assert row['total_s'] == row['share_pct'] == ''
            assert all(row[key] == '' for key in COST_LABELS)


@pytest.mark.parametrize('mode', ['sweep', 'size-su', 'feedback'])
def test_candidate_exports_keep_operator_denominators_separate(mode):
    if mode == 'feedback':
        result = feedback(inference_configuration(), single_search(prefill_lengths=[4, 8]))
    else:
        candidates = [{'result': run(inference_configuration({'workload': {'input_length': length}}))}
                      for length in (4, 8)]
        result = {'mode': mode, 'status': 'completed', 'candidates': candidates} if mode == 'sweep' else {
            'mode': mode, 'status': 'completed', 'searches': {
                'fixed': {'status': 'completed', 'minimum_su_size': 1, 'candidates': candidates}}}
    with artifact_directory() as directory:
        write_report(result, directory)
        with (directory / 'operator_breakdown.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        groups = {(row['candidate_id'], row['metric']) for row in rows}
        assert len(groups) == 4
        for candidate, metric in groups:
            same = [row for row in rows if row['candidate_id'] == candidate and row['metric'] == metric]
            assert sum(float(row['share_pct']) for row in same) == pytest.approx(100)
            assert sum(float(row['total_s']) for row in same) == pytest.approx(float(same[0]['window_total_s']))

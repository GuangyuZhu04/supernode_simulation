"""Resource lower bounds preserve work while constraining shared service."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from supernode_sim.analysis import compile_graph, execution_metrics, run
from supernode_sim.calibration import apply_kernel_profiles
from supernode_sim.config import Config, KernelResourceDemand, KernelTraceOperation
from supernode_sim.core import Event, Resource
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import window_breakdown
from supernode_sim.operator_trace import apply_kernel_traces
from supernode_sim.streaming import run_streaming
from tests.test_operator_trace import fixture, scoped_trace
from tests.test_structured import configuration


def demand(resource, service_s, kind='memory'):
    return dict(resource=resource, service_s=service_s, kind=kind,
                source='Synthetic service demand; no measured hardware evidence.')


def single_operation(demands=(), dependency=0):
    graph, config = fixture()
    operation = dict(kernel='dense.single', flops=100, memory_bytes=100,
                     startup_s=.1, resource_demands=list(demands), dependency_floor_s=dependency)
    config['hardware']['kernel_traces']['component'][0]['operations'] = [operation]
    return graph, config


def test_shared_demands_sum_independent_demands_overlap_without_added_hbm():
    graph, config = single_operation([demand('mte1', 2), demand('mte1', 3), demand('cube', 4, 'compute')])
    apply_kernel_traces(graph, config)
    result = simulate(graph)
    assert result['duration_s'] == pytest.approx(5.3)
    event = next(e for e in graph.events if e.metadata.get('kernel'))
    assert event.work['npu0.kernel_resource.mte1'] == 5
    assert event.work['npu0.kernel_resource.cube'] == 4
    assert result['resource_work']['npu0.hbm'] == 100
    assert result['resource_work']['npu0.compute'] == 100
    assert graph.metadata['operator_trace']['additional_memory_bytes'] == 0
    assert graph.metadata['operator_trace']['resource_modeled_operations'] == 1
    assert event.metadata['resource_demands'][0]['source'].startswith('Synthetic')


def test_same_physical_resource_is_shared_between_concurrent_events():
    graph, config = single_operation([demand('l2', 3)])
    apply_kernel_traces(graph, config)
    first = next(e for e in graph.events if e.metadata.get('kernel'))
    # The trace currently preserves exclusive compute-stream ordering. Exercise
    # sharing with another graph event, as a concurrent producer/consumer would.
    first.slots = ()
    first.startup_s = 0
    graph.events.append(Event('external', work={'npu0.kernel_resource.l2': 3}))
    result = simulate(graph)
    assert result['duration_s'] == pytest.approx(6)
    assert result['resource_work']['npu0.kernel_resource.l2'] == 6


def test_different_physical_npus_have_independent_resource_capacity():
    graph, config = single_operation([demand('l2', 3)])
    apply_kernel_traces(graph, config)
    graph.resources['npu1.kernel_resource.l2'] = Resource(
        'npu1.kernel_resource.l2', 1, 'second/second', 'microarchitecture_memory')
    graph.events.append(Event('other_npu', work={'npu1.kernel_resource.l2': 3}))
    assert simulate(graph)['duration_s'] == pytest.approx(3.3)


def test_dependency_is_private_max_floor_with_startup_charged_once():
    graph, config = single_operation([demand('cube', 2, 'compute')], dependency=4)
    apply_kernel_traces(graph, config)
    result = simulate(graph)
    assert result['duration_s'] == pytest.approx(4.3)
    assert graph.metadata['operator_trace']['dependency_modeled_operations'] == 1
    event = next(e for e in graph.events if e.metadata.get('kernel'))
    assert event.metadata['dependency_floor_s'] == 4
    costs = window_breakdown(graph, result, 'after')['operator_breakdown']['components_s']
    assert costs['launch_s'] == pytest.approx(.3)
    assert costs['unattributed_service_s'] == pytest.approx(4)


@pytest.mark.parametrize('kind,expected', [('memory', 'memory_access_s'),
                                          ('compute', 'compute_s'), ('issue', 'compute_s')])
def test_limiting_resource_has_explicit_operator_cost_attribution(kind, expected):
    graph, config = single_operation([demand('limiter', 3, kind)])
    apply_kernel_traces(graph, config)
    result = simulate(graph)
    costs = window_breakdown(graph, result, 'after')['operator_breakdown']['components_s']
    assert costs[expected] == pytest.approx(3)
    assert costs['other_s'] == 0
    assert sum(costs.values()) == pytest.approx(result['duration_s'])


@pytest.mark.parametrize('dependency', [0, 4])
def test_profile_must_obey_resource_and_dependency_floor(dependency):
    graph, config = single_operation([demand('compute', 3, 'compute')], dependency=dependency)
    config['hardware']['kernel_profiles'] = {'dense.single': [dict(
        flops=100, memory_bytes=100, duration_s=2, startup_s=.05)]}
    apply_kernel_traces(graph, config)
    with pytest.raises(ValueError, match='physical'):
        apply_kernel_profiles(graph, config)
    config['hardware']['kernel_profiles']['dense.single'][0]['duration_s'] = 5
    apply_kernel_profiles(graph, config)
    assert simulate(graph)['duration_s'] == pytest.approx(5.05)
    assert graph.metadata['kernel_calibration']['matched_events'] == 1


@pytest.mark.parametrize('change', [
    {'resource': ''}, {'resource': 'a.b'}, {'resource': '1bank'},
    {'source': ''}, {'source': '  '}, {'service_s': -1},
    {'service_s': float('nan')}, {'service_s': float('inf')},
    {'kind': 'network'}, {'evidence_kind': 'measurement'},
])
def test_schema_rejects_invalid_resource_declarations(change):
    with pytest.raises(ValueError):
        KernelResourceDemand.model_validate({**demand('cube', 1), **change})


def test_schema_rejects_negative_dependency_conflicting_kinds_and_overflow():
    operation = dict(kernel='gemm', flops=100, memory_bytes=100)
    with pytest.raises(ValueError):
        KernelTraceOperation.model_validate({**operation, 'dependency_floor_s': -1})
    with pytest.raises(ValueError, match='kind'):
        KernelTraceOperation.model_validate({**operation, 'resource_demands': [
            demand('shared', 1, 'memory'), demand('shared', 1, 'compute')]})
    with pytest.raises(ValueError, match='finite'):
        KernelTraceOperation.model_validate({**operation, 'resource_demands': [
            demand('shared', 1e308), demand('shared', 1e308)]})


def test_conflicting_shared_kind_between_operations_is_rejected():
    graph, config = fixture()
    operations = config['hardware']['kernel_traces']['component'][0]['operations']
    operations[0]['resource_demands'] = [demand('shared', 2, 'compute')]
    operations[1]['resource_demands'] = [demand('shared', 3, 'memory')]
    with pytest.raises(ValueError, match='Conflicting shared kernel resource'):
        apply_kernel_traces(graph, config)


def test_omitted_and_empty_demands_produce_identical_graphs():
    graph, config = single_operation()
    previous = deepcopy(graph)
    empty = deepcopy(config)
    operation = config['hardware']['kernel_traces']['component'][0]['operations'][0]
    del operation['resource_demands']
    del operation['dependency_floor_s']
    apply_kernel_traces(graph, config)
    apply_kernel_traces(previous, empty)
    assert asdict(graph) == asdict(previous)


def test_training_model_numerator_and_physical_bytes_are_not_resource_seconds():
    config = configuration()
    baseline = run(config)
    parent = next(e for e in compile_graph(config).events
                  if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'backward')
    trace = scoped_trace(config, parent)
    trace['operations'][0]['resource_demands'] = [demand('cube', .01, 'compute'), demand('l2', .02)]
    trace['operations'][0]['dependency_floor_s'] = .03
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    actual = run(Config.model_validate(data))
    for key in ('executed_npu_flops', 'model_forward_backward_flops', 'model_recompute_flops'):
        assert actual['metrics']['training_accounting'][key] == baseline['metrics']['training_accounting'][key]
    assert actual['metrics']['training_accounting']['unknown_compute_resources'] == []
    assert actual['model_metadata']['fidelity']['hardware_performance'] == 'uncalibrated'
    for key, amount in baseline['simulation']['resource_work'].items():
        if key.endswith('.hbm'):
            assert actual['simulation']['resource_work'][key] == pytest.approx(amount)


def test_streaming_cache_preserves_demands_dependency_and_summary():
    config = configuration({'workload': {'scenario': 'inference', 'output_length': 5}})
    parent = next(e for e in compile_graph(config).events
                  if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'decode')
    trace = scoped_trace(config, parent)
    trace['operations'][0]['resource_demands'] = [demand('cube', .001, 'compute')]
    trace['operations'][0]['dependency_floor_s'] = .002
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    graph = compile_graph(data)
    expected = simulate(graph)
    metrics = execution_metrics(Config.model_validate(data), graph, expected)
    cache = {}
    for _ in range(2):
        actual = run_streaming(data, cache=cache)
        assert actual['simulation']['duration_s'] == pytest.approx(expected['duration_s'])
        assert actual['metrics']['tpot_s'] == pytest.approx(metrics['tpot_s'])
        assert actual['model_metadata']['operator_trace'] == graph.metadata['operator_trace']
        for resource in set(actual['simulation']['resource_work']) | set(expected['resource_work']):
            assert actual['simulation']['resource_work'].get(resource, 0) == pytest.approx(expected['resource_work'].get(resource, 0))
    assert any(chunk['timing_cache_hit'] for chunk in actual['chunks'])
    changed = deepcopy(data)
    changed['hardware']['kernel_traces']['dense'][0]['operations'][0]['resource_demands'][0]['service_s'] = .005
    actual = run_streaming(changed, cache=cache)
    assert actual['simulation']['duration_s'] == pytest.approx(simulate(compile_graph(changed))['duration_s'])
    assert actual['simulation']['duration_s'] > expected['duration_s']

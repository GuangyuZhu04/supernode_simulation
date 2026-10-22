"""Conserve full-graph speculative execution, including sparse EP ownership."""
from collections import defaultdict

import pytest

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from tests.test_speculative import spec


@pytest.mark.parametrize('batch', [1, 3, 4])
@pytest.mark.parametrize('sp', [False, True])
@pytest.mark.parametrize('algorithm', ['pairwise', 'direct'])
def test_streamed_rounds_match_full_graph_and_warm_decode(batch, sp, algorithm):
    value = spec({'hardware': {'npu_count': 4, 'su_size': 2},
        'model': {'layers': 4, 'architecture': {
            'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA'],
            'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE'], 'attnres_block_size': 2}},
        'execution': {'tp': 2, 'ep': 2, 'sequence_parallel': sp,
            'all_to_all_algorithm': algorithm, 'features': {'speculative_decode': {'draft': {
                'prefilled': False, 'prefill_cost': {'flops': 8192, 'memory_bytes': 4096}}}}},
        'workload': {'batch_size': batch, 'input_lengths': [8] * batch, 'output_lengths': [8] * batch}})
    graph = compile_graph(value)
    full = simulate(graph)
    expected = execution_metrics(value, graph, full)
    cache = {}
    for _ in range(2):
        actual = run_streaming(value, cache=cache)
        assert full['status'] == actual['status'] == 'feasible'
        for key in ('duration_s', 'ttft_s', 'tpot_s'):
            assert actual['metrics'][key] == pytest.approx(expected[key], rel=1e-10, abs=1e-12)
        for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
            for resource in set(full[key]) | set(actual['simulation'][key]):
                assert actual['simulation'][key].get(resource, 0) == pytest.approx(full[key].get(resource, 0))
        for metric in ('ttft', 'tpot'):
            breakdown = actual['metrics']['latency_breakdown'][metric]
            assert sum(breakdown['components_s'].values()) == pytest.approx(expected[metric + '_s'])
        traffic = sum(e.metadata.get('payload_bytes', 0) for e in graph.events)
        assert sum(p['communication']['transport_payload_bytes'] for p in actual['phase_metrics'].values()) == pytest.approx(traffic)
    warm = run_streaming(value, phase='decode', cache=cache)
    assert warm['metrics']['ttft_s'] is None
    assert warm['metrics']['tpot_s'] == pytest.approx(expected['tpot_s'])
    assert warm['metrics']['duration_s'] == pytest.approx(expected['duration_s'] - expected['ttft_s'])
    assert len(warm['metrics']['decode_intervals_s']) == 7
    assert warm['simulation']['memory_final_bytes'] == pytest.approx(full['memory_final_bytes'])


def test_sparse_ep_snapshot_uses_owned_sequences_and_final_state_is_conserved():
    value = spec({'hardware': {'npu_count': 4, 'su_size': 4}, 'execution': {'tp': 1, 'ep': 4},
        'workload': {'batch_size': 1, 'input_lengths': [8], 'output_lengths': [8]}})
    graph = compile_graph(value)
    snapshots = [a for e in graph.events for a in e.allocs if a.category == 'speculative_snapshot']
    assert snapshots and {a.pool for a in snapshots} == {'npu0.hbm'}
    full = simulate(graph)
    raw = value.model_dump()
    raw['execution']['features']['speculative_decode']['enabled'] = False
    baseline = simulate(compile_graph(Config.model_validate(raw)))
    for rank in range(4):
        key = f'npu{rank}.hbm'
        assert full['memory_final_bytes'][key] == pytest.approx(baseline['memory_final_bytes'][key] + 4096)


def test_streamed_draft_capacity_failure_is_not_reported_as_success():
    value = spec({'execution': {'features': {'speculative_decode': {'draft': {'resident_bytes_per_rank': 1e20}}}}})
    result = run_streaming(value)
    assert result['status'] == 'infeasible'
    assert result['metrics']['ttft_s'] is result['metrics']['tpot_s'] is None
    assert result['simulation']['diagnostics'][0]['code'] == 'initial_memory_exceeded'


def test_direct_all_to_all_preserves_payload_and_uses_shared_injection():
    from supernode_sim.core import Graph
    from supernode_sim.topology import Topology
    value = spec({'hardware': {'npu_count': 4, 'su_size': 4}, 'execution': {'ep': 4}}).model_dump()
    durations, work = {}, {}
    for algorithm in ('pairwise', 'direct'):
        topo = Topology(value['hardware'], value['execution']['precision'])
        graph = Graph()
        topo.populate(graph)
        topo.collective(graph, 'exchange', 'all_to_all_v', [f'npu{i}' for i in range(4)], 0,
                        payload_matrix=[[0 if i == j else 1000 for j in range(4)] for i in range(4)],
                        algorithm=algorithm)
        result = simulate(graph)
        durations[algorithm] = result['duration_s']
        work[algorithm] = result['resource_work']
        assert sum(e.metadata['payload_bytes'] for e in graph.events) == 12000
    assert work['direct'] == work['pairwise']
    assert durations['direct'] < durations['pairwise']

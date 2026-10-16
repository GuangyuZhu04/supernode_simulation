"""Latency ledgers survive bounded streaming, cache reuse and feedback export."""
import math

import pytest

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.engine import simulate
from supernode_sim.feedback import _decode_metrics, feedback
from supernode_sim.streaming import run_streaming
from supernode_sim.topology import Topology
from tests.test_structured import configuration


def inference_config(patch=None):
    from supernode_sim.config import deep_merge
    return configuration(deep_merge({
        'workload': {'scenario': 'inference', 'output_length': 3},
        'hardware': {'npu_count': 4, 'su_size': 2},
        'execution': {'tp': 2, 'ep': 2, 'features': {'kv_offload': 'host'}},
    }, patch or {}))


def assert_record(actual, expected):
    assert actual['total_s'] == pytest.approx(expected['total_s'], abs=1e-12)
    assert actual['components_s'] == pytest.approx(expected['components_s'], abs=1e-12)
    assert math.fsum(actual['components_s'].values()) == pytest.approx(actual['total_s'], abs=1e-12)
    assert all(value >= 0 for value in actual['components_s'].values())


@pytest.mark.parametrize('ragged', [False, True])
def test_cached_streaming_and_phase_windows_match_full_graph(ragged):
    config = inference_config({'workload': {'input_lengths': [8, 16], 'output_lengths': [4, 2]}}
                              if ragged else {})
    graph = compile_graph(config)
    expected = execution_metrics(config, graph, simulate(graph))
    cache = {}
    first = run_streaming(config, cache=cache)
    repeated = run_streaming(config, cache=cache)
    assert first['metrics'] == repeated['metrics']
    assert any(row['timing_cache_hit'] for row in repeated['chunks'])
    for metric in ('ttft', 'tpot'):
        actual = first['metrics']['latency_breakdown'][metric]
        assert_record(actual, expected['latency_breakdown'][metric])
        assert actual['total_s'] == pytest.approx(first['metrics'][metric + '_s'], abs=1e-12)
        assert actual['sequence'] == expected['latency_breakdown'][metric]['sequence']
    prefill = run_streaming(config, phase='prefill', cache=cache)['metrics']
    decode = run_streaming(config, phase='decode', cache=cache)['metrics']
    assert_record(prefill['latency_breakdown']['ttft'], expected['latency_breakdown']['ttft'])
    assert_record(decode['latency_breakdown']['tpot'], expected['latency_breakdown']['tpot'])
    assert prefill['tpop_s'] is prefill['tpot_s'] is prefill['latency_breakdown']['tpot'] is None
    assert decode['ttft_s'] is decode['latency_breakdown']['ttft'] is None
    assert decode['tpop_s'] == decode['tpot_s']


def test_component_macro_replay_conserves_latency_categories():
    config = inference_config({'model': {'layers': 6, 'architecture': {
        'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA', 'MLA', 'KDA'],
        'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE', 'MoE', 'MoE'],
        'attnres_block_size': 2}}, 'execution': {'sequence_parallel': True}})
    graph = compile_graph(config)
    expected = execution_metrics(config, graph, simulate(graph))
    actual = run_streaming(config)
    assert any(row['component_macro_hit'] for row in actual['chunks'])
    for metric in ('ttft', 'tpot'):
        assert_record(actual['metrics']['latency_breakdown'][metric], expected['latency_breakdown'][metric])


def test_timing_cache_identity_includes_latency_domain_semantics(monkeypatch):
    config = inference_config()
    cache = {}
    first = run_streaming(config, cache=cache)
    original = Topology.transfer

    def reclassify(self, *args, **kwargs):
        event = original(self, *args, **kwargs)
        event.metadata['resource_domains'] = {key: 'other' for key in event.work}
        event.metadata['startup_s_by_domain'] = {'other': event.startup_s}
        return event

    monkeypatch.setattr(Topology, 'transfer', reclassify)
    cached = run_streaming(config, cache=cache)
    uncached = run_streaming(config, cache={})
    assert cached['metrics'] == uncached['metrics']
    assert cached['metrics']['ttft_s'] == first['metrics']['ttft_s']
    before = first['metrics']['latency_breakdown']['ttft']['components_s']
    after = cached['metrics']['latency_breakdown']['ttft']['components_s']
    assert before['communication_scaleout_s'] > 0
    assert after['communication_scaleout_s'] == 0
    assert after['communication_other_s'] > before['communication_other_s']


def test_infeasible_streaming_has_no_finite_token_latency():
    metrics = run_streaming(inference_config({'hardware': {'npu': {'hbm_bytes': 100}}}))['metrics']
    assert metrics['ttft_s'] is metrics['tpot_s'] is metrics['tpop_s'] is None
    assert metrics['latency_breakdown']['ttft'] is metrics['latency_breakdown']['tpot'] is None
    assert metrics['latency_breakdown']['unavailable_reason'] == 'execution_incomplete'


def test_feedback_preserves_latency_breakdown_and_alias():
    metrics = run_streaming(inference_config())['metrics']
    reduced = _decode_metrics(metrics)
    assert reduced['latency_breakdown'] == metrics['latency_breakdown']
    assert reduced['tpop_s'] == metrics['tpot_s']
    assert 'requests' not in reduced
    result = feedback(inference_config(), {
        'ttft_s': 1.0, 'tpop_s': 1.0, 'prefill_lengths': [8], 'decode_batch_sizes': [2],
        'tp_sizes': [2], 'ep_sizes': [2], 'sp_enabled': [False], 'engine': 'streaming',
    })
    assert result['status'] == 'feasible'
    for row in [result['best'], *result['candidates']]:
        assert row['metrics']['tpop_s'] == row['metrics']['tpot_s']
        assert row['metrics']['latency_breakdown']['ttft']['total_s'] == pytest.approx(row['metrics']['ttft_s'])
        assert row['metrics']['latency_breakdown']['tpot']['total_s'] == pytest.approx(row['metrics']['tpot_s'])

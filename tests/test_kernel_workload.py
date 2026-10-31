"""Shape guards distinguish attention workloads that share token counts."""
from copy import deepcopy

import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, load_config
from supernode_sim.kernel_workload import kernel_workload_shape


def inference_config(lengths, outputs=None):
    data = load_config('examples/features/synthetic-training.yaml').model_dump()
    data['execution']['features']['optimizer_offload'] = False
    data['execution']['features']['activation_checkpointing'] = False
    data['workload'].update(scenario='inference', batch_size=len(lengths),
        input_lengths=lengths, output_lengths=outputs or [3] * len(lengths))
    return Config.model_validate(data)


def kernels(config, name='gqa', phase='prefill'):
    return [e for e in compile_graph(config).events
            if e.metadata.get('kernel') == name and e.metadata.get('phase') == phase]


def test_equal_tokens_different_ragged_shapes_change_attention_not_dense_work():
    balanced, ragged = inference_config([4, 4]), inference_config([2, 6])
    first, second = kernels(balanced)[0], kernels(ragged)[0]
    a, b = first.metadata['workload_shape'], second.metadata['workload_shape']
    assert a['query_tokens'] == b['query_tokens'] == 8
    assert a['dense_causal_pairs_reference'] == 20
    assert b['dense_causal_pairs_reference'] == 24
    assert a['sequence_signature'] != b['sequence_signature']
    assert first.work['npu0.compute'] < second.work['npu0.compute']
    assert kernels(balanced, 'dense')[0].work['npu0.compute'] == kernels(ragged, 'dense')[0].work['npu0.compute']


def test_decode_context_grows_without_increasing_mlp_tokens_and_finished_requests_exit():
    config = inference_config([2, 6], [2, 3])
    events = kernels(config, phase='decode')
    first, second = [e.metadata['workload_shape'] for e in events]
    assert first['source_batch_size'] == 2
    assert first['query_tokens'] == first['operator_tokens'] == 2
    assert first['kv_cache_tokens_before'] == 8
    assert second['source_batch_size'] == 1
    assert second['query_tokens'] == second['operator_tokens'] == 1
    assert second['context_tokens'] == 8
    short, long = inference_config([2]), inference_config([6])
    assert kernels(short, 'dense', 'decode')[0].work['npu0.compute'] == kernels(long, 'dense', 'decode')[0].work['npu0.compute']
    assert kernels(short, phase='decode')[0].work['npu0.hbm'] < kernels(long, phase='decode')[0].work['npu0.hbm']


def test_training_scope_and_verification_queries_are_explicit():
    training = kernel_workload_shape(phase='backward', tokens=8, lengths=[8, 8], training=True, ep=2)
    assert training['sequence_scope'] == 'dp_microbatch_before_ep'
    assert training['query_tokens'] == 16
    assert training['operator_tokens'] == 8
    assert training['training_ep_divisor'] == 2
    verify = kernel_workload_shape(phase='verify', tokens=5, lengths=[7, 10], query_lengths=[2, 3])
    assert verify['kv_cache_tokens_before'] == 12
    assert verify['dense_causal_pairs_reference'] == 13 + 27
    with pytest.raises(ValueError, match='explicit query'):
        kernel_workload_shape(phase='verify', tokens=2, lengths=[7])


def test_kernel_profile_guard_rejects_different_sequence_distribution():
    config = inference_config([4, 4]).model_dump()
    baseline = kernels(config, 'dense')[0]
    config['hardware']['kernel_profiles'] = {'dense': [{
        'flops': baseline.work['npu0.compute'], 'memory_bytes': baseline.work['npu0.hbm'],
        'duration_s': 1.0, 'phase': 'prefill',
        'metadata_match': {'workload_shape': baseline.metadata['workload_shape']},
        'source': 'synthetic scope regression; not a measured kernel',
    }]}
    matched = compile_graph(config).metadata['kernel_calibration']
    assert matched['matched_events'] == 1
    changed = deepcopy(config)
    changed['workload']['input_lengths'] = [2, 6]
    rejected = compile_graph(changed).metadata['kernel_calibration']
    assert rejected['matched_events'] == 0
    assert rejected['scope_mismatch_reasons']['metadata_mismatch'] == 1

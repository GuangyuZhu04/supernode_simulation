"""End-to-end resource and timing consistency for optional sparse attention."""
import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config
from supernode_sim.streaming import run_streaming
from tests.test_sparse_attention import sparse_model


def config_for(kind, tp=1):
    model = sparse_model(kind)
    model['layers'] = 2
    model['architecture']['attention_by_layer'] = [kind, kind]
    model['architecture']['ffn_by_layer'] = ['Dense', 'Dense']
    return Config.model_validate({
        'model': model,
        'hardware': {'npu_count': tp, 'su_size': tp, 'npus_per_host': tp,
                     'npu': {'runtime_bytes': 0, 'workspace_bytes': 0}},
        'workload': {'scenario': 'inference', 'batch_size': 2,
                     'input_lengths': [7, 9], 'output_lengths': [4, 2]},
        'execution': {'tp': tp},
    })


@pytest.mark.parametrize('kind', ['CSA', 'HCA'])
@pytest.mark.parametrize('tp', [1, 2])
def test_sparse_ragged_full_and_streaming_agree(kind, tp):
    config = config_for(kind, tp)
    full, streaming = run(config), run_streaming(config)
    assert full['status'] == streaming['status'] == 'feasible'
    for field in ('ttft_s', 'tpot_s', 'duration_s'):
        assert streaming['metrics'][field] == pytest.approx(full['metrics'][field])
    for field in ('memory_final_bytes', 'memory_peaks_bytes', 'resource_work'):
        expected, actual = full['simulation'][field], streaming['simulation'][field]
        for key in set(expected) | set(actual):
            assert actual.get(key, 0) == pytest.approx(expected.get(key, 0), abs=1e-6)
    assert full['model_metadata']['kernel_calibration'] == streaming['model_metadata']['kernel_calibration']
    assert full['metrics']['output_tokens'] == 6
    assert [len(row['token_completion_s']) for row in full['metrics']['requests']] == [4, 2]
    assert any(row['selected'] and row['name'] == 'attention'
               for row in streaming['model_metadata']['feature_plan']['features'])


def test_csa_inference_accounts_index_selection_workspace_and_communication():
    graph = compile_graph(config_for('CSA', 2))
    assert any(event.work.get('npu0.selection', 0) > 0 for event in graph.events)
    assert any('.index_scores.tp' in event.id and event.metadata.get('payload_bytes', 0) > 0
               for event in graph.events)
    workspace = {a.id for event in graph.events for a in event.allocs if a.category == 'indexer_score_workspace'}
    released = [aid for event in graph.events for aid in event.releases]
    assert workspace and all(released.count(aid) == 1 for aid in workspace)


def test_sparse_decode_only_has_no_ttft_and_prefill_has_no_tpot():
    config = config_for('CSA')
    prefill = run_streaming(config, phase='prefill')
    decode = run_streaming(config, phase='decode')
    assert prefill['status'] == decode['status'] == 'feasible'
    assert prefill['metrics']['tpot_s'] is None
    assert decode['metrics']['ttft_s'] is None
    assert decode['metrics']['tpot_s'] > 0


def test_index_scores_keep_compute_dtype_even_when_activation_wire_is_quantized():
    data = config_for('CSA', 2).model_dump()
    data['execution']['precision']['communication'] = 'fp8'
    graph = compile_graph(data)
    messages = [event for event in graph.events if 'infer.d0.s0.l0.e0.index_scores.tp' in event.id]
    # 7*floor(7/4) + 9*floor(9/4) = 25 scores. FP16 all-reduce on TP2
    # transports half the tensor in each of two ring phases per rank.
    assert len(messages) == 4
    assert all(event.metadata['payload_bytes'] == 25 for event in messages)
    assert all(allocation.dtype == 'fp16' for event in messages for allocation in event.allocs)


def test_larger_hca_position_table_increases_residency_not_single_row_lookup_work():
    graphs = []
    for ratio in (8, 128):
        model = sparse_model('HCA')
        model['architecture']['hca']['compress_ratio'] = ratio
        graphs.append(compile_graph({'model': model,
            'hardware': {'npu_count': 1, 'su_size': 1, 'npu': {'runtime_bytes': 0, 'workspace_bytes': 0}},
            'workload': {'scenario': 'inference', 'input_length': 1, 'output_length': 1}}))
    kernels = [next(e for e in graph.events if e.metadata.get('kernel') == 'hca') for graph in graphs]
    assert kernels[0].work == kernels[1].work
    assert sum(a.size_bytes for a in graphs[1].initial_allocations) > sum(a.size_bytes for a in graphs[0].initial_allocations)

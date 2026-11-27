from pathlib import Path

import pytest
from pydantic import ValidationError

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config, load_config
from supernode_sim.rank_layout import shard_pairs
from supernode_sim.streaming import run_streaming


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'independent_parallel_groups.yaml'


def config(*, training=False, **axes):
    data = load_config(EXAMPLE).model_dump()
    data['execution'].update(axes)
    if training:
        data['workload'].update(scenario='training', batch_size=1, sequence_length=8,
                                input_lengths=[], output_lengths=[])
    return Config.model_validate(data)


def test_physical_membership_residency_and_stage_cardinality():
    spec = config(rank_order=list(reversed(range(8))))
    graph = compile_graph(spec)
    layout = graph.metadata['parallel_groups']
    groups = layout['stages'][0]['groups']
    assert layout['stage_width'] == 8
    assert groups['attention_tp'][0] == ['npu7', 'npu6']
    assert groups['cp'][0] == ['npu7', 'npu5']
    assert groups['attention_dp'][0] == ['npu7', 'npu3']
    assert groups['ep'] == [[f'npu{i}' for i in reversed(range(8))]]
    for axis, members in groups.items():
        assert sorted(r for group in members for r in group) == [f'npu{i}' for i in range(8)], axis
    weights = [a for a in graph.initial_allocations if a.category == 'weight' and a.metadata.get('shard', {}).get('group') == 'expert']
    assert len(weights) == 8
    assert all(a.size_bytes == 3*16*8*2 for a in weights)  # One entire gate/up/down expert per rank.
    prefill = [e for e in graph.events if e.metadata.get('phase') == 'prefill']
    for operation in ('dispatch', 'combine'):
        transfers = [e for e in prefill if e.metadata.get('operation') == operation and e.metadata.get('type') == 'transfer']
        assert {tuple(e.metadata['communicator']) for e in transfers} == {tuple(groups['ep'][0])}
        assert sum(e.metadata['payload_bytes'] for e in transfers) == 13*2*16*2*7/8
    kv = [a for e in graph.events if '.s0.' in e.id for a in e.allocs if a.category == 'kv_cache']
    # Each position is stored once per attention TP lane, each lane stores one K/V head.
    assert sum(a.size_bytes for a in kv) == 13 * 2 * (2*4) * 2


def test_training_query_ownership_and_causal_triangles():
    graph = compile_graph(config(training=True))
    contexts = [e for e in graph.events if e.metadata.get('kernel') == 'gqa_context' and e.metadata['phase'] == 'forward']
    assert [e.metadata['tokens'] for e in contexts] == [4]*4
    assert [e.metadata['workload_shape']['dense_causal_pairs_reference'] for e in contexts] == [16, 16, 20, 20]
    assert not any(e.metadata.get('kernel') == 'gqa_context' and '.npu4.' in e.id for e in graph.events)
    assert any(e.metadata.get('operation') == 'cp_kv_gradient' for e in graph.events)
    assert run(config(training=True), full_detail=False)['status'] == 'feasible'
    # The original orthogonal mesh still puts this whole sequence on a single attention replica.
    data = config(training=True).model_dump()
    data['execution'] = {'tp': 1, 'ep': 8}
    legacy = compile_graph(Config.model_validate(data))
    attention = [e for e in legacy.events if e.metadata.get('kernel') == 'gqa' and e.metadata['phase'] == 'forward']
    assert [(e.id.split('.')[-2], e.metadata['tokens']) for e in attention] == [('npu0', 8)]


def test_expert_tensor_and_data_groups_are_independent_of_attention():
    spec = config(training=True, tp=1, cp=2, attention_dp=4, expert_tp=2, ep=2, expert_dp=2)
    graph = compile_graph(spec)
    axes = graph.metadata['parallel_groups']['stages'][0]['groups']
    assert axes['ep'][0] == ['npu0', 'npu2'] and axes['expert_tp'][0] == ['npu0', 'npu1']
    assert axes['expert_dp'][0] == ['npu0', 'npu4']
    for axis in ('expert_tp', 'ep', 'expert_dp'):
        events = [e for e in graph.events if e.metadata.get('type') == 'transfer' and e.metadata.get('parallel_axis') == axis]
        assert events and all(e.metadata['communicator'] in axes[axis] for e in events)
    assert any(e.metadata.get('parallel_axis') == 'layout_exchange' for e in graph.events)
    assert run(spec, full_detail=False)['status'] == 'feasible'


def test_expert_order_drives_regrouping_and_collective_endpoints():
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    spec = config(ep=4, expert_tp=2, expert_rank_order=order)
    graph = compile_graph(spec)
    groups = graph.metadata['parallel_groups']['stages'][0]['groups']
    assert groups['expert_tp'][0] == ['npu0', 'npu4']
    assert groups['attention_tp'][0] == ['npu0', 'npu1']
    assert groups['ep'] == [['npu0', 'npu1', 'npu2', 'npu3'], ['npu4', 'npu5', 'npu6', 'npu7']]
    transfers = [e.metadata for e in graph.events if e.metadata.get('parallel_axis') == 'layout_exchange']
    assert any(e['src'] == 'npu0' and e['dst'] == 'npu4' for e in transfers)
    assert run(spec, full_detail=False)['status'] == 'feasible'
    with pytest.raises(ValidationError, match='expert_rank_order'):
        config(expert_rank_order=[0]*8)


@pytest.mark.parametrize('axes', [{'attention_dp': 3}, {'expert_dp': 2}, {'expert_tp': 2}, {'cp': 3}])
def test_both_decompositions_must_cover_the_stage(axes):
    with pytest.raises(ValidationError):
        config(**axes)


def test_feature_input_and_streaming_use_the_same_placement():
    spec = config()
    raw = spec.model_dump()
    axes = ('tp', 'cp', 'attention_dp', 'ep', 'expert_tp', 'expert_dp')
    raw['features'] = {'parallelism': {k:raw['execution'].pop(k) for k in axes}}
    assert Config.model_validate(raw).model_dump() == spec.model_dump()
    with pytest.raises(ValidationError, match='Conflicting feature selection'):
        Config.model_validate({**spec.model_dump(), 'features': {'parallelism': {'cp': 1}}})
    full = run(spec, full_detail=False)
    streamed = run_streaming(spec)
    assert streamed['status'] == full['status'] == 'feasible'
    for key in ('ttft_s', 'tpot_s'):
        assert streamed['metrics'][key] == pytest.approx(full['metrics'][key])
    assert run_streaming(spec, phase='decode')['metrics']['tpot_s'] == pytest.approx(full['metrics']['tpot_s'])


def test_cyclic_window_count_includes_only_owned_query_positions():
    for length in range(1, 12):
        for queries in range(1, length+1):
            for cp in (1, 2, 3, 5):
                for shard in range(cp):
                    for window in (None, 2, 5):
                        expected = sum(min(i+1, window or length) for i in range(length-queries, length) if i%cp == shard)
                        assert shard_pairs(length, queries, shard, cp, window) == expected


def test_model_flops_conserved_across_context_and_expert_meshes():
    def model_work(spec):
        result = run(spec, full_detail=False)
        return result['metrics']['training_accounting']['npu_flops_by_role']['model_forward_backward']
    a = config(training=True)
    b = config(training=True, cp=1, attention_dp=4)
    assert model_work(a) == pytest.approx(model_work(b))


def test_sparse_microbatch_legacy_budget_has_single_attention_owner():
    data = config(training=True).model_dump()
    data['model'] = {'name': 'dense-budget', 'layers': 1, 'hidden_dim': 16, 'total_params': 1000,
                     'num_experts': 8, 'num_heads': 4, 'kv_heads': 2, 'head_dim': 4}
    data['execution'] = {'ep': 8}
    graph = compile_graph(Config.model_validate(data))
    kernels = [e for e in graph.events if e.metadata.get('kernel') == 'training_forward']
    assert [e.metadata['attention_tokens'] for e in kernels] == [8, 0, 0, 0, 0, 0, 0, 0]
    assert [e.metadata['flops'] for e in kernels] == [2*1000*8 + 4*4*4*36, 0, 0, 0, 0, 0, 0, 0]

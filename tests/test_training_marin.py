"""Physical counts and MFU conventions for mixed attention and nonuniform EP capacity."""
from copy import deepcopy

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config
from supernode_sim.training_attention import causal_pairs, causal_tiles


def workload(structured):
    model = {'name': 'mixed-moe', 'layers': 2, 'hidden_dim': 16, 'intermediate_dim': 16,
             'num_heads': 4, 'kv_heads': 2, 'head_dim': 4}
    if structured:
        model['architecture'] = {'vocab_size': 32, 'attention_by_layer': ['GQA'] * 2,
            'ffn_by_layer': ['MoE'] * 2, 'moe': {'num_experts': 8, 'top_k': 2, 'num_shared_experts': 1,
                                                  'intermediate_dim': 16, 'latent_dim': 8}}
    else:
        model.update(total_params=16000, num_experts=8, top_k=2, expert_parameter_fraction=0.8, moe_layer_fraction=1)
    return {'model': model, 'hardware': {'npu_count': 4, 'su_size': 2},
            'workload': {'scenario': 'training', 'batch_size': 2, 'sequence_length': 7},
            'execution': {'ep': 2, 'dp': 2, 'features': {'activation_checkpointing': True}}}


def test_window_tiles_match_per_document_brute_force_with_ragged_edges():
    for lengths in ([1], [2, 7, 13]):
        for blocks in (1, 2, 5, 20):
            for window in (None, 1, 2, 6, 30):
                expected = 0
                for length in lengths:
                    pairs = sum(1 for q in range(length) for k in range(q+1) if window is None or q-k < window)
                    assert causal_pairs(length, window) == pairs
                    expected += pairs
                tiles = causal_tiles(lengths, blocks, window)
                assert sum(v[0] for _, v in tiles) == expected
                assert sum(v[2] for _, v in tiles) == sum(v[3] for _, v in tiles) == sum(lengths)
    assert len(causal_tiles([13], 8, 1)) == 8


@pytest.mark.parametrize('structured,blocks', [(False, 0), (True, 0), (True, 3)])
def test_window_changes_only_attention_work_on_selected_layer(structured, blocks):
    data = workload(structured)
    data['execution']['flash_attention_blocks'] = blocks
    base = run(data)['metrics']['training_accounting']
    data['execution']['training_attention_windows'] = [2, None]
    window = run(data)['metrics']['training_accounting']
    # 28 causal positions at length seven, 13 valid in a two-token window.
    removed = (28 - 13) * 4 * 4 * 4 * 2 * 2  # 4*heads*head_dim, 2 sequences, 2 replicas
    assert base['model_forward_backward_flops'] - window['model_forward_backward_flops'] == 3 * removed
    assert base['model_recompute_flops'] - window['model_recompute_flops'] == removed


@pytest.mark.parametrize('structured', [False, True])
def test_routed_work_partitions_wire_padding_and_useful_compute(structured):
    data = workload(structured)
    wire_width = 8
    ragged = {'mode': 'ragged', 'routed_width': wire_width}
    fixed = {**ragged, 'mode': 'fixed_capacity', 'sender_capacity_factor': 1.25,
             'expert_capacity_factor': 1.5, 'assignment_drop_fraction': .25}
    tokens = 2 * 7 * 2  # global microbatch, EP does not multiply it
    active_expert_params = 2 * 3 * 8 * 16 * 2 if structured else 16000 * .8 * 2 / 8
    requested_fb = 6 * tokens * active_expert_params
    for policy, wire_factor in [(ragged, 1), (fixed, 1.25), ({**ragged, 'assignment_drop_fraction': .25}, .75)]:
        data['execution']['training_moe_routing'] = policy
        cfg = Config.model_validate(data)
        graph = compile_graph(cfg)
        for phase in ('forward', 'backward', 'recompute'):
            for op in ('dispatch', 'combine'):
                payload = sum(e.metadata['payload_bytes'] for e in graph.events if e.metadata.get('phase') == phase
                    and e.metadata.get('type') == 'transfer' and op in (e.metadata.get('operation'), e.metadata.get('semantic')))
                assert payload == tokens * 2 * 2 * wire_width * 2 * .5 * wire_factor  # layers, top-k, BF16/FP16, off rank
        result = run(cfg)
        exposed = {s['name'] for s in result['training_summary']['groups']['exposed_ops']['segments']}
        assert {'ep / all_to_all_v (dispatch)', 'ep / all_to_all_v (combine)'} <= exposed
        accounting = result['metrics']['training_accounting']
        roles = accounting['npu_flops_by_role']
        assert sum(roles.values()) == pytest.approx(accounting['executed_npu_flops'])
        if policy == ragged:
            reference = accounting
            continue
        assert reference['model_forward_backward_flops'] - accounting['model_forward_backward_flops'] == pytest.approx(requested_fb * .25)
        assert reference['model_recompute_flops'] - accounting['model_recompute_flops'] == pytest.approx(requested_fb * .25 / 3)
        assert accounting['moe_routing']['dropped_forward_backward_flops'] == pytest.approx(requested_fb * .25)
        padding = requested_fb * (4 / 3) * .75 if policy == fixed else 0
        assert accounting['moe_routing']['expert_padding_flops'] == pytest.approx(padding)
        assert result['training_summary']['training_compute']['moe_routing'] == accounting['moe_routing']


def test_external_mfu_uses_full_step_and_declared_counter_in_both_compilers():
    for structured in (False, True):
        data = workload(structured)
        data['workload'].update(gradient_accumulation=2, training_steps=2)
        reference = {'name': 'external counter', 'dtype': 'fp16', 'sequence_length': 7, 'model_flops_per_token': 12345}
        data['execution']['training_mfu_reference'] = reference
        result = run(data)
        ref = result['training_summary']['training_compute']['reference_mfu']
        assert ref['tokens'] == 2 * 7 * 2 * 2 * 2
        assert ref['ratio'] == pytest.approx(ref['tokens'] * 12345 / (4 * ref['peak_flops_per_device_s'] * result['simulation']['duration_s']))
        data['workload']['sequence_length'] = 8
        with pytest.raises(ValueError, match='sequence length'):
            Config.model_validate(data)


def test_opt_in_scope_and_legacy_serialization():
    data = workload(True)
    assert not {'training_moe_routing', 'training_attention_windows', 'training_mfu_reference'} & Config.model_validate(data).model_dump()['execution'].keys()
    selected = deepcopy(data)
    selected['features'] = {'training_attention_windows': [2, None], 'training_moe_routing': {'mode': 'ragged'}}
    cfg = Config.model_validate(selected)
    assert cfg.execution.training_attention_windows == [2, None]
    assert cfg.execution.training_moe_routing.mode == 'ragged'
    for invalid in ({'training_attention_windows': [2]}, {'training_moe_routing': {'mode': 'ragged', 'sender_capacity_factor': 2}},
                    {'training_moe_routing': {'mode': 'ragged', 'routed_width': 16}}):
        with pytest.raises(ValueError):
            Config.model_validate({**data, 'execution': {**data['execution'], **invalid}})

@pytest.mark.parametrize('blocks', [0, 3])
def test_active_kv_slices_work_and_activation_but_keeps_stored_weights(blocks):
    data = workload(True)
    data['execution']['flash_attention_blocks'] = blocks
    base = run(data)
    data['execution']['training_active_kv_heads'] = [2, 1]
    sliced = run(data)
    base_flops = base['metrics']['training_accounting']['model_forward_backward_flops']
    active_flops = sliced['metrics']['training_accounting']['model_forward_backward_flops']
    # Remove one K and one V head in just one layer, at 6 FLOPs per parameter/token.
    assert base_flops - active_flops == 2 * 4 * 16 * 6 * 28
    assert sliced['model_metadata']['component_inventory'] == base['model_metadata']['component_inventory']
    data['execution']['training_active_kv_heads'] = [2, 4]
    with pytest.raises(ValueError, match='Active GQA'):
        Config.model_validate(data)


def test_marin_published_counter_at_each_context():
    from examples.marin_training import logger_flops_per_token
    # Independent outputs from the GRUG + Levanter counter at the cited revision.
    assert [logger_flops_per_token(n) for n in (4096, 16384, 65536)] == [
        149_736_259_584, 160_671_596_544, 204_412_944_384]

def test_refined_moe_trace_conserves_padding_and_dropped_counters():
    data = workload(True)
    data['execution']['training_moe_routing'] = {'mode': 'fixed_capacity', 'expert_capacity_factor': 1.25,
                                                  'assignment_drop_fraction': .125}
    cfg = Config.model_validate(data)
    parent = next(e for e in compile_graph(cfg).events if e.metadata.get('phase') == 'forward' and e.metadata.get('kernel') == 'moe')
    flops = parent.work['npu0.compute']
    memory = parent.work['npu0.hbm']
    trace = {'id': 'split-moe', 'source': 'Synthetic work conservation check', 'hardware_name': cfg.hardware.name,
             'dtype': 'fp16', 'phase': 'forward', 'metadata_match': {'component': 'MoE'},
             'flops': flops, 'memory_bytes': memory,
             'operations': [{'kernel': 'first', 'flops': flops / 4, 'memory_bytes': memory / 4},
                            {'kernel': 'second', 'flops': 3 * flops / 4, 'memory_bytes': 3 * memory / 4}]}
    data['hardware']['kernel_traces'] = {'moe': [trace]}
    original_result, traced_result = run(cfg), run(data)
    assert traced_result['model_metadata']['operator_trace']['matched_events'] > 0
    for component in ('npu_flops_by_role', 'moe_routing'):
        original = original_result['metrics']['training_accounting'][component]
        refined = traced_result['metrics']['training_accounting'][component]
        assert refined == original

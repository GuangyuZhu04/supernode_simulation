"""Optional feature inputs resolve strictly to rerunnable canonical settings."""
import copy
import json

import pytest

from supernode_sim.cli import main, parser
from supernode_sim.config import ArchitectureSpec, Config, CSASpec, HCASpec, load_config
from supernode_sim.features import build_feature_plan, feature_catalog


def baseline(scenario='inference'):
    return {
        'model': {'name': 'synthetic-feature-test', 'total_params': 1024,
                  'layers': 1, 'hidden_dim': 8, 'num_heads': 2, 'kv_heads': 1, 'head_dim': 4},
        'hardware': {'npu_count': 1, 'su_size': 1},
        'workload': {'scenario': scenario, 'input_length': 4, 'output_length': 4, 'sequence_length': 4},
    }


def structured(scenario='inference'):
    data = baseline(scenario)
    data['model'].pop('total_params')
    data['model']['architecture'] = {
        'vocab_size': 16, 'attention_by_layer': ['GQA'], 'ffn_by_layer': ['Dense'],
        'dense_intermediate_dim': 16,
    }
    return data


def speculative_selection():
    return {'enabled': True, 'method': 'dspark', 'draft_tokens': 2, 'accepted_length': 2,
            'draft': {'resident_bytes_per_rank': 1024,
                      'prefill_cost': {'duration_s': 0.000002, 'memory_bytes': 64},
                      'cost': {'duration_s': 0.001, 'memory_bytes': 64}}}


def test_legacy_input_remains_valid_and_optional_input_is_not_serialized():
    data = baseline()
    old = Config.model_validate(data).model_dump()
    data['features'] = {}
    resolved = Config.model_validate(data).model_dump()
    assert resolved == old
    assert 'features' not in resolved
    assert Config.model_validate(resolved).model_dump() == resolved


def test_feature_precision_expands_only_explicit_fields_and_preserves_other_settings():
    data = baseline()
    data['execution'] = {'precision': {'weight': 'bf16'}}
    data['features'] = {'precision': {'compute': 'bf16', 'kv': 'int8'},
                        'kv_offload': {'location': 'host', 'fraction': 0.25, 'page_tokens': 8}}
    config = Config.model_validate(data)
    assert config.execution.precision.weight == 'bf16'
    assert config.execution.precision.compute == 'bf16'
    assert config.execution.precision.kv == 'int8'
    assert config.execution.features.kv_offload == 'host'
    assert config.execution.features.kv_offload_fraction == 0.25
    assert config.execution.features.kv_page_tokens == 8
    assert 'execution' not in data['features']


def test_conflicting_canonical_and_feature_values_are_rejected_but_equal_allowed():
    data = baseline()
    data['execution'] = {'precision': {'compute': 'fp16'}}
    data['features'] = {'precision': {'compute': 'bf16'}}
    with pytest.raises(ValueError, match='execution.precision.compute'):
        Config.model_validate(data)
    data['features']['precision']['compute'] = 'fp16'
    assert Config.model_validate(data).execution.precision.compute == 'fp16'


def test_resolved_config_can_change_canonical_option_without_replaying_old_selection():
    data = baseline()
    data['features'] = {'precision': {'compute': 'bf16'}}
    resolved = Config.model_validate(data).model_dump()
    resolved['execution']['precision']['compute'] = 'fp16'
    rerun = Config.model_validate(resolved)
    assert rerun.execution.precision.compute == 'fp16'
    assert rerun.features is None


def test_training_checkpoint_offload_fsdp_and_parallelism_map_to_existing_fields():
    data = baseline('training')
    data['hardware'].update(npu_count=2, su_size=2)
    data['features'] = {
        'activation_checkpointing': True, 'optimizer_offload': True,
        'parallelism': {'dp': 2}, 'fsdp': {'degree': 2, 'reshard_after_forward': False},
        'optimizer': {'kind': 'adam'}, 'quantization': {'weight_scale_block': 64, 'extra_weight_copy_dtype': 'bf16'},
    }
    config = Config.model_validate(data)
    assert config.execution.fsdp == config.execution.dp == 2
    assert not config.execution.reshard_after_forward
    assert config.execution.features.activation_checkpointing
    assert config.execution.features.optimizer_offload
    assert config.execution.features.weight_scale_block == 64


@pytest.mark.parametrize('selection', [
    {'activation_checkpointing': True}, {'optimizer_offload': True},
    {'fsdp': {'degree': 2}}, {'optimizer': {'kind': 'adam'}},
])
def test_training_only_feature_inputs_reject_inference(selection):
    data = baseline()
    data['features'] = selection
    with pytest.raises(ValueError, match='only supported for training'):
        Config.model_validate(data)


@pytest.mark.parametrize('selection', [
    {'kv_offload': {'location': 'host'}}, {'engram': {'enabled': True}},
    {'sequence_parallel': True},
])
def test_inference_only_feature_inputs_reject_training(selection):
    data = baseline('training')
    data['features'] = selection
    with pytest.raises(ValueError, match='only supported for inference'):
        Config.model_validate(data)


def test_disabled_training_only_options_can_be_explicit_in_inference():
    data = baseline()
    data['features'] = {'activation_checkpointing': False, 'optimizer_offload': False, 'fsdp': {'degree': 1}}
    assert Config.model_validate(data).execution.fsdp == 1


@pytest.mark.parametrize('selection', [
    {'unknown_accelerator': True}, {'continuous_batching': True},
    {'speculative_decode': True}, {'kernel_profiles': True},
    {'activation_checkpointing': 'true'}, {'precision': {'imaginary_dtype': 'fp8'}},
])
def test_unknown_features_and_boolean_shortcuts_are_rejected(selection):
    data = baseline()
    data['features'] = selection
    with pytest.raises(ValueError):
        Config.model_validate(data)


def test_attention_input_selects_explicit_kda_shapes():
    data = structured()
    data['model']['architecture'].pop('attention_by_layer')
    data['features'] = {'attention': {
        'attention_by_layer': ['KDA'],
        'kda': {'num_heads': 2, 'head_k_dim': 4, 'head_v_dim': 4, 'gate_low_rank_dim': 4},
    }}
    config = Config.model_validate(data)
    assert config.model.architecture.attention_by_layer == ['KDA']
    assert config.model.architecture.kda.num_heads == 2
    assert config.model.total_params > 0
    assert Config.model_validate(config.model_dump()).model_dump() == config.model_dump()


def test_attention_input_cannot_silently_replace_an_explicit_layer_map():
    data = structured()
    data['features'] = {'attention': {'attention_by_layer': ['KDA']}}
    with pytest.raises(ValueError, match='model.architecture.attention_by_layer'):
        Config.model_validate(data)


def test_sequence_parallel_unified_selection_keeps_shape_and_topology_validation():
    data = structured()
    data['hardware'].update(npu_count=2, su_size=2)
    data['features'] = {'parallelism': {'tp': 2}, 'sequence_parallel': True}
    config = Config.model_validate(data)
    assert config.execution.sequence_parallel
    assert config.execution.tp == 2
    data['features']['parallelism']['tp'] = 1
    data['hardware'].update(npu_count=1, su_size=1)
    with pytest.raises(ValueError, match='requires TP > 1'):
        Config.model_validate(data)


def test_kernel_profiles_and_feature_plan_do_not_claim_measured_serving_fidelity():
    data = baseline()
    data['features'] = {'kernel_profiles': {'decode': [{'flops': 10, 'memory_bytes': 20, 'duration_s': 0.1}]}}
    config = Config.model_validate(data)
    coverage = {'matched_events': 0, 'unmatched_events': 5}
    plan = build_feature_plan(config, coverage)
    kernel = next(row for row in plan['features'] if row['name'] == 'kernel_profiles')
    assert kernel['selected']
    assert kernel['modeling'] == 'profile_driven'
    assert plan['kernel_calibration_coverage'] == coverage
    assert plan['coverage_reference'] == 'model_metadata.kernel_calibration'
    assert all(row['modeling'] not in ('measured', 'verified') for row in plan['features'])


def test_speculative_selection_requires_cost_residency_and_acceptance():
    data = structured()
    data['features'] = {'speculative_decode': speculative_selection()}
    config = Config.model_validate(data)
    assert config.execution.features.speculative_decode.method == 'dspark'
    assert config.execution.features.speculative_decode.accepted_length == 2
    for patch in ({'draft': None}, {'accepted_length': None}, {'accepted_lengths': [1, 2]}, {'accepted_length': 4}):
        invalid = copy.deepcopy(data)
        invalid['features']['speculative_decode'].update(patch)
        with pytest.raises(ValueError):
            Config.model_validate(invalid)


def test_speculative_selection_rejects_unmodeled_kv_offload_and_nonuniform_batch():
    data = structured()
    data['features'] = {'speculative_decode': speculative_selection(), 'kv_offload': {'location': 'host'}}
    with pytest.raises(ValueError, match='kv_offload=none'):
        Config.model_validate(data)
    data['features'].pop('kv_offload')
    data['workload'].update(batch_size=2, input_lengths=[4, 5], output_lengths=[4, 4])
    with pytest.raises(ValueError, match='uniform'):
        Config.model_validate(data)


def test_speculative_draft_prompt_preparation_must_be_explicit():
    data = structured()
    selection = speculative_selection()
    selection['draft'].pop('prefill_cost')
    data['features'] = {'speculative_decode': selection}
    with pytest.raises(ValueError, match='draft.prefill_cost'):
        Config.model_validate(data)
    selection['draft']['prefilled'] = True
    assert Config.model_validate(data).execution.features.speculative_decode.draft.prefilled
    selection['draft']['prefill_cost'] = {'duration_s': 0.000002}
    with pytest.raises(ValueError, match='mutually exclusive'):
        Config.model_validate(data)


def test_sparse_attention_requires_declared_dimensions_and_valid_layout():
    for kind in ('CSA', 'HCA'):
        with pytest.raises(ValueError, match='explicit architecture'):
            ArchitectureSpec.model_validate({'vocab_size': 16, 'attention_by_layer': [kind], 'ffn_by_layer': ['Dense']})
    common = {'q_lora_rank': 4, 'head_dim': 4, 'rope_head_dim': 2, 'o_groups': 2, 'o_lora_rank': 4, 'window_size': 4}
    csa = CSASpec.model_validate(common | {'index_num_heads': 2, 'index_head_dim': 2, 'index_topk': 2})
    assert csa.compress_ratio == 4
    assert HCASpec.model_validate(common).compress_ratio == 128
    with pytest.raises(ValueError, match='compress_ratio=4'):
        HCASpec.model_validate(common | {'compress_ratio': 4})
    with pytest.raises(ValueError, match='index_head_dim'):
        CSASpec.model_validate(common | {'index_num_heads': 2, 'index_head_dim': 1, 'index_topk': 2})


@pytest.mark.parametrize('kind', ['CSA', 'HCA'])
@pytest.mark.parametrize('unsupported', ['mixed_kv_dtype', 'kv_offload'])
def test_sparse_attention_rejects_unmodeled_cache_conversion_and_offload(kind, unsupported):
    data = structured()
    dimensions = {'q_lora_rank': 4, 'head_dim': 4, 'rope_head_dim': 2,
                  'o_groups': 2, 'o_lora_rank': 4, 'window_size': 4}
    if kind == 'CSA':
        dimensions.update(index_num_heads=2, index_head_dim=2, index_topk=2)
    data['model']['architecture'].update(attention_by_layer=[kind], **{kind.lower(): dimensions})
    if unsupported == 'mixed_kv_dtype':
        data['features'] = {'precision': {'compute': 'bf16', 'kv': 'fp16'}}
        message = 'KV dtype equal to compute dtype'
    else:
        data['features'] = {'kv_offload': {'location': 'host'}}
        message = 'kv_offload=none'
    with pytest.raises(ValueError, match=message):
        Config.model_validate(data)


def test_feature_file_merges_inline_selection_and_rejects_conflicting_overrides(tmp_path):
    source = tmp_path / 'scenario.json'
    features = tmp_path / 'features.json'
    data = baseline()
    data['features'] = {'precision': {'compute': 'bf16'}}
    source.write_text(json.dumps(data), encoding='utf-8')
    features.write_text(json.dumps({'precision': {'weight': 'bf16'}}), encoding='utf-8')
    config = load_config(source, features)
    assert config.execution.precision.compute == config.execution.precision.weight == 'bf16'
    features.write_text(json.dumps({'precision': {'compute': 'fp16'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='features.precision.compute'):
        load_config(source, features)


def test_cli_feature_file_validation_catalog_and_schema(tmp_path, capsys):
    source = tmp_path / 'scenario.json'
    features = tmp_path / 'features.json'
    source.write_text(json.dumps(baseline()), encoding='utf-8')
    features.write_text(json.dumps({'precision': {'compute': 'bf16'}}), encoding='utf-8')
    assert main(['validate', str(source), '--features', str(features)]) == 0
    validated = json.loads(capsys.readouterr().out)
    precision = next(row for row in validated['feature_plan']['features'] if row['name'] == 'precision')
    assert precision['parameters']['execution.precision']['compute'] == 'bf16'
    assert main(['list-features']) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert next(row for row in catalog['features'] if row['name'] == 'continuous_batching')['support_status'] == 'unsupported'
    assert main(['schema']) == 0
    schema = json.loads(capsys.readouterr().out)
    assert 'features' in schema['properties']
    assert 'speculative_decode' in schema['$defs']['FeatureSelectionSpec']['properties']
    assert 'CSA' in schema['$defs']['ArchitectureSpec']['properties']['attention_by_layer']['items']['enum']


def test_all_config_commands_accept_the_same_feature_file_option():
    for command in ('validate', 'estimate', 'simulate', 'sweep', 'size-su'):
        extra = ['--axes', 'axes.yaml'] if command == 'sweep' else []
        args = parser().parse_args([command, 'scenario.yaml', '--features', 'features.yaml', *extra])
        assert args.features.name == 'features.yaml'


def test_catalog_has_explicit_unsupported_options_and_no_nominal_speedups():
    catalog = feature_catalog()
    unsupported = {row['name'] for row in catalog['features'] if row['support_status'] == 'unsupported'}
    assert {'continuous_batching', 'prefix_cache', 'chunked_prefill', 'mhc'} <= unsupported
    assert 'speedup_factor' not in json.dumps(catalog)


def megakernel_selection():
    return {'enabled': True, 'groups': [
        {'name': 'synthetic-local-chain', 'kernels': ['first_kernel', 'second_kernel'],
         'phases': ['decode'], 'layers': [0], 'ranks': ['npu0'],
         'fused_startup_s': 0.000001, 'saved_hbm_bytes': 16, 'workspace_bytes': 32},
    ]}


def pd_scenario():
    data = structured()
    data['hardware'].update(npu_count=2, su_size=2)
    data['features'] = {'prefill_decode_disaggregation': {'enabled': True}}
    return data


def test_megakernel_selection_expands_and_survives_canonical_replay():
    data = structured()
    data['features'] = {'megakernel': megakernel_selection()}
    config = Config.model_validate(data)
    group = config.execution.features.megakernel.groups[0]
    assert group.kernels == ['first_kernel', 'second_kernel']
    assert group.fused_startup_s == 0.000001
    assert group.source.status == 'assumed'
    resolved = config.model_dump()
    assert 'features' not in resolved
    assert Config.model_validate(resolved).model_dump() == resolved
    rows = build_feature_plan(config)['features']
    fused = next(row for row in rows if row['name'] == 'megakernel')
    assert fused['selected'] and fused['modeling'] == 'analytical'
    assert fused['coverage_reference'] == 'model_metadata.megakernel'


def test_megakernel_explicit_duration_is_profile_driven_without_claiming_measured():
    data = structured('training')
    selection = megakernel_selection()
    selection['groups'][0].update(phases=['forward'], duration_s=0.001)
    data['features'] = {'megakernel': selection}
    row = next(row for row in build_feature_plan(Config.model_validate(data))['features'] if row['name'] == 'megakernel')
    assert row['modeling'] == 'profile_driven'
    assert row['parameters']['execution.features.megakernel']['groups'][0]['source']['status'] == 'assumed'


@pytest.mark.parametrize('change', [
    {'kernels': ['only_one']}, {'kernels': ['a', '']}, {'name': '  '},
    {'phases': ['']}, {'layers': [-1]}, {'layers': [1]},
    {'ranks': ['cpu0']}, {'ranks': ['npu1']}, {'fused_startup_s': -1},
    {'duration_s': 0}, {'saved_hbm_bytes': -1}, {'speedup_factor': 2},
])
def test_invalid_megakernel_groups_are_rejected(change):
    data = structured()
    selection = megakernel_selection()
    selection['groups'][0].update(change)
    data['features'] = {'megakernel': selection}
    with pytest.raises(ValueError):
        Config.model_validate(data)


def test_megakernel_requires_explicit_groups_and_startup_and_unique_names():
    data = structured()
    for selection in (True, {'enabled': True}):
        data['features'] = {'megakernel': selection}
        with pytest.raises(ValueError):
            Config.model_validate(data)
    selection = megakernel_selection()
    selection['groups'][0].pop('fused_startup_s')
    data['features'] = {'megakernel': selection}
    with pytest.raises(ValueError, match='fused_startup_s'):
        Config.model_validate(data)
    selection = megakernel_selection()
    selection['groups'].append(copy.deepcopy(selection['groups'][0]))
    data['features'] = {'megakernel': selection}
    with pytest.raises(ValueError, match='unique'):
        Config.model_validate(data)


def test_megakernel_repeated_kernel_pattern_is_legal_and_conflicts_remain_strict():
    data = structured()
    selection = megakernel_selection()
    selection['groups'][0]['kernels'] = ['same_kernel', 'same_kernel']
    data['features'] = {'megakernel': selection}
    assert Config.model_validate(data).execution.features.megakernel.enabled
    data['execution'] = {'features': {'megakernel': {'enabled': False}}}
    with pytest.raises(ValueError, match='execution.features.megakernel.enabled'):
        Config.model_validate(data)


def test_pd_default_pool_placement_uses_total_physical_count_and_replays_stably():
    config = Config.model_validate(pd_scenario())
    pd = config.execution.features.prefill_decode_disaggregation
    assert pd.prefill_ranks == [0]
    assert pd.decode_ranks == [1]
    assert pd.token_id_bytes == 8
    assert config.hardware.npu_count == 2 * config.execution.tp * config.execution.ep
    resolved = config.model_dump()
    assert Config.model_validate(resolved).model_dump() == resolved
    row = next(row for row in build_feature_plan(config)['features'] if row['name'] == 'prefill_decode_disaggregation')
    assert row['selected'] and row['support_status'] == 'supported'
    assert row['physical_npu_count'] == 2
    assert row['stage_parallelism'] == {'tp': 1, 'ep': 1, 'pp': 1, 'dp': 1}


def test_pd_explicit_rank_order_with_sp_and_ragged_lengths_is_preserved():
    data = pd_scenario()
    data['hardware'].update(npu_count=4, su_size=4)
    data['features'].update(parallelism={'tp': 2}, sequence_parallel=True)
    data['features']['prefill_decode_disaggregation'].update(prefill_ranks=[3, 1], decode_ranks=[0, 2], token_id_bytes=4)
    data['workload'].update(batch_size=2, input_lengths=[3, 4], output_lengths=[1, 3])
    config = Config.model_validate(data)
    pd = config.execution.features.prefill_decode_disaggregation
    assert pd.prefill_ranks == [3, 1]
    assert pd.decode_ranks == [0, 2]
    assert config.execution.sequence_parallel
    assert pd.token_id_bytes == 4


@pytest.mark.parametrize('change', [
    {'prefill_ranks': [0]}, {'decode_ranks': [1]},
    {'prefill_ranks': [0], 'decode_ranks': [0]},
    {'prefill_ranks': [0], 'decode_ranks': [2]},
    {'prefill_ranks': [-1], 'decode_ranks': [1]},
    {'prefill_ranks': [True], 'decode_ranks': [1]},
    {'prefill_ranks': [0, 1], 'decode_ranks': [2, 3]},
    {'token_id_bytes': 2}, {'prefill_tp': 2},
])
def test_pd_invalid_or_ambiguous_pool_layouts_are_rejected(change):
    data = pd_scenario()
    data['features']['prefill_decode_disaggregation'].update(change)
    with pytest.raises(ValueError):
        Config.model_validate(data)


def test_pd_does_not_relax_ordinary_rank_count_or_attention_divisibility():
    data = structured()
    data['hardware'].update(npu_count=2, su_size=2)
    with pytest.raises(ValueError, match='must equal hardware.npu_count'):
        Config.model_validate(data)
    data['features'] = {'prefill_decode_disaggregation': {'enabled': True}}
    data['hardware'].update(npu_count=1, su_size=1)
    with pytest.raises(ValueError, match=r'2 \* TP \* EP'):
        Config.model_validate(data)
    data['hardware'].update(npu_count=6, su_size=6)
    data['features']['parallelism'] = {'tp': 3}
    with pytest.raises(ValueError, match='num_heads divisible by TP'):
        Config.model_validate(data)


@pytest.mark.parametrize('extra', [
    {'parallelism': {'dp': 2}}, {'parallelism': {'pp': 2}},
    {'parallelism': {'rank_order': [0, 1]}},
    {'kv_offload': {'location': 'host'}}, {'engram': {'enabled': True}},
    {'speculative_decode': speculative_selection()},
])
def test_pd_rejects_unsupported_execution_combinations(extra):
    data = pd_scenario()
    data['features'].update(extra)
    with pytest.raises(ValueError, match='prefill_decode_disaggregation'):
        Config.model_validate(data)


def test_pd_requires_structured_inference_and_can_combine_with_megakernel():
    data = baseline()
    data['hardware'].update(npu_count=2, su_size=2)
    data['features'] = {'prefill_decode_disaggregation': {'enabled': True}}
    with pytest.raises(ValueError, match='structured inference'):
        Config.model_validate(data)
    data = pd_scenario()
    data['workload']['scenario'] = 'training'
    with pytest.raises(ValueError, match='only supported for inference'):
        Config.model_validate(data)
    data = pd_scenario()
    data['features']['megakernel'] = megakernel_selection()
    assert Config.model_validate(data).execution.features.megakernel.enabled


def test_new_features_are_disabled_by_default_and_pd_is_no_longer_unsupported():
    config = Config.model_validate(baseline())
    assert not config.execution.features.megakernel.enabled
    assert not config.execution.features.prefill_decode_disaggregation.enabled
    catalog = {row['name']: row for row in feature_catalog()['features']}
    assert catalog['prefill_decode_disaggregation']['support_status'] == 'supported'
    assert catalog['megakernel']['support_status'] == 'supported'
    schema = Config.model_json_schema()
    assert {'megakernel', 'prefill_decode_disaggregation'} <= schema['$defs']['FeatureSelectionSpec']['properties'].keys()

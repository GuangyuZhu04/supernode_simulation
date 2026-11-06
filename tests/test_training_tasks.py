"""Task semantics, real cost sensitivity, operator identity and lifetime checks."""
import pytest

from supernode_sim.analysis import compile_graph, execution_metrics, run
from supernode_sim.components import parameter_inventory
from supernode_sim.config import Config, deep_merge
from supernode_sim.engine import simulate


def model(scalar=False):
    result = {'name': 'synthetic-task-model', 'layers': 1, 'hidden_dim': 16,
              'num_heads': 4, 'kv_heads': 2, 'head_dim': 4,
              'architecture': {'vocab_size': 32, 'attention_by_layer': ['GQA'],
                               'ffn_by_layer': ['Dense'], 'dense_intermediate_dim': 32}}
    if scalar:
        result['architecture']['output_head_dim'] = 1
    return result


def configuration(task=None, patch=None):
    return Config.model_validate(deep_merge({
        'model': model(),
        'hardware': {'name': 'synthetic-task-device', 'npu_count': 1, 'su_size': 1,
                     'npus_per_host': 1, 'npu': {'runtime_bytes': 64, 'workspace_bytes': 32}},
        'workload': {'scenario': 'training', 'batch_size': 2, 'sequence_length': 8,
                     'training': task or {'kind': 'pretrain'}},
    }, patch or {}))


def rl(**patch):
    return {'kind': 'rl', 'data': {'batching': 'packed'}, 'rl': {
        'prompts_per_step': 2, 'samples_per_prompt': 2, 'prompt_length': 4,
        'response_lengths': [2, 4, 1, 3], 'rollout_batch_size': 3, **patch}}


def execute(config):
    graph = compile_graph(config)
    result = simulate(graph)
    assert result['status'] == 'feasible', result['diagnostics']
    resident = {pool: sum(a.size_bytes for a in graph.initial_allocations if a.pool == pool)
                for pool in graph.memory_caps}
    assert {pool: result['memory_final_bytes'].get(pool, 0) for pool in resident} == pytest.approx(resident)
    return graph, result, execution_metrics(config, graph, result)


@pytest.mark.parametrize('task', [
    {'kind': 'pretrain', 'pretrain': {'dataset_tokens': 1000, 'epochs': 2}},
    {'kind': 'sft', 'sft': {'supervised_tokens': [2, 3]}},
    {'kind': 'dpo', 'dpo': {'chosen_lengths': [5, 7], 'rejected_lengths': [6, 8]}},
    rl(), rl(algorithm='ppo', critic_model=model(True), reward='model', reward_model=model(True)),
])
def test_all_tasks_execute_multiple_steps_and_release_transients(task):
    graph, result, metrics = execute(configuration(task, {'workload': {'training_steps': 2},
        'execution': {'features': {'activation_checkpointing': True, 'optimizer_offload': True}}}))
    assert metrics['training_task']['kind'] == task['kind']
    assert sum(metrics['step_times_s']) == pytest.approx(result['duration_s'])
    assert sum(metrics['training_task']['stage_times_s'].values()) == pytest.approx(result['duration_s'])
    assert metrics['training_accounting']['status'] == 'complete'
    assert metrics['training_accounting']['unclassified_phase_names'] == []
    assert len([a for a in graph.initial_allocations if a.category == 'runtime']) == 1


def test_sft_mask_changes_supervision_without_pruning_backbone_or_output():
    first = execute(configuration({'kind': 'sft', 'sft': {'supervised_tokens': [2, 3]}}))
    second = execute(configuration({'kind': 'sft', 'sft': {'supervised_tokens': [1, 1]}}))
    assert first[1]['resource_work'] == second[1]['resource_work']
    assert first[1]['duration_s'] == second[1]['duration_s']
    assert first[2]['training_task']['supervised_tokens'] == 5
    assert second[2]['training_task']['supervised_tokens'] == 2
    assert first[2]['training_tokens'] == second[2]['training_tokens'] == 16


def test_packing_uses_independent_causal_segments_and_real_compute_tokens():
    task = {'kind': 'sft', 'data': {'sequence_lengths': [3, 6]}, 'sft': {'supervised_tokens': [1, 2]}}
    padded = execute(configuration(task))
    task['data']['batching'] = 'packed'
    packed = execute(configuration(task))
    attention = next(e for e in packed[0].events if e.metadata.get('kernel') == 'gqa' and e.metadata['phase'] == 'forward')
    dense = next(e for e in packed[0].events if e.metadata.get('kernel') == 'dense' and e.metadata['phase'] == 'forward')
    padded_dense = next(e for e in padded[0].events if e.metadata.get('kernel') == 'dense' and e.metadata['phase'] == 'forward')
    assert attention.metadata['workload_shape']['dense_causal_pairs_reference'] == 6 + 21
    assert dense.metadata['flops'] / padded_dense.metadata['flops'] == pytest.approx(9 / 16)
    assert packed[2]['training_tokens'] == 9
    assert padded[2]['training_tokens'] == 16
    assert packed[2]['training_task']['effective_tokens'] == padded[2]['training_task']['effective_tokens'] == 9


def test_rl_epochs_repeat_updates_not_rollouts_and_keep_microbatch_shapes():
    base = execute(configuration(rl(), {'workload': {'batch_size': 1, 'gradient_accumulation': 2}}))
    doubled = execute(configuration(rl(update_epochs=2), {'workload': {'batch_size': 1, 'gradient_accumulation': 2}}))
    first, second = (result[2]['training_task'] for result in (base, doubled))
    assert first['optimizer_steps'] == 2
    assert second['optimizer_steps'] == 4
    assert first['rollout_generated_tokens'] == second['rollout_generated_tokens'] == 10
    assert first['processed_tokens'] == 26
    assert second['processed_tokens'] == 52
    assert second['stage_times_s']['rollout'] == pytest.approx(first['stage_times_s']['rollout'])
    assert second['stage_times_s']['actor_update'] == pytest.approx(first['stage_times_s']['actor_update'] * 2)
    assert base[2]['training_accounting']['model_forward_backward_flops'] * 2 == pytest.approx(
        doubled[2]['training_accounting']['model_forward_backward_flops'])


def test_rl_reference_is_frozen_and_excluded_from_model_mfu_numerator():
    full = execute(configuration(rl()))
    no_reference = execute(configuration(rl(reference_policy=False)))
    assert full[2]['training_accounting']['model_forward_backward_flops'] == no_reference[2]['training_accounting']['model_forward_backward_flops']
    assert full[2]['training_accounting']['executed_npu_flops'] > no_reference[2]['training_accounting']['executed_npu_flops']
    reference = [a for a in full[0].initial_allocations if a.metadata.get('training_role') == 'reference']
    assert reference and not any(a.category in ('gradient', 'master_weight', 'optimizer_momentum', 'optimizer_variance') for a in reference)


def test_rule_reward_latency_and_experience_capacity_have_real_effects():
    base = execute(configuration(rl()))
    changed = execute(configuration(rl(rule_reward_latency_s=.1, reward_workers=2, experience_bytes_per_token=48)))
    assert changed[1]['duration_s'] - base[1]['duration_s'] == pytest.approx(.2)
    assert changed[1]['memory_peaks_bytes']['npu0.hbm'] > base[1]['memory_peaks_bytes']['npu0.hbm']
    assert changed[2]['training_task']['stage_times_s']['rule_reward'] == pytest.approx(.2)


def test_scalar_reward_and_value_heads_preserve_full_vocabulary_embeddings():
    spec = configuration(rl(algorithm='ppo', critic_model=model(True), reward='model', reward_model=model(True)))
    tensors = parameter_inventory(spec.workload.training.rl.critic_model.model_dump())
    assert next(t for t in tensors if t.name == 'word_embeddings.weight').elements == 32 * 16
    assert next(t for t in tensors if t.name == 'output.weight').elements == 16
    graph, _, metrics = execute(spec)
    assert metrics['training_task']['critic_optimizer_steps'] == 2
    assert {'actor', 'critic', 'reference', 'reward', 'experience'} <= set(metrics['training_accounting']['npu_flops_by_training_role'])


def test_dpo_cached_reference_removes_reference_execution_and_residency():
    task = {'kind': 'dpo', 'dpo': {'chosen_lengths': [5, 7], 'rejected_lengths': [6, 8]}}
    online = execute(configuration(task))
    task['dpo']['reference_log_probs'] = 'precomputed'
    cached = execute(configuration(task))
    assert online[2]['training_task']['preference_pairs'] == cached[2]['training_task']['preference_pairs'] == 2
    assert online[2]['training_tokens'] == cached[2]['training_tokens'] == 32
    assert 'reference' not in cached[2]['training_task']['stage_times_s']
    assert cached[2]['training_task']['supervised_tokens_s'] is None
    assert cached[1]['duration_s'] < online[1]['duration_s']
    assert online[2]['training_accounting']['model_forward_backward_flops'] == cached[2]['training_accounting']['model_forward_backward_flops']


def test_tp_dp_and_fsdp_sft_costs_and_rl_tp_work_with_scalar_head():
    execute(configuration({'kind': 'sft', 'sft': {'supervised_tokens': [2, 3]}}, {
        'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'dp': 2, 'fsdp': 2}}))
    graph, _, metrics = execute(configuration(rl(algorithm='ppo', critic_model=model(True)), {
        'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'dp': 2}}))
    assert metrics['training_task']['rollout_generated_tokens'] == 20
    assert metrics['training_task']['processed_tokens'] == 52
    assert any('value_sum' in event.id for event in graph.events)


@pytest.mark.parametrize('task,patch', [
    ({'kind': 'sft', 'sft': {'supervised_tokens': [9, 1]}}, {}),
    ({'kind': 'sft', 'sft': {'supervised_tokens': [0, 0]}}, {}),
    ({'kind': 'pretrain', 'sft': {'supervised_tokens': [1, 1]}}, {}),
    ({'kind': 'pretrain', 'data': {'sequence_lengths': [0, 8]}}, {}),
    (rl(samples_per_prompt=1), {}),
    (rl(algorithm='ppo'), {}),
    (rl(reward='model'), {}),
    (rl(critic_model=model(True)), {}),
    (rl(), {'workload': {'sequence_length': 7}}),
    (rl(), {'workload': {'gradient_accumulation': 3}}),
    (rl(), {'hardware': {'npu_count': 2, 'su_size': 2}, 'execution': {'dp': 2, 'fsdp': 2}}),
    ({'kind': 'pretrain'}, {'workload': {'scenario': 'inference'}}),
])
def test_invalid_or_unsupported_task_combinations_fail_validation(task, patch):
    with pytest.raises(ValueError):
        configuration(task, patch)


def test_feature_input_roundtrips_and_budget_projection_uses_effective_tokens():
    data = configuration().model_dump()
    data['workload'].pop('training')
    data['features'] = {'training_task': {'kind': 'pretrain', 'data': {'sequence_lengths': [3, 6]},
                                        'pretrain': {'dataset_tokens': 100, 'epochs': 2}}}
    config = Config.model_validate(data)
    result = run(config)
    assert Config.model_validate(config.model_dump()).model_dump() == config.model_dump()
    assert result['metrics']['training_task']['budget_projection']['optimizer_steps'] == 23
    assert next(f for f in result['model_metadata']['feature_plan']['features'] if f['name'] == 'training_task')['selected']


def test_operator_trace_keeps_task_role_identity_and_mfu_conservation():
    config = configuration(rl())
    graph, result, metrics = execute(config)
    event = next(e for e in graph.events if e.metadata.get('kernel') == 'dense' and e.metadata['training_stage'] == 'actor_update' and e.metadata['phase'] == 'backward')
    rank = next(key.split('.')[0] for key in event.work if key.endswith('.compute'))
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [{
        'id': 'task-backward-synthetic', 'source': 'synthetic conservation check',
        'hardware_name': config.hardware.name, 'dtype': config.execution.precision.compute, 'phase': 'backward',
        'flops': event.work[rank + '.compute'], 'memory_bytes': event.work[rank + '.hbm'],
        'metadata_match': {key: event.metadata[key] for key in ('training_task', 'training_role', 'training_stage', 'workload_shape')},
        'operations': [{'kernel': 'dense.dgrad', 'flops': event.work[rank + '.compute'] / 2,
                        'memory_bytes': event.work[rank + '.hbm'] / 2, 'startup_s': .001},
                       {'kernel': 'dense.wgrad', 'flops': event.work[rank + '.compute'] / 2,
                        'memory_bytes': event.work[rank + '.hbm'] / 2}],
    }]}
    traced = execute(Config.model_validate(data))
    assert traced[0].metadata['operator_trace']['matched_events'] > 0
    assert traced[1]['duration_s'] > result['duration_s']
    assert traced[2]['training_accounting']['model_forward_backward_flops'] == metrics['training_accounting']['model_forward_backward_flops']
    assert all(e.metadata['training_role'] == 'actor' for e in traced[0].events if e.metadata.get('trace_id'))


def test_ep_moe_rl_and_pipeline_dpo_execute_complete_role_lifetimes():
    moe = {'architecture': {'ffn_by_layer': ['MoE'], 'moe': {
        'num_experts': 4, 'top_k': 2, 'intermediate_dim': 8, 'num_shared_experts': 1}}}
    graph, _, metrics = execute(configuration(rl(rollout_batch_size=3), {
        'model': moe, 'hardware': {'npu_count': 4, 'su_size': 2}, 'execution': {'tp': 2, 'ep': 2}}))
    assert metrics['training_task']['rollout_generated_tokens'] == 10
    assert any('advantages.group_stats' in e.id and e.work for e in graph.events)
    execute(configuration({'kind': 'dpo', 'dpo': {'chosen_lengths': [5, 7], 'rejected_lengths': [6, 8]}}, {
        'model': {'layers': 2, 'architecture': {'attention_by_layer': ['GQA', 'GQA'], 'ffn_by_layer': ['Dense', 'Dense']}},
        'hardware': {'npu_count': 2, 'su_size': 2}, 'execution': {'pp': 2}}))


def test_task_report_and_compact_web_preserve_numerators_and_stage_identity(tmp_path):
    from supernode_sim.report import write_report
    from supernode_sim.web_worker import compact_result
    import csv
    result = run(configuration(rl()))
    write_report(result, tmp_path)
    with (tmp_path / 'training_task_stages.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert sum(float(row['duration_s']) for row in rows) == pytest.approx(result['metrics']['duration_s'])
    compact = compact_result(result)
    for key in ('processed_tokens', 'supervised_tokens', 'rollout_generated_tokens', 'stage_times_s'):
        assert compact['metrics']['training_task'][key] == result['metrics']['training_task'][key]
    assert any(e['metadata'].get('training_stage') == 'rollout' for e in compact['trace']['events'])
    assert 'Actor' in (tmp_path / 'report.md').read_text(encoding='utf-8')


def test_logits_live_until_loss_and_sampling_consumers_complete():
    graph, result, _ = execute(configuration(rl(), {'execution': {'features': {'activation_checkpointing': True}}}))
    times = {event['id']: event for event in result['trace']}
    for event in graph.events:
        if event.metadata.get('kernel') not in ('rollout_sampling', 'policy_loss', 'reference_log_probs'):
            continue
        assert event.deps
        assert all(times[dep]['end_s'] <= times[event.id]['start_s'] + 1e-12 for dep in event.deps)
    # Each Output allocation is retained beyond its producer, including checkpointed updates.
    releases = {aid: event.id for event in graph.events for aid in event.releases}
    for event in graph.events:
        if event.metadata.get('kernel') == 'output':
            for allocation in event.allocs:
                if allocation.metadata.get('component') == 'Output':
                    assert releases[allocation.id] != event.id

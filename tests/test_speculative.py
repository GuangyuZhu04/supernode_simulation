"""Resource and output conservation for explicit draft/verify/commit rounds."""
import pytest

from supernode_sim.analysis import execution_metrics
from supernode_sim.components import parameter_inventory
from supernode_sim.config import Config
from supernode_sim.core import Event
from supernode_sim.engine import simulate
from supernode_sim.speculative import build_speculative, speculative_plan, _profile_event
from supernode_sim.structured import Builder, build_structured
from supernode_sim.topology import Topology
from tests.test_sparse_attention import sparse_model
from tests.test_structured import configuration


def spec(patch=None):
    data = configuration({'workload': {'scenario': 'inference', 'output_length': 8},
        'execution': {'features': {'speculative_decode': {
            'enabled': True, 'method': 'dspark', 'draft_tokens': 3,
            'accepted_lengths': [2, 4, 4],
            'draft': {'cost': {'duration_s': 0.001}, 'prefilled': True, 'resident_bytes_per_rank': 4096,
                      'workspace_bytes_per_rank': 256},
            'commit_cost': {'duration_s': 0.0002}, 'replay_cost': {'duration_s': 0.0003},
        }}}}).model_dump()
    from supernode_sim.config import deep_merge
    data = deep_merge(data, patch or {})
    data['model']['total_params'] = data['model']['active_params'] = None
    return Config.model_validate(data)


def compile_spec(value):
    data = value.model_dump()
    return build_speculative(data, Topology(data['hardware'], data['execution']['precision']))


def test_round_plan_preserves_output_count_and_eos_truncates_queries():
    value = spec()
    plan = speculative_plan(value.model_dump())
    assert [(r['query_tokens'], r['committed_tokens']) for r in plan] == [(8, 1), (4, 2), (4, 4), (1, 1)]
    assert [r['context_before'] for r in plan] == [0, 8, 10, 14]
    assert plan[-1]['draft_tokens'] == 0
    assert plan[-1]['eos_truncated']
    assert sum(r['committed_tokens'] for r in plan) == value.workload.output_length


def test_exhausted_acceptance_trace_does_not_repeat_or_shorten_workload():
    value = spec({'execution': {'features': {'speculative_decode': {'accepted_lengths': [1]}}}})
    with pytest.raises(ValueError, match='trace ended'):
        compile_spec(value)


def test_draft_verify_acceptance_and_replay_are_scheduled_before_commit():
    value = spec()
    graph = compile_spec(value)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    trace = {row['id']: row for row in result['trace']}
    for item in graph.metadata['speculative_decode']['rounds']:
        prefix = f"infer.d0.s{item['round_index'] + 1}"
        verify = trace[prefix + '.l0.npu0.MLA']
        drafts = [row for name, row in trace.items() if name.startswith(prefix + '.draft')]
        if drafts:
            assert verify['start_s'] >= max(row['end_s'] for row in drafts)
        accept = trace[prefix + '.acceptance']
        assert accept['start_s'] >= verify['end_s']
        commit = trace[prefix + '.commit']
        if item['rejected_queries']:
            replay = trace[prefix + '.npu0.replay']
            assert replay['start_s'] >= trace[prefix + '.npu0.restore']['end_s']
            assert commit['start_s'] >= replay['end_s']
        else:
            assert prefix + '.npu0.replay' not in trace
    metrics = execution_metrics(value, graph, result)
    assert metrics['output_tokens'] == 16
    assert all(len(request['token_completion_s']) == 8 for request in metrics['requests'])
    assert any(interval == 0 for interval in metrics['requests'][0]['itl_s'])
    assert metrics['tpot_s'] == pytest.approx((metrics['duration_s'] - metrics['ttft_s']) / 7)
    # Six real draft steps, three commit profiles, and one rejection replay.
    assert metrics['duration_s'] - metrics['ttft_s'] >= 6 * 0.001 + 3 * 0.0002 + 0.0003


def test_target_verification_computes_all_queries_and_causal_attention_pairs():
    value = spec()
    graph = compile_spec(value)
    event = next(e for e in graph.events if e.id == 'infer.d0.s1.l0.npu0.MLA')
    model = value.model.model_dump()
    local_parameters = sum(t.local_elements() for t in parameter_inventory(model)
                           if t.layer == 0 and t.component == 'MLA' and not t.alias_of)
    batch, q, context = 2, 4, 8
    pairs = batch * (q * context + q * (q + 1) / 2)
    attention = model['architecture']['mla']
    width = attention['qk_nope_head_dim'] + attention['qk_rope_head_dim'] + attention['v_head_dim']
    expected = 2 * local_parameters * batch * q + 2 * model['num_heads'] * width * pairs
    assert event.work['npu0.compute'] == expected
    output = next(e for e in graph.events if e.id == 'infer.d0.s1.l1.npu0.Output')
    assert output.metadata['tokens'] == batch * q


def test_rejected_candidates_do_not_grow_final_cache_and_temporary_storage_is_released():
    value = spec()
    graph = compile_spec(value)
    result = simulate(graph)
    data = value.model_dump()
    data['execution']['features']['speculative_decode']['enabled'] = False
    ordinary = build_structured(data, Topology(data['hardware'], data['execution']['precision']))
    expected = simulate(ordinary)
    assert result['status'] == expected['status'] == 'feasible'
    assert result['memory_final_bytes']['npu0.hbm'] == expected['memory_final_bytes']['npu0.hbm'] + 4096
    assert any(a.category == 'speculative_candidates' for e in graph.events for a in e.allocs)
    assert any(a.category == 'speculative_snapshot' for e in graph.events for a in e.allocs)
    assert result['memory_peaks_bytes']['npu0.hbm'] > result['memory_final_bytes']['npu0.hbm']


def test_draft_resident_budget_can_make_otherwise_fitting_target_infeasible():
    value = spec()
    graph = compile_spec(value)
    resident = sum(a.size_bytes for a in graph.initial_allocations if a.pool == 'npu0.hbm')
    graph.memory_caps['npu0.hbm'] = resident - 1
    result = simulate(graph)
    assert result['status'] == 'infeasible'
    assert result['diagnostics'][0]['code'] == 'initial_memory_exceeded'


def test_draft_hbm_work_contends_with_other_device_traffic():
    value = spec({'hardware': {'npu': {'hbm_Bps': 100}}})
    data = value.model_dump()
    builder = Builder(data, Topology(data['hardware'], data['execution']['precision']))
    profile = {'flops': 0, 'memory_bytes': 100, 'duration_s': 1.0, 'startup_s': 0}
    _profile_event(builder, 'draft', 'npu0', profile, [], 'draft_forward')
    builder.g.events.append(Event('other_hbm', work={'npu0.hbm': 100}))
    result = simulate(builder.g)
    assert result['status'] == 'feasible'
    assert result['duration_s'] == pytest.approx(2.0)


def test_analytic_draft_flops_reserve_the_target_compute_stream():
    value = spec({'hardware': {'npu': {'flops': {'fp16': 1000}, 'efficiency': 0.5}}})
    data = value.model_dump()
    builder = Builder(data, Topology(data['hardware'], data['execution']['precision']))
    profile = {'flops': 100, 'memory_bytes': 0, 'duration_s': None, 'startup_s': 0.003}
    _profile_event(builder, 'draft', 'npu0', profile, [], 'draft_forward')
    builder.g.events.append(Event('target', work={'npu0.compute': 100}, slots=('npu0.compute_stream',)))
    result = simulate(builder.g)
    assert result['status'] == 'feasible'
    assert result['duration_s'] == pytest.approx(0.403)


def test_service_time_cannot_hide_physical_work_behind_startup():
    value = spec({'execution': {'features': {'speculative_decode': {'draft': {
        'cost': {'flops': 1e18, 'duration_s': 1e-9, 'startup_s': 100}}}}}})
    with pytest.raises(ValueError, match='physical'):
        compile_spec(value)


def test_per_round_profile_is_not_discounted_when_last_round_is_shortened():
    patch = {'workload': {'output_length': 7, 'output_lengths': [7, 7]},
             'execution': {'features': {'speculative_decode': {'accepted_lengths': [4, 4]}}}}
    step_graph = compile_spec(spec(patch))
    patch['execution']['features']['speculative_decode']['draft'] = {'cost_unit': 'per_round'}
    round_graph = compile_spec(spec(patch))
    step_events = [e for e in step_graph.events if e.metadata.get('operation') == 'draft_forward']
    round_events = [e for e in round_graph.events if e.metadata.get('operation') == 'draft_forward']
    assert len(step_events) == 4
    assert len(round_events) == 2
    assert all(e.metadata['service_time_s'] == 0.001 for e in round_events)


def test_batch_tp_ep_and_sequence_parallel_output_queries_are_preserved():
    value = spec({'hardware': {'npu_count': 4, 'su_size': 4},
        'execution': {'tp': 2, 'ep': 2, 'sequence_parallel': True},
        'workload': {'batch_size': 4, 'input_lengths': [8] * 4, 'output_lengths': [8] * 4}})
    graph = compile_spec(value)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert execution_metrics(value, graph, result)['output_tokens'] == 32
    projections = [e for e in graph.events if e.metadata.get('component') == 'Output'
                   and e.metadata.get('kernel') == 'output' and e.metadata.get('phase') == 'verify']
    assert any(e.metadata['tokens'] == 8 for e in projections)


@pytest.mark.parametrize('kind', ['CSA', 'HCA'])
def test_sparse_compressor_and_window_are_snapshotted_with_correct_dtypes(kind):
    data = spec().model_dump()
    data['model'] = sparse_model(kind)
    value = Config.model_validate(data)
    graph = compile_spec(value)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    from supernode_sim.sparse_attention import sparse_attention_state
    state = sparse_attention_state(value.model.model_dump(), 0, [8, 8])
    expected = state['recurrent_elements'] * 4 + state['window_elements'] * 2
    snapshot = next(a for e in graph.events for a in e.allocs if a.id == 'infer.d0.s1.npu0.snapshot')
    assert snapshot.size_bytes == expected
    data['execution']['features']['speculative_decode']['enabled'] = False
    ordinary = build_structured(data, Topology(data['hardware'], data['execution']['precision']))
    baseline = simulate(ordinary)
    assert result['memory_final_bytes']['npu0.hbm'] == baseline['memory_final_bytes']['npu0.hbm'] + 4096


def test_missing_replay_cost_rejects_mutable_state_rollback():
    value = spec({'execution': {'features': {'speculative_decode': {'replay_cost': None}}}})
    with pytest.raises(ValueError, match='requires replay_cost'):
        compile_spec(value)


def test_pure_kv_truncation_can_omit_replay_but_does_not_claim_profiled_commit():
    value = spec({'model': {'architecture': {'attention_by_layer': ['MLA', 'MLA']}},
        'execution': {'features': {'speculative_decode': {'commit_cost': None, 'replay_cost': None}}}})
    graph = compile_spec(value)
    assert graph.metadata['speculative_decode']['commit_cost'] == 'not_modeled'
    assert graph.metadata['speculative_decode']['replay_cost'] == 'not_modeled'
    assert graph.metadata['speculative_decode']['acceptance_source']['status'] == 'assumed'
    assert simulate(graph)['status'] == 'feasible'


def test_draft_prefill_adds_to_ttft_and_uses_workspace():
    seeded = spec()
    seeded_graph = compile_spec(seeded)
    seeded_metrics = execution_metrics(seeded, seeded_graph, simulate(seeded_graph))
    measured = spec({'execution': {'features': {'speculative_decode': {'draft': {
        'prefilled': False, 'prefill_cost': {'duration_s': 0.005}}}}}})
    graph = compile_spec(measured)
    result = simulate(graph)
    metrics = execution_metrics(measured, graph, result)
    assert metrics['ttft_s'] - seeded_metrics['ttft_s'] == pytest.approx(0.005)
    assert metrics['tpot_s'] == pytest.approx(seeded_metrics['tpot_s'])
    assert graph.metadata['speculative_decode']['draft_prefill'] == 'profile_driven'
    assert any(a.category == 'draft_workspace' and 'prefill' in a.id for e in graph.events for a in e.allocs)

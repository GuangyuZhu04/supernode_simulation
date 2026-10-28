"""Wire bytes, conversion work and phase coverage for asymmetric MoE transport."""
import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming


def configuration(structured=True, *, scenario='inference', precision=None, checkpoint=False, speculative=False):
    model = {'name': 'wire-test', 'layers': 1, 'hidden_dim': 16,
             'num_heads': 4, 'kv_heads': 2, 'head_dim': 4, 'intermediate_dim': 8}
    if structured:
        model['architecture'] = {'vocab_size': 32, 'attention_by_layer': ['GQA'], 'ffn_by_layer': ['MoE'],
                                 'moe': {'num_experts': 4, 'top_k': 2, 'intermediate_dim': 8}}
    else:
        model.update(total_params=4096, num_experts=4, top_k=2,
                     moe_layer_fraction=1, expert_parameter_fraction=0.8)
    features = {'activation_checkpointing': checkpoint}
    if speculative:
        features['speculative_decode'] = {'enabled': True, 'method': 'dspark', 'draft_tokens': 1,
            'accepted_lengths': [2], 'draft': {'cost': {'duration_s': 0.0001}, 'prefilled': True,
                                             'resident_bytes_per_rank': 4096}}
    return Config.model_validate({
        'model': model,
        'hardware': {'npu_count': 2, 'su_size': 2, 'npus_per_host': 2,
                     'npu': {'runtime_bytes': 0, 'workspace_bytes': 0}},
        'workload': {'scenario': scenario, 'batch_size': 2, 'sequence_length': 8,
                     'input_lengths': [8, 8] if speculative else [8, 4],
                     'output_lengths': [3, 3] if speculative else [2, 2]},
        'execution': {'ep': 2, 'features': features,
                      'precision': {'weight': 'bf16', 'compute': 'bf16', 'activation': 'bf16',
                                    'communication': 'bf16', **(precision or {})}},
    })


ASYMMETRIC = {'moe_dispatch': 'fp8', 'moe_combine': 'bf16'}


def network(graph, operation, phase=None):
    return [event for event in graph.events if event.metadata.get('type') == 'transfer'
            and event.metadata.get('operation', event.metadata.get('semantic')) == operation
            and (phase is None or event.metadata.get('phase') == phase)]


def assert_no_transient_leak(graph, result):
    assert result['status'] == 'feasible', result['diagnostics']
    persistent = sum(allocation.size_bytes for allocation in graph.initial_allocations)
    persistent += sum(allocation.size_bytes for event in graph.events for allocation in event.allocs
                      if allocation.category in ('kv', 'kv_cache', 'page_table'))
    assert sum(result['memory_final_bytes'].values()) == pytest.approx(persistent)


@pytest.mark.parametrize('structured', [False, True])
def test_dispatch_and_combine_use_independent_precision_and_ragged_reverse_flow(structured):
    graph = compile_graph(configuration(structured, precision=ASYMMETRIC))
    dispatch, combine = network(graph, 'dispatch', 'prefill'), network(graph, 'combine', 'prefill')
    assert sum(event.metadata['payload_bytes'] for event in dispatch) == 12 * 2 * 16 / 2
    assert sum(event.metadata['payload_bytes'] for event in combine) == 12 * 2 * 16 * 2 / 2
    backwards = {(event.metadata['dst'], event.metadata['src']): event.metadata['payload_bytes'] for event in combine}
    for event in dispatch:
        assert backwards[event.metadata['src'], event.metadata['dst']] == 2 * event.metadata['payload_bytes']
    for operation, dtype in [('dispatch', 'fp8'), ('combine', 'bf16')]:
        for event in network(graph, operation):
            assert event.metadata['wire_dtype'] == dtype
            assert event.metadata['buffer_dtype'] == dtype
            assert all(allocation.dtype == dtype and allocation.size_bytes == event.metadata['payload_bytes']
                       for allocation in event.allocs)
    packs = [event for event in graph.events if event.metadata.get('kernel') == 'communication_pack']
    assert packs and all('.dispatch.' in event.id for event in packs)
    assert {event.metadata['phase'] for event in packs} == {'prefill', 'decode'}
    assert_no_transient_leak(graph, simulate(graph))


@pytest.mark.parametrize('structured', [False, True])
def test_metadata_contributes_wire_and_buffer_bytes_without_becoming_tensor_elements(structured):
    base = compile_graph(configuration(structured, precision=ASYMMETRIC))
    graph = compile_graph(configuration(structured, precision={**ASYMMETRIC,
        'moe_dispatch_metadata_bytes_per_token': 8, 'moe_combine_metadata_bytes_per_token': 4}))
    for operation, expected in [('dispatch', 12 * 2 * (16 + 8) / 2),
                                ('combine', 12 * 2 * (16 * 2 + 4) / 2)]:
        events = network(graph, operation, 'prefill')
        assert sum(event.metadata['payload_bytes'] for event in events) == expected
        assert all(allocation.size_bytes == event.metadata['payload_bytes'] for event in events for allocation in event.allocs)
    reference = {event.id: event for event in base.events}
    conversions = [event for event in graph.events if event.metadata.get('kernel') in ('communication_pack', 'communication_unpack')]
    assert conversions
    for event in conversions:
        original = reference[event.id]
        # Metadata copying adds HBM work, never fake floating-point conversion elements.
        assert event.metadata['tensor_elements'] == original.metadata['tensor_elements']
        assert event.metadata['wire_metadata_bytes'] > 0
        for key in event.work:
            if '.compute' in key:
                assert event.work[key] == original.work[key]
            if key.endswith('.hbm'):
                assert event.work[key] - original.work[key] == pytest.approx(2 * event.metadata['wire_metadata_bytes'])
    assert_no_transient_leak(graph, simulate(graph))


@pytest.mark.parametrize('structured', [False, True])
def test_unspecified_moe_precision_matches_explicit_communication_default(structured):
    implicit = compile_graph(configuration(structured))
    explicit = compile_graph(configuration(structured, precision={'moe_dispatch': 'bf16', 'moe_combine': 'bf16'}))
    assert [(event.id, event.work, event.startup_s) for event in implicit.events] == [
        (event.id, event.work, event.startup_s) for event in explicit.events]
    assert simulate(implicit)['duration_s'] == simulate(explicit)['duration_s']


@pytest.mark.parametrize('structured', [False, True])
def test_training_and_checkpoint_recompute_keep_phase_specific_asymmetric_flow(structured):
    graph = compile_graph(configuration(structured, scenario='training', precision=ASYMMETRIC, checkpoint=True))
    for phase in ('forward', 'backward', 'recompute'):
        dispatch, combine = network(graph, 'dispatch', phase), network(graph, 'combine', phase)
        assert dispatch and combine
        assert sum(event.metadata['payload_bytes'] for event in combine) == 2 * sum(
            event.metadata['payload_bytes'] for event in dispatch)
        assert all(event.metadata['wire_dtype'] == 'fp8' for event in dispatch)
        assert all(event.metadata['wire_dtype'] == 'bf16' for event in combine)
    assert_no_transient_leak(graph, simulate(graph))


@pytest.mark.parametrize('speculative', [False, True])
def test_streaming_and_speculative_paths_preserve_wire_work(speculative):
    config = configuration(precision={**ASYMMETRIC, 'moe_dispatch_metadata_bytes_per_token': 8}, speculative=speculative)
    graph = compile_graph(config)
    full = simulate(graph)
    streamed = run_streaming(config)
    assert full['status'] == streamed['status'] == 'feasible'
    assert streamed['simulation']['duration_s'] == pytest.approx(full['duration_s'])
    for resource, work in full['resource_work'].items():
        if resource.startswith('link:'):
            assert streamed['simulation']['resource_work'].get(resource, 0) == pytest.approx(work)
    if speculative:
        assert network(graph, 'dispatch', 'verify')
        assert all(event.metadata['wire_dtype'] == 'fp8' for event in network(graph, 'dispatch', 'verify'))


@pytest.mark.parametrize('key,value', [('moe_dispatch', 'fp4'), ('moe_combine', 'fp4'),
    ('moe_dispatch_metadata_bytes_per_token', -1), ('moe_combine_metadata_bytes_per_token', True),
    ('moe_dispatch_metadata_bytes_per_token', 0.5)])
def test_moe_wire_schema_rejects_unmodeled_or_invalid_formats(key, value):
    with pytest.raises(ValueError):
        configuration(precision={key: value})


def test_communication_override_still_controls_other_collectives():
    data = configuration(precision=ASYMMETRIC).model_dump()
    data['hardware'].update(npu_count=4, su_size=4)
    data['execution']['tp'] = 2
    graph = compile_graph(Config.model_validate(data))
    non_moe = [event for event in graph.events if event.metadata.get('collective') == 'all_reduce']
    assert non_moe
    assert all(allocation.dtype == 'bf16' for event in non_moe for allocation in event.allocs)


@pytest.mark.parametrize('structured,scenario', [(False, 'inference'), (True, 'inference'),
                                                (False, 'training'), (True, 'training')])
@pytest.mark.parametrize('measured_duration', [0.001, 1e-30])
def test_moe_call_profile_encloses_pack_and_unpack_without_double_charging(structured, scenario, measured_duration):
    config = configuration(structured, scenario=scenario, precision=ASYMMETRIC,
                           checkpoint=scenario == 'training')
    baseline_graph = compile_graph(config)
    baseline = simulate(baseline_graph)
    phase = 'prefill' if scenario == 'inference' else 'forward'
    events = network(baseline_graph, 'dispatch', phase)
    ranks = events[0].metadata['communicator']
    sends = [sum(event.metadata['payload_bytes'] for event in events if event.metadata['src'] == rank) for rank in ranks]
    receives = [sum(event.metadata['payload_bytes'] for event in events if event.metadata['dst'] == rank) for rank in ranks]
    data = config.model_dump()
    data['hardware']['collective_profiles'] = [{
        'id': 'synthetic-call-boundary-test', 'collective': 'all_to_all_v', 'semantic': 'dispatch',
        'phase': phase, 'world_size': len(ranks), 'dtype': 'fp8', 'algorithm': 'pairwise',
        'timing_scope': 'activation_collective', 'total_payload_bytes': sum(sends),
        'send_payload_bytes': sends, 'recv_payload_bytes': receives, 'duration_s': measured_duration,
        'hardware_name': config.hardware.name, 'source': 'synthetic unit-test duration; no hardware claim',
    }]
    graph = compile_graph(Config.model_validate(data))
    result = simulate(graph)
    assert result['status'] == 'feasible'
    timers = [event for event in graph.events if event.metadata.get('type') == 'collective_timing']
    assert len(timers) == 1  # prefill/forward calibration must not match decode/backward/recompute.
    timer = timers[0]
    join = next(event for event in graph.events if event.id == timer.metadata['collective_calibration']['completion_event'])
    assert not any('.pack' in dep for dep in timer.deps)
    terminals = [dep for dep in join.deps if dep != timer.id]
    assert terminals and all(dep.endswith('.unpack') for dep in terminals)
    before = {row['id']: row for row in baseline['trace']}
    physical_envelope = max(before[dep]['end_s'] for dep in terminals) - max(
        (before[dep]['end_s'] for dep in timer.deps), default=0)
    after = {row['id']: row for row in result['trace']}
    elapsed = after[join.id]['end_s'] - after[timer.id]['start_s']
    assert elapsed == pytest.approx(max(physical_envelope, measured_duration), abs=1e-15)

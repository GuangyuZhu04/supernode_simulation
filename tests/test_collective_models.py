"""Environment-scoped payload interpolation on real simulator collectives."""
from copy import deepcopy

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.collective_calibration import collective_family_signature, summarize_collective_calibration
from supernode_sim.config import CALIBRATION_CONTEXT_FIELDS, Config, HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from supernode_sim.topology import Topology
from tests.test_moe_wire_precision import configuration
from tests.test_topology import hardware


RANKS = ['npu0', 'npu1']
PRECISION = {'compute': 'bf16', 'activation': 'bf16', 'communication': 'fp8'}


def context():
    return {**{key: 'synthetic-' + key for key in CALIBRATION_CONTEXT_FIELDS}, 'timer_boundary': 'whole_call'}


def params():
    return hardware(npu_count=2, su_size=2, name='curve-test', calibration_context=context())


def matrix(total):
    return [[total / 4, total / 2], [total / 2, total / 4]]


def model(hw, *, payload_matrix=None, samples=None, metadata_match=None, scope='transport', phase='decode', algorithm='direct'):
    metadata_match = metadata_match or {}
    return {'id': 'synthetic-curve', 'collective': 'all_to_all_v', 'semantic': 'dispatch',
            'phase': phase, 'world_size': 2, 'dtype': 'fp8', 'algorithm': algorithm,
            'timing_scope': scope, 'hardware_name': hw['name'],
            'family_signature': collective_family_signature(hw, RANKS, 'all_to_all_v', 0,
                payload_matrix or matrix(100), precision=PRECISION, metadata_match=metadata_match),
            'metadata_match': metadata_match,
            'samples': samples or [{'payload_bytes': 100, 'duration_s': 2}, {'payload_bytes': 300, 'duration_s': 6}],
            'source': 'Synthetic response-curve test; no hardware measurement claim'}


def build(hw, total=200, *, ranks=RANKS, payload_matrix=None, metadata=None):
    graph = Graph()
    topology = Topology(hw, PRECISION)
    topology.populate(graph)
    topology.collective(graph, 'dispatch', 'all_to_all_v', ranks, 0,
        payload_matrix=payload_matrix or matrix(total), algorithm='direct',
        metadata={'phase': 'decode', 'operation': 'dispatch', 'wire_dtype': 'fp8', **(metadata or {})})
    return graph


@pytest.mark.parametrize('payload,method,elapsed', [(50, 'out_of_range', .25), (100, 'exact_model', 2),
    (200, 'interpolated', 4), (300, 'exact_model', 6), (400, 'out_of_range', 2)])
def test_curve_is_exact_at_samples_linear_inside_and_never_extrapolates(payload, method, elapsed):
    hw = params()
    before = simulate(build(hw, payload))
    hw['collective_models'] = [model(hw)]
    graph = build(hw, payload)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert result['duration_s'] == pytest.approx(elapsed)
    assert result['resource_work'] == before['resource_work']
    assert result['memory_peaks_bytes'] == before['memory_peaks_bytes']
    summary = summarize_collective_calibration(graph, result)
    assert summary[method + '_collectives'] == 1
    record = next(event.metadata['collective_calibration'] for event in graph.events if 'collective_calibration' in event.metadata)
    assert record['calibration_context'] == hw['calibration_context']
    if method == 'out_of_range':
        assert summary['matched_collectives'] == 0
        assert record['status'] == 'out_of_range'
        assert not any(event.metadata.get('type') == 'collective_timing' for event in graph.events)
    else:
        assert summary['matched_collectives'] == summary['guarded_matched_collectives'] == 1
        assert summary['matched_model_ids'] == ['synthetic-curve']
        assert record['method'] == method
        assert record['service_duration_s'] == elapsed
        assert 'measured_duration_s' not in record


def test_exact_profile_takes_precedence_over_curve():
    hw = params()
    hw['collective_models'] = [model(hw)]
    hw['collective_profiles'] = [{'id': 'exact', 'collective': 'all_to_all_v', 'semantic': 'dispatch',
        'phase': 'decode', 'world_size': 2, 'dtype': 'fp8', 'algorithm': 'direct', 'timing_scope': 'transport',
        'hardware_name': hw['name'], 'source': 'Synthetic exact fixture', 'duration_s': 3,
        'total_payload_bytes': 200, 'send_payload_bytes': [100, 100], 'recv_payload_bytes': [100, 100]}]
    graph = build(hw)
    result = simulate(graph)
    assert result['duration_s'] == 3
    summary = summarize_collective_calibration(graph, result)
    assert summary['exact_profile_collectives'] == 1
    assert summary['interpolated_collectives'] == 0
    assert summary['matched_model_ids'] == []


@pytest.mark.parametrize('key', CALIBRATION_CONTEXT_FIELDS)
def test_any_software_transport_timer_or_device_context_change_falls_back(key):
    hw = params()
    hw['collective_models'] = [model(hw)]
    hw['calibration_context'][key] = 'device_only' if key == 'timer_boundary' else 'different-' + key
    graph = build(hw)
    result = simulate(graph)
    assert result['duration_s'] == 1
    assert summarize_collective_calibration(graph)['unmatched_reasons'] == {'family_mismatch': 1}


def test_additional_environment_keys_and_nonuniform_peer_shape_are_locked():
    hw = params()
    hw['collective_models'] = [model(hw)]
    for ranks, other_matrix in [(list(reversed(RANKS)), matrix(200)), (RANKS, [[50, 80], [120, 50]]),
                                (RANKS, [[0, 100], [100, 0]])]:
        report = summarize_collective_calibration(build(hw, ranks=ranks, payload_matrix=other_matrix))
        assert report['unmatched_reasons'] == {'family_mismatch': 1}
    hw['calibration_context']['firmware'] = 'another-firmware'
    assert summarize_collective_calibration(build(hw))['unmatched_reasons'] == {'family_mismatch': 1}


def test_normalized_family_tolerates_rescaling_but_preserves_tiny_routes():
    hw = params()
    first = [[0, .1], [.2, .3]]
    scaled = [[value * 37 for value in row] for row in first]
    fingerprint = lambda values: collective_family_signature(hw, RANKS, 'all_to_all_v', 0, values, precision=PRECISION)
    assert fingerprint(first) == fingerprint(scaled)
    tiny = deepcopy(first)
    tiny[0][0] = 1e-30
    assert fingerprint(first) != fingerprint(tiny)
    with_models = deepcopy(hw)
    with_models['collective_models'] = [model(hw)]
    assert fingerprint(first) == collective_family_signature(with_models, RANKS, 'all_to_all_v', 0, first, precision=PRECISION)


def test_metadata_subset_locks_activation_shape_and_distinguishes_boolean_from_integer():
    hw = params()
    expected = {'moe_hidden_width': 16, 'format': {'scaled': False, 'layout': [1, 2]}}
    hw['collective_models'] = [model(hw, metadata_match=expected)]
    matching = build(hw, metadata={**expected, 'extra_runtime_tag': 'allowed'})
    assert summarize_collective_calibration(matching)['interpolated_collectives'] == 1
    changed = deepcopy(expected)
    changed['format']['scaled'] = 0
    for tags in ({}, {'moe_hidden_width': 32}, changed):
        assert summarize_collective_calibration(build(hw, metadata=tags))['unmatched_reasons'] == {'metadata_mismatch': 1}
    assert model(hw, metadata_match={'moe_hidden_width': 16})['family_signature'] != model(hw, metadata_match={'moe_hidden_width': 32})['family_signature']


@pytest.mark.parametrize('change', [
    lambda h: h['collective_models'][0].update(samples=[{'payload_bytes': 100, 'duration_s': 1}]),
    lambda h: h['collective_models'][0].update(samples=[{'payload_bytes': 100, 'duration_s': 1}, {'payload_bytes': 100, 'duration_s': 2}]),
    lambda h: h['collective_models'][0]['samples'][1].update(payload_bytes=50),
    lambda h: h['collective_models'][0]['samples'][0].update(duration_s=float('nan')),
    lambda h: h['collective_models'][0]['samples'][0].update(payload_bytes=True),
    lambda h: h['collective_models'][0].update(extrapolation='linear'),
    lambda h: h['collective_models'][0].update(metadata_match={'value': float('inf')}),
    lambda h: h['calibration_context'].pop('timer_boundary'),
    lambda h: h['calibration_context'].update(cann_version='unknown'),
    lambda h: h['calibration_context'].update(hccl_version=8),
    lambda h: h['collective_models'].append(deepcopy(h['collective_models'][0])),
])
def test_invalid_curves_and_environment_fail_raw_and_canonical_inputs(change):
    hw = params()
    hw['collective_models'] = [model(hw)]
    change(hw)
    with pytest.raises(ValueError):
        HardwareSpec.model_validate(hw)
    with pytest.raises(ValueError):
        Topology(hw, PRECISION)


def test_overlapping_model_domains_are_ambiguous_and_short_timing_never_bypasses_physics():
    hw = params()
    first = model(hw, samples=[{'payload_bytes': 100, 'duration_s': .01}, {'payload_bytes': 300, 'duration_s': .02}])
    hw['collective_models'] = [first]
    graph = build(hw)
    result = simulate(graph)
    assert result['duration_s'] == 1
    assert summarize_collective_calibration(graph, result)['physical_limited_collectives'] == 1
    hw['collective_models'].append({**first, 'id': 'overlap'})
    with pytest.raises(ValueError, match='Ambiguous collective models'):
        build(hw)


def moe_config(*, structured=True, scenario='inference', scopes=('transport',), outer_out_of_range=False, layers=1):
    data = configuration(structured, scenario=scenario, precision={'moe_dispatch': 'fp8', 'moe_combine': 'bf16'},
                         checkpoint=scenario == 'training').model_dump()
    data['hardware']['calibration_context'] = context()
    data['hardware']['name'] = 'curve-test'
    if layers != 1:
        data['model']['layers'] = layers
        data['model']['architecture']['attention_by_layer'] = ['GQA'] * layers
        data['model']['architecture']['ffn_by_layer'] = ['MoE'] * layers
        data['model']['total_params'] = data['model']['active_params'] = None
    config = Config.model_validate(data)
    original = compile_graph(config)
    phase = 'decode' if scenario == 'inference' else 'forward'
    events = [event for event in original.events if event.metadata.get('type') == 'transfer'
              and event.metadata.get('operation', event.metadata.get('semantic')) == 'dispatch'
              and event.metadata.get('phase') == phase and event.metadata['layer'] == 0]
    ranks = events[0].metadata['communicator']
    # Structured/legacy inference pack the diagonal; legacy training only packs remote traffic.
    payload = sum(event.metadata['payload_bytes'] for event in events)
    cell = payload / 2
    diagonal = 0 if scenario == 'training' and not structured else cell
    values = [[diagonal, cell], [cell, diagonal]]
    shapes = {'moe_hidden_width': 16, 'moe_tensor_bytes_per_token': 16, 'moe_metadata_bytes_per_token': 0,
              'moe_top_k': 2, 'moe_num_experts': 4}
    entries = []
    for scope in scopes:
        samples = [{'payload_bytes': payload / 2, 'duration_s': .001}, {'payload_bytes': payload * 2, 'duration_s': .004}]
        if scope == 'activation_collective' and outer_out_of_range:
            samples = [{'payload_bytes': payload * 2, 'duration_s': .002}, {'payload_bytes': payload * 4, 'duration_s': .004}]
        entry = model(data['hardware'], payload_matrix=values, samples=samples, metadata_match=shapes,
                      scope=scope, phase=phase, algorithm='pairwise')
        entry['id'] = scope
        entry['family_signature'] = collective_family_signature(data['hardware'], ranks, 'all_to_all_v', 0, values,
            precision=data['execution']['precision'], metadata_match=shapes)
        entries.append(entry)
    data['hardware']['collective_models'] = entries
    return Config.model_validate(data)


@pytest.mark.parametrize('structured,scenario', [(False, 'inference'), (True, 'inference'), (False, 'training'), (True, 'training')])
@pytest.mark.parametrize('scopes,outer_out_of_range,selected', [(('transport',), False, 'transport'),
    (('transport', 'activation_collective'), False, 'activation_collective'),
    (('transport', 'activation_collective'), True, 'transport')])
def test_real_moe_selects_one_correct_boundary_and_counts_once(structured, scenario, scopes, outer_out_of_range, selected):
    config = moe_config(structured=structured, scenario=scenario, scopes=scopes, outer_out_of_range=outer_out_of_range)
    graph = compile_graph(config)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    timers = [event for event in graph.events if event.metadata.get('type') == 'collective_timing']
    assert len(timers) == 1
    record = timers[0].metadata['collective_calibration']
    assert record['selected_scope'] == selected
    assert record['scope_selection']['selected_scope'] == selected
    assert record['method'] == 'interpolated'
    assert any('.pack' in dep for dep in timers[0].deps) == (selected == 'transport')
    if outer_out_of_range:
        assert record['scope_selection']['activation_collective_status'] == 'out_of_range'
    assert summarize_collective_calibration(graph, result)['matched_collectives'] == 1


def test_streaming_and_macro_replay_preserve_model_statistics():
    config = moe_config(layers=3)
    ordinary = run(config)
    cache = {}
    streamed = run_streaming(config, cache=cache)
    replay = run_streaming(config, cache=cache)
    summary = ordinary['model_metadata']['collective_calibration']
    assert summary['interpolated_collectives'] == 3
    assert summary['exact_profile_collectives'] == 0
    assert streamed['model_metadata']['collective_calibration'] == summary
    assert replay['model_metadata']['collective_calibration'] == summary
    assert streamed['simulation']['duration_s'] == pytest.approx(ordinary['simulation']['duration_s'])
    assert any(chunk['component_macro_hit'] for chunk in streamed['chunks'])
    assert any(chunk['timing_cache_hit'] for chunk in replay['chunks'])


@pytest.mark.parametrize('kind,factor', [('all_gather', 1), ('reduce_scatter', 1), ('all_reduce', 2)])
def test_ring_curves_use_total_traffic_across_ranks_and_rounds(kind, factor):
    hw = params()
    size = 400 / factor
    curve = model(hw, samples=[{'payload_bytes': 200, 'duration_s': 2}, {'payload_bytes': 600, 'duration_s': 6}])
    curve.update(collective=kind, semantic=kind, algorithm='ring',
        family_signature=collective_family_signature(hw, RANKS, kind, size, precision=PRECISION))
    hw['collective_models'] = [curve]
    topo, graph = Topology(hw, PRECISION), Graph()
    topo.populate(graph)
    topo.collective(graph, 'ring', kind, RANKS, size,
        metadata={'phase': 'decode', 'operation': kind, 'wire_dtype': 'fp8'})
    result = simulate(graph)
    assert result['duration_s'] == 4
    assert sum(event.metadata.get('payload_bytes', 0) for event in graph.events) == 400
    assert summarize_collective_calibration(graph, result)['interpolated_collectives'] == 1


@pytest.mark.parametrize('boundary,includes_host', [('device_only', False), ('whole_call', True)])
def test_source_timer_boundary_is_not_mislabeled_as_simulation_or_client_boundary(boundary, includes_host):
    hw = params()
    hw['calibration_context']['timer_boundary'] = boundary
    hw['collective_models'] = [model(hw)]
    graph = build(hw)
    record = next(event.metadata['collective_calibration'] for event in graph.events if event.metadata.get('type') == 'collective_timing')
    assert record['source_timer_boundary'] == boundary
    assert record['source_includes_host_startup'] is includes_host
    assert record['simulated_timer_boundary'] == 'dependency_ready_to_collective_completion'
    assert 'includes_startup' not in record


@pytest.mark.parametrize('field', ['moe_hidden_width', 'moe_tensor_bytes_per_token', 'moe_metadata_bytes_per_token',
                                 'moe_top_k', 'moe_num_experts'])
def test_activation_moe_curves_require_known_shape_and_expert_layout(field):
    data = moe_config(scopes=('activation_collective',)).model_dump()
    del data['hardware']['collective_models'][0]['metadata_match'][field]
    with pytest.raises(ValueError, match='five shape/format'):
        HardwareSpec.model_validate(data['hardware'])
    with pytest.raises(ValueError, match='five shape/format'):
        Topology(data['hardware'], data['execution']['precision'])


@pytest.mark.parametrize('patch', [{'moe_top_k': 5}, {'moe_tensor_bytes_per_token': 32},
                                 {'moe_hidden_width': True}, {'moe_metadata_bytes_per_token': -1}])
def test_activation_moe_shape_must_be_physically_consistent(patch):
    data = moe_config(scopes=('activation_collective',)).model_dump()
    data['hardware']['collective_models'][0]['metadata_match'].update(patch)
    with pytest.raises(ValueError):
        Topology(data['hardware'], data['execution']['precision'])

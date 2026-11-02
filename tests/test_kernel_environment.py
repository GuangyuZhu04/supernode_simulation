"""Kernel service records remain scoped when Ascend software/hardware changes."""
from copy import deepcopy

import pytest

from supernode_sim.calibration import (_same_declaration, apply_kernel_profiles, calibration_summary,
                                      merge_calibration_summaries, model_shape_signature)
from supernode_sim.config import KernelProfile
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate


def context():
    return dict(device_sku='synthetic-Ascend-A3-like', cann_version='test-cann-1',
                hccl_version='test-hccl-1', driver_version='test-driver-1',
                runtime='synthetic-runtime', runtime_version='test-runtime-1',
                transport='HCCS', mode='graph', timer_boundary='device_only')


def fixture(*, guarded=True):
    graph = Graph()
    graph.resources = {'npu0.compute': Resource('npu0.compute', 60, 'FLOP/s', 'compute'),
                       'npu0.hbm': Resource('npu0.hbm', 1000, 'byte/s', 'bandwidth')}
    graph.memory_caps = {'npu0.hbm': 1e6}
    graph.events = [Event('test', work={'npu0.compute': 100, 'npu0.hbm': 50}, startup_s=.1,
                         metadata={'kernel': 'test_gemm', 'phase': 'decode',
                                   'component': 'MoE', 'tokens': 8, 'shape': [8, 16], 'flag': True})]
    profile = dict(flops=100, memory_bytes=50, duration_s=3, dtype='bf16', phase='decode')
    if guarded:
        profile.update(hardware_name='synthetic-chip', calibration_context=context(),
                       metadata_match={'tokens': 8, 'shape': [8, 16]},
                       source='synthetic test fixture; no measured hardware')
    config = {'hardware': {'name': 'synthetic-chip', 'calibration_context': context(),
                          'npu': {'flops': {'bf16': 100}}, 'kernel_profiles': {'test_gemm': [profile]}},
              'execution': {'precision': {'compute': 'bf16'}}}
    return graph, config


def test_matching_scope_changes_execution_and_reports_the_guard():
    graph, config = fixture()
    apply_kernel_profiles(graph, config)
    assert simulate(graph)['duration_s'] == pytest.approx(3.1)
    summary = graph.metadata['kernel_calibration']
    assert summary['matched_events'] == summary['guarded_matched_events'] == 1
    assert summary['unguarded_matched_events'] == 0
    assert graph.events[0].metadata['calibration']['guard_fields'] == ['hardware_name', 'calibration_context', 'metadata_match']


@pytest.mark.parametrize('field', list(context()))
def test_any_environment_change_falls_back_without_silent_legacy_bypass(field):
    graph, config = fixture()
    legacy = {key: value for key, value in config['hardware']['kernel_profiles']['test_gemm'][0].items()
              if key not in {'hardware_name', 'calibration_context', 'metadata_match', 'source'}}
    config['hardware']['kernel_profiles']['test_gemm'].append(legacy)
    config['hardware']['calibration_context'][field] += '-different'
    apply_kernel_profiles(graph, config)
    assert simulate(graph)['duration_s'] == pytest.approx(100/60 + .1)
    summary = graph.metadata['kernel_calibration']
    assert summary['matched_events'] == 0
    assert summary['scope_mismatched_events'] == 1
    assert summary['scope_mismatch_reasons'] == {'environment_mismatch': 1}


def test_same_work_but_different_shape_or_sku_does_not_match():
    for change, reason in [('shape', 'metadata_mismatch'), ('sku', 'hardware_mismatch')]:
        graph, config = fixture()
        if change == 'shape':
            graph.events[0].metadata['shape'] = [4, 32]
        else:
            config['hardware']['name'] = 'other-chip'
        apply_kernel_profiles(graph, config)
        assert graph.metadata['kernel_calibration']['scope_mismatch_reasons'] == {reason: 1}


def test_missing_metadata_and_bool_integer_confusion_are_rejected():
    graph, config = fixture()
    config['hardware']['kernel_profiles']['test_gemm'][0]['metadata_match'] = {'flag': 1}
    apply_kernel_profiles(graph, config)
    assert graph.metadata['kernel_calibration']['matched_events'] == 0
    graph, config = fixture()
    graph.events[0].metadata.pop('shape')
    apply_kernel_profiles(graph, config)
    assert graph.metadata['kernel_calibration']['matched_events'] == 0


@pytest.mark.parametrize('left,right,expected', [
    (8.0, 8, True), (0, -0.0, True), (8, 8.000000001, False),
    (True, 1, False), (False, 0.0, False), (True, True, True),
    ({'shape': [8.0, {'tokens': 16.0, 'enabled': True}], 'empty': None},
     {'empty': None, 'shape': [8, {'enabled': True, 'tokens': 16}]}, True),
    ({'shape': [8, 16]}, {'shape': [16, 8]}, False),
    ({'flag': True}, {'flag': 1.0}, False), ({'x': None}, {}, False),
    ([8, 16], (8, 16), False), ('8', 8, False),
    (2 ** 53 + 1, float(2 ** 53 + 1), False),
    (float('inf'), float('inf'), False), (float('nan'), float('nan'), False),
])
def test_json_declaration_equivalence_is_numeric_but_not_boolean_or_structural(left, right, expected):
    assert _same_declaration(left, right) is expected
    assert _same_declaration(right, left) is expected


def test_kernel_and_operator_trace_guards_survive_browser_integer_serialization():
    from tests.test_operator_trace import fixture as trace_fixture
    from supernode_sim.operator_trace import apply_kernel_traces

    graph, config = fixture()
    graph.events[0].metadata.update(tokens=8.0, shape=[8.0, 16.0],
                                    workload={'tokens': 8.0, 'shape': [8.0, 16.0], 'flag': True})
    profile = config['hardware']['kernel_profiles']['test_gemm'][0]
    profile['metadata_match']['workload'] = {'tokens': 8, 'shape': [8, 16], 'flag': True}
    apply_kernel_profiles(graph, config)
    assert graph.metadata['kernel_calibration']['matched_events'] == 1
    assert simulate(graph)['duration_s'] == pytest.approx(3.1)
    graph, config = trace_fixture()
    graph.events[1].metadata['shape'] = [4.0, 4.0]
    apply_kernel_traces(graph, config)
    assert graph.metadata['operator_trace']['matched_events'] == 1
    assert graph.metadata['operator_trace']['generated_operations'] == 2


def test_legacy_is_compatible_and_summary_merges_cached_scoped_counts():
    target = calibration_summary()
    for guarded in (True, False, True):
        graph, config = fixture(guarded=guarded)
        apply_kernel_profiles(graph, config)
        merge_calibration_summaries(target, graph.metadata['kernel_calibration'])
    assert target['matched_events'] == 3
    assert target['guarded_matched_events'] == 2
    assert target['unguarded_matched_events'] == 1
    assert target['by_phase']['decode']['guarded_matched_events'] == 2
    assert target['coverage_fraction'] == 1


def test_physical_floor_and_startup_contract_still_apply():
    graph, config = fixture()
    config['hardware']['kernel_profiles']['test_gemm'][0]['duration_s'] = .5
    with pytest.raises(ValueError, match='physical'):
        apply_kernel_profiles(graph, config)


def test_schema_rejects_partial_unknown_or_nonfinite_context_metadata():
    _, config = fixture()
    profile = config['hardware']['kernel_profiles']['test_gemm'][0]
    assert KernelProfile.model_validate(profile).calibration_context == context()
    for invalid in ({'cann_version': 'only-one-field'}, {**context(), 'mode': 'unknown'}):
        changed = deepcopy(profile)
        changed['calibration_context'] = invalid
        with pytest.raises(ValueError):
            KernelProfile.model_validate(changed)
    changed = deepcopy(profile)
    changed['metadata_match'] = {'nested': [float('nan')]}
    with pytest.raises(ValueError):
        KernelProfile.model_validate(changed)


def test_model_shape_guard_changes_with_structure_not_display_name():
    first = {'name': 'a', 'hidden_dim': 16, 'num_heads': 4, 'reported_total_params': 200}
    assert model_shape_signature(first) == model_shape_signature({**first, 'name': 'alias', 'reported_total_params': 201})
    assert model_shape_signature(first) != model_shape_signature({**first, 'num_heads': 8})


def test_structured_events_expose_real_shape_and_parallel_metadata():
    from supernode_sim.analysis import compile_graph
    from supernode_sim.config import load_config
    config = load_config('examples/features/synthetic-deepinfer-stages.yaml')
    graph = compile_graph(config)
    event = next(event for event in graph.events if event.metadata.get('kernel') == 'moe')
    assert event.metadata['model_shape_signature'] == model_shape_signature(config.model.model_dump())
    assert event.metadata['tensor_parallel_degree'] == config.execution.tp
    assert event.metadata['expert_parallel_degree'] == config.execution.ep

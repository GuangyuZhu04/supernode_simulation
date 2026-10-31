"""Independent MFU numerator, precision and execution-boundary checks."""
import pytest

from supernode_sim.analysis import compile_graph, execution_metrics, run
from supernode_sim.config import Config
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.training_accounting import training_accounting
from tests.test_structured import configuration
from tests.test_operator_trace import scoped_trace


def hand_fixture():
    config = configuration({'hardware': {'npu_count': 2, 'su_size': 2,
        'npu': {'flops': {'fp16': 100, 'fp32': 10}, 'efficiency': .25},
        'cpu': {'flops': 20, 'efficiency': .5}}, 'execution': {'dp': 2}})
    resources = {
        key: Resource(key, rate, 'FLOP/s', 'compute') for key, rate in
        [('npu0.compute', 25), ('npu0.compute.fp32', 2.5), ('cpu0.compute', 10), ('cpu1.compute', 10)]}
    resources['kernel_limit:forward'] = Resource('kernel_limit:forward', 1, 'second/second', 'kernel_timing')
    graph = Graph(resources=resources, events=[
        Event('forward', work={'npu0.compute': 100, 'kernel_limit:forward': 10000},
              metadata={'phase': 'forward', 'kernel': 'training_forward', 'flops': 100}),
        Event('recompute', work={'npu0.compute': 100},
              metadata={'phase': 'recompute', 'kernel': 'training_recompute', 'flops': 100}),
        Event('optimizer', work={'npu0.compute.fp32': 10},
              metadata={'phase': 'optimizer', 'kernel': 'adam', 'flops': 10}),
        Event('auxiliary', work={'npu0.compute': 40}, metadata={'phase': 'other', 'kernel': 'unknown'}),
        Event('host_optimizer', work={'cpu0.compute': 40}, metadata={'phase': 'optimizer', 'kernel': 'adam'}),
    ])
    return config, graph


def test_mixed_precision_uses_declared_peaks_configured_rank_count_and_separate_cpu():
    config, graph = hand_fixture()
    result = training_accounting(config, graph, {'status': 'feasible', 'duration_s': 2})
    assert result['model_mfu'] == pytest.approx(.25)  # (100/100)/(2 ranks*2s)
    assert result['executed_compute_utilization'] == pytest.approx(.85)  # (240/100+10/10)/4
    assert result['executed_npu_flops'] == 250
    assert result['model_recompute_flops'] == 100
    assert result['npu_count'] == 2
    assert result['active_npu_ranks'] == ['npu0']
    assert result['unclassified_phase_names'] == ['other']
    assert result['cpu']['device_count'] == 2
    assert result['cpu']['executed_flops'] == 40
    assert result['cpu']['executed_compute_utilization'] == pytest.approx(.5)
    assert result['npu_by_dtype']['fp32']['executed_flops'] == 10
    assert result['npu_flops_by_role'] == {'model_forward_backward': 100, 'model_recompute': 100,
                                         'optimizer': 10, 'auxiliary': 40}
    assert sum(result['npu_flops_by_role'].values()) == result['executed_npu_flops']


def test_unknown_compute_identity_is_explicit_and_ratios_are_unavailable():
    config, graph = hand_fixture()
    graph.resources['npu0.compute.custom'] = Resource('npu0.compute.custom', 1, 'FLOP/s', 'compute')
    graph.events.append(Event('unknown', work={'npu0.compute.custom': 50}))
    result = training_accounting(config, graph, {'status': 'feasible', 'duration_s': 2})
    assert result['status'] == 'incomplete_resource_identity'
    assert result['unknown_compute_resources'] == ['npu0.compute.custom']
    assert result['unknown_compute_flops'] == {'npu0.compute.custom': 50}
    assert result['model_mfu'] is result['executed_compute_utilization'] is None


@pytest.mark.parametrize('status,duration', [('infeasible', 2), ('error', 2), ('feasible', 0)])
def test_incomplete_or_zero_duration_execution_has_no_mfu(status, duration):
    config, graph = hand_fixture()
    result = training_accounting(config, graph, {'status': status, 'duration_s': duration})
    assert result['status'] == 'unavailable'
    assert result['model_mfu'] is result['executed_compute_utilization'] is None
    if status != 'feasible':
        assert execution_metrics(config, graph, {'status': status, 'duration_s': duration}) == {}


def test_checkpoint_changes_executed_work_without_inflating_model_numerator():
    baseline = run(configuration())['metrics']['training_accounting']
    checkpoint = run(configuration({'execution': {'features': {'activation_checkpointing': True}}}))['metrics']['training_accounting']
    assert checkpoint['model_forward_backward_flops'] == baseline['model_forward_backward_flops']
    assert baseline['model_recompute_flops'] == 0
    assert checkpoint['model_recompute_flops'] == pytest.approx(
        baseline['npu_flops_by_phase']['forward'])
    assert checkpoint['executed_npu_flops'] > baseline['executed_npu_flops']
    assert checkpoint['model_mfu'] < baseline['model_mfu']


def test_cpu_optimizer_offload_is_excluded_from_npu_model_and_execution_flops():
    base = run(configuration())['metrics']['training_accounting']
    offload_result = run(configuration({'execution': {'features': {'optimizer_offload': True}}}))
    offload = offload_result['metrics']['training_accounting']
    assert offload['model_forward_backward_flops'] == base['model_forward_backward_flops']
    assert offload['cpu']['optimizer_phase_flops'] > 0
    assert base['cpu']['executed_flops'] == 0
    assert offload['npu_flops_by_phase'].get('optimizer', 0) == 0
    actual_npu_work = sum(value for key, value in offload_result['simulation']['resource_work'].items()
                          if key.startswith('npu') and '.compute' in key)
    assert offload['executed_npu_flops'] == pytest.approx(actual_npu_work)


def test_trace_and_timing_profile_preserve_logical_model_work_and_exclude_pack():
    config = configuration()
    baseline = run(config)['metrics']['training_accounting']
    parent = next(event for event in compile_graph(config).events
                  if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'backward')
    data = config.model_dump()
    trace = scoped_trace(config, parent)
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    data['hardware']['kernel_profiles'] = {'dense.part0': [dict(
        flops=trace['flops'], memory_bytes=0, duration_s=.01, dtype='fp16', phase='backward',
        metadata_match={'trace_id': trace['id']}, source='Synthetic timing, not measured.') ]}
    actual = run(Config.model_validate(data))['metrics']['training_accounting']
    for key in ('executed_npu_flops', 'model_forward_backward_flops', 'model_recompute_flops'):
        assert actual[key] == baseline[key]
    assert actual['model_mfu'] < baseline['model_mfu']
    config, graph = hand_fixture()
    graph.events.append(Event('pack', work={'npu0.compute': 200},
        metadata={'kernel': 'communication_pack', 'component': 'Dense', 'phase': 'forward', 'flops': 200}))
    summary = training_accounting(config, graph, {'status': 'feasible', 'duration_s': 2})
    assert summary['model_forward_backward_flops'] == 100
    assert summary['npu_flops_by_phase']['forward'] == 300


def test_fusion_member_provenance_conserves_model_work_without_counting_barriers():
    config = configuration()
    parent = next(event for event in compile_graph(config).events
                  if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'forward')
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [scoped_trace(config, parent)]}
    before = run(Config.model_validate(data))['metrics']['training_accounting']
    data['execution']['features']['megakernel'] = {'enabled': True, 'groups': [
        dict(name='dense-trace', kernels=['dense.part0', 'dense.part1'], phases=['forward'],
             fused_startup_s=0, saved_hbm_bytes=0, workspace_bytes=0)]}
    result = run(Config.model_validate(data))
    assert result['model_metadata']['megakernel']['matched_groups'] == 1
    after = result['metrics']['training_accounting']
    assert after['model_forward_backward_flops'] == before['model_forward_backward_flops']
    assert after['executed_npu_flops'] == before['executed_npu_flops']


@pytest.mark.parametrize('timing_source', ['group_duration', 'kernel_profile'])
def test_explicit_fusion_timing_resources_preserve_mfu_and_executed_work(timing_source):
    config = configuration()
    baseline = run(config)['metrics']['training_accounting']
    parent = next(event for event in compile_graph(config).events
                  if event.metadata.get('kernel') == 'dense' and event.metadata.get('phase') == 'forward')
    data = config.model_dump()
    trace = scoped_trace(config, parent)
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    data['execution']['features']['megakernel'] = {'enabled': True, 'groups': [
        dict(name='dense-trace', kernels=['dense.part0', 'dense.part1'], phases=['forward'],
             fused_startup_s=0, saved_hbm_bytes=0, workspace_bytes=0, duration_s=.01)]}
    if timing_source == 'kernel_profile':
        data['hardware']['kernel_profiles'] = {'megakernel:dense-trace': [dict(
            flops=trace['flops'], memory_bytes=trace['memory_bytes'], duration_s=.02,
            dtype=config.execution.precision.compute, phase='forward',
            source='Synthetic fused timing test; not a measurement.')]}
    spec = Config.model_validate(data)
    graph = compile_graph(spec)
    fused = next(event for event in graph.events if event.metadata.get('kernel') == 'megakernel:dense-trace')
    suffix = 'megakernel' if timing_source == 'group_duration' else 'calibrated'
    assert f'npu0.compute.{suffix}.fp16' in fused.work
    result = run(spec)
    after = result['metrics']['training_accounting']
    assert after['status'] == 'complete'
    assert after['unknown_compute_resources'] == []
    assert after['unknown_compute_flops'] == {}
    assert after['model_forward_backward_flops'] == baseline['model_forward_backward_flops']
    assert after['executed_npu_flops'] == baseline['executed_npu_flops']
    physical_work = sum(amount for key, amount in result['simulation']['resource_work'].items()
                        if key.startswith('npu') and graph.resources[key].kind == 'compute'
                        and graph.resources[key].unit == 'FLOP/s')
    assert after['executed_npu_flops'] == pytest.approx(physical_work)
    denominator = spec.hardware.npu_count * result['simulation']['duration_s']
    assert after['model_mfu'] == pytest.approx(
        baseline['model_forward_backward_flops'] / spec.hardware.npu.flops['fp16'] / denominator)
    expected_executed = sum(row['executed_flops'] / spec.hardware.npu.flops[dtype]
                            for dtype, row in after['npu_by_dtype'].items()) / denominator
    assert after['executed_compute_utilization'] == pytest.approx(expected_executed)
    # The timer duration is not FLOP work, even though it shares the event.
    assert any(key.startswith('kernel_limit:') for key in fused.work)

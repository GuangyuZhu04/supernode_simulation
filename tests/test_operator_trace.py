"""Operator traces refine execution without inventing timing evidence."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.calibration import apply_kernel_profiles
from supernode_sim.config import Config, KernelTrace, KernelTraceOperation
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.operator_trace import apply_kernel_traces
from supernode_sim.streaming import run_streaming
from tests.test_structured import configuration


def fixture():
    graph = Graph(
        resources={'npu0.compute': Resource('npu0.compute', 100, 'FLOP/s', 'compute'),
                   'npu0.hbm': Resource('npu0.hbm', 100, 'byte/s', 'bandwidth')},
        memory_caps={'npu0.hbm': 1000},
        events=[Event('before'), Event('parent', ('before',), {'npu0.compute': 100, 'npu0.hbm': 100},
            startup_s=.2, slots=('npu0.compute_stream',),
            allocs=(Allocation('saved', 'npu0.hbm', 50, 'activation'),), releases=('saved',),
            metadata={'kernel': 'component', 'component': 'Dense', 'phase': 'backward',
                      'layer': 0, 'shape': [4, 4]}), Event('after', ('parent',))])
    trace = dict(id='synthetic-bwd', source='Synthetic serial trace, not device measurements.',
                 hardware_name='synthetic-chip', dtype='fp16', phase='backward',
                 flops=100, memory_bytes=100, metadata_match={'shape': [4, 4]},
                 operations=[dict(kernel='dense.dgrad', flops=100, memory_bytes=0,
                                  startup_s=.1, workspace_bytes=20),
                             dict(kernel='dense.wgrad', flops=0, memory_bytes=100,
                                  startup_s=.2, workspace_bytes=30)])
    config = {'hardware': {'name': 'synthetic-chip', 'npu': {'flops': {'fp16': 100}},
                          'kernel_profiles': {},
                          'kernel_traces': {'component': [KernelTrace.model_validate(trace).model_dump()]}},
              'execution': {'precision': {'compute': 'fp16'}}}
    return graph, config


def test_serial_trace_has_real_timing_and_preserves_dependencies_and_lifetimes():
    graph, config = fixture()
    baseline = simulate(deepcopy(graph))
    apply_kernel_traces(graph, config)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert baseline['duration_s'] == pytest.approx(1.2)
    assert result['duration_s'] == pytest.approx(2.5)
    assert result['resource_work'] == baseline['resource_work']
    assert result['memory_peaks_bytes']['npu0.hbm'] == 80
    assert result['memory_final_bytes']['npu0.hbm'] == 0
    events = {e.id: e for e in graph.events}
    assert events['parent.operator_trace.0'].deps == ('before',)
    assert events['parent.operator_trace.1'].deps == ('parent.operator_trace.0',)
    assert events['parent'].deps == ('parent.operator_trace.1',)
    assert events['after'].deps == ('parent',)
    assert events['parent'].releases == ('saved',)
    assert 'kernel' not in events['parent'].metadata
    assert events['parent.operator_trace.0'].metadata['shape'] == [4, 4]
    summary = graph.metadata['operator_trace']
    assert summary['matched_events'] == 1
    assert summary['generated_operations'] == 2
    assert summary['additional_startup_s'] == pytest.approx(.3)


def test_added_hbm_traffic_and_workspace_affect_resources_and_admission():
    graph, config = fixture()
    config['hardware']['kernel_traces']['component'][0]['operations'][1]['memory_bytes'] = 200
    apply_kernel_traces(graph, config)
    result = simulate(graph)
    assert result['duration_s'] == pytest.approx(3.5)
    assert result['resource_work']['npu0.hbm'] == 200
    assert graph.metadata['operator_trace']['additional_memory_bytes'] == 100
    graph.memory_caps['npu0.hbm'] = 79
    assert simulate(graph)['status'] == 'infeasible'


@pytest.mark.parametrize('field,value,reason', [
    ('phase', 'forward', 'work_phase_or_dtype_mismatch'),
    ('shape', [2, 8], 'metadata_mismatch'),
    ('shape', [True, 4], 'metadata_mismatch'),
])
def test_same_work_with_wrong_phase_or_shape_falls_back(field, value, reason):
    graph, config = fixture()
    graph.events[1].metadata[field] = value
    previous = asdict(graph)
    apply_kernel_traces(graph, config)
    assert [asdict(e) for e in graph.events] == previous['events']
    assert graph.metadata['operator_trace']['matched_events'] == 0
    assert graph.metadata['operator_trace']['skip_reasons'][reason] == 1


def test_hardware_dtype_and_runtime_context_are_exact_guards():
    for change, reason in [('hardware', 'hardware_mismatch'), ('dtype', 'work_phase_or_dtype_mismatch'),
                           ('context', 'environment_mismatch')]:
        graph, config = fixture()
        if change == 'hardware':
            config['hardware']['name'] = 'other-chip'
        elif change == 'dtype':
            config['execution']['precision']['compute'] = 'bf16'
        else:
            from tests.test_kernel_environment import context
            config['hardware']['kernel_traces']['component'][0]['calibration_context'] = context()
            config['hardware']['calibration_context'] = {**context(), 'runtime_version': 'different'}
        apply_kernel_traces(graph, config)
        assert graph.metadata['operator_trace']['matched_events'] == 0
        assert graph.metadata['operator_trace']['skip_reasons'][reason] == 1


def test_unsupported_resources_are_not_discarded_and_ambiguity_is_rejected():
    graph, config = fixture()
    graph.resources['npu0.selection'] = Resource('npu0.selection', 10, 'operation/s', 'selection')
    graph.events[1].work['npu0.selection'] = 20
    apply_kernel_traces(graph, config)
    assert graph.metadata['operator_trace']['skip_reasons'] == {'unsupported_resource': 1}
    assert simulate(graph)['resource_work']['npu0.selection'] == 20
    graph, config = fixture()
    duplicate = deepcopy(config['hardware']['kernel_traces']['component'][0])
    duplicate['id'] = 'second'
    config['hardware']['kernel_traces']['component'].append(duplicate)
    with pytest.raises(ValueError, match='Ambiguous kernel trace'):
        apply_kernel_traces(graph, config)


def test_suboperators_use_existing_profiles_and_physical_floors():
    graph, config = fixture()
    config['hardware']['kernel_profiles'] = {
        'component': [dict(flops=100, memory_bytes=100, duration_s=100)],
        'dense.dgrad': [dict(flops=100, memory_bytes=0, duration_s=3, phase='backward',
                            dtype='fp16', metadata_match={'trace_id': 'synthetic-bwd'})]}
    apply_kernel_traces(graph, config)
    apply_kernel_profiles(graph, config)
    assert simulate(graph)['duration_s'] == pytest.approx(4.5)
    assert graph.metadata['kernel_calibration']['matched_events'] == 1
    graph, config = fixture()
    config['hardware']['kernel_profiles'] = {'dense.dgrad': [dict(flops=100, memory_bytes=0, duration_s=.5)]}
    apply_kernel_traces(graph, config)
    with pytest.raises(ValueError, match='physical'):
        apply_kernel_profiles(graph, config)


def test_missing_traces_make_no_graph_changes():
    graph, config = fixture()
    config['hardware']['kernel_traces'] = {}
    previous = asdict(graph)
    apply_kernel_traces(graph, config)
    assert asdict(graph) == previous


def test_generated_event_and_workspace_ids_cannot_collide():
    for collision in ('event', 'allocation'):
        graph, config = fixture()
        if collision == 'event':
            graph.events.append(Event('parent.operator_trace.0'))
        else:
            graph.initial_allocations.append(Allocation('parent.operator_trace.0.workspace', 'npu0.hbm', 1))
        with pytest.raises(ValueError, match='duplicate'):
            apply_kernel_traces(graph, config)


def test_schema_rejects_inconsistent_work_shapes_and_missing_scope():
    _, config = fixture()
    trace = config['hardware']['kernel_traces']['component'][0]
    for change, message in [
        ({'flops': 99}, 'conserve'),
        ({'memory_bytes': 101}, 'undercount'),
        ({'metadata_match': {}}, 'metadata_match'),
        ({'metadata_match': {'shape': [float('nan')]}}, 'finite'),
        ({'calibration_context': {'runtime': 'partial'}}, 'missing required'),
        ({'source': ' '}, 'nonempty'),
    ]:
        with pytest.raises(ValueError, match=message):
            KernelTrace.model_validate({**trace, **change})
    op = KernelTraceOperation.model_validate(dict(kernel='gemm', flops=64, memory_bytes=100,
                                                matmul=dict(m=2, n=4, k=4)))
    assert op.matmul.batch == 1
    with pytest.raises(ValueError, match='matmul FLOPs'):
        KernelTraceOperation.model_validate({**op.model_dump(), 'flops': 63})
    with pytest.raises(ValueError):
        KernelTraceOperation.model_validate({**op.model_dump(), 'matmul': dict(m=True, n=4, k=4)})
    with pytest.raises(ValueError, match='matmul FLOPs'):
        KernelTraceOperation.model_validate({**op.model_dump(), 'matmul': dict(m=10 ** 500, n=4, k=4)})
    huge = dict(kernel='huge', flops=1e308, memory_bytes=1)
    with pytest.raises(ValueError, match='finite'):
        KernelTrace.model_validate({**trace, 'flops': 1e308, 'operations': [huge, huge]})


def scoped_trace(config, event, *, extra_bytes=0, workspace=128):
    compute = next(key for key in event.work if '.compute' in key)
    rank = compute.split('.')[0]
    flops, memory = event.work[compute], event.work[rank + '.hbm']
    return dict(id='synthetic-' + event.metadata['phase'], source='Synthetic operator decomposition; no measured time.',
                hardware_name=config.hardware.name, phase=event.metadata['phase'],
                dtype='fp32' if compute.endswith('.fp32') else config.execution.precision.compute,
                flops=flops, memory_bytes=memory,
                metadata_match={key: event.metadata[key] for key in
                                ('model_shape_signature', 'tokens', 'tensor_parallel_degree', 'expert_parallel_degree')},
                operations=[dict(kernel='dense.part0', flops=flops, memory_bytes=0,
                                 workspace_bytes=workspace, startup_s=1e-6),
                            dict(kernel='dense.part1', flops=0, memory_bytes=memory + extra_bytes)])


@pytest.mark.parametrize('scenario,phase', [('training', 'forward'), ('training', 'backward'),
                                         ('training', 'recompute'), ('inference', 'prefill'), ('inference', 'decode')])
def test_real_compiler_applies_distinct_training_and_inference_phases(scenario, phase):
    config = configuration({'workload': {'scenario': scenario},
                            'execution': {'features': {'activation_checkpointing': scenario == 'training'}}})
    baseline = compile_graph(config)
    event = next(e for e in baseline.events if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == phase)
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [scoped_trace(config, event, extra_bytes=123)]}
    graph = compile_graph(data)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert graph.metadata['operator_trace']['matched_events'] > 0
    assert graph.metadata['operator_trace']['by_phase'][phase]['matched_events'] > 0
    assert all(counts['matched_events'] == 0 for other, counts in graph.metadata['operator_trace']['by_phase'].items()
               if other != phase)
    assert graph.metadata['fidelity']['hardware_performance'] == 'uncalibrated'
    assert result['memory_final_bytes'] == simulate(baseline)['memory_final_bytes']


def test_streaming_and_shared_cache_match_full_graph_with_traces():
    config = configuration({'workload': {'scenario': 'inference', 'output_length': 5}})
    baseline = compile_graph(config)
    event = next(e for e in baseline.events if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'decode')
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [scoped_trace(config, event, extra_bytes=1234)]}
    config = Config.model_validate(data)
    graph = compile_graph(config)
    expected = simulate(graph)
    expected_metrics = execution_metrics(config, graph, expected)
    cache = {}
    for _ in range(2):
        actual = run_streaming(config, cache=cache)
        assert actual['status'] == 'feasible'
        assert actual['simulation']['duration_s'] == pytest.approx(expected['duration_s'])
        assert actual['metrics']['ttft_s'] == pytest.approx(expected_metrics['ttft_s'])
        assert actual['metrics']['tpot_s'] == pytest.approx(expected_metrics['tpot_s'])
        for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
            for resource in set(actual['simulation'][key]) | set(expected[key]):
                assert actual['simulation'][key].get(resource, 0) == pytest.approx(expected[key].get(resource, 0))
        assert actual['model_metadata']['operator_trace'] == graph.metadata['operator_trace']
        assert actual['model_metadata']['kernel_calibration'] == graph.metadata['kernel_calibration']
    assert any(chunk['timing_cache_hit'] for chunk in actual['chunks'])
    changed = config.model_dump()
    changed['hardware']['kernel_traces']['dense'][0]['operations'][0]['startup_s'] = .01
    changed_graph = compile_graph(changed)
    changed_full = simulate(changed_graph)
    changed_stream = run_streaming(changed, cache=cache)
    assert changed_stream['simulation']['duration_s'] == pytest.approx(changed_full['duration_s'])
    assert changed_full['duration_s'] > expected['duration_s']


def test_streaming_preserves_suboperator_calibration_resources_and_coverage():
    config = configuration({'workload': {'scenario': 'inference', 'output_length': 3}})
    parent = next(e for e in compile_graph(config).events
                  if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'decode')
    data = config.model_dump()
    trace = scoped_trace(config, parent)
    data['hardware']['kernel_traces'] = {'dense': [trace]}
    data['hardware']['kernel_profiles'] = {'dense.part0': [dict(
        flops=trace['flops'], memory_bytes=0, duration_s=.01, startup_s=2e-6,
        phase='decode', dtype=config.execution.precision.compute,
        metadata_match={'trace_id': trace['id']}, hardware_name=config.hardware.name,
        source='Synthetic test timing; not a measurement.')]}
    full_graph = compile_graph(data)
    full = simulate(full_graph)
    streamed = run_streaming(data)
    assert streamed['simulation']['duration_s'] == pytest.approx(full['duration_s'])
    assert streamed['model_metadata']['operator_trace'] == full_graph.metadata['operator_trace']
    assert streamed['model_metadata']['kernel_calibration'] == full_graph.metadata['kernel_calibration']
    assert streamed['model_metadata']['kernel_calibration']['matched_events'] == 2
    for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
        for resource in set(streamed['simulation'][key]) | set(full[key]):
            assert streamed['simulation'][key].get(resource, 0) == pytest.approx(full[key].get(resource, 0))


def test_trace_suboperators_can_use_explicit_megakernel_groups():
    config = configuration({'workload': {'scenario': 'inference'}})
    base = compile_graph(config)
    event = next(e for e in base.events if e.metadata.get('kernel') == 'dense' and e.metadata.get('phase') == 'prefill')
    data = config.model_dump()
    data['hardware']['kernel_traces'] = {'dense': [scoped_trace(config, event)]}
    data['execution']['features']['megakernel'] = {'enabled': True, 'groups': [
        dict(name='dense-trace', kernels=['dense.part0', 'dense.part1'], phases=['prefill'],
             fused_startup_s=1e-6, saved_hbm_bytes=0, workspace_bytes=0)]}
    graph = compile_graph(data)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert graph.metadata['operator_trace']['matched_events'] == 1
    assert graph.metadata['megakernel']['matched_groups'] == 1
    parent = next(e for e in graph.events if e.id == event.id)
    assert parent.metadata['operator_trace_completion'] is True
    assert next(e for e in graph.events if e.id == parent.deps[0]).metadata['kernel'] == 'megakernel:dense-trace'
    assert result['memory_final_bytes'] == simulate(base)['memory_final_bytes']

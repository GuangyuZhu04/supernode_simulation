"""Peak decompositions must describe simultaneous memory, not summed maxima."""
import csv
import json
from pathlib import Path
import shutil
import uuid
import gc
import tracemalloc

import pytest

from supernode_sim.core import Allocation, Event, Graph
from supernode_sim.engine import simulate
from supernode_sim.memory_breakdown import build_memory_breakdown
from supernode_sim.report import markdown_report, write_report


@pytest.fixture
def report_directory():
    # Avoid Windows mode-0700 temporary-directory DACLs, which exclude the
    # restricted runner token. Keep artifacts inside this test's workspace.
    base = Path(__file__).resolve().parent / '.memory_report_tmp'
    base.mkdir(exist_ok=True)
    path = base / ('report_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.parent != base.resolve() or not resolved.name.startswith('report_'):
            raise RuntimeError('Unsafe memory report temporary directory cleanup')
        shutil.rmtree(resolved)
        try:
            base.rmdir()
        except OSError:
            pass


def tensor(name, size, component=None, state='weights', pool='npu0.hbm', category='weights'):
    metadata = {'state': state, 'layer': 0, 'tensor_name': name}
    if component is not None:
        metadata['component'] = component
    return Allocation(name, pool, size, category, 'fp16', (int(size / 2),), metadata)


def log_row(sequence, time, action, allocation):
    return {'sequence': sequence, 'time_s': time, 'action': action,
            'allocation_id': allocation.id, 'pool': allocation.pool, 'bytes': allocation.size_bytes}


def nonsimultaneous_graph():
    w = tensor('attention.weight', 20, 'Attention')
    a = tensor('attention.activation', 80, 'Attention', 'activation', category='activations')
    b = tensor('ffn.activation', 60, 'FFN', 'activation', category='activations')
    graph = Graph(memory_caps={'npu0.hbm': 200}, initial_allocations=[w], events=[
        Event('attention', allocs=(a,), releases=(a.id,)),
        Event('ffn', deps=('attention',), allocs=(b,), releases=(b.id,))])
    log = [log_row(0, 0, 'allocate', w), log_row(1, 0, 'allocate', a),
           log_row(2, 1, 'release', a), log_row(3, 1, 'allocate', b), log_row(4, 2, 'release', b)]
    return graph, {'status': 'feasible', 'memory_log': log}


def test_actual_peak_is_a_concurrent_component_state_snapshot():
    graph, simulation = nonsimultaneous_graph()
    result = build_memory_breakdown(graph, simulation)
    pool = result['pools'][0]
    assert result['mode'] == 'simulated'
    assert pool['peak_bytes'] == 100
    assert pool['composition_at_peak'] == [
        {'component': 'Attention', 'state': 'activation', 'bytes': 80},
        {'component': 'Attention', 'state': 'weights', 'bytes': 20}]
    assert pool['component_independent_peak_bytes'] == {'Attention': 100, 'FFN': 60}
    assert sum(pool['component_independent_peak_bytes'].values()) == 160
    assert sum(row['bytes_at_pool_peak'] for row in result['component_state_rows']) == 100
    assert result['cluster']['peak_bytes'] == 100
    ffn = next(row for row in result['tensors'] if row['component'] == 'FFN')
    assert (ffn['allocated_at_s'], ffn['released_at_s']) == (1, 2)


def test_cluster_peak_does_not_sum_nonsimultaneous_device_peaks():
    a = tensor('device0.work', 100, 'Attention', 'activation', pool='npu0.hbm')
    b = tensor('device1.work', 120, 'FFN', 'activation', pool='npu1.hbm')
    graph = Graph(memory_caps={'npu0.hbm': 200, 'npu1.hbm': 200}, events=[
        Event('a', allocs=(a,), releases=(a.id,)), Event('b', deps=('a',), allocs=(b,), releases=(b.id,))])
    simulation = {'status': 'feasible', 'memory_log': [
        log_row(0, 0, 'allocate', a), log_row(1, 1, 'release', a),
        log_row(2, 1, 'allocate', b), log_row(3, 2, 'release', b)]}
    result = build_memory_breakdown(graph, simulation)
    assert [p['peak_bytes'] for p in result['pools']] == [100, 120]
    assert result['cluster']['peak_bytes'] == 120
    assert result['cluster']['composition_at_peak'] == [{'component': 'FFN', 'state': 'activation', 'bytes': 120}]


def test_same_time_order_uses_engine_sequence_instead_of_event_name():
    graph, simulation = nonsimultaneous_graph()
    simulation['memory_log'].reverse()
    result = build_memory_breakdown(graph, simulation)
    assert result['pools'][0]['peak_bytes'] == 100


def test_real_engine_log_reproduces_engine_pool_peaks():
    graph, _ = nonsimultaneous_graph()
    simulation = simulate(graph)
    assert simulation['status'] == 'feasible', simulation['diagnostics']
    result = build_memory_breakdown(graph, simulation)
    for pool in result['pools']:
        assert pool['peak_bytes'] == simulation['memory_peaks_bytes'][pool['pool']]
        assert sum(row['bytes'] for row in pool['composition_at_peak']) == pool['peak_bytes']


def test_estimate_is_resident_only_without_fabricating_optimistic_composition():
    graph, _ = nonsimultaneous_graph()
    estimate = {'optimistic_schedule_memory_peak_bytes': {'npu0.hbm': 170}}
    result = build_memory_breakdown(graph, estimate=estimate)
    assert result['mode'] == 'resident_only'
    assert result['pools'][0]['peak_bytes'] == 20
    assert result['pools'][0]['optimistic_peak_bytes'] == 170
    assert result['pools'][0]['composition_at_peak'] == [{'component': 'Attention', 'state': 'weights', 'bytes': 20}]
    assert len(result['tensors']) == 3
    assert all(not row['observed'] for row in result['tensors'])


def test_legacy_attribution_does_not_invent_model_or_optimizer_structure():
    graph = Graph(memory_caps={'npu0.hbm': 100}, initial_allocations=[
        Allocation('unknown.weights', 'npu0.hbm', 20, 'weights'),
        Allocation('legacy.optimizer', 'npu0.hbm', 40, 'optimizer_states'),
        Allocation('runtime', 'npu0.hbm', 10, 'runtime_reserve')])
    result = build_memory_breakdown(graph)
    ledger = {row['allocation_id']: row for row in result['tensors']}
    assert ledger['unknown.weights']['component'] == 'Unassigned'
    assert ledger['legacy.optimizer']['component'] == 'Unassigned'
    assert ledger['legacy.optimizer']['state'] == 'optimizer_combined'
    assert ledger['runtime']['component'] == 'System'
    assert not any(row['state'] in ('adam_momentum', 'adam_variance', 'muon_momentum') for row in ledger.values())


def test_explicit_muon_adam_recurrent_and_quantization_states_remain_separate():
    states = ['weights', 'gradients', 'master_weights', 'adam_momentum', 'adam_variance',
              'muon_momentum', 'activation', 'kv_cache', 'recurrent_state', 'quantization_metadata']
    graph = Graph(memory_caps={'npu0.hbm': 1000}, initial_allocations=[
        tensor('layer0.' + state, (i + 1) * 2, 'Explicit component', state)
        for i, state in enumerate(states)])
    result = build_memory_breakdown(graph)
    assert set(result['pools'][0]['state_bytes_at_peak']) == set(states)
    assert result['pools'][0]['peak_bytes'] == 110


def test_equivalent_device_grouping_scales_without_losing_pool_ledger():
    count = 8192
    graph = Graph(memory_caps={f'npu{i}.hbm': 100 for i in range(count)}, initial_allocations=[
        tensor(f'npu{i}.weight', 20, 'Attention', pool=f'npu{i}.hbm') for i in range(count)])
    result = build_memory_breakdown(graph)
    assert len(result['pools']) == count
    assert len(result['equivalent_pool_groups']) == 1
    group = result['equivalent_pool_groups'][0]
    assert group['pool_count'] == count
    assert group['peak_bytes_per_pool'] == 20
    assert len(group['pool_label']) < 100


def test_partial_execution_peak_is_marked_and_keeps_unobserved_planned_tensors():
    graph, simulation = nonsimultaneous_graph()
    simulation['status'] = 'infeasible'
    simulation['memory_log'] = simulation['memory_log'][:1]
    result = build_memory_breakdown(graph, simulation)
    assert result['mode'] == 'observed_partial'
    assert result['pools'][0]['peak_bytes'] == 20
    assert sum(row['observed'] for row in result['tensors']) == 1


@pytest.mark.parametrize('mutation,match', [
    ('unknown_id', 'unknown allocation'), ('duplicate_sequence', 'sequence numbers'),
    ('bad_bytes', 'disagrees'), ('release_first', 'before allocation'),
    ('missing_initial', 'omitted initial'),
])
def test_memory_log_contract_is_validated(mutation, match):
    graph, simulation = nonsimultaneous_graph()
    if mutation == 'unknown_id':
        simulation['memory_log'][0]['allocation_id'] = 'missing'
    elif mutation == 'duplicate_sequence':
        simulation['memory_log'][1]['sequence'] = 0
    elif mutation == 'bad_bytes':
        simulation['memory_log'][0]['bytes'] = 300
    elif mutation == 'release_first':
        simulation['memory_log'][0]['action'] = 'release'
    else:
        simulation['memory_log'] = simulation['memory_log'][1:]
    with pytest.raises(ValueError, match=match):
        build_memory_breakdown(graph, simulation)


def report_result(breakdown):
    return {'schema_version': '1.0', 'mode': 'estimate', 'status': 'estimated',
        'config': {'model': {'name': 'synthetic-explicit'}, 'workload': {'scenario': 'training'},
                   'execution': {'precision': {'weight': 'fp16'}}},
        'model_metadata': {}, 'metrics': {}, 'sources': {}, 'requirements': {'target_window_s': None},
        'estimate': {'duration_lower_bound_s': 0, 'resource_time_lower_bounds_s': {}, 'resident_memory_bytes': {},
                     'optimistic_schedule_memory_peak_bytes': {},
                     'communication': {'physical_bytes_by_domain': {}, 'payload_by_semantic_bytes': {}}},
        'memory_capacities_bytes': {'npu0.hbm': 200}, 'memory_breakdown': breakdown}


def test_chinese_report_and_four_csvs_preserve_concurrent_attribution(report_directory):
    graph, simulation = nonsimultaneous_graph()
    breakdown = build_memory_breakdown(graph, simulation)
    result = report_result(breakdown)
    report = markdown_report(result)
    assert '组件 × 状态' in report
    assert '各组件独立峰值（不可相加）' in report
    assert 'Adam 一阶动量' not in report  # No optimizer tensors were declared.
    write_report(result, report_directory)
    for name in ('component_memory.csv', 'tensors.csv', 'state_memory.csv', 'memory_peaks.csv'):
        assert (report_directory / name).exists()
    with (report_directory / 'component_memory.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert sum(float(row['bytes_at_pool_peak']) for row in rows) == 100
    with (report_directory / 'tensors.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert json.loads(rows[0]['shape']) == [10]
    assert all(row['attribution'] == 'explicit' for row in rows)


def test_logical_model_inventory_and_physical_member_links_export_separately(report_directory):
    graph, simulation = nonsimultaneous_graph()
    graph.metadata['component_inventory'] = {
        'mode': 'shape_derived', 'total_params': 10,
        'components': {'Attention': {'params': 10, 'active_params': 10, 'tensor_count': 1, 'layers': [0]}},
        'tensors': [{'name': 'layers.0.query.weight', 'component': 'Attention', 'layer': 0,
                     'shape': [2, 5], 'elements': 10, 'accounted_elements': 10, 'alias_of': None}]}
    graph.metadata['logical_component_state_bytes'] = {'Attention': {'weights': 20, 'muon_momentum': 40}}
    graph.initial_allocations[0].metadata['tensor_members'] = ['layers.0.query.weight']
    breakdown = build_memory_breakdown(graph, simulation)
    assert breakdown['logical_component_state_bytes']['Attention']['muon_momentum'] == 40
    write_report(report_result(breakdown), report_directory)
    with (report_directory / 'model_parameters.csv').open(encoding='utf-8-sig', newline='') as handle:
        logical = list(csv.DictReader(handle))
    assert len(logical) == 1
    assert json.loads(logical[0]['shape']) == [2, 5]
    with (report_directory / 'tensors.csv').open(encoding='utf-8-sig', newline='') as handle:
        physical = list(csv.DictReader(handle))
    assert json.loads(physical[0]['tensor_members']) == ['layers.0.query.weight']
    report = (report_directory / 'report.md').read_text(encoding='utf-8')
    assert '全模型逻辑状态容量' in report
    assert 'Muon 动量' in report


def test_lean_details_preserve_all_buffer_contributions_and_model_lifetimes():
    weight = tensor('weight', 20, 'Attention')
    initial_buffer = tensor('workspace', 4, 'System', 'buffers', category='workspace')
    activation = tensor('activation', 80, 'Attention', 'activation', category='activations')
    dma_buffer = tensor('collective_buffer', 200, 'System', 'buffers', category='communication_buffer')
    graph = Graph(memory_caps={'npu0.hbm': 1000}, initial_allocations=[weight, initial_buffer], events=[
        Event('model', allocs=(activation,), releases=(activation.id,)),
        Event('collective', deps=('model',), allocs=(dma_buffer,), releases=(dma_buffer.id,))])
    simulation = simulate(graph)
    full = build_memory_breakdown(graph, simulation)
    lean = build_memory_breakdown(graph, simulation, include_ephemeral_details=False)
    for key in ('pools', 'equivalent_pool_groups', 'component_state_rows', 'state_rows', 'cluster',
                'component_inventory', 'logical_component_state_bytes', 'mode'):
        assert lean[key] == full[key]
    assert lean['pools'][0]['peak_bytes'] == 224
    assert lean['pools'][0]['state_bytes_at_peak']['buffers'] == 204
    assert lean['tensor_detail_export']['total_rows'] == 4
    assert lean['tensor_detail_export']['retained_rows'] == 3
    assert lean['tensor_detail_export']['omitted_rows'] == 1
    assert lean['tensor_detail_export']['include_ephemeral_details'] is False
    assert {row['allocation_id'] for row in lean['tensors']} == {'weight', 'workspace', 'activation'}
    full_retained = [row for row in full['tensors'] if row['allocation_id'] != 'collective_buffer']
    assert lean['tensors'] == full_retained


def test_lean_replay_still_rejects_repeated_omitted_buffer_ids():
    buffer = tensor('omitted', 20, 'System', 'buffers', category='communication_buffer')
    graph = Graph(memory_caps={'npu0.hbm': 100}, events=[Event('e', allocs=(buffer,), releases=(buffer.id,))])
    simulation = {'status': 'feasible', 'memory_log': [log_row(0, 0, 'allocate', buffer),
        log_row(1, 1, 'release', buffer), log_row(2, 2, 'allocate', buffer)]}
    with pytest.raises(ValueError, match='repeated'):
        build_memory_breakdown(graph, simulation, include_ephemeral_details=False)


def test_lean_construction_has_bounded_incremental_memory_for_buffer_heavy_graph():
    graph = Graph(memory_caps={'npu0.hbm': 1000}, events=[
        Event(f'e{i}', allocs=(tensor(f'buffer{i}', 8, 'System', 'buffers', category='communication_buffer'),),
              releases=(f'buffer{i}',)) for i in range(2000)])
    tracemalloc.start()
    full = build_memory_breakdown(graph)
    _, full_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(full['tensors']) == 2000
    del full
    gc.collect()
    tracemalloc.start()
    lean = build_memory_breakdown(graph, include_ephemeral_details=False)
    _, lean_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert lean['tensor_detail_export']['omitted_rows'] == 2000
    assert lean['tensors'] == []
    assert lean_peak < full_peak / 2

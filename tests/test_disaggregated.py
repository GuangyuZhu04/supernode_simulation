"""Physical pool, state-transfer and timing invariants for fixed-batch PD."""
import pytest

from supernode_sim.analysis import execution_metrics
from supernode_sim.config import Config, deep_merge
from supernode_sim.disaggregated import build_disaggregated
from supernode_sim.engine import simulate
from supernode_sim.structured import build_structured
from supernode_sim.topology import Topology
from tests.test_sparse_attention import sparse_model
from tests.test_structured import configuration


def specification(patch=None):
    value = configuration({'workload': {'scenario': 'inference', 'output_length': 4},
        'hardware': {'npu_count': 2, 'su_size': 2,
                     'scale_up': {'bandwidth_Bps': 1024, 'injection_Bps': 1024,
                                  'efficiency': 1, 'latency_s': 0.01}},
        'execution': {'features': {'prefill_decode_disaggregation': {'enabled': True}}}}).model_dump()
    value = deep_merge(value, patch or {})
    value['model']['total_params'] = value['model']['active_params'] = None
    return Config.model_validate(value)


def compile_pd(value):
    data = value.model_dump()
    return build_disaggregated(data, Topology(data['hardware'], data['execution']['precision']))


def ordinary(value):
    data = value.model_dump()
    data['execution']['features']['prefill_decode_disaggregation']['enabled'] = False
    data['hardware']['npu_count'] = value.execution.tp * value.execution.ep
    data['hardware']['su_size'] = data['hardware']['npu_count']
    spec = Config.model_validate(data)
    data = spec.model_dump()
    return spec, build_structured(data, Topology(data['hardware'], data['execution']['precision']))


def transfers(graph):
    return [event for event in graph.events if event.metadata.get('operation') == 'pd_state_handoff']


def test_two_physical_pools_have_two_resident_model_copies_but_one_logical_inventory():
    value = specification()
    graph = compile_pd(value)
    _, baseline = ordinary(value)
    weights = lambda g: sum(a.size_bytes for a in g.initial_allocations if a.category == 'weight')
    assert weights(graph) == 2 * weights(baseline)
    assert graph.metadata['logical_memory_bytes'] == baseline.metadata['logical_memory_bytes']
    assert graph.metadata['prefill_decode_disaggregation']['model_copies'] == 2
    assert graph.metadata['prefill_decode_disaggregation']['physical_npu_count'] == 2
    assert all(a.category not in ('kv_cache', 'recurrent_state', 'convolution_state')
               for a in graph.initial_allocations)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    # Decode residency is an independent capacity constraint before any work.
    graph.memory_caps['npu1.hbm'] = sum(a.size_bytes for a in graph.initial_allocations
                                      if a.pool == 'npu1.hbm') - 1
    rejected = simulate(graph)
    assert rejected['diagnostics'][0]['code'] == 'initial_memory_exceeded'
    assert rejected['diagnostics'][0]['pool'] == 'npu1.hbm'


def test_handoff_uses_real_topology_and_transfers_all_persistent_state():
    graph = compile_pd(specification())
    [transfer] = transfers(graph)
    payload = transfer.metadata['payload_bytes']
    assert transfer.metadata['src'] == 'npu0'
    assert transfer.metadata['dst'] == 'npu1'
    assert transfer.metadata['path']
    assert transfer.metadata['physical_bytes_by_domain'] == {'SU': payload}
    assert transfer.work['npu0.hbm'] == transfer.work['npu1.hbm'] == payload
    assert sum(value for key, value in transfer.work.items() if key.startswith('link:')) == payload
    states = {row['state'] for row in transfer.metadata['state_breakdown']}
    assert {'kv_cache', 'page_table', 'recurrent_state', 'convolution_state'} <= states
    assert transfer.metadata['token_id_payload_bytes'] == 2 * 8
    assert payload == sum(row['bytes'] for row in transfer.metadata['state_breakdown']) + 16
    result = simulate(graph)
    trace = {row['id']: row for row in result['trace']}
    assert trace[transfer.id]['duration_s'] == pytest.approx(payload / 1024 + 0.01)
    handoff_end = trace[transfer.id]['end_s']
    assert all(row['start_s'] >= handoff_end for row in result['trace']
               if row['metadata'].get('execution_pool') == 'decode')


def test_destination_is_admitted_before_copy_and_source_released_only_after_copy():
    graph = compile_pd(specification())
    result = simulate(graph)
    [transfer] = transfers(graph)
    row = next(item for item in result['trace'] if item['id'] == transfer.id)
    allocations = {entry['allocation_id']: entry['time_s'] for entry in result['memory_log']
                   if entry['action'] == 'allocate'}
    releases = {entry['allocation_id']: entry['time_s'] for entry in result['memory_log']
                if entry['action'] == 'release'}
    for allocation in transfer.allocs:
        assert allocations[allocation.id] == row['start_s']
    for allocation_id in transfer.releases:
        assert releases[allocation_id] == row['end_s']
    first_decode_id = graph.metadata['prefill_decode_disaggregation']['first_decode_completion_event']
    first_decode_end = next(item['end_s'] for item in result['trace'] if item['id'] == first_decode_id)
    destination_token = next(a.id for a in transfer.allocs if a.id.endswith('decode_token_ids'))
    assert releases[destination_token] == first_decode_end


def test_decode_cache_capacity_is_checked_at_transfer_admission():
    graph = compile_pd(specification())
    [transfer] = transfers(graph)
    resident = sum(a.size_bytes for a in graph.initial_allocations if a.pool == 'npu1.hbm')
    received = sum(a.size_bytes for a in transfer.allocs if a.pool == 'npu1.hbm')
    graph.memory_caps['npu1.hbm'] = resident + received - 1
    result = simulate(graph)
    assert result['status'] == 'infeasible'
    assert result['diagnostics'][0]['code'] == 'memory_admission_deadlock'
    assert transfer.id not in {row['id'] for row in result['trace']}
    released = {row['allocation_id'] for row in result['memory_log'] if row['action'] == 'release'}
    assert not released.intersection(transfer.releases)


def test_cross_su_handoff_charges_scale_out_links_and_host_nics():
    value = specification({'hardware': {'su_size': 1,
        'scale_out': {'bandwidth_Bps': 2048, 'injection_Bps': 2048, 'efficiency': 1, 'latency_s': 0.02}}})
    graph = compile_pd(value)
    [transfer] = transfers(graph)
    payload = transfer.metadata['payload_bytes']
    assert transfer.metadata['domains'] == ['SO']
    assert transfer.metadata['physical_bytes_by_domain'] == {'SO': payload * len(transfer.metadata['path'])}
    assert transfer.work['cpu0.nic.tx'] == payload
    assert transfer.work['cpu1.nic.rx'] == payload
    assert transfer.work['npu0.SO.tx'] == payload
    assert transfer.work['npu1.SO.rx'] == payload
    assert simulate(graph)['status'] == 'feasible'


def test_slower_handoff_changes_first_decode_gap_and_tpot_but_not_ttft():
    value = specification()
    graph = compile_pd(value)
    first = execution_metrics(value, graph, simulate(graph))
    slower = specification({'hardware': {'scale_up': {'bandwidth_Bps': 512, 'injection_Bps': 512}}})
    slow_graph = compile_pd(slower)
    second = execution_metrics(slower, slow_graph, simulate(slow_graph))
    payload = graph.metadata['prefill_decode_disaggregation']['handoff_payload_bytes']
    extra = payload / 512 - payload / 1024
    assert second['ttft_s'] == pytest.approx(first['ttft_s'])
    assert second['tpot_s'] - first['tpot_s'] == pytest.approx(extra / 3)
    assert second['requests'][0]['itl_s'][0] - first['requests'][0]['itl_s'][0] == pytest.approx(extra)
    assert second['requests'][0]['itl_s'][1:] == pytest.approx(first['requests'][0]['itl_s'][1:])


def test_target_arithmetic_and_final_decode_state_match_unsplit_execution():
    value = specification()
    graph = compile_pd(value)
    _, baseline = ordinary(value)
    compute = lambda g: sum(amount for event in g.events for key, amount in event.work.items()
                            if key.startswith('npu') and '.compute' in key)
    assert compute(graph) == compute(baseline)
    actual, expected = simulate(graph), simulate(baseline)
    assert actual['status'] == expected['status'] == 'feasible'
    assert actual['memory_final_bytes']['npu0.hbm'] == sum(
        a.size_bytes for a in graph.initial_allocations if a.pool == 'npu0.hbm')
    assert actual['memory_final_bytes']['npu1.hbm'] == expected['memory_final_bytes']['npu0.hbm']
    metrics = execution_metrics(value, graph, actual)
    assert metrics['output_tokens'] == 8
    assert all(len(row['token_completion_s']) == 4 for row in metrics['requests'])


def test_single_token_requests_need_no_handoff_but_decode_weights_stay_reserved():
    value = specification({'workload': {'output_length': 1, 'output_lengths': [1, 1]}})
    graph = compile_pd(value)
    assert not transfers(graph)
    detail = graph.metadata['prefill_decode_disaggregation']
    assert detail['handoff_completion_event'] is None
    assert detail['first_decode_completion_event'] is None
    assert detail['handoff_payload_bytes'] == 0
    assert any(a.pool == 'npu1.hbm' and a.category == 'weight' for a in graph.initial_allocations)
    assert not any(event.metadata.get('execution_pool') == 'decode' for event in graph.events)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert execution_metrics(value, graph, result)['tpot_s'] is None


def test_ragged_batch_preserves_completion_lengths_and_only_sends_needed_token_ids():
    value = specification({'workload': {'input_lengths': [3, 9], 'output_lengths': [1, 4]}})
    graph = compile_pd(value)
    [transfer] = transfers(graph)
    assert transfer.metadata['decode_sequences'] == [1]
    assert transfer.metadata['token_id_payload_bytes'] == 8
    result = simulate(graph)
    assert result['status'] == 'feasible'
    metrics = execution_metrics(value, graph, result)
    assert sorted(len(row['token_completion_s']) for row in metrics['requests']) == [1, 4]
    assert metrics['output_tokens'] == 5


def test_tp_ep_sp_use_explicit_pool_ranks_and_skip_inactive_ep_state_handoff():
    value = specification({'hardware': {'npu_count': 8, 'su_size': 8},
        'execution': {'tp': 2, 'ep': 2, 'sequence_parallel': True,
                      'features': {'prefill_decode_disaggregation': {'prefill_ranks': [], 'decode_ranks': []}}},
        'workload': {'batch_size': 4, 'input_lengths': [3, 4, 5, 6], 'output_lengths': [1, 3, 1, 3]}})
    graph = compile_pd(value)
    assert {(e.metadata['src'], e.metadata['dst']) for e in transfers(graph)} == {('npu2', 'npu6'), ('npu3', 'npu7')}
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert execution_metrics(value, graph, result)['output_tokens'] == 8
    assert len({event.id for event in graph.events}) == len(graph.events)


def test_noncontiguous_pool_rank_order_is_honored():
    value = specification({'hardware': {'npu_count': 4, 'su_size': 4},
        'execution': {'tp': 2, 'features': {'prefill_decode_disaggregation': {
            'prefill_ranks': [1, 3], 'decode_ranks': [0, 2], 'token_id_bytes': 4}}}})
    graph = compile_pd(value)
    assert {(e.metadata['src'], e.metadata['dst']) for e in transfers(graph)} == {('npu1', 'npu0'), ('npu3', 'npu2')}
    assert all(e.metadata['token_id_payload_bytes'] == 8 for e in transfers(graph))
    assert simulate(graph)['status'] == 'feasible'


@pytest.mark.parametrize('kind', ['CSA', 'HCA'])
def test_compressed_attention_handoff_includes_compressor_window_pools_and_scales(kind):
    data = specification().model_dump()
    data['model'] = sparse_model(kind)
    data['execution']['precision']['kv'] = 'fp8'
    data['execution']['precision']['compute'] = 'fp8'
    value = Config.model_validate(data)
    graph = compile_pd(value)
    [transfer] = transfers(graph)
    states = {row['state']: row for row in transfer.metadata['state_breakdown']}
    assert {'window_elements', 'compressed_pool_elements', 'compression_state_elements', 'kv_scales'} <= states.keys()
    assert states['compression_state_elements']['dtype'] == 'fp32'
    assert states['window_elements']['dtype'] == 'fp8'
    if kind == 'CSA':
        assert {'indexer_pool_elements', 'indexer_compression_state_elements'} <= states.keys()
    result = simulate(graph)
    assert result['status'] == 'feasible'
    _, baseline = ordinary(value)
    expected = simulate(baseline)
    assert result['memory_final_bytes']['npu1.hbm'] == expected['memory_final_bytes']['npu0.hbm']

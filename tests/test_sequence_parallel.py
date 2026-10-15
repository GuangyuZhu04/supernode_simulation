"""Megatron sequence parallelism shares the TP ranks, with explicit padding."""
import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, deep_merge
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from supernode_sim.structured import Builder
from supernode_sim.topology import Topology
from tests.test_structured import configuration


def sp_configuration(tp=4, ep=2, patch=None):
    values = {
        'model': {'hidden_dim': 64, 'num_heads': 32, 'kv_heads': 32,
                  'architecture': {'vocab_size': 64, 'dense_intermediate_dim': 64,
                      'kda': {'num_heads': 32},
                      'moe': {'num_experts': 128, 'intermediate_dim': 32}}},
        'hardware': {'npu_count': tp * ep, 'su_size': min(64, tp * ep),
                     'npus_per_host': min(8, tp * ep)},
        'workload': {'scenario': 'inference', 'batch_size': 2,
                     'input_length': 64, 'output_length': 2},
        'execution': {'tp': tp, 'ep': ep, 'sequence_parallel': True},
    }
    return configuration(deep_merge(values, patch or {}))


@pytest.mark.parametrize('tp,ep,hosts_per_tp', [(16, 64, 2), (32, 32, 4)])
def test_sp_uses_tp_ranks_and_preserves_1024_card_hierarchy(tp, ep, hosts_per_tp):
    config = sp_configuration(tp, ep)
    data = config.model_dump()
    topology = Topology(data['hardware'], data['execution']['precision'])
    builder = Builder(data, topology)
    assert config.execution.sequence_parallel is True
    assert config.execution.tp * config.execution.ep == 1024
    assert len(builder.coords) == len(set(builder.ranks.values())) == 1024
    tp_groups, ep_groups = builder.groups(0, 'tp'), builder.groups(0, 'ep')
    assert len(tp_groups) == ep and all(len(group) == tp for group in tp_groups)
    assert len(ep_groups) == tp and all(len(group) == ep for group in ep_groups)
    for group in tp_groups:
        indices = [int(rank[3:]) for rank in group]
        assert len({index // 8 for index in indices}) == hosts_per_tp
        assert len({index // 64 for index in indices}) == 1
    for group in ep_groups:
        assert len({int(rank[3:]) // 64 for rank in group}) == 16
    assert all(topology.links[eid]['domain'] == 'SU'
               for eid in topology.path('npu0', f'npu{tp - 1}'))
    assert all(topology.links[eid]['domain'] == 'SO'
               for eid in topology.path('npu0', 'npu64'))


@pytest.mark.parametrize('patch', [
    {'workload': {'scenario': 'training'}},
    {'execution': {'tp': 1}, 'hardware': {'npu_count': 2, 'su_size': 2, 'npus_per_host': 2}},
    {'model': {'architecture': None, 'total_params': 1000, 'active_params': 1000}},
    {'execution': {'pp': 2}, 'hardware': {'npu_count': 16, 'su_size': 16}},
])
def test_sequence_parallel_rejects_unmodeled_cases(patch):
    with pytest.raises(ValueError, match='sequence_parallel'):
        sp_configuration(patch=patch)


def main_kernel(graph, event_id):
    return next(event for event in graph.events if event.id == event_id)


def traffic(graph, fragment):
    return sum(event.metadata.get('payload_bytes', 0) for event in graph.events if fragment in event.id)


def allocation_map(graph, categories):
    return {allocation.id: allocation.size_bytes for event in graph.events for allocation in event.allocs
            if allocation.category in categories}


def test_sp_shards_only_pointwise_work_and_keeps_valid_attention_state():
    patch = {'model': {'architecture': {'attnres_block_size': 1}}}
    config = sp_configuration(patch=patch)
    sp_graph = compile_graph(config)
    raw = config.model_dump()
    raw['execution']['sequence_parallel'] = False
    tp_graph = compile_graph(Config.model_validate(raw))
    for step, expected_tokens in ((0, 16), (1, 1)):
        for component in ('PreNorm', 'PostNorm', 'AttnRes'):
            eid = f'infer.d0.s{step}.l0.npu0.{component}'
            current, baseline = main_kernel(sp_graph, eid), main_kernel(tp_graph, eid)
            assert current.metadata['tokens'] == expected_tokens
            assert current.metadata['sequence_parallel_degree'] == 4
            ratio = 1 / 4 if step == 0 else 1
            assert current.metadata['flops'] == pytest.approx(baseline.metadata['flops'] * ratio)
            assert sum(a.size_bytes for a in current.allocs) == pytest.approx(
                sum(a.size_bytes for a in baseline.allocs) * ratio)
        for layer, component in ((0, 'MLA'), (0, 'Dense'), (1, 'KDA'), (1, 'MoE')):
            eid = f'infer.d0.s{step}.l{layer}.npu0.{component}'
            assert main_kernel(sp_graph, eid).work == main_kernel(tp_graph, eid).work
    caches = {'kv_cache', 'kv_scales', 'page_table', 'recurrent_state', 'convolution_state'}
    assert allocation_map(sp_graph, caches) == allocation_map(tp_graph, caches)
    residual_sp = allocation_map(sp_graph, {'attnres_residual'})
    residual_tp = allocation_map(tp_graph, {'attnres_residual'})
    for aid, size in residual_sp.items():
        assert size == residual_tp[aid] / (4 if '.s0.' in aid else 1)


def test_sp_padding_is_integer_and_output_gathers_only_selected_rows():
    config = sp_configuration(patch={'workload': {'input_length': 65}})
    graph = compile_graph(config)
    assert main_kernel(graph, 'infer.d0.s0.l0.npu0.PreNorm').metadata['tokens'] == 17
    assert main_kernel(graph, 'infer.d0.s1.l0.npu0.PreNorm').metadata['tokens'] == 1
    assert main_kernel(graph, 'infer.d0.s0.l0.npu0.MLA').metadata['tokens'] == 65
    assert main_kernel(graph, 'infer.d0.s1.l0.npu0.MLA').metadata['tokens'] == 1
    gathered = {a.id: a for event in graph.events for a in event.allocs if a.category == 'sp_gathered_input'}
    prefill = gathered['infer.d0.s0.l0.MLA.npu0.sp_gathered_input']
    decode = gathered['infer.d0.s1.l0.MLA.npu0.sp_gathered_input']
    output = gathered['infer.d0.s0.l1.Output.npu0.sp_gathered_input']
    assert (prefill.metadata['valid_source_tokens'], prefill.metadata['padded_tokens']) == (65, 68)
    assert (decode.metadata['valid_source_tokens'], decode.metadata['padded_tokens']) == (1, 4)
    assert (output.metadata['valid_source_tokens'], output.metadata['padded_tokens']) == (1, 4)
    assert prefill.size_bytes == 68 * 64 * 2
    assert decode.size_bytes == output.size_bytes == 4 * 64 * 2
    assert traffic(graph, 'infer.d0.s0.l0.MLA.e0.sp_all_gather') == 3 * 68 * 64 * 2
    assert traffic(graph, 'infer.d0.s1.l0.MLA.e0.sp_all_gather') == 3 * 4 * 64 * 2
    assert traffic(graph, 'infer.d0.s0.l1.Output.e0.sp_all_gather') == 3 * 4 * 64 * 2
    assert main_kernel(graph, 'infer.d0.s0.l1.npu0.Output').metadata['tokens'] == 1


def test_embedding_enters_sp_and_moe_gathers_before_dispatch():
    graph = compile_graph(sp_configuration())
    assert traffic(graph, 'infer.d0.s0.l0.WordEmbedding.e0.tp.reduce_scatter') == 3 * 64 * 64 * 2
    assert not any(event.metadata.get('collective') == 'all_reduce' for event in graph.events)
    events = {event.id: event for event in graph.events}

    def ancestors(eid):
        found, pending = set(), list(events[eid].deps)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(events[current].deps)
        return found

    dispatch = next(event for event in graph.events if '.l1.t0.dispatch.' in event.id and event.metadata.get('payload_bytes'))
    assert any('.l1.MoE.e0.sp_all_gather.' in aid for aid in ancestors(dispatch.id))
    release = events['infer.d0.s0.l1.MoE.sp_gather_release']
    assert any('.l1.MoE.e0.tp.reduce_scatter.' in aid for aid in ancestors(release.id))
    assert release.releases
    assert all(aid.endswith('.sp_gathered_input') for aid in release.releases)


def test_sp_collective_conversion_uses_local_gather_input_and_scatter_output():
    graph = compile_graph(sp_configuration(patch={'execution': {'precision': {'communication': 'fp32'}}}))
    gather_pack = main_kernel(graph, 'infer.d0.s0.l0.MLA.e0.sp_all_gather.npu0.wire.pack')
    gather_unpack = main_kernel(graph, 'infer.d0.s0.l0.MLA.e0.sp_all_gather.npu0.unpack')
    scatter_pack = main_kernel(graph, 'infer.d0.s0.l0.MLA.e0.tp.npu0.wire.pack')
    scatter_unpack = main_kernel(graph, 'infer.d0.s0.l0.MLA.e0.tp.npu0.unpack')
    assert sum(a.size_bytes for a in gather_pack.allocs) == 16 * 64 * 4
    assert gather_pack.work['npu0.compute.fp32'] == 2 * 16 * 64
    assert gather_unpack.work['npu0.compute.fp32'] == 2 * 64 * 64
    assert sum(a.size_bytes for a in scatter_pack.allocs) == 64 * 64 * 4
    assert scatter_unpack.work['npu0.compute.fp32'] == 2 * 16 * 64


def test_sp_streaming_and_full_graph_preserve_schedule_memory_and_traffic():
    config = sp_configuration(patch={
        'model': {'layers': 4, 'architecture': {
            'attention_by_layer': ['MLA', 'KDA', 'MLA', 'KDA'],
            'ffn_by_layer': ['Dense', 'MoE', 'MoE', 'MoE'], 'attnres_block_size': 2}},
        'workload': {'input_lengths': [65, 9], 'output_lengths': [3, 2]},
        'execution': {'precision': {'communication': 'fp32'}}})
    graph = compile_graph(config)
    expected = simulate(graph)
    actual = run_streaming(config)
    assert expected['status'] == actual['status'] == 'feasible'
    assert actual['metrics']['duration_s'] == pytest.approx(expected['duration_s'], rel=1e-11)
    for key in ('resource_work', 'memory_peaks_bytes', 'memory_final_bytes'):
        for resource in set(expected[key]) | set(actual['simulation'][key]):
            assert actual['simulation'][key].get(resource, 0) == pytest.approx(expected[key].get(resource, 0), rel=1e-10, abs=1e-7)
    assert sum(phase['communication']['transport_payload_bytes'] for phase in actual['phase_metrics'].values()) == pytest.approx(
        sum(event.metadata.get('payload_bytes', 0) for event in graph.events))
    assert any(chunk['component_macro_hit'] for chunk in actual['chunks'])


@pytest.mark.parametrize('tp', [16, 32])
def test_requested_sp_degrees_pad_single_token_decode_to_full_tp_wire(tp):
    graph = compile_graph(sp_configuration(tp=tp, ep=1, patch={
        'workload': {'batch_size': 1, 'input_length': 128},
        'model': {'architecture': {'attnres_block_size': 1}}}))
    assert graph.metadata['sequence_parallel']['degree'] == tp
    prefill = main_kernel(graph, 'infer.d0.s0.l0.npu0.PreNorm')
    decode = main_kernel(graph, 'infer.d0.s1.l0.npu0.PreNorm')
    assert prefill.metadata['tokens'] == 128 // tp
    assert decode.metadata['tokens'] == 1
    assert main_kernel(graph, 'infer.d0.s1.l0.npu0.MLA').metadata['tokens'] == 1
    gathers = {a.id: a for event in graph.events for a in event.allocs if a.category == 'sp_gathered_input'}
    allocation = gathers['infer.d0.s1.l0.MLA.npu0.sp_gathered_input']
    assert allocation.metadata['padded_tokens'] == tp
    assert allocation.metadata['valid_source_tokens'] == 1
    assert allocation.size_bytes == tp * 64 * 2
    assert traffic(graph, 'infer.d0.s1.l0.MLA.e0.sp_all_gather') == (tp - 1) * tp * 64 * 2
    assert traffic(graph, 'infer.d0.s0.l1.Output.e0.sp_all_gather') == (tp - 1) * tp * 64 * 2

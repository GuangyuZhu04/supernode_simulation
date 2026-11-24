"""Schedule, memory and fabric invariants independent of hardware calibration."""
from copy import deepcopy
from pathlib import Path

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config, load_config
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.search import size_su
from supernode_sim.training_attention import causal_tiles


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/training_overlap_qos.yaml'


def case(structured=True, **execution):
    data = load_config(EXAMPLE).model_dump()
    data['execution'].update(flash_attention_blocks=0)
    data['execution'].update(execution)
    data['workload']['gradient_accumulation'] = 1
    if not structured:
        data['model'] = {k: v for k, v in data['model'].items() if k != 'architecture'}
        data['model']['active_params'] = data['model']['total_params']
    return data


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_gather_window_and_gradient_lifetimes(structured, depth):
    data = case(structured, prefetch_layers=depth)
    graph = compile_graph(data)
    events = {event.id: event for event in graph.events}
    start = '.gather_allocate' if structured else '.gather.start'
    done = '.gather_done' if structured else '.gather.done'
    for phase, layers in [('forward', [0, 1, 2]), ('backward', [2, 1, 0])]:
        prefix = f'train.s0.p0.{phase}.m0.l'
        last = events[prefix + str(layers[-1]) + start]
        if depth:
            assert prefix + str(layers[1]) + done in last.deps
        if depth < 2:
            assert prefix + str(layers[1 - depth]) + '.done' in last.deps
        else:
            assert all(not dep.endswith(f'.l{layers[0]}.done') for dep in last.deps)
    sim = simulate(graph)
    assert sim['status'] == 'feasible', sim['diagnostics']
    for pool, amount in sim['memory_final_bytes'].items():
        assert amount == pytest.approx(sum(a.size_bytes for a in graph.initial_allocations if a.pool == pool))
    trace = {row['id']: row for row in sim['trace']}
    gradients = [row['end_s'] for eid, row in trace.items() if eid.endswith('.gradient.done')]
    updates = [row['start_s'] for row in trace.values() if row['metadata'].get('phase') == 'optimizer']
    assert gradients and min(updates) >= max(gradients) - 1e-12


@pytest.mark.parametrize('structured', [False, True])
def test_hierarchical_dp_work_and_placement(structured):
    data = case(structured, fsdp=1, dp_domain_size=2)
    result = run(data)
    assert result['status'] == 'feasible'
    rows = [row['metadata'] for row in result['simulation']['trace'] if row['metadata'].get('parallel_axis') == 'dp_replica']
    assert {m['collective'] for m in rows if m['dp_reduction_scope'] == 'within_domain'} == {'reduce_scatter', 'all_gather'}
    assert {m['collective'] for m in rows if m['dp_reduction_scope'] == 'between_domains'} == {'all_reduce'}
    assert all(len(m['communicator']) == 2 for m in rows)
    data['execution']['dp_domain_size'] = None
    flat = run(data)
    payload = lambda r: r['metrics']['training_communication']['by_class']['dp']['payload_bytes']
    assert payload(result) == pytest.approx(payload(flat))
    data['execution']['dp_domain_size'] = 3
    with pytest.raises(ValueError, match='dp_domain_size'):
        Config.model_validate(data)


def test_causal_flash_pipeline_conserves_work_and_overlaps_dma():
    data = case(flash_attention_blocks=3)
    data['workload'].update(training={'kind': 'pretrain', 'data': {'batching': 'packed', 'sequence_lengths': [7, 3]}},
                            batch_size=2, input_lengths=[], output_lengths=[])
    data['execution']['features']['activation_checkpointing'] = True
    tiles = causal_tiles([7, 3], 3)
    assert sum(row[0] for _, row in tiles) == 28 + 6
    result = run(data)
    data['execution']['flash_attention_blocks'] = 0
    base = run(data)
    for key in ('model_forward_backward_flops', 'model_recompute_flops'):
        assert result['metrics']['training_accounting'][key] == pytest.approx(base['metrics']['training_accounting'][key])
    trace = {row['id']: row for row in result['simulation']['trace']}
    for phase in ('forward', 'recompute', 'backward'):
        loads = [row for row in trace.values() if row['metadata'].get('kernel') == 'flash_tile_load'
                 and row['metadata']['phase'] == phase and row['id'].endswith('.q1.k0.load')]
        assert loads
        if phase != 'backward':  # backward also depends on the earlier dKV write
            assert any(row['start_s'] < trace[row['id'].replace('.q1.k0.load', '.q0.k0.compute')]['end_s']
                       for row in loads)
    assert result['metrics']['training_tokens'] == base['metrics']['training_tokens']


def test_effective_hbm_is_used_for_residents_transients_and_search():
    data = case()
    baseline = run(data)
    physical = data['hardware']['npu']['hbm_bytes']
    budget = baseline['metrics']['training_hbm']
    assert budget['usable_bytes_per_npu'] == physical * .9
    resident = max(v for k, v in baseline['estimate']['resident_memory_bytes'].items() if k.endswith('.hbm'))
    data['hardware']['npu']['hbm_bytes'] = resident / .95
    blocked = run(data)
    assert blocked['simulation']['diagnostics'][0]['code'] == 'initial_memory_exceeded'
    search = size_su(Config.model_validate(data), 'multi')['searches']['multi']
    assert all(r['result']['pruned'] == 'resident_memory_lower_bound' for r in search['candidates'])
    data['hardware']['npu']['hbm_bytes'] = resident / .9 + 1
    transient = run(data)
    assert transient['simulation']['diagnostics'][0]['code'] == 'memory_admission_deadlock'
    data['execution']['training_hbm_fraction'] = .85
    assert run(data, 'estimate')['memory_capacities_bytes']['npu0.hbm'] == pytest.approx(data['hardware']['npu']['hbm_bytes'] * .85)


def test_search_fixed_batch_pins_and_replays_best_placement():
    data = case(fsdp=1)
    data['targets'].update(candidate_dp_domain_sizes=[2], training_global_batch_size=8,
                           parallel_candidates=[{'pp': 2, 'dp': 4}, {'pp': 4, 'dp': 2}])
    search = size_su(Config.model_validate(data), 'multi')['searches']['multi']
    assert {r['placement'] for r in search['candidates']} == {'model_parallel_first', 'dp_domain_first'}
    assert {r['samples_per_accumulation'] for r in search['candidates']} == {8}
    fastest = search['highest_throughput']
    best = search['candidates'][fastest['candidate_index']]
    assert run(best['result']['config'])['metrics']['throughput_tokens_s'] == fastest['tokens_s']
    assert fastest['tokens_s'] == max(r['result']['metrics']['throughput_tokens_s'] for r in search['candidates'])
    pinned = list(reversed(range(8)))
    data['execution']['rank_order'] = pinned
    rows = size_su(Config.model_validate(data), 'multi')['searches']['multi']['candidates']
    assert all(r['parallelism']['rank_order'] == pinned for r in rows)


def test_qos_weighted_link_and_nonpreemptive_slot():
    fabric = 'shared:uplink'
    resource = Resource(fabric, 100, 'byte/s')
    graph = Graph(resources={fabric: resource}, events=[
        Event('a_bulk', work={fabric: 100}, metadata={'qos_weight': 1}),
        Event('b_latency', work={fabric: 100}, metadata={'qos_weight': 3})])
    end = lambda result: {row['id']: row['end_s'] for row in result['trace']}
    assert end(simulate(graph)) == pytest.approx({'a_bulk': 2, 'b_latency': 2})
    graph.metadata['training_qos'] = {'enabled': True}
    qos = simulate(graph)
    assert end(qos) == pytest.approx({'a_bulk': 2, 'b_latency': 4 / 3})
    assert qos['resource_work'][fabric] == pytest.approx(200)
    for event in graph.events:
        event.slots = ('npu0.collective.tx',)
    assert end(simulate(graph)) == pytest.approx({'a_bulk': 2, 'b_latency': 1})
    graph.events.append(Event('arrives', startup_s=.5))
    graph.events[1].deps = ('arrives',)
    assert end(simulate(graph)) == pytest.approx({'a_bulk': 1, 'b_latency': 2, 'arrives': .5})
    # With only HBM contention, traffic weights have no effect on service.
    for event in graph.events[:2]:
        event.deps, event.slots, event.work = (), (), {'npu0.hbm': 100}
    graph.resources['npu0.hbm'] = Resource('npu0.hbm', 100, 'byte/s')
    assert end(simulate(graph)) == pytest.approx({'a_bulk': 2, 'b_latency': 2, 'arrives': .5})


def test_qos_feature_preserves_payload_and_reports_overlapping_rank_time():
    data = case(overlap_gradient_sync=True)
    base = deepcopy(data)
    base['execution']['features']['training_qos']['enabled'] = False
    a, b = run(base), run(data)
    payloads = lambda r: {k: v['payload_bytes'] for k, v in r['metrics']['training_communication']['by_class'].items()}
    assert payloads(a) == payloads(b)
    assert set(payloads(b)) == {'pp', 'fsdp_gather', 'fsdp_reduce_scatter', 'dp'}
    assert 0 < b['metrics']['training_communication']['overlap_fraction'] <= 1
    assert a['estimate']['resource_work'] == b['estimate']['resource_work']
    data['execution']['features']['training_qos']['weights'] = {'unknown': 2}
    with pytest.raises(ValueError):
        Config.model_validate(data)


@pytest.mark.parametrize('structured', [False, True])
def test_retained_weights_sync_each_microbatch_and_blocking_pp(structured):
    data = case(structured, prefetch_layers=2, reshard_after_forward=False, sync_each_microbatch=True,
                overlap_gradient_sync=True, overlap_pipeline_p2p=False)
    data['workload'].update(gradient_accumulation=3, training_steps=2)
    graph = compile_graph(data)
    gathers = [e for e in graph.events if e.metadata.get('traffic_class') == 'fsdp_gather']
    assert gathers and all('.forward.m0.' in e.id for e in gathers)
    sim = simulate(graph)
    assert sim['status'] == 'feasible', sim['diagnostics']
    for pool, amount in sim['memory_final_bytes'].items():
        assert amount == pytest.approx(sum(a.size_bytes for a in graph.initial_allocations if a.pool == pool))
    ends = {row['id']: row['end_s'] for row in sim['trace']}
    for step in range(2):
        for stage in range(2):
            for micro in range(3):
                prefix = f'train.s{step}.p{stage}.backward.m{micro}'
                assert all(time <= ends[prefix + '.done'] for eid, time in ends.items()
                           if eid.startswith(prefix + '.') and eid.endswith('.gradient.done'))


def test_qos_tp_ep_identity_uses_logical_coordinates_with_physical_permutation():
    data = case(False, tp=2, ep=2, pp=2, dp=2, fsdp=2, rank_order=list(reversed(range(16))))
    data['model'].update(num_experts=4, top_k=2, expert_parameter_fraction=.5, moe_layer_fraction=.5)
    data['hardware']['npu_count'] = 16
    data['execution']['features']['training_qos']['weights'] = {'tp': 5, 'ep': 3, 'pp': 2}
    graph = compile_graph(data)
    weights = {(e.metadata['traffic_class'], e.metadata['qos_weight']) for e in graph.events if 'traffic_class' in e.metadata}
    assert weights == {('tp', 5), ('ep', 3), ('pp', 2), ('fsdp_gather', 1), ('fsdp_reduce_scatter', 1)}


def test_attention_tp_streams_completed_query_blocks_and_drains_before_consumer():
    data = case(tp=2, pp=1, dp=1, fsdp=1, flash_attention_blocks=3, overlap_attention_tp=True)
    data['hardware'].update(npu_count=2, su_size=2)
    data['execution']['features']['activation_checkpointing'] = True
    split = run(data)
    data['execution']['overlap_attention_tp'] = False
    whole = run(data)
    a, b = split['metrics'], whole['metrics']
    assert a['training_accounting']['model_forward_backward_flops'] == pytest.approx(b['training_accounting']['model_forward_backward_flops'])
    tp_payload = lambda m: m['training_communication']['by_class']['tp']['payload_bytes']
    assert tp_payload(a) == pytest.approx(tp_payload(b))
    trace = {r['id']: r for r in split['simulation']['trace']}
    for phase, layer, replay in [('forward', 0, ''), ('backward', 5, '.recompute')]:
        prefix = f'train.s0.p0.{phase}.m0.l{layer}'
        first_transfer = trace[prefix + replay + '.GQA.tp0.q0.reduce_scatter.s0.r0']
        attention_end = trace[prefix + '.npu0' + replay + '.GQA']['end_s']
        assert first_transfer['start_s'] < attention_end
        assert trace[prefix + replay + '.GQA.tp0.q2.done']['end_s'] >= attention_end - 1e-12
    assert not any('query_block' in r['metadata'] for r in trace.values()
                   if r['metadata'].get('traffic_class') == 'tp' and r['metadata'].get('phase') == 'backward')
    for pool, amount in split['simulation']['memory_final_bytes'].items():
        assert amount == pytest.approx(split['estimate']['resident_memory_bytes'].get(pool, 0.0))

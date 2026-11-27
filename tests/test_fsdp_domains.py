from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config, load_config
from supernode_sim.rank_layout import configured_groups
from supernode_sim.components import parameter_inventory


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'training_cp_ep_fsdp.yaml'


def case(mode='fsdp2', **domain):
    data = load_config(EXAMPLE).model_dump()
    data['execution']['fsdp_domain'].update(implementation=mode, **domain)
    return data


@pytest.mark.parametrize('mode', ['fsdp', 'fsdp2'])
def test_shared_pool_domains_conserve_placement_storage_and_communication(mode):
    spec = Config.model_validate(case(mode))
    graph = compile_graph(spec)
    result = run(spec, full_detail=False)
    assert result['status'] == 'feasible'
    validated = configured_groups(spec.execution.model_dump(), 8)['fsdp_domains']
    assert validated == result['training_summary']['fsdp_domains'] == graph.metadata['fsdp_domains']
    assert spec.execution.dp == spec.execution.pp == 1
    dense_groups = [[f'npu{i}' for i in range(start, start+4)] for start in (0, 4)]
    expert_groups = [[f'npu{i}', f'npu{i+4}'] for i in range(4)]
    assert validated['dense']['stages'][0]['shard_groups'] == dense_groups
    assert validated['expert']['stages'][0]['shard_groups'] == expert_groups
    assert validated['dense']['stages'][0]['replica_groups'] == expert_groups
    assert not any(e.metadata.get('operation') == 'pipeline' for e in graph.events)
    tensors = parameter_inventory(spec.model.model_dump())
    for family, groups, routed in [('dense', dense_groups, False), ('expert', expert_groups, True)]:
        size = len(groups[0])
        local = sum(t.local_elements(1, 4) for t in tensors if t.ep_shard == routed)
        weights = [a for a in graph.initial_allocations if a.category == 'weight' and a.metadata['shard']['fsdp_family'] == family]
        assert sum(a.size_bytes for a in weights) == len(groups) * local * 2
        for e in graph.events:
            m = e.metadata
            if m.get('fsdp_family') == family and m.get('type') == 'transfer':
                expected = validated[family]['stages'][0]
                assert m['communicator'] in expected['replica_groups' if m['category'] == 'data_parallel_gradients' else 'shard_groups']
                assert m['traffic_class'] == {'fsdp_weights':'fsdp_gather','fsdp_gradients':'fsdp_reduce_scatter',
                                              'data_parallel_gradients':'dp'}[m['category']]
        traffic = result['training_summary']['fsdp_sent_bytes'][family]
        ring_bytes = len(groups) * (size-1) * local * 2
        assert traffic['forward']['fsdp_weights'] == traffic['backward']['fsdp_weights'] == 2 * ring_bytes
        assert traffic['backward']['fsdp_gradients'] == ring_bytes
        assert traffic['backward'].get('data_parallel_gradients', 0) == (2*local*2 if not routed else 0)
    assert result['metrics']['training_communication']['by_class']['ep']['payload_bytes'] > 0
    assert result['metrics']['training_communication']['by_class']['cp']['payload_bytes'] > 0


def test_modes_partition_uneven_weights_and_communicate_padded_shards():
    data = case()
    data['hardware'].update(npu_count=4, su_size=4)
    data['model'] = {'name':'odd-shaped', 'layers':1, 'hidden_dim':3, 'num_heads':1, 'kv_heads':1, 'head_dim':3,
                     'architecture':{'vocab_size':5, 'dense_intermediate_dim':7, 'attention_by_layer':['GQA'], 'ffn_by_layer':['Dense']}}
    data['execution'] = {'fsdp_domain':{'dense_degree':4, 'expert_degree':1}, 'cp':2}
    wire_bytes = {}
    for mode, expected in [('fsdp', [30,0,0,0]), ('fsdp2', [12,12,6,0])]:
        data['execution']['fsdp_domain']['implementation'] = mode
        cfg = Config.model_validate(data)
        graph = compile_graph(cfg)
        weights = [a.size_bytes for a in graph.initial_allocations if a.category == 'weight' and a.metadata['component'] == 'WordEmbedding']
        assert weights == expected
        result = run(cfg, full_detail=False)
        assert result['status'] == 'feasible'
        wire_bytes[mode] = result['training_summary']['fsdp_sent_bytes']['dense']['forward']['fsdp_weights']
        initial = Counter()
        for a in graph.initial_allocations:
            initial[a.pool] += a.size_bytes
        for pool, amount in result['simulation']['memory_final_bytes'].items():
            assert amount == pytest.approx(initial[pool])
    assert wire_bytes['fsdp2'] > wire_bytes['fsdp']


@pytest.mark.parametrize('change,match', [
    ({'expert_groups':[[0,1],[2,3],[4,5],[6,7]]}, 'different pre-FSDP'),
    ({'expert_groups':[[0,4],[1,5],[2,6],[3,3]]}, 'exactly once'),
    ({'expert_groups':[[0,4],[1,5]]}, 'cover every'),
    ({'expert_degree':4, 'expert_groups':None}, 'copies'),
    ({'dense_groups':[[0,1,2,3,4,5,6,7]]}, 'degree'),
])
def test_invalid_domains_fail_at_validation(change, match):
    with pytest.raises(ValidationError, match=match):
        Config.model_validate(case(**change))


def test_custom_rank_order_preserves_shard_indices_across_replica_groups():
    groups = [[7,0,6,1], [2,5,3,4]]
    data = case(dense_groups=groups, expert_groups=None, expert_degree=2)
    data['execution'].update(rank_order=[7,6,5,4,3,2,1,0], expert_rank_order=[0,4,1,5,2,6,3,7])
    fsdp = compile_graph(Config.model_validate(data)).metadata['fsdp_domains']
    assert fsdp['dense']['stages'][0]['replica_groups'] == [[f'npu{x}', f'npu{y}'] for x,y in zip(*groups)]
    assert fsdp['expert']['stages'][0]['shard_groups'][0] == ['npu7','npu5']


@pytest.mark.parametrize('retain', [False, True])
def test_tied_embedding_is_owned_once_and_available_to_both_layers(retain):
    data = case()
    data['model']['architecture']['tie_embeddings'] = True
    data['model'].update(total_params=None, active_params=None)
    data['execution'].update(reshard_after_forward=not retain, sync_each_microbatch=True,
                             overlap_gradient_sync=True, prefetch_layers=1)
    data['execution']['features']['optimizer_offload'] = True
    cfg = Config.model_validate(data)
    graph = compile_graph(cfg)
    ops = [e.metadata for e in graph.events if e.metadata.get('tied_parameter') and e.metadata.get('type') == 'transfer']
    gathers = {(m['layer'],m['phase']) for m in ops if m['category']=='fsdp_weights'}
    assert gathers == ({(0,'forward')} if retain else {(i, phase) for i in (0,1) for phase in ('forward','backward')})
    assert {m['layer'] for m in ops if m['category']=='fsdp_gradients'} == {0}
    result = run(cfg, full_detail=False)
    assert result['status'] == 'feasible'
    peaks = Counter()
    for a in graph.initial_allocations:
        peaks[a.pool] += a.size_bytes
    for pool, amount in result['simulation']['memory_final_bytes'].items():
        assert amount == pytest.approx(peaks[pool])


def test_feature_input_and_scalar_compatibility():
    data = case()
    canonical = Config.model_validate(data)
    domain = data['execution'].pop('fsdp_domain')
    data['features'] = {'fsdp': {'domain':domain}}
    assert Config.model_validate(data).model_dump() == canonical.model_dump()
    data.pop('features')
    data['execution']['fsdp'] = 2
    with pytest.raises(ValidationError, match='fsdp must divide dp'):
        Config.model_validate(data)
    data['execution']['fsdp_domain'] = {'implementation':'fsdp'}
    assert compile_graph(Config.model_validate(data)).metadata['fsdp_domains']['dense']['degree'] == 2
    data['workload']['scenario'] = 'inference'
    with pytest.raises(ValidationError, match='requires a training workload'):
        Config.model_validate(data)


def test_explicit_members_respect_pipeline_and_tensor_shards():
    data = case(expert_groups=None, expert_degree=2, dense_degree=2)
    data['hardware']['npu_count'] = 16
    data['execution'].update(pp=2, tp=2, attention_dp=2)
    spec = Config.model_validate(data)
    domains = configured_groups(spec.execution.model_dump(), 16)['fsdp_domains']
    assert domains['dense']['stages'][1]['shard_groups'][0] == ['npu8', 'npu10']
    groups = [[int(r[3:]) for r in group] for stage in domains['dense']['stages'] for group in stage['shard_groups']]
    for wrong in (1, 8):
        # Swap two ranks, preserving full coverage and degree but violating ownership.
        altered = [group[:] for group in groups]
        j = next(i for i, group in enumerate(altered) if wrong in group)
        k = altered[j].index(wrong)
        altered[0][0], altered[j][k] = altered[j][k], altered[0][0]
        data['execution']['fsdp_domain']['dense_groups'] = altered
        with pytest.raises(ValidationError, match='different pre-FSDP'):
            Config.model_validate(data)


def test_hybrid_residual_domains_and_default_shard_size():
    data = case(expert_groups=None, expert_degree=2)
    data['hardware']['npu_count'] = 32
    data['execution'].update(dp=4, dp_domain_size=2)
    spec = Config.model_validate(data)
    graph = compile_graph(spec)
    phases = {e.metadata['dp_reduction_scope'] for e in graph.events
              if e.metadata.get('type') == 'transfer' and e.metadata.get('fsdp_family') == 'dense'
              and e.metadata.get('parallel_axis') == 'dp_replica'}
    assert phases == {'within_domain', 'between_domains'}
    data['execution']['fsdp_domain']['dense_degree'] = None
    cfg = Config.model_validate(data)
    desc = configured_groups(cfg.execution.model_dump(), 32)['fsdp_domains']
    assert desc['dense']['degree'] == 16
    data['execution']['fsdp_domain']['dense_degree'] = 32
    with pytest.raises(ValidationError, match='dp_domain_size'):
        Config.model_validate(data)

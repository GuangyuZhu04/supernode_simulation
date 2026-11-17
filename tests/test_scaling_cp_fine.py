"""Semantic invariants for the independent fine CP experiment adapter."""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.operator_trace import apply_kernel_traces
from scripts.scaling_context_parallel import build_cp_layer
from scripts.scaling_cp_fine_ops import build_fine_cp_layer
from scripts.scaling_kimi_attention import default_profile as kimi_profile
from scripts.scaling_v41_attention import default_profile as v41_profile
from scripts.scaling_study_engine import make_config, MODELS


KINDS=[('kimi',kind) for kind in kimi_profile()['counts']]+[('v41',kind) for kind in v41_profile()['counts']]


def inputs(family):
    profile=kimi_profile() if family=='kimi' else v41_profile()
    config=make_config(MODELS[0],'960DT',5,attention_profile=profile).model_dump()
    # Small host graph, full 1M geometry and unchanged exact parameter budget.
    # The roomy capacity is synthetic test hardware, not a production result.
    config['hardware'].update(name='synthetic-fine-CP-semantic-test',npu_count=16,su_size=16)
    config['hardware']['npu']['hbm_bytes']=2e12
    config['execution'].update(tp=4,ep=4)
    config['workload'].update(batch_size=2,input_lengths=[1048576]*2,output_lengths=[2]*2)
    return Config.model_validate(config),profile


def compute(event):return sum(v for k,v in event.work.items() if '.compute' in k)
def close(a,b):assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-6),(a,b)
def allocations(graph):
    return {a.id:a for a in graph.initial_allocations+[a for e in graph.events for a in e.allocs]}
def event_signature(event):
    return (event.id,event.deps,event.startup_s,event.work,event.allocs,event.releases)
def ancestors(events,eid):
    found=set();todo=list(events[eid].deps)
    while todo:
        current=todo.pop()
        if current not in found:found.add(current);todo.extend(events[current].deps)
    return found


@pytest.mark.parametrize('family,kind',KINDS)
def test_work_network_and_lifetime_invariants(family,kind):
    config,profile=inputs(family)
    coarse=build_cp_layer(config,profile,kind)
    fine,audit=build_fine_cp_layer(config,profile,kind)
    old_events={e.id:e for e in coarse.events};new_events={e.id:e for e in fine.events}
    assert set(old_events)<=set(new_events)
    for event in coarse.events:
        if event.metadata.get('type')=='transfer':assert event_signature(event)==event_signature(new_events[event.id])
    before=allocations(coarse);after=allocations(fine)
    for key,allocation in before.items():assert after[key]==allocation
    for event in coarse.events:
        assert set(event.releases)<=set(new_events[event.id].releases)
    old_parameter=sum(compute(e)*e.metadata.get('symmetry_multiplicity',1) for e in coarse.events
                      if e.metadata.get('kernel') in ('experts','shared_projection_pre','shared_projection_post'))
    new_parameter=0.
    for event in fine.events:
        if event.metadata.get('fine_extra'):
            parameter=event.metadata.get('fine_work_classification')=='parameter_work_reassigned_from_shared_D'
        else:
            parameter=bool(event.work) and event.metadata.get('original_kernel') in ('experts','shared_projection_pre','shared_projection_post')
        if parameter:new_parameter+=compute(event)*event.metadata.get('symmetry_multiplicity',1)
        if event.metadata.get('matmul'):
            shape=event.metadata['matmul']
            assert all(isinstance(shape[k],int) and shape[k]>0 for k in ('batch','m','n','k'))
            close(compute(event),2*shape['batch']*shape['m']*shape['n']*shape['k'])
        if event.metadata.get('original_kernel') in ('experts','shared_projection_pre','shared_projection_post'):
            assert 'matmul' not in event.metadata
    close(new_parameter,old_parameter)
    # Weight traffic is a separately conserved quantity, not merely hidden
    # beneath a larger total that includes new activation materialization.
    for parent in coarse.events:
        if parent.metadata.get('kernel')!='experts':continue
        children=[e for e in fine.events if e.metadata.get('trace_parent_event')==parent.id and e.work and not e.metadata.get('fine_extra')]
        assert len(children)==3
        old_weight=sum(v for r,v in parent.work.items() if r.endswith('.hbm'))
        close(sum(e.metadata['implementation']['preserved_weight_read_bytes'] for e in children),old_weight)
        geometry=children[0].metadata['implementation']['logical_geometry']
        assignments=geometry['expected_routed_token_assignments_per_rank'];width=geometry['logical_intermediate_width']
        hidden=geometry['hidden_dim']
        close(sum(e.metadata['implementation']['added_activation_hbm_bytes'] for e in children),6*assignments*(hidden+width))
        scalar=[e for e in fine.events if e.metadata.get('trace_parent_event')==parent.id and e.metadata.get('fine_extra')]
        assert len(scalar)==1
        close(compute(scalar[0]),(12 if family=='kimi' else 8)*assignments*width)
        close(sum(v for r,v in scalar[0].work.items() if r.endswith('.hbm')),6*assignments*width)
    shared_before=sum(v for e in coarse.events if e.metadata.get('kernel') in ('shared_projection_pre','shared_projection_post') for r,v in e.work.items() if r.endswith('.hbm'))
    shared_after=sum(v for e in fine.events if e.metadata.get('original_kernel') in ('shared_projection_pre','shared_projection_post') for r,v in e.work.items() if r.endswith('.hbm'))
    close(shared_after+audit['reassigned_conv_parameter_flops']/2,shared_before)
    close(sum(compute(e) for e in fine.events),sum(compute(e) for e in coarse.events)+audit['added_nonparameter_flops'])
    assert audit['added_hbm_bytes']>=0
    assert audit['native_trace_summary']['matched_events']==audit['native_trace_summary']['eligible_events']
    # Every introduced materialization is produced before its last consumer.
    for buffer in audit['cross_child_buffers']:
        assert buffer['producer'] in ancestors(new_events,buffer['consumer'])
        assert buffer['id'] in {a.id for a in new_events[buffer['producer']].allocs}
        assert buffer['id'] in new_events[buffer['consumer']].releases
    result=simulate(fine,retain_event_payload=False)
    assert result['status']=='feasible'
    live=set()
    for action in result['memory_log']:
        if action['action']=='allocate':live.add(action['allocation_id'])
        else:live.remove(action['allocation_id'])
    assert not (live & (set(after)-set(before)))
    assert result['completed_events']==len(fine.events)
    if kind!='KDA':
        for parent,event in old_events.items():
            if event.metadata.get('kernel')!='cp_attention_core':continue
            qk=parent+'.operator_trace.0';av=parent+'.operator_trace.1'
            stats=parent.replace('cp_attention_core','cp_local_softmax_stats')
            softmax=[e.id for e in fine.events if e.metadata.get('trace_parent_event')==stats and e.work]
            assert all(qk in ancestors(new_events,eid) for eid in softmax)
            assert all(eid in ancestors(new_events,av) for eid in softmax)
            assert av in ancestors(new_events,parent)
            assert parent in ancestors(new_events,stats)
    else:
        k=profile['kda'];channels=k['num_heads']*(2*k['head_k_dim']+k['head_v_dim'])
        close(audit['reassigned_conv_parameter_flops'],2*channels*k.get('conv_kernel',4))
        conv=[e for e in fine.events if e.metadata.get('fine_work_classification')=='parameter_work_reassigned_from_shared_D']
        assert len(conv)==1 and any(k.endswith('.compute.fp32') for k in conv[0].work)
        # FP32 matrix state is read/written in each unfused decay/update/readout.
        core=[e for e in fine.events if e.metadata.get('original_kernel')=='kda_recurrent' and e.work and not e.metadata.get('fine_extra')]
        state_bytes=k['num_heads']*k['head_k_dim']*k['head_v_dim']*4
        assert sum(v for e in core for r,v in e.work.items() if r.endswith('.hbm'))>=6*state_bytes


def test_launch_and_network_factors_are_independent():
    config,profile=inputs('kimi')
    base,_=build_fine_cp_layer(config,profile,'MLA',alpha=1.,launch_s=0.)
    launched,_=build_fine_cp_layer(config,profile,'MLA',alpha=.3,launch_s=1e-6)
    indexed={e.id:e for e in base.events}
    for event in launched.events:
        original=indexed[event.id]
        assert event.work==original.work and event.deps==original.deps
        if event.metadata.get('type')=='transfer':close(event.startup_s,.3*original.startup_s)
        elif event.metadata.get('kernel'):
            assert event.startup_s==1e-6
        else:assert event.startup_s==0
    assert all(e.startup_s==0 for e in base.events if e.metadata.get('kernel'))


@pytest.mark.parametrize('change',('hardware','phase','dtype','context_signature'))
def test_native_trace_guards_reject_stale_scope(change):
    config,profile=inputs('kimi')
    _,audit=build_fine_cp_layer(config,profile,'MLA')
    trace_config=deepcopy(audit['trace_config'])
    graph=build_cp_layer(config,profile,'MLA')
    traces=trace_config['hardware']['kernel_traces']
    signature=next(iter(traces.values()))[0]['metadata_match']['fine_workload_signature']
    target=next(e for e in graph.events if e.metadata.get('kernel')=='cp_attention_core')
    for event in graph.events:
        if event.metadata.get('kernel'):
            event.metadata.update(fine_workload_signature=signature,fine_parent_kernel=event.metadata['kernel'])
            if change=='phase':event.metadata['phase']='prefill'
            if change=='context_signature':event.metadata['fine_workload_signature']='different-context'
    if change=='hardware':trace_config['hardware']['name']='different-hardware'
    if change=='dtype':trace_config['execution']['precision']['compute']='fp16'
    apply_kernel_traces(graph,trace_config)
    assert not any(e.metadata.get('trace_parent_event')==target.id for e in graph.events)


def test_same_flops_and_bytes_different_head_geometry_has_distinct_guard():
    config,profile=inputs('kimi');alternate=deepcopy(profile)
    alternate['num_heads']=64
    alternate['mla'].update(qk_nope_head_dim=224,qk_rope_head_dim=64,v_head_dim=192)
    original=build_cp_layer(config,profile,'MLA');changed=build_cp_layer(config,alternate,'MLA')
    a=next(e for e in original.events if e.metadata.get('kernel')=='cp_attention_core')
    b=next(e for e in changed.events if e.metadata.get('kernel')=='cp_attention_core')
    assert a.work==b.work
    _,left=build_fine_cp_layer(config,profile,'MLA');_,right=build_fine_cp_layer(config,alternate,'MLA')
    assert left['identity']!=right['identity']
    first=lambda value:next(iter(value['trace_config']['hardware']['kernel_traces'].values()))[0]['metadata_match']['fine_workload_signature']
    assert first(left)!=first(right)

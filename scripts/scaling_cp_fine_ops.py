"""Scoped operator refinement of the frozen owner/star CP experiment.

The native kernel-trace expander creates serial child Events.  Extra vector
operators and cross-child tensor lifetimes are explicit graph additions.  No
hardware measurements, binary layouts or integer expert matrix shapes are
inferred from the user's parameter budgets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math

from supernode_sim.config import KernelTrace
from supernode_sim.core import Allocation, Event
from supernode_sim.operator_trace import apply_kernel_traces
from scripts.scaling_context_parallel import build_cp_layer
from scripts.cp_fine_trace_specs import operations_for as shared_operations_for


def _operation(name,flops,memory,**extra):
    return {'kernel':name,'flops':float(flops),'memory_bytes':float(memory),**extra}


def _resources(event):
    compute=next(k for k in event.work if k.endswith(('.compute','.compute.fp32')))
    rank=compute.split('.')[0]
    return rank,compute,float(event.work[compute]),float(event.work.get(rank+'.hbm',0))


def _keep_memory(operations,minimum):
    short=max(0.,minimum-math.fsum(x['memory_bytes'] for x in operations))
    if short:
        operations[0]['memory_bytes']+=short
        implementation=operations[0].setdefault('implementation',{})
        implementation['retained_parent_hbm_proxy_bytes']=short
        implementation['retained_hbm_scope']='Unresolved parent HBM envelope retained for conservation; not asserted to be this child kernel actual transactions.'
    return operations


def _extra(name,flops,memory,after,*,fp32=True,buffer_specs=(),release_names=(),classification='nonparameter'):
    return {'name':name,'flops':float(flops),'memory_bytes':float(memory),'after_index':after,
            'fp32':fp32,'buffers':list(buffer_specs),'release_names':list(release_names),'classification':classification}


def _recipe(event,c,profile,kind):
    rank,compute,F,M=_resources(event);kernel=event.metadata['kernel']
    m=c['model'];cp=c['execution']['tp'];ep=c['execution']['ep'];batch=c['workload']['batch_size']
    context=c['workload']['input_length']+1
    ops=[];carry=[];extras=[];notes=[];geometry={};budget_only=False
    shared=shared_operations_for(event,c,profile,kind)
    if shared is not None:
        return {'operations':shared['operations'],
                'carry_buffers':[{'name':b['name'],'bytes':b['size_bytes'],'dtype':b['dtype'],
                                  'producer':b['producer_index'],'consumer':b['consumer_index']} for b in shared['carry_buffers']],
                'extras':[{**e,'name':e['kernel'],'classification':'additional_nonparameter'} for e in shared['extra_after_ops']],
                'geometry':shared['logical_geometry'],'notes':shared['notes'],
                'budget_only':kernel.startswith('shared_projection_')}
    if kernel=='cp_attention_core':
        heads=profile['num_heads']
        if profile['family']=='kimi_k3':
            rows=math.ceil(context/cp);qk=profile['mla']['qk_nope_head_dim']+profile['mla']['qk_rope_head_dim'];v=profile['mla']['v_head_dim']
        else:
            ratio=2 if 'ENCODER' in kind else 1
            rows=math.ceil((min(context,profile['window_size'])+(0 if kind=='SWA' else min(profile['index_topk'],context//ratio)))/cp)
            qk=v=profile['head_dim']
        score=4*heads*rows
        f_qk=2*heads*rows*qk;f_av=2*heads*rows*v
        assert math.isclose(f_qk+f_av,F,rel_tol=1e-12)
        geometry={'dimension_basis':'reference_attention_algebra_not_binary_or_compressed_MLA_layout',
                  'heads':heads,'local_context_or_selected_rows':rows,'qk_width':qk,'value_width':v}
        ops=[_operation('attention.qk_dot',f_qk,M+score,matmul={'m':1,'n':rows,'k':qk,'batch':heads}),
             _operation('attention.av_dot',f_av,M+score+heads*v*4,matmul={'m':1,'n':v,'k':rows,'batch':heads},
                        implementation={'output_dtype':'fp32','output_role':'local_unnormalized_softmax_numerator'})]
        carry=[{'name':'scores_or_exponentials','bytes':score,'dtype':'fp32','producer':0,'consumer':1}]
        notes=['Unfused QK and AV each reread the declared compressed/sparse cache envelope; the FP32 score/exponential array spans the intervening local softmax.',
               'Matmul fields describe reference algebra only; no real MLA compressed-cache kernel or binary tiling is asserted.']
    elif kernel=='cp_local_softmax_stats':
        heads=profile['num_heads'];elements=F/5
        assert math.isclose(elements,round(elements),abs_tol=1e-7)
        ops=[_operation('attention.local_reduce_max',elements,4*elements+4*heads),
             _operation('attention.exp_shift',3*elements,8*elements+4*heads),
             _operation('attention.local_reduce_sum',elements,4*elements+4*heads)]
        notes=['Five parent scalar operations per score are retained as max (1), subtract/exp proxy (3), and sum (1). No instruction-cycle claim.']
    elif kernel=='kda_recurrent':
        k=profile['kda'];H=k['num_heads'];dk=k['head_k_dim'];dv=k['head_v_dim'];n=H*dk*dv;R=n*4
        assert math.isclose(F,8*n,rel_tol=1e-12)
        ops=[_operation('kda.state_decay',n,2*R+H*dk*4),
             _operation('kda.state_prediction',2*n,R+H*dk*2+H*dv*4),
             _operation('kda.delta_rank1_update',3*n,2*R+H*dk*2+H*dv*4),
             _operation('kda.state_readout',2*n,R+H*dk*2+H*dv*2)]
        carry=[{'name':'predicted_value','bytes':H*dv*4,'dtype':'fp32','producer':1,'consumer':2},
               {'name':'delta_value','bytes':H*dv*4,'dtype':'fp32','producer':1,'consumer':2}]
        extras=[_extra('kda.value_residual_beta',3*H*dv,4*H*dv*3,1),
                _extra('kda.output_norm_gate',6*H*dv,H*dv*8,3)]
        geometry={'dimension_basis':'reference_KDA_state_algebra','heads':H,'head_k_dim':dk,'head_v_dim':dv,
                  'state_dtype':'fp32','core_compute_resource_policy':'unchanged_from_coarse_parent'}
        notes=['Core 8*heads*dk*dv work is conserved across decay/prediction/rank-one-update/readout; state reads and writes are FP32 bytes.',
               'Core arithmetic capacity is inherited for a controlled granularity comparison; FP32 state does not establish the unknown fused kernel arithmetic throughput.']
    elif kernel=='cp_local_topk':
        H=profile['index_num_heads'];keys=F/(3*H+11);keep=min(profile['index_topk'],keys)
        ops=[_operation('index.head_weight_reduce',3*H*keys,4*keys*H+4*keys),
             _operation('index.local_topk',9*keys,8*keys+12*keep),
             _operation('index.pack_candidates',2*keys,12*keep*2)]
        notes=['The existing scalar proxy is conserved; sorting/select implementation and real instruction count remain uncalibrated.']
    elif kernel=='cp_global_topk':
        candidates=F/11
        ops=[_operation('index.global_topk',9*candidates,M),
             _operation('index.global_pack_indices',2*candidates,profile['index_topk']*8*2)]
    elif kernel=='cp_index_score':
        H=profile['index_num_heads'];D=profile['index_head_dim'];keys=F/(2*H*D)
        assert math.isclose(keys,round(keys),abs_tol=1e-7);keys=int(round(keys))
        ops=[_operation('index.score_dot',F,M+4*keys*H,
                        matmul={'m':1,'n':keys,'k':D,'batch':H})]
        # The next parent is local_topk, which consumes per-head scores.
        carry=[{'name':'per_head_index_scores','bytes':4*keys*H,'dtype':'fp32','producer':0,'consumer':'next_local_topk'}]
    elif kernel=='cp_exp_rescale':
        H=profile['num_heads'];v=profile['mla']['v_head_dim'] if profile['family']=='kimi_k3' else profile['head_dim']
        ops=[_operation('attention.global_max_rescale_factor',H*8,H*12),
             _operation('attention.rescale_numerator_denominator',H*v,H*(v+1)*8+H*4)]
        carry=[{'name':'rescale_factor','bytes':H*4,'dtype':'fp32','producer':0,'consumer':1}]
    elif kernel=='cp_softmax_merge':
        H=profile['num_heads'];v=profile['mla']['v_head_dim'] if profile['family']=='kimi_k3' else profile['head_dim'];N=H*(v+1)
        ops=[_operation('attention.reduce_numerator_denominator',H*(cp-1)*(v+1),cp*N*4+N*4),
             _operation('attention.normalize_output',H*v,N*4+H*v*2)]
        carry=[{'name':'global_numerator_denominator','bytes':N*4,'dtype':'fp32','producer':0,'consumer':1}]
    else:
        ops=[_operation('atomic.'+kernel,F,M)]
    _keep_memory(ops,M)
    assert math.isclose(math.fsum(op['flops'] for op in ops),F,rel_tol=1e-10,abs_tol=1e-6),(kernel,F,ops)
    return {'operations':ops,'carry_buffers':carry,'extras':extras,'geometry':geometry,'notes':notes,'budget_only':budget_only}


def _signature(events):
    return [(e.id,e.deps,e.startup_s,dict(e.work),[(a.id,a.pool,a.size_bytes) for a in e.allocs],e.releases)
            for e in events if e.metadata.get('type')=='transfer']


def _work(events):
    answer=defaultdict(float)
    for e in events:
        for key,value in e.work.items():
            label='hbm_bytes' if key.endswith('.hbm') else 'flops' if '.compute' in key else 'network_or_other'
            answer[label]+=value
    return dict(answer)


def _expanded_work(events):
    answer=defaultdict(float)
    for e in events:
        multiple=e.metadata.get('symmetry_multiplicity',1)
        for key,value in e.work.items():
            label=('hbm_bytes' if key.endswith('.hbm') else 'fp32_flops' if key.endswith('.compute.fp32')
                   else 'inherited_compute_flops' if key.endswith('.compute') else 'other_resource_work')
            answer[label]+=value*multiple
    return dict(answer)


def build_fine_cp_layer(config,profile,kind,alpha=1.,launch_s=0.):
    """Return (executed-fine-ready Graph, auditable recipe/config dictionary)."""
    if launch_s<0:raise ValueError('launch_s must be nonnegative')
    c=config.model_dump() if hasattr(config,'model_dump') else deepcopy(config)
    graph=build_cp_layer(config,profile,kind,alpha)
    coarse_events=deepcopy(graph.events)
    network_before=_signature(coarse_events);coarse_work=_work(coarse_events)
    cp=c['execution']['tp'];ep=c['execution']['ep'];batch=c['workload']['batch_size']
    identity={'model':c['model']['name'],'total_params':c['model']['total_params'],
              'active_params':c['model']['active_params'],'kind':kind,'context':c['workload']['input_length'],
              'cp':cp,'ep':ep,'batch':batch,'adapter':'cp-fine-v1',
              'model_sha256':hashlib.sha256(json.dumps(c['model'],sort_keys=True).encode()).hexdigest(),
              'attention_profile_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
              'precision_sha256':hashlib.sha256(json.dumps(c['execution']['precision'],sort_keys=True).encode()).hexdigest()}
    workload_signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    recipes={};traces=defaultdict(list);seen=set();unresolved=[];reassigned=0.;kda_pre_extras=[]
    if kind=='KDA':
        # The known depthwise convolution parameters belong to the aggregate
        # D budget. Move their work rather than count these parameters twice.
        k=profile['kda'];channels=k['num_heads']*(2*k['head_k_dim']+k['head_v_dim']);history=k['conv_state_tokens']
        conv_kernel=k.get('conv_kernel',4)  # Kimi reference short-convolution width; history is allocated state, not kernel width.
        conv_params=channels*conv_kernel;conv_F=2*conv_params
        pre=next(e for e in graph.events if e.metadata.get('kernel')=='shared_projection_pre')
        rank,compute,F,M=_resources(pre)
        if F<=conv_F or M<=conv_params:raise ValueError('Shared budget cannot contain reference depthwise convolution')
        pre.work[compute]-=conv_F;pre.work[rank+'.hbm']-=conv_params
        reassigned=conv_F
        qkv_bytes=channels*2;gate_bytes=k['num_heads']*(k['head_k_dim']+1)*4
        kda_pre_extras=[
            {'name':'kda.depthwise_conv_shift','flops':conv_F,'memory_bytes':conv_params+2*channels*history*2+2*qkv_bytes,
             'classification':'parameter_work_reassigned_from_shared_D','buffers':[('qkv',qkv_bytes,'fp16')]},
            {'name':'kda.conv_silu','flops':5*channels,'memory_bytes':2*qkv_bytes,'classification':'additional_nonparameter'},
            {'name':'kda.qk_l2norm_gate','flops':5*k['num_heads']*2*k['head_k_dim']+8*k['num_heads']*(k['head_k_dim']+1),
             'memory_bytes':2*qkv_bytes+gate_bytes,'classification':'additional_nonparameter','buffers':[('gates',gate_bytes,'fp32')]}]
    for event in graph.events:
        if not event.metadata.get('kernel'):continue
        recipe=_recipe(event,c,profile,kind);recipes[event.id]=recipe
        event.metadata.update(fine_workload_signature=workload_signature,fine_parent_kernel=event.metadata['kernel'])
        rank,compute,F,M=_resources(event);dtype='fp32' if compute.endswith('.fp32') else c['execution']['precision']['compute']
        operations=deepcopy(recipe['operations'])
        for op in operations:
            op['startup_s']=float(launch_s)
            op['implementation']={**op.get('implementation',{}),'evidence_kind':'assumed','mapping_scope':recipe['geometry'] or {'basis':'parent_budget_or_existing_atomic_op'},
                                  'is_measured_kernel':False,'source':'scripts/scaling_cp_fine_ops.py; repository reference algebra and retained budget'}
        key=(event.metadata['kernel'],dtype,F,M,json.dumps(operations,sort_keys=True))
        if key not in seen:
            seen.add(key)
            trace={'id':'cp-fine-'+hashlib.sha256(repr(key).encode()).hexdigest()[:16],
                   'source':'Synthetic scoped operator trace; exact parent matrix/core work, declared additional materialization; no hardware timing calibration.',
                   'hardware_name':c['hardware']['name'],'phase':'decode','dtype':dtype,'flops':F,'memory_bytes':M,
                   'metadata_match':{'fine_workload_signature':workload_signature,'fine_parent_kernel':event.metadata['kernel']},
                   'operations':operations}
            traces[event.metadata['kernel']].append(KernelTrace.model_validate(trace).model_dump())
        if recipe['budget_only']:unresolved.append({'parent':event.id,'kernel':event.metadata['kernel'],'flops':F,'memory_bytes':M})
    trace_config={'hardware':{'name':c['hardware']['name'],'kernel_traces':dict(traces)},
                  'execution':{'precision':{'compute':c['execution']['precision']['compute']}}}
    apply_kernel_traces(graph,trace_config)
    events={e.id:e for e in graph.events};added=[];buffer_rows=[];extra_rows=[]

    def attach_buffer(parent,name,size,dtype,producer,consumer):
        node=next(k.split('.')[0] for k in events[producer].work if '.compute' in k)
        aid=parent+'.fine_tensor.'+name
        a=Allocation(aid,node+'.hbm',float(size),'fine_intermediate',dtype,
                     metadata={'component':events[parent].metadata.get('component'),'trace_parent_event':parent,
                               'lifetime_scope':'explicit_cross_child_materialization','conservative_old_arena_retained':True})
        events[producer].allocs+= (a,);events[consumer].releases+=(aid,)
        buffer_rows.append({'id':aid,'bytes':size,'dtype':dtype,'producer':producer,'consumer':consumer})

    def extra_event(parent,spec,dep):
        base=events[parent];node=next(eid.split('.')[0] for eid in events[parent+'.operator_trace.0'].work if '.compute' in eid)
        resource=node+('.compute.fp32' if spec.get('fp32',True) else '.compute')
        eid=parent+'.fine_extra.'+spec['name']
        e=Event(eid,(dep,),{resource:float(spec['flops']),node+'.hbm':float(spec['memory_bytes'])},launch_s,
                slots=(node+'.compute_stream',),metadata={**{k:v for k,v in base.metadata.items() if k!='operator_trace_completion'},'kernel':spec['name'],
                    'component':base.metadata.get('component'),'fine_stage':spec['name'],
                    'trace_parent_event':parent,'original_kernel':base.metadata.get('original_kernel'),
                    'fine_extra':True,'fine_work_classification':spec.get('classification','additional_nonparameter'),
                    'implementation':spec.get('implementation',{}),
                    'is_measured_kernel':False})
        added.append(e);events[eid]=e
        extra_rows.append({'event':eid,'flops':spec['flops'],'memory_bytes':spec['memory_bytes'],
                           'classification':spec.get('classification','additional_nonparameter'),'compute_resource':resource})
        return eid

    # Extra nonlinear operators execute between real trace children; original
    # parent IDs remain as completion sentinels for all network consumers.
    for parent,recipe in recipes.items():
        for spec in recipe['extras']:
            index=spec['after_index'];previous=parent+f'.operator_trace.{index}'
            extra=extra_event(parent,spec,previous)
            following=parent+f'.operator_trace.{index+1}' if index+1<len(recipe['operations']) else parent
            events[following].deps=(extra,)
        for b in recipe['carry_buffers']:
            consumer=(parent+f'.operator_trace.{b["consumer"]}' if isinstance(b['consumer'],int) else
                      parent.replace('cp_index_score','cp_local_topk'))
            attach_buffer(parent,b['name'],b['bytes'],b['dtype'],parent+f'.operator_trace.{b["producer"]}',consumer)

    dependency_refinements=[]
    for parent,recipe in recipes.items():
        if events[parent].metadata.get('original_kernel')!='cp_attention_core':continue
        rank=parent.rsplit('.',1)[0];stats=rank+'.cp_local_softmax_stats'
        qk=parent+'.operator_trace.0';av=parent+'.operator_trace.1'
        stats_first=stats+'.operator_trace.0';stats_last=stats+f'.operator_trace.{len(recipes[stats]["operations"])-1}'
        assert events[stats_first].deps==(parent,)
        events[stats_first].deps=(qk,)
        events[av].deps=(stats_last,)
        events[stats].deps=(parent,)
        dependency_refinements.append({'rank':rank,'order':[qk,stats_first,stats_last,av,parent,stats],
                                      'external_completion_ids_retained':True})

    if kind=='KDA':
        parent=next(p for p in recipes if events[p].metadata.get('original_kernel')=='kda_recurrent')
        first=parent+'.operator_trace.0';dep=events[first].deps[0];held=[]
        for spec in kda_pre_extras:
            dep=extra_event(parent,spec,dep)
            for name,size,dtype in spec.get('buffers',[]):
                attach_buffer(parent,name,size,dtype,dep,parent)
        events[first].deps=(dep,)
    graph.events.extend(added)
    assert _signature(graph.events)==network_before,'Fine operator refinement changed a communication event'
    fine_work=_work(graph.events)
    extra_nonparam=sum(r['flops'] for r in extra_rows if r['classification']!='parameter_work_reassigned_from_shared_D')
    assert math.isclose(fine_work['flops'],coarse_work['flops']+extra_nonparam,rel_tol=1e-10,abs_tol=1e-5)
    assert fine_work['hbm_bytes']>=coarse_work['hbm_bytes']
    names=[a.id for a in graph.initial_allocations]+[a.id for e in graph.events for a in e.allocs]
    assert len(names)==len(set(names)),'Duplicate fine tensor allocation'
    summary={
        'method':'native_kernel_trace_children_plus_explicit_scalar_events_and_tensor_lifetimes',
        'identity':identity,'launch_s_per_real_child_or_extra':launch_s,'network_startup_factor':alpha,
        'coarse_work':coarse_work,'fine_work':fine_work,'added_nonparameter_flops':extra_nonparam,
        'coarse_expanded_work':_expanded_work(coarse_events),'fine_expanded_work':_expanded_work(graph.events),
        'expanded_work_scope':'Symmetry multiplicity applied to resource work. Other-resource work is not transport payload; original cp_study transport remains authoritative.',
        'reassigned_conv_parameter_flops':reassigned,'added_hbm_bytes':fine_work['hbm_bytes']-coarse_work['hbm_bytes'],
        'network_events_unchanged':True,'parameter_flops_conserved':True,
        'fine_child_events':sum('trace_parent_event' in e.metadata and not e.metadata.get('fine_extra') for e in graph.events),
        'fine_extra_events':len(added),'cross_child_buffers':buffer_rows,'extra_operations':extra_rows,
        'dependency_refinements':dependency_refinements,'budget_unresolved':unresolved,
        'native_trace_summary':deepcopy(graph.metadata['operator_trace']),
        'coverage_scope':'Trace matches are execution coverage; single-node shared budgets are not claimed to be tensor-shape resolved.',
        'fidelity_notes':[
            'Analytical scoped child operators, not instruction-level execution, measured kernels or NPUSim calibration.',
            'Compressed MLA QK/AV matmul annotations are reference algebra, not a claimed binary memory layout.',
            'KDA core retains its coarse compute resource; FP32 persistent state traffic does not prove exact arithmetic throughput.',
            'Known KDA depthwise-convolution parameter work is moved out of shared D, preserving aggregate parameter FLOPs and weight residency.',
            'Shared D remains an unresolved budget GEMM. MoE logical intermediate width is budget-derived and may be fractional; no integer expert GEMM shape is fabricated.',
            'All original memory reservations remain; explicit new unfused materializations are additional, so capacity is conservative.',
            'Any retained_parent_hbm_proxy_bytes assigned to a child preserve unresolved parent traffic, not an exact child transaction count.',
            'Additional nonlinear/scalar operation counts are declared proxies; launch_s is independent of network alpha and is an assumption.']}
    graph.metadata['cp_study']['expanded_work_scope']='Original coarse CP parent graph before fine operator expansion; see fine_operator_study.fine_expanded_work for this execution.'
    graph.metadata['fine_operator_study']=summary
    return graph,{**summary,'trace_config':trace_config,
                  'recipe_notes':{parent:r['notes'] for parent,r in recipes.items() if r['notes']}}

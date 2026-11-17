"""Conditional owner/star context-parallel warm-decode experiment.

TP is completely removed. Shared and expert weights are replicated across CP.
One owner per EP lane executes the shared projection budget and the full routed
expert shard. Only context attention is split. This is a declared scheduling
model, not native runtime support or a claim of checkpoint-quality equivalence.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.core import Allocation,Event,Graph,Resource
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from supernode_sim.latency_breakdown import window_breakdown,combine_breakdowns
from supernode_sim.config import load_config
from scripts.scaling_kimi_attention import attention_state as kimi_state
from scripts.scaling_v41_attention import attention_state as v41_state


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def build_cp_layer(config,profile,kind,alpha=1.):
    c=config.model_dump() if hasattr(config,'model_dump') else deepcopy(config)
    m,h,x,w=(c[k] for k in ('model','hardware','execution','workload'))
    assert h['npu_count']==h['su_size'] and h['topology']=='full_mesh'
    assert x['pp']==x['dp']==1 and not x.get('rank_order')
    cp,ep,batch,layers=x['tp'],x['ep'],w['batch_size'],m['layers']
    assert batch<=ep and cp*ep==h['npu_count']
    assert m['total_params']==2.8e12,'Only the four 2.8T points pass the mandatory shared-weight bound'
    assert 0<alpha<=1
    h=deepcopy(h);h['scale_up']['latency_s']*=alpha
    topo=Topology(h,x['precision']);g=Graph();topo.populate(g)
    for index in range(h['npu_count']):
        key=f'npu{index}.compute.fp32'
        g.resources[key]=Resource(key,h['npu']['flops']['fp32']*h['npu']['efficiency'],'FLOP/s','compute')
    dense=m['total_params']*(1-m['expert_parameter_fraction']);expert=m['total_params']-dense
    expert_active=m['active_params']-dense
    context=w['input_length'];prefix=context+1
    precision=x['precision'];feature=x['features']
    params_per_card=dense+expert/ep
    owner='npu0'
    semantic=defaultdict(float);expanded_work=defaultdict(float)
    state=(kimi_state if profile['family']=='kimi_k3' else v41_state)(profile,prefix,1)['categories_bytes']
    shardable=({'kv_cache','kv_scales','page_table'} if profile['family']=='kimi_k3' else
               {'sliding_window_kv','shared_compressed_kv','shared_index_cache','page_table','shared_rope_table'})

    def rank(e,s=0):return f'npu{e*cp+s}'
    def alloc(aid,node,size,category,dtype=None):
        return Allocation(aid,node+'.hbm',float(size),category,dtype)
    for e in range(ep):
        for s in range(cp):
            node=rank(e,s)
            g.initial_allocations.extend([
                alloc(node+'.weight',node,params_per_card,'weight','fp8'),
                alloc(node+'.scales',node,math.ceil(params_per_card/feature['weight_scale_block'])*feature['scale_bytes'],'quantization_metadata'),
                alloc(node+'.runtime',node,h['npu']['runtime_bytes'],'runtime'),
                alloc(node+'.workspace',node,h['npu']['workspace_bytes'],'workspace')])
            if e<batch:
                for key,amount in state.items():
                    if key in shardable:
                        # Bytes are integer and every shard is provisioned for
                        # the largest balanced shard, including page rounding.
                        amount=math.ceil(amount/cp)
                    elif s:
                        continue
                    dtype='fp32' if key in ('recurrent_state','compressor_state','shared_rope_table') else 'fp16'
                    g.initial_allocations.append(alloc(node+'.'+key,node,amount,key,dtype))

    def add(event,multiplicity=1):
        event.metadata.setdefault('phase','decode');event.metadata.setdefault('layer',0)
        event.metadata['symmetry_multiplicity']=multiplicity
        for resource,value in event.work.items():expanded_work[resource.split('.',1)[-1]]+=value*multiplicity
        g.events.append(event);return event.id
    def barrier(name,deps,allocations=(),releases=()):
        return add(Event(name,tuple(deps),allocs=tuple(allocations),releases=tuple(releases)))
    def kernel(name,node,deps,flops=0.,memory=0.,fp32=False,component='Attention',allocations=(),releases=(),multiplicity=batch):
        resource=node+('.compute.fp32' if fp32 else '.compute')
        return add(Event(name,tuple(deps),{resource:float(flops),node+'.hbm':float(memory)},
            slots=(node+'.compute_stream',),allocs=tuple(allocations),releases=tuple(releases),
            metadata={'operation':name.split('.')[-1],'kernel':name.split('.')[-1],'component':component}),multiplicity)
    def transfers(name,flows,deps,multiplicity=batch):
        if not flows:return barrier(name+'.empty',deps)
        work=defaultdict(float);allocations=[];releases=[];payload=0.;physical=0.;startup=0.
        for i,(src,dst,size) in enumerate(flows):
            if size<=0:continue
            event=topo.transfer(f'{name}.{i}',src,dst,size,tuple(deps),
                {'operation':name,'component':'MoE' if name in ('dispatch','combine') else 'Attention',
                 'phase':'decode','collective':'all_to_all_v' if name in ('dispatch','combine') else name,'algorithm':'direct_star'})
            for key,value in event.work.items():work[key]+=value
            startup=max(startup,event.startup_s)
            payload+=size;physical+=event.metadata['physical_bytes']
            for role,node in (('tx',src),('rx',dst)):
                a=alloc(f'{name}.{i}.{role}',node,size,'communication_buffer','fp16')
                allocations.append(a);releases.append(a.id)
        semantic[name]+=payload*multiplicity
        return add(Event(name,tuple(deps),dict(work),startup,
            allocs=tuple(allocations),releases=tuple(releases),metadata={
                'type':'transfer','domains':['SU'],'operation':name,'component':'MoE' if name in ('dispatch','combine') else 'Attention','phase':'decode',
                'payload_bytes':payload,'physical_bytes':physical,'physical_bytes_by_domain':{'SU':physical},
                'startup_s_by_domain':{'SU':startup},'algorithm':'direct_star',
                'scope':'Nonblocking star resource envelope; all symmetric independent request groups expand by batch.'}),multiplicity)
    def broadcast(name,size,deps):return transfers(name,[(owner,rank(0,s),size) for s in range(1,cp)],deps)
    def gather(name,size,deps):return transfers(name,[(rank(0,s),owner,size) for s in range(1,cp)],deps)

    active=[]
    for e in range(batch):
        for s in range(cp):active.append(alloc(f'{rank(e,s)}.activation',rank(e,s),4*m['hidden_dim'],'activation','fp16'))
    ready=barrier('layer.ready',(),active)
    # Equal pre/post budget split is an explicit scheduling envelope. Their
    # combined matrix work is exactly 2*D/L and weight reads D/L per request.
    pre=kernel('owner.shared_projection_pre',owner,(ready,),dense/layers,dense/(2*layers)+4*m['hidden_dim'])
    if kind=='KDA':
        spec=profile['kda'];heads=spec['num_heads'];dk=spec['head_k_dim'];dv=spec['head_v_dim']
        recurrent=heads*dk*dv*4;conv=heads*(2*dk+dv)*spec['conv_state_tokens']*2
        merged=kernel('owner.kda_recurrent',owner,(pre,),8*heads*dk*dv,2*(recurrent+conv))
    else:
        is_kimi=profile['family']=='kimi_k3'
        heads=profile['num_heads'];dv=profile['mla']['v_head_dim'] if is_kimi else profile['head_dim']
        qwidth=(heads*(profile['mla']['qk_nope_head_dim']+profile['mla']['qk_rope_head_dim'])+
                profile['kv_latent_dim']+profile['kv_rope_dim']) if is_kimi else (
                heads*profile['head_dim']+profile['index_num_heads']*profile['index_head_dim']+profile['head_dim']+profile['index_head_dim'])
        query=broadcast('cp_query_broadcast',qwidth*2,(pre,))
        # The appended token belongs to exactly one interleaved context shard.
        # Other peers receive its frame but do not create duplicate cache rows.
        append_node=rank(0,context%cp)
        new_elements=(profile['kv_latent_dim']+profile['kv_rope_dim'] if is_kimi else profile['head_dim'])
        if not is_kimi and kind=='CSA2_DECODER_FULL_INDEX':
            new_elements+=profile['head_dim']+profile['index_head_dim']
        appended=kernel(f'{append_node}.cp_cache_append',append_node,(query,),new_elements,
                        new_elements*4)
        query=barrier('cp.cache_ready',(query,appended))
        context_rows=math.ceil(prefix/cp)
        if not is_kimi and 'INDEX' in kind:
            ratio=2 if 'ENCODER' in kind else 1
            keys=prefix//ratio
            if 'CANDIDATE' in kind:keys=min(keys,profile['candidate_rows'])
            local_keys=math.ceil(keys/cp);keep=min(profile['index_topk'],local_keys)
            index_ready=[]
            for s in range(cp):
                node=rank(0,s)
                dot=kernel(f'{node}.cp_index_score',node,(query,),2*local_keys*profile['index_num_heads']*profile['index_head_dim'],
                           local_keys*profile['index_head_dim']*2+local_keys*4)
                index_ready.append(kernel(f'{node}.cp_local_topk',node,(dot,),local_keys*(3*profile['index_num_heads']+11),
                                          local_keys*8+keep*12,True))
            gathered=gather('cp_index_candidates_gather',keep*12,index_ready)
            chosen=kernel('owner.cp_global_topk',owner,(gathered,),cp*keep*11,cp*keep*12+profile['index_topk']*8,True)
            query=broadcast('cp_selected_indices_broadcast',profile['index_topk']*8,(chosen,))
        core=[]
        for s in range(cp):
            node=rank(0,s)
            if is_kimi:
                spec=profile['mla'];width=profile['kv_latent_dim']+profile['kv_rope_dim']
                flops=2*heads*(spec['qk_nope_head_dim']+spec['qk_rope_head_dim']+spec['v_head_dim'])*context_rows
                memory=context_rows*width*2+qwidth*2
                local_pairs=context_rows
            else:
                ratio=2 if 'ENCODER' in kind else 1
                global_pairs=0 if kind=='SWA' else min(profile['index_topk'],prefix//ratio)
                local_pairs=math.ceil((min(prefix,profile['window_size'])+global_pairs)/cp)
                flops=4*local_pairs*heads*profile['head_dim']
                memory=2*local_pairs*profile['head_dim']*2+qwidth*2
            stat_bytes=heads*(dv+2)*4
            a=alloc(node+'.softmax_statistics',node,stat_bytes,'attention_workspace','fp32')
            dot=kernel(f'{node}.cp_attention_core',node,(query,),flops,memory,allocations=(a,))
            core.append(kernel(f'{node}.cp_local_softmax_stats',node,(dot,),local_pairs*heads*5,
                               stat_bytes*2,True))
        maxima=gather('cp_max_gather',heads*4,core)
        maximum=kernel('owner.cp_global_max',owner,(maxima,),max(0,cp-1)*heads,cp*heads*4+heads*4,True)
        max_bcast=broadcast('cp_max_broadcast',heads*4,(maximum,))
        rescaled=[]
        for s in range(cp):
            node=rank(0,s)
            rescaled.append(kernel(f'{node}.cp_exp_rescale',node,(max_bcast,),heads*(dv+8),heads*(dv+2)*4*2,True))
        partials=gather('cp_numerator_denominator_gather',heads*(dv+1)*4,rescaled)
        merged=kernel('owner.cp_softmax_merge',owner,(partials,),heads*((cp-1)*(dv+1)+dv),
                      cp*heads*(dv+1)*4+heads*dv*2,True,
                      releases=tuple(rank(0,s)+'.softmax_statistics' for s in range(cp)))
    output=kernel('owner.shared_projection_post',owner,(merged,),dense/layers,dense/(2*layers)+4*m['hidden_dim'])

    expert_allocs=[]
    for e in range(ep):
        node=rank(e)
        expert_allocs.extend([
            alloc(node+'.expert_buffer',node,batch*m['top_k']*m['hidden_dim']*2/ep,'expert_buffer','fp16'),
            alloc(node+'.expert_intermediate',node,batch*m['top_k']*m['intermediate_dim']*2/ep,'activation','fp16')])
    expert_ready=barrier('expert.ready',(output,),expert_allocs)
    unit=m['hidden_dim']*m['top_k']*2/ep
    dispatch=transfers('dispatch',[(rank(src),rank(dst),unit) for src in range(batch) for dst in range(ep) if src!=dst],(expert_ready,),1)
    touched=-math.expm1(batch*math.log1p(-m['top_k']/m['num_experts']))
    experts=[]
    for e in range(ep):
        experts.append(kernel(f'{rank(e)}.experts',rank(e),(dispatch,),2*expert_active*batch/(layers*ep),
                              expert*touched/(layers*ep),component='MoE',multiplicity=1))
    combine=transfers('combine',[(rank(src),rank(dst),unit) for src in range(ep) for dst in range(batch) if src!=dst],experts,1)
    final=barrier('infer.d0.token1',(combine,),releases=tuple(a.id for a in active+expert_allocs))
    g.metadata={'scenario':'inference','token_completion_events':{'0':[final]},
        'sequence_token_completion_events':{'0':{str(i):[final] for i in range(batch)}},
        'cp_study':{'kind':kind,'cp':cp,'tp':1,'ep':ep,'batch_size':batch,'alpha':alpha,
            'dense_params':dense,'expert_params':expert,'resident_weight_params_per_card':params_per_card,
            'transport_by_semantic_bytes':dict(semantic),'expanded_work_by_resource_kind':dict(expanded_work),
            'context_state_categories':state,'context_sharded_categories':sorted(shardable),
            'owner_only_categories':sorted(set(state)-shardable),'query_owner':owner,
            'communication_algorithm':'direct_owner_star','state_partition':'balanced/interleaved context rows; ceil per shard',
            'projection_budget_split':'half before context attention, half after; exact combined D reads/2D FLOPs',
            'ffn_policy':'Only s=0 owner executes the whole EP expert shard; weights remain replicated on every CP rank',
            'softmax_merge':'localmax gather, ownermax broadcast, local exp-rescale, numerator/denominator gather, owner sum/divide',
            'scheduling':'Serial phase barriers; independent request CP groups are congruent and represented by one group with batch multiplicity'}}
    return g


def run_case(config,profile,alpha=1.,evidence=None):
    records=[];kinds={};memory={};traffic=defaultdict(float);startups=defaultdict(float)
    for kind,count in profile['counts'].items():
        g=build_cp_layer(config,profile,kind,alpha);s=simulate(g,retain_event_payload=False)
        rec=window_breakdown(g,s,'infer.d0.token1') if s['status']=='feasible' else None
        if rec:records.extend([rec]*count)
        for k,v in s['memory_peaks_bytes'].items():memory[k]=max(memory.get(k,0),v)
        for k,v in g.metadata['cp_study']['transport_by_semantic_bytes'].items():traffic[k]+=v*count
        for event in g.events:
            if event.metadata.get('payload_bytes'):
                startups[event.metadata['operation']]+=event.startup_s*count
        kinds[kind]={'count':count,'status':s['status'],'layer_s':s['duration_s'],
                     'diagnostics':s['diagnostics'],'metadata':g.metadata['cp_study']}
        if evidence:
            evidence.mkdir(parents=True,exist_ok=True)
            with gzip.open(evidence/f'{kind}.graph-and-simulation.json.gz','wt',encoding='utf-8',compresslevel=1) as f:
                from dataclasses import asdict
                json.dump({'graph':asdict(g),'simulation':s},f,ensure_ascii=False,allow_nan=False)
        del g,s
    complete=all(r['status']=='feasible' for r in kinds.values())
    rec=combine_breakdowns(records,divisor=1) if complete else None
    total=rec['total_s'] if rec else None
    communication=0.
    if rec:
        for op in rec['operator_breakdown']['operators']:
            if op['operator'] in startups:communication+=op['total_s']
    target=config.targets.tpot_s
    return {'status':('feasible' if total<=target else 'slo_failed') if complete else 'infeasible',
        'capacity_status':'feasible' if complete else 'infeasible','tpot_ms':total*1000 if total else None,
        'target_ms':target*1000,'target_satisfied':bool(total and total<=target),
        'communication_ms':communication*1000 if rec else None,'startup_ms':sum(startups.values())*1000 if rec else None,
        'transfer_ms':(communication-sum(startups.values()))*1000 if rec else None,
        'latency_breakdown':rec,'kinds':kinds,'peak_hbm_GB':max(memory.values(),default=0)/1e9,
        'transport_by_semantic_bytes':dict(traffic),'startup_by_operator_ms':{k:v*1000 for k,v in startups.items()}}


def fingerprints():
    names=['scripts/scaling_context_parallel.py','scripts/analyze_scaling_sp.py','scripts/scaling_kimi_attention.py',
           'scripts/scaling_v41_attention.py','supernode_sim/core.py','supernode_sim/engine.py','supernode_sim/topology.py',
           'supernode_sim/latency_breakdown.py']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--skip-evidence',action='store_true')
    args=parser.parse_args();study=args.study;out=study/'sp-megakernel/cp-owner-star'
    manifest=json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    selected=[r for r in manifest['cases'] if r['model_label']=='2.8T']
    hashes=fingerprints();rows=[];gc.disable()
    for case in selected:
        cfg=load_config(study/case['config_path']);profile=json.loads((study/case['attention_profile_path']).read_text(encoding='utf-8'))
        for alpha in (1.,.5,.3):
            assert hashes==fingerprints(),'CP sources changed during execution'
            evidence=out/'evidence'/case['id'] if alpha==1 and not args.skip_evidence else None
            row=run_case(cfg,profile,alpha,evidence)
            row.update(case_id=case['id'],architecture_id=case['architecture_id'],hardware=case['hardware_label'],
                       cp=cfg.execution.tp,tp=1,ep=cfg.execution.ep,startup_factor=alpha,
                       config_source=case['config_path'],profile_source=case['attention_profile_path'])
            name=case['id']+f'-alpha{alpha:g}'
            save(out/'cases'/f'{name}.json',row)
            save(out/'configs'/f'{name}.json',{'native_budget_config':cfg.model_dump(),'attention_profile':profile,
                'cp_sidecar':{'tp':1,'cp':cfg.execution.tp,'startup_factor':alpha,'algorithm':'owner_star'},'source_sha256':hashes})
            rows.append(row)
            save(out/'summary.json',{'complete':False,'rows':rows,'source_sha256':hashes})
            print(json.dumps({k:row[k] for k in ('case_id','startup_factor','tpot_ms','target_satisfied','startup_ms','transfer_ms')}),flush=True)
    summary={'complete':True,'rows':rows,'source_sha256':hashes,
        'scope':'Four 2.8T points not excluded by mandatory shared-weight-read lower bound; twelve conditional CP executions.',
        'excluded_case_bounds':'../sp-analysis/pure-cp-bounds.csv',
        'assumptions':['Pure replacement TP1/CP=formerTP; all shared/expert weights replicated, no weight HBM read divided by CP.',
            'Owner executes shared projection budget and KDA recurrent work; CP peers execute only context attention.',
            'One full-batch EP expert shard executes at each owner, without FFN tensor parallelism or token spreading across CP.',
            'Balanced/interleaved context rows and selected keys; no hot shard, scheduling queue, quantization-quality or runtime-layout validation.',
            'Stable softmax and global top-k merges are explicit owner-star transfers plus FP32 work.',
            'Comparing this CP owner-star path to TP ring changes both partitioning and collective algorithm.',
            'Startup factors are hypothetical communication improvements, not evidence that local megakernels reduce network latency.',
            'Shared D is an exact aggregate budget, half charged before and half after attention; no claim of shape-exact projection reconstruction.',
            'Warm decode only; successful CP TPOT does not establish CP Prefill or end-to-end service feasibility.']}
    save(out/'summary.json',summary)
    fields=['case_id','architecture_id','hardware','cp','tp','ep','startup_factor','status','capacity_status','tpot_ms',
            'target_ms','target_satisfied','communication_ms','startup_ms','transfer_ms','peak_hbm_GB']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in rows)
    print(json.dumps({'complete':True,'rows':len(rows),'output':str(out)}))


if __name__=='__main__':main()

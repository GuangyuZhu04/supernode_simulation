"""Execute full-context Prefill for the frozen scaling study's budget templates.

The extension preserves all requested tokens, weights and FLOPs. It explicitly
uses reusable 64 KiB per-peer transport staging and tiled index-score staging;
the original whole-message staging is separately checked for admission.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import replace
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supernode_sim.config import load_config
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import combine_breakdowns, window_breakdown
from supernode_sim.report import write_latency_csv, latency_markdown
from scripts.scaling_study_graph import build_layer, attribution_graph
import scripts.scaling_kimi_attention as kimi
import scripts.scaling_v41_attention as v41

ORIGINAL_KIMI_WORK = kimi.attention_work
COMMUNICATION_OPS = {'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
STAGING_BYTES = 64 * 1024


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')


def kimi_prefill_work(profile, kind, model, context, tp, phase):
    row = ORIGINAL_KIMI_WORK(profile,kind,model,context,tp,phase)
    if phase != 'prefill':
        return row
    # components.component_activations(..., training=False), original K3 widths.
    # Inference does not retain chunk-boundary recurrent states for backward.
    if kind == 'KDA':
        elements = context * (96 * (3*128 + 3*128 + 1)/tp + 128)
    else:
        elements = context * (1536 + 576 + 96 * (2*192 + 2*128)/tp + 96/tp + 96*128/tp)
    workspace = elements * 2
    row['workspace_bytes'] += workspace
    row['memory_bytes'] += 2 * workspace
    row['prefill_workspace_formula'] = 'K3 component_activations attention buffers, FP16; HBM reads+writes=2*working bytes; no training-only recurrent chunk checkpoints'
    return row


def index_geometry(q, ratio, candidate, q_tile=256, key_tile=65536):
    # Exactly the repository V4.1 index_geometry(...,bulk=True) loop. Visibility
    # at the tile end is its declared causal tile envelope, not per-row exactness.
    pairs = tiles = key_reads = maximum = 0
    for begin in range(0,q,q_tile):
        count = min(q_tile,q-begin)
        keys = (begin+count)//ratio
        if candidate:
            keys=min(keys,16384)
        pairs += count*keys
        tiles += math.ceil(keys/key_tile) if keys else 0
        key_reads += keys
        maximum=max(maximum,count*min(keys,key_tile))
    return {'pairs':pairs,'tiles':tiles,'key_reads':key_reads,'max_score_tile_elements':maximum}


def v41_prefill_work(profile, kind, model, context, tp, phase):
    if phase!='prefill':
        raise ValueError('This independent extension only executes Prefill')
    q=context
    d=profile['head_dim']; heads=profile['num_heads']/tp
    ratio=2 if 'ENCODER' in kind else 1
    window=profile['window_size']; topk=profile['index_topk']
    window_pairs=q*(q+1)/2 if q<window else window*q-window*(window-1)/2
    # Exact bounded prefix sum of min(index_topk,position//ratio).
    cutoff=min(q,topk*ratio)
    global_pairs=(sum(min(topk,pos//ratio) for pos in range(1,cutoff+1))+
                  max(0,q-cutoff)*topk) if kind!='SWA' else 0
    pairs=window_pairs+global_pairs
    geo=index_geometry(q,ratio,'CANDIDATE' in kind) if 'INDEX' in kind else {
        'pairs':0,'tiles':0,'key_reads':0,'max_score_tile_elements':0}
    index_heads=profile['index_num_heads']/tp; index_d=profile['index_head_dim']
    index_pairs=geo['pairs']
    index_flops=2*index_pairs*index_heads*index_d
    index_memory=geo['key_reads']*index_d*2+index_pairs*4+q*index_heads*index_d*2 if index_pairs else 0
    selection=index_pairs*(3*index_heads+math.ceil(math.log2(topk))+2)
    selection_memory=index_pairs*8+q*topk*4+q*2048*4 if index_pairs else 0
    shared_emit=math.ceil(q/ratio) if 'FULL_INDEX' in kind or 'ENCODER_INDEX' in kind else 0
    cache_writes=q*d*2+shared_emit*(d+index_d)*2
    # Score tiles are reusable. Selected indices remain resident for the full
    # query set until the later sparse-attention stage consumes them, matching
    # this graph's score -> select -> attention barriers.
    query_tile=min(q,256)
    workspace=(geo['max_score_tile_elements']*4 +
               min(geo['key_reads'],65536)*index_d*2+
               q*(topk+2048)*4 + query_tile*(heads+1)*d*2)
    return {'query_tokens':q,'kind':kind,'kernel_name':kind.lower(),
        'extra_compute_flops':4*pairs*heads*d,
        'extra_fp32_flops':pairs*heads*5+q*(heads+1)*d*10,
        'memory_bytes':2*pairs*d*2+q*heads*d*2+2*workspace,
        'fp32_memory_bytes':q*(heads+1)*d*2,
        'cache_write_bytes':cache_writes,'workspace_bytes':workspace,
        'index_reduce_bytes':index_pairs*4,'index_reduce_repeats':geo['tiles'],
        'index_flops':index_flops,'index_memory_bytes':index_memory,
        'index_selection_flops':selection,'index_selection_memory_bytes':selection_memory,
        'cache_conversion_flops':0.,'cache_raw_bytes':0.,'cache_storage_bytes':0.,
        'index_score_tile_bytes':geo['max_score_tile_elements']*4,
        'index_geometry':geo,'sparse_attention_pairs':pairs,
        'precision_note':'Controlled FP8 projections/dot; FP16 KV/index; FP32 score/select. All40 budget layers execute; CED phase skipping is not implemented.'}


@contextmanager
def prefill_callbacks():
    old_kimi,old_v41=kimi.attention_work,v41.attention_work
    kimi.attention_work=kimi_prefill_work
    v41.attention_work=v41_prefill_work
    try:
        yield
    finally:
        kimi.attention_work=old_kimi
        v41.attention_work=old_v41


def stream_transport(graph, profile, kind, config, afd):
    """Change scratch capacity only; preserve every byte of resource demand."""
    before={event.id:(dict(event.work),event.startup_s,event.deps) for event in graph.events}
    tp=afd['tp'] if afd else config.execution.tp
    work=(kimi_prefill_work if profile['family']=='kimi_k3' else v41_prefill_work)(
        profile,kind,config.model.model_dump(),config.workload.input_length,tp,'prefill')
    staging_before=staging_after=0.
    for event in graph.events:
        op=event.metadata.get('operation')
        if op not in COMMUNICATION_OPS:
            continue
        if op=='tp_index_scores':
            cap=max(1.,work.get('index_score_tile_bytes',0)/tp)
        else:
            cap=STAGING_BYTES
        allocations=[]
        for allocation in event.allocs:
            if allocation.category=='communication_buffer':
                staging_before += allocation.size_bytes
                size=min(allocation.size_bytes,cap)
                allocations.append(replace(allocation,size_bytes=size,
                    metadata={**allocation.metadata,'streamed_staging':True,'full_payload_buffer_bytes':allocation.size_bytes,
                              'buffer_cap_bytes':cap,'scope':'Reusable in-flight tile; complete payload remains charged to endpoint HBM and network resources'}))
                staging_after += size
            else:
                allocations.append(allocation)
        event.allocs=tuple(allocations)
        event.metadata['prefill_transport_staging']='single reusable tile per peer; complete work and logical startup unchanged'
    assert all(before[e.id]==(e.work,e.startup_s,e.deps) for e in graph.events)
    graph.metadata['prefill_transport']={'strategy':'streamed-communication-prefill-template',
        'peer_staging_bytes':STAGING_BYTES,'index_score_buffer':'one streamed score tile per ring rank',
        'sum_original_staging_allocations_bytes':staging_before,'sum_streamed_staging_allocations_bytes':staging_after,
        'resource_work_startups_dependencies_unchanged':True,
        'note':'Buffers are reused while payload streams from resident activations into resident expert buffers or source accumulation output; no startup per 64KiB tile is added.'}


def startup_by_operator(graph, record):
    if not record:
        return None
    startup=defaultdict(float)
    for event in graph.events:
        op=event.metadata.get('operation')
        if op in COMMUNICATION_OPS:
            # Coalesced collective/ring envelopes each occur once on this
            # globally serialized representative layer's critical path.
            startup[op]+=event.startup_s
    result={}
    for operator in record['operator_breakdown']['operators']:
        name=operator['operator']
        if name not in COMMUNICATION_OPS:
            continue
        costs=operator['components_s']
        wait=sum(costs.get(k,0) for k in ('scheduling_wait_s','memory_wait_s','synchronization_wait_s','other_s','launch_s','unattributed_service_s'))
        transfer=operator['total_s']-startup[name]-wait
        assert transfer >= -1e-8
        result[name]={'communication_total_s':operator['total_s'],'startup_s':startup[name],
                      'transfer_s':max(0.,transfer),'other_wait_s':wait,
                      'method':'actual_graph_collective_envelope_startup; transfer includes endpoint/shared-bandwidth contention'}
    return result


def run_case(case_id,config,profile,afd,directory):
    started=time.perf_counter(); records=[]; kinds={}; diagnostics=[]
    comm=defaultdict(lambda:defaultdict(float)); peaks=defaultdict(float); monolithic=[]
    output=directory/'cases'/case_id; output.mkdir(parents=True,exist_ok=True)
    for kind,count in profile['counts'].items():
        print(json.dumps({'case':case_id,'kind':kind,'stage':'build'}),flush=True)
        with prefill_callbacks():
            graph=build_layer(config,phase='prefill',afd=afd,attention_profile=profile,attention_kind=kind)
        original=simulate(graph,retain_event_payload=False)
        monolithic.append({'kind':kind,'status':original['status'],'diagnostics':original['diagnostics']})
        del original
        stream_transport(graph,profile,kind,config,afd)
        simulation=simulate(graph,retain_event_payload=False)
        record=window_breakdown(attribution_graph(graph),simulation,'infer.d0.token1')
        full=combine_breakdowns([record]*count,divisor=1) if record else None
        records.append(full)
        kinds[kind]={'count':count,'status':simulation['status'],
                     'layer_duration_s':record['total_s'] if record else None,
                     'duration_s':full['total_s'] if full else None,
                     'transport':graph.metadata['prefill_transport'],
                     'startup_by_operator_s':{k:v['startup_s']*count for k,v in (startup_by_operator(graph,record) or {}).items()},
                     'source_artifact':f'prefill-study/cases/{case_id}/prefill.{kind}.representative-layer.json.gz'}
        for name,values in (startup_by_operator(graph,record) or {}).items():
            for field,value in values.items():
                if isinstance(value,(int,float)):
                    comm[name][field]+=value*count
        for key,value in simulation['memory_peaks_bytes'].items(): peaks[key]=max(peaks[key],value)
        diagnostics.extend({'kind':kind,**item} for item in simulation['diagnostics'])
        # The complete actual representative execution remains inspectable.
        payload={'scope':'prefill_representative_kind_with_full_model_state_reserved',
                 'graph_metadata':graph.metadata,'simulation':simulation,'latency_breakdown':record,
                 'startup_by_operator':startup_by_operator(graph,record)}
        with gzip.open(output/f'prefill.{kind}.representative-layer.json.gz','wb',compresslevel=1) as handle:
            handle.write(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        print(json.dumps({'case':case_id,'kind':kind,'stage':'complete','status':simulation['status'],
                          'duration_s':full['total_s'] if full else None}),flush=True)
        del graph,simulation,payload
        gc.collect()
    feasible=all(r is not None for r in records)
    full_record=combine_breakdowns(records,divisor=1) if feasible else None
    if full_record: full_record.update(replica=0,sequence=0)
    row={'case_id':case_id,'architecture_id':profile['architecture_id'],
         'status':'feasible' if feasible else 'infeasible','ttft_s':full_record['total_s'] if full_record else None,
         'latency_breakdown':full_record,'diagnostics':diagnostics,
         'source_artifact':f'prefill-study/cases/{case_id}/result.json',
         'attention_kinds':kinds,'monolithic_admission':monolithic,
         'communication_by_operator':dict(comm) if feasible else None,
         'startup_by_operator_s':{k:v['startup_s'] for k,v in comm.items()} if feasible else None,
         'max_hbm_bytes':max(peaks.values(),default=0),'wall_s':time.perf_counter()-started,
         'scope':'All input tokens execute through the declared attention-kind template with full weight/state capacity reserved and streamed transport staging. No CED skipping, cold loading, online queue, or original-checkpoint full tensor reconstruction.'}
    result={'status':row['status'],'config':config.model_dump(),'attention_profile':profile,'afd_sidecar':afd,
            'metrics':{'ttft_s':row['ttft_s'],'tpot_s':None,'latency_breakdown':{'ttft':full_record,'tpot':None,
                'ttft_unavailable_reason':None if feasible else 'prefill_memory_admission_failed',
                'tpot_unavailable_reason':'This artifact only executes Prefill; use original Decode result.'}},
            'prefill':row,'memory_peaks_bytes':dict(peaks),'provenance':fingerprints()}
    save(output/'result.json',result); save(output/'config.resolved.json',config.model_dump())
    write_latency_csv(result,output)
    with (output/'memory.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['pool','peak_bytes']);writer.writeheader()
        writer.writerows({'pool':key,'peak_bytes':value} for key,value in peaks.items())
    text=['# Prefill / TTFT 模板仿真','',f'场景：{case_id}；状态：{row["status"]}；TTFT：{row["ttft_s"]} s。','',
          '完整 1M 输入 × POD 总 batch32；weights 预加载，所有模型状态预留容量；不代表缓存已提前完成计算。',
          '通信使用每 peer 64 KiB 可复用暂存，源 activation 和目的 expert buffer 完整驻留；payload、HBM需求、逻辑启动次数和依赖保持不变。V4索引归约使用单tile暂存。',
          '同时保存未采用流式暂存时的整消息缓冲准入结果。缓冲策略是显式执行假设，不是硬件扩容。',
          'Kimi增加原组件推理attention工作区及2×工作区HBM访问；V4使用仓库因果tile几何、全40预算层，未实现CED阶段跳层。',
          'TTFT以该模板完整前向窗口结束计，未逐张量重建Embedding/LM head，不是原始checkpoint的实机TTFT。','',
          '## 整消息缓冲准入','',*('- '+item['kind']+': '+item['status'] for item in monolithic),'',
          '## 流式执行诊断','',json.dumps(diagnostics,ensure_ascii=False),*latency_markdown(result['metrics'])]
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return row


def fingerprints():
    paths=['scripts/scaling_prefill_study.py','scripts/scaling_study_graph.py','scripts/scaling_kimi_attention.py',
           'scripts/scaling_v41_attention.py','supernode_sim/engine.py','supernode_sim/components.py',
           'supernode_sim/case_studies/v41_fp4/target.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--case',action='append')
    args=parser.parse_args(); directory=args.output/'prefill-study'; directory.mkdir(exist_ok=True)
    manifest=json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    initial=fingerprints(); rows=[]; cache={}; gc.disable()
    for case in manifest['cases']:
        if args.case and case['id'] not in args.case: continue
        assert fingerprints()==initial,'Source changed during Prefill execution'
        config=load_config(args.output/case['config_path'])
        profile=json.loads((args.output/case['attention_profile_path']).read_text(encoding='utf-8'))
        afd=json.loads((args.output/case['afd_path']).read_text()) if case.get('afd_path') else None
        key=(case['architecture_id'],config.model.total_params,config.hardware.name,bool(afd))
        if key in cache:
            prior=cache[key]
            row=deepcopy(prior);row['case_id']=case['id'];row['reused_identical_physical_case']=prior['case_id']
            row['source_artifact']=prior['source_artifact']
            output=directory/'cases'/case['id'];output.mkdir(parents=True,exist_ok=True)
            save(output/'result-reference.json',row)
            (output/'report.md').write_text(f'# Prefill结果复用\n\n与 `{prior["case_id"]}` 的模型、负载、硬件和执行策略完全一致；仅TPOT目标不同。\n\nTTFT：{row["ttft_s"]} s；状态：{row["status"]}。\n\n原始结果：`{prior["source_artifact"]}`。\n',encoding='utf-8')
        else:
            row=run_case(case['id'],config,profile,afd,directory);cache[key]=row
        rows.append(row)
        save(directory/'summary.json',{'rows':rows,'complete':False,'source_sha256':initial})
        save(directory/'component-ledgers.json',[{'scope':'prefill','case_id':r['case_id'],
            'architecture_id':r['architecture_id'],'variant':'baseline','metric':'ttft','record':r['latency_breakdown'],
            'source_artifact':r['source_artifact'],'unavailable_reason':None if r['ttft_s'] is not None else 'Prefill capacity admission failed'} for r in rows])
    save(directory/'summary.json',{'rows':rows,'complete':len(rows)==20,'source_sha256':initial,
        'strategy':'streamed-communication-prefill-template','peer_staging_bytes':STAGING_BYTES})
    fields=['case_id','architecture_id','status','ttft_s','max_hbm_bytes','source_artifact']
    with (directory/'summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({'complete':len(rows)==20,'rows':len(rows),'feasible':sum(r['status']=='feasible' for r in rows)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

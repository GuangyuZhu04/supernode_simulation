"""Decode model-scaling control with fixed POD size and fixed EP=128.

Preserves each original model/profile/workload/hardware definition. Only EP
and its implied TP change. The four 2.8T inputs already use EP128 and can be
reused exactly; the other eight cases execute the existing frozen event graph.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import window_breakdown,combine_breakdowns
from supernode_sim.report import write_latency_csv,latency_markdown
from scripts.scaling_study_graph import build_layer,attribution_graph

COMM={'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
MOE={'experts','dispatch','combine','tp_ffn','afd_additional_partial_merge'}
FILES=['scripts/scaling_fixed_ep.py','scripts/scaling_study_graph.py',
    'scripts/scaling_kimi_attention.py','scripts/scaling_v41_attention.py',
    'supernode_sim/config.py','supernode_sim/core.py','supernode_sim/engine.py',
    'supernode_sim/topology.py','supernode_sim/inference.py','supernode_sim/moe_wire.py',
    'supernode_sim/latency_breakdown.py','supernode_sim/operator_costs.py']


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def fingerprints():return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}


def execute(config,profile,folder):
    records=[];kinds={};startup=defaultdict(float);traffic=defaultdict(float)
    peaks={};status='feasible';diagnostics=[]
    for kind,count in profile['counts'].items():
        started=time.perf_counter()
        graph=build_layer(config,phase='decode',attention_profile=profile,attention_kind=kind)
        sim=simulate(graph,retain_event_payload=False)
        rec=window_breakdown(attribution_graph(graph),sim,'infer.d0.token1')
        if sim['status']!='feasible':status=sim['status'];diagnostics+=sim['diagnostics']
        if rec:records.append(combine_breakdowns([rec]*count,divisor=1))
        events={e.id:e for e in graph.events};kind_startup=defaultdict(float)
        for trace in sim['trace']:
            event=events[trace['id']];op=event.metadata.get('operation') or event.metadata.get('kernel')
            if op in COMM:kind_startup[op]+=min(trace['duration_s'],event.startup_s)*count*1000
        for op,value in kind_startup.items():startup[op]+=value
        for op,value in graph.metadata['study']['per_semantic_bytes_total'].items():traffic[op]+=value*count
        for pool,value in sim['memory_peaks_bytes'].items():peaks[pool]=max(peaks.get(pool,0),value)
        kinds[kind]={'count':count,'status':sim['status'],'layer_s':sim['duration_s'],
            'total_s':sim['duration_s']*count if rec else None,'startup_by_operator_ms':dict(kind_startup),
            'latency_breakdown':rec,'event_count':len(graph.events)}
        trace_path=folder/f'{kind}.representative-trace.json.gz'
        with gzip.open(trace_path,'wb',compresslevel=1) as f:
            f.write(json.dumps({'scope':'One representative kind layer with whole-model state resident; times multiply by recorded count',
                'kind':kind,'count':count,'trace':sim['trace'],'status':sim['status'],
                'layer_duration_s':sim['duration_s'],'diagnostics':sim['diagnostics']},
                ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        print(json.dumps({'kind':kind,'count':count,'wall_s':time.perf_counter()-started,'status':sim['status']}),flush=True)
        del graph,sim;gc.collect()
    return {'capacity_status':status,'latency_breakdown':combine_breakdowns(records,divisor=1) if status=='feasible' else None,
        'startup_by_operator_ms':dict(startup),'transport_by_semantic_bytes':dict(traffic),
        'memory_peaks_bytes':peaks,'attention_kinds':kinds,'diagnostics':diagnostics}


def split(record,startup):
    ops=record['operator_breakdown']['operators'];window=record['total_s']*1000
    comm=sum(o['total_s'] for o in ops if o['operator'] in COMM)*1000
    compute=sum(o['components_s'].get('compute_s',0) for o in ops if o['operator'] not in COMM)*1000
    memory=sum(o['components_s'].get('memory_access_s',0) for o in ops if o['operator'] not in COMM)*1000
    other=window-comm-compute-memory
    result={'tpot_ms':window,'communication_ms':comm,'comm_total_ms':comm,
        'communication_startup_ms':sum(startup.values()),'communication_transfer_ms':comm-sum(startup.values()),
        'compute_ms':compute,'memory_ms':memory,'other_ms':max(0.,other)}
    assert math.isclose(comm+compute+memory+max(0.,other),window,abs_tol=1e-8)
    for k in ('communication','communication_startup','communication_transfer','compute','memory','other'):
        result[k+'_pct']=100*result[k+'_ms']/window
    result.update(startup_ms=result['communication_startup_ms'],transfer_ms=result['communication_transfer_ms'],
                  startup_pct=result['communication_startup_pct'],transfer_pct=result['communication_transfer_pct'])
    components=[]
    op_totals=defaultdict(float)
    for o in ops:op_totals[o['operator']]+=o['total_s']*1000
    for name in ('attention','moe'):
        selected=[o for o in ops if (o['operator'] in MOE)==(name=='moe')]
        total=sum(o['total_s'] for o in selected)*1000
        c=sum(o['total_s'] for o in selected if o['operator'] in COMM)*1000
        f=sum(o['components_s'].get('compute_s',0) for o in selected if o['operator'] not in COMM)*1000
        m=sum(o['components_s'].get('memory_access_s',0) for o in selected if o['operator'] not in COMM)*1000
        s=sum(startup.get(o['operator'],0)*o['total_s']*1000/op_totals[o['operator']] for o in selected if o['operator'] in COMM)
        row={'component':name,'total_ms':total,'window_share_pct':100*total/window,
             'compute_ms':f,'communication_ms':c,'memory_ms':m,'other_ms':max(0.,total-c-f-m),
             'communication_startup_ms':s,'communication_transfer_ms':c-s}
        for field in ('compute','communication','memory','other'):row[field+'_pct']=100*row[field+'_ms']/total if total else None
        components.append(row)
        result[name+'_ms']=total;result[name+'_pct']=100*total/window
    result['component_breakdown']=components
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();study=args.study_root;out=args.output or study/'fixed-ep-128'
    out.mkdir(parents=True,exist_ok=True);frozen=fingerprints();gc.disable()
    manifest=load(study/'manifest.json')
    cases=[r for r in manifest['cases'] if r['group']=='model_scaling'];assert len(cases)==12
    comm=load(study/'communication_breakdown.json')['rows']
    startup_by_case=defaultdict(lambda:defaultdict(float))
    for r in comm:
        if r['scope']=='main' and r['metric'].lower()=='tpot':startup_by_case[r['case_id']][r['operator']]+=r['startup_ms']
    rows=[];checks=[]
    for item in cases:
        assert fingerprints()==frozen,'Numerical files changed during execution'
        original=load(study/item['config_path']);raw=load(study/item['config_path'])
        raw['execution']['ep']=128;raw['execution']['tp']=raw['hardware']['npu_count']//128
        config=Config.model_validate(raw);profile=load(study/item['attention_profile_path'])
        unchanged=json.loads(json.dumps(raw));unchanged['execution']['ep']=original['execution']['ep'];unchanged['execution']['tp']=original['execution']['tp']
        assert unchanged==original,'A field other than EP/TP was changed'
        case_id=item['id']+'-fixed-ep128';folder=out/'cases'/case_id;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'config.json',raw);save(folder/'attention-profile.json',profile)
        started=time.perf_counter()
        if original['execution']['ep']==128:
            base=load(study/item['result_path']);assert raw==original
            asset={'capacity_status':base['simulation']['status'],'latency_breakdown':base['metrics']['latency_breakdown']['tpot'],
                'startup_by_operator_ms':dict(startup_by_case[item['id']]),
                'transport_by_semantic_bytes':base['study']['decode']['communication']['payload_by_semantic_bytes'],
                'memory_peaks_bytes':base['simulation']['memory_peaks_bytes'],
                'attention_kinds':base['study']['decode'].get('attention_kinds',{}),'diagnostics':base['simulation']['diagnostics']}
            execution='reused_identical_original_config_and_executed_ledger'
        else:
            asset=execute(config,profile,folder);execution='executed_frozen_event_engine'
        record=asset['latency_breakdown'];assert record is not None,'Keep any actual failure explicitly; do not synthesize a duration'
        values=split(record,asset['startup_by_operator_ms']);m=raw['model'];h=raw['hardware']
        row={'case_id':case_id,'original_case_id':item['id'],'architecture_id':item['architecture_id'],
            'hardware':item['hardware_label'],'model_T':m['total_params']/1e12,'total_params':m['total_params'],
            'active_params':m['active_params'],'top_k':m['top_k'],'num_experts':m['num_experts'],
            'batch_size':raw['workload']['batch_size'],'input_length':raw['workload']['input_length'],
            'cards':h['npu_count'],'ep':128,'tp':raw['execution']['tp'],'original_ep':original['execution']['ep'],
            'original_tp':original['execution']['tp'],'target_tpot_ms':5.,**values,
            'status':'feasible' if values['tpot_ms']<=5 else 'slo_failed','capacity_status':asset['capacity_status'],
            'peak_hbm_GB':max((v for k,v in asset['memory_peaks_bytes'].items() if k.endswith('.hbm')),default=0)/1e9,
            'startup_by_operator_ms':asset['startup_by_operator_ms'],'transport_by_semantic_bytes':asset['transport_by_semantic_bytes'],
            'latency_breakdown':record,'attention_kinds':asset['attention_kinds'],'diagnostics':asset['diagnostics'],
            'execution':execution,'original_result_artifact':item['result_path'],'source_sha256':frozen,
            'original_config_sha256':digest(original),'config_sha256':digest(raw),'attention_profile_sha256':digest(profile),
            'changed_fields':[] if raw==original else ['execution.ep','execution.tp'],
            'wall_s':time.perf_counter()-started,'config_path':str((folder/'config.json').relative_to(out)).replace('\\','/'),
            'result_path':str((folder/'result.json').relative_to(out)).replace('\\','/')}
        save(folder/'result.json',row)
        result={'status':row['status'],'config':raw,'metrics':{'tpot_s':row['tpot_ms']/1000,
            'latency_breakdown':{'ttft':None,'tpot':record,'ttft_unavailable_reason':'Decode-only fixed-EP control'}}}
        write_latency_csv(result,folder)
        lines=['# 固定卡数、固定EP128的模型扩展控制实验','',
            f"{item['architecture_id']} / {item['hardware_label']} / {m['total_params']/1e12:g}T：EP128、TP{raw['execution']['tp']}。",
            f"TPOT {row['tpot_ms']:.6f} ms；通信占比 {row['communication_pct']:.6f}%；访存占比 {row['memory_pct']:.6f}%。",'',
            '仅EP及由卡数决定的TP变化，模型参数、top-k、专家数、结构模板和硬件/精度保持原输入。Attention/KV模板、均匀预算分片及未实机校准的限制继续适用。',
            *latency_markdown(result['metrics'])]
        (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        rows.append(row);save(out/'summary.json',{'complete':False,'rows':rows,'source_sha256':frozen})
        checks.append({'case_id':case_id,'only_EP_TP_changed':True,'source_consistent':True,
            'additive_latency_and_components':True,'execution':execution})
        print(json.dumps({k:row[k] for k in ('case_id','tpot_ms','communication_pct','memory_pct','wall_s','execution')}),flush=True)
    for arch in ('kimi-k3','deepseek-v41'):
        for hw in ('960DT','970DT'):
            group=[r for r in rows if r['architecture_id']==arch and r['hardware']==hw]
            assert len(group)==3
            assert max(r['communication_startup_ms'] for r in group)-min(r['communication_startup_ms'] for r in group)<1e-8
    verification={'status':'passed','case_count':12,'executed_new_case_count':8,'reused_identical_case_count':4,
        'same_hardware_EP_and_TP_within_each_three_model_group':True,'network_startup_constant_within_each_group':True,'checks':checks}
    save(out/'summary.json',{'complete':True,'rows':rows,'source_sha256':frozen,'verification':verification,
        'scope':'Fixed physical POD and fixed EP128; TP32 on4096 cards or TP96 on12288 cards. Three original user model scales retain their original parameter budgets, top-k, expert count, hidden widths and Attention/KV template.',
        'interpretation':'Communication is whole communication-operator critical-path time; compute/memory are noncommunication limiting-resource attribution. Startup and transfer are nested communication subitems.'})
    save(out/'verification.json',verification)
    fields=['case_id','architecture_id','hardware','model_T','total_params','active_params','top_k','num_experts','cards','ep','tp',
        'tpot_ms','target_tpot_ms','status','communication_ms','communication_pct','communication_startup_ms','communication_startup_pct',
        'communication_transfer_ms','communication_transfer_pct','startup_ms','startup_pct','transfer_ms','transfer_pct',
        'compute_ms','compute_pct','memory_ms','memory_pct','other_ms','other_pct',
        'attention_ms','attention_pct','moe_ms','moe_pct','peak_hbm_GB','execution','config_path','result_path']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({'complete':True,'rows':len(rows)}),flush=True)


if __name__=='__main__':main()

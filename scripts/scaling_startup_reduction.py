"""Run the frozen scaling graphs with communication startup at 50% / 30%.

This is a communication-runtime sensitivity experiment, not evidence that a
local megakernel can shorten physical links or collective synchronization.
All physical resource work, traffic, allocations and dependencies are kept.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.scaling_study_graph import build_layer as build_main, attribution_graph
from scripts.scaling_afd_ratios import build_layer as build_ratio, CANDIDATES
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import window_breakdown, combine_breakdowns
from supernode_sim.report import write_latency_csv, latency_markdown

COMM = {'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
MOE = {'experts','dispatch','combine','tp_ffn','afd_additional_partial_merge'}
ALPHAS = (.5,.3)
SOURCE_FILES = ['scripts/scaling_startup_reduction.py','scripts/scaling_study_graph.py',
    'scripts/scaling_afd_ratios.py','scripts/scaling_kimi_attention.py','scripts/scaling_v41_attention.py',
    'supernode_sim/core.py','supernode_sim/engine.py','supernode_sim/topology.py',
    'supernode_sim/latency_breakdown.py','supernode_sim/operator_costs.py']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def identity(event):
    return event.metadata.get('operation') or event.metadata.get('kernel') or ''


def invariant(graph):
    """Fingerprint everything the intervention is forbidden to change."""
    digest=hashlib.sha256()
    for event in graph.events:
        value=(event.id,event.deps,event.work,event.slots,
               [(a.id,a.pool,a.size_bytes,a.category,a.dtype) for a in event.allocs],event.releases)
        digest.update(json.dumps(value,sort_keys=True).encode())
    for allocation in graph.initial_allocations:
        digest.update(repr((allocation.id,allocation.pool,allocation.size_bytes,allocation.category,allocation.dtype)).encode())
    digest.update(json.dumps(graph.memory_caps,sort_keys=True).encode())
    digest.update(json.dumps({k:(r.capacity,r.unit,r.kind) for k,r in graph.resources.items()},sort_keys=True).encode())
    return digest.hexdigest()


def scale_startup(graph,factor):
    changed=0
    for event in graph.events:
        if identity(event) not in COMM:
            continue
        if event.startup_s:
            changed+=1
        event.startup_s*=factor
        if 'startup_s_by_domain' in event.metadata:
            event.metadata['startup_s_by_domain']={k:v*factor for k,v in event.metadata['startup_s_by_domain'].items()}
        representative=event.metadata.get('representative_flow')
        if representative and 'startup_s_by_domain' in representative['metadata']:
            representative['metadata']['startup_s_by_domain']={k:v*factor for k,v in representative['metadata']['startup_s_by_domain'].items()}
    return changed


def run_case(config,profile,*,case_id,scope,target_ms,base_seconds,base_startup_ms,
             afd=None,tp_a=None,tp_f=None):
    records={a:[] for a in ALPHAS}; details={a:{} for a in ALPHAS}
    startup={a:defaultdict(float) for a in ALPHAS}
    peak={a:0. for a in ALPHAS};diagnostics={a:[] for a in ALPHAS}
    kernel_startup=0.;kernel_events=0;graph_checks=[]
    for kind,count in profile['counts'].items():
        if scope=='main':
            graph=build_main(config,phase='decode',afd=afd,attention_profile=profile,attention_kind=kind)
            terminal='infer.d0.token1'
        else:
            graph,_,_=build_ratio(config,profile,kind,tp_a,tp_f)
            terminal='done'
        original=invariant(graph)
        kernel_startup+=count*sum(e.startup_s for e in graph.events if identity(e) not in COMM)
        kernel_events+=count*sum(bool(e.work) for e in graph.events if identity(e) not in COMM)
        previous=1.
        for alpha in ALPHAS:
            scale_startup(graph,alpha/previous);previous=alpha
            assert invariant(graph)==original,'Physical work, memory or dependencies changed'
            simulation=simulate(graph,retain_event_payload=False)
            rec=window_breakdown(attribution_graph(graph) if scope=='main' else graph,simulation,terminal)
            if rec:
                records[alpha].append(combine_breakdowns([rec]*count,divisor=1))
            diagnostics[alpha]+=simulation['diagnostics']
            peak[alpha]=max(peak[alpha],max((v for k,v in simulation['memory_peaks_bytes'].items() if k.endswith('.hbm')),default=0))
            events={e.id:e for e in graph.events}
            # Uniform coalesced network phase has one actual critical event;
            # there is no sum across concurrent ranks or repeated flow copies.
            for row in simulation['trace']:
                event=events[row['id']]
                op=identity(event)
                if op in COMM:
                    startup[alpha][op]+=min(row['duration_s'],event.startup_s)*count*1000
            details[alpha][kind]={'count':count,'status':simulation['status'],
                'layer_seconds':simulation['duration_s'],'record':rec,
                'physical_invariant_sha256':original,
                'event_count':len(graph.events),'completed_events':simulation['completed_events']}
            graph_checks.append({'kind':kind,'alpha':alpha,'physical_invariant':True})
        del graph,simulation
        gc.collect()
    result=[]
    for alpha in ALPHAS:
        feasible=all(x['status']=='feasible' for x in details[alpha].values())
        record=combine_breakdowns(records[alpha],divisor=1) if feasible else None
        seconds=record['total_s'] if record else None
        operators=record['operator_breakdown']['operators'] if record else []
        communication=sum(o['total_s'] for o in operators if o['operator'] in COMM)*1000 if record else None
        startup_ms=sum(startup[alpha].values())
        expected=base_seconds*1000-(1-alpha)*base_startup_ms
        assert seconds is not None and math.isclose(seconds*1000,expected,rel_tol=2e-10,abs_tol=1e-8),(case_id,alpha,seconds,expected)
        assert math.isclose(startup_ms,alpha*base_startup_ms,rel_tol=2e-10,abs_tol=1e-8),(case_id,alpha,startup_ms,base_startup_ms)
        assert math.isclose(sum(o['total_s'] for o in operators),seconds,rel_tol=1e-10)
        attention=sum(o['total_s'] for o in operators if o['operator'] not in MOE)*1000
        moe=sum(o['total_s'] for o in operators if o['operator'] in MOE)*1000
        result.append({'scope':scope,'case_id':case_id,'architecture_id':profile['architecture_id'],
            'startup_remaining_fraction':alpha,'variant':f'communication_startup_{round(alpha*100)}pct',
            'tp_a':tp_a,'tp_f':tp_f,'ratio':None if tp_a is None else f'{tp_a}:{tp_f}',
            'target_tpot_ms':target_ms,'tpot_ms':seconds*1000,'baseline_tpot_ms':base_seconds*1000,
            'target_passed':seconds*1000<=target_ms*(1+1e-12),'capacity_status':'feasible' if feasible else 'infeasible',
            'status':'feasible' if seconds*1000<=target_ms*(1+1e-12) else 'slo_failed',
            'communication_ms':communication,'communication_startup_ms':startup_ms,
            'communication_transfer_ms':communication-startup_ms if communication is not None else None,
            'communication_startup_by_operator_ms':dict(startup[alpha]),
            'attention_side_ms':attention,'moe_side_ms':moe,'peak_hbm_GB':peak[alpha]/1e9,
            'modeled_noncommunication_startup_ms':kernel_startup*1000,
            'modeled_noncommunication_work_event_count':kernel_events,
            'latency_breakdown':record,'attention_kinds':details[alpha],'diagnostics':diagnostics[alpha],
            'verification':{'physical_work_memory_dependencies_preserved':True,
                'independent_saved_startup_formula_tpot_ms':expected,'formula_agrees_with_event_execution':True,
                'stage_records_and_operators_additive':True},
            'scope_note':'Communication-event startup sensitivity only; not a native local megakernel transformation or measured speedup.'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--scope',choices=('all','main','ratio'),default='all')
    args=parser.parse_args(); root=args.study_root
    out=args.output or root/'sp-megakernel/startup-reduction'
    hashes=source_hashes();manifest=load(root/'manifest.json')
    split=load(root/'communication_breakdown.json')['totals']
    main_split={r['case_id']:r for r in split if r['scope']=='main'}
    ratio_split={(r['architecture_id'],r['tp_a'],r['tp_f']):r for r in split if r['scope']=='ratio' and r['variant']=='baseline'}
    ratio_original={(r['architecture_id'],r['tp_a'],r['tp_f']):r for r in load(root/'ratio-search/summary.json')['rows'] if r['variant']=='baseline'}
    tasks=[]
    if args.scope in ('main','all'):
        for item in manifest['cases']:
            config=load(root/item['config_path']);Config.model_validate(config)
            profile=load(root/item['attention_profile_path']);afd=load(root/item['afd_path']) if item.get('afd_path') else None
            original=main_split[item['id']]
            tasks.append((config,profile,{'case_id':item['id'],'scope':'main','target_ms':item['target_tpot_ms'],
                'base_seconds':original['tpot_ms']/1000,'base_startup_ms':original['startup_ms'],'afd':afd}))
    if args.scope in ('ratio','all'):
        for item in (r for r in manifest['cases'] if r.get('afd_path')):
            config=load(root/item['config_path']);profile=load(root/item['attention_profile_path'])
            for a,f in CANDIDATES:
                original=ratio_original[profile['architecture_id'],a,f]
                startup=ratio_split[profile['architecture_id'],a,f]['startup_ms']
                tasks.append((config,profile,{'case_id':f'{profile["architecture_id"]}-a{a}-f{f}',
                    'scope':'ratio','target_ms':1.,'base_seconds':original['tpot_ms']/1000,
                    'base_startup_ms':startup,'tp_a':a,'tp_f':f}))
    out.mkdir(parents=True,exist_ok=True);gc.disable();rows=[]
    for config,profile,kwargs in tasks:
        assert source_hashes()==hashes,'Source changed while experiment runs'
        start=time.perf_counter()
        task_rows=run_case(config,profile,**kwargs)
        for row in task_rows:
            name=row['case_id']+'-'+str(round(row['startup_remaining_fraction']*100))+'pct'
            folder=out/'cases'/name
            save(folder/'result.json',row)
            save(folder/'config.json',{'native_budget_config':config,'attention_profile':profile,
                'intervention':{'startup_remaining_fraction':row['startup_remaining_fraction'],
                    'scope':'communication_event_startup_only'},'layout':kwargs,'source_sha256':hashes})
            standard={'status':row['status'],'config':config,'metrics':{'ttft_s':None,'tpot_s':row['tpot_ms']/1000,
                'latency_breakdown':{'ttft':None,'tpot':row['latency_breakdown'],'ttft_unavailable_reason':'TPOT-only startup sensitivity'}}}
            write_latency_csv(standard,folder)
            (folder/'report.md').write_text('\n'.join(['# 通信启动缩减仿真','',
                f"启动剩余{row['startup_remaining_fraction']:.0%}，TPOT={row['tpot_ms']:.6f} ms，目标={row['target_tpot_ms']} ms，状态={row['status']}。",'',
                '本实验仅缩减通信事件启动，保持物理工作量、字节、内存和依赖，不宣称局部megakernel可消除物理链路时延。',
                *latency_markdown(standard['metrics'])]),encoding='utf-8')
            row['result_path']=str((folder/'result.json').relative_to(out)).replace('\\','/')
            row['config_path']=str((folder/'config.json').relative_to(out)).replace('\\','/')
            rows.append(row)
        save(out/'summary.json',{'complete':False,'rows':rows,'source_sha256':hashes})
        print(json.dumps({'case':kwargs['case_id'],'scope':kwargs['scope'],'elapsed_s':time.perf_counter()-start,
            'results':[{k:r[k] for k in ('startup_remaining_fraction','tpot_ms','target_passed')} for r in task_rows]}),flush=True)
    best={}
    for arch in ('kimi-k3','deepseek-v41'):
        best[arch]={}
        for a in ALPHAS:
            valid=[r for r in rows if r['scope']=='ratio' and r['architecture_id']==arch and r['startup_remaining_fraction']==a and r['capacity_status']=='feasible']
            if valid:
                b=min(valid,key=lambda r:r['tpot_ms'])
                best[arch][str(a)]={k:v for k,v in b.items() if k not in ('latency_breakdown','attention_kinds','diagnostics')}
    public=[{k:v for k,v in r.items() if k not in ('latency_breakdown','attention_kinds','diagnostics')} for r in rows]
    save(out/'summary.json',{'complete':True,'rows':rows,'best_by_architecture':best,'source_sha256':hashes,
        'kernel_launch_audit':{'all_noncommunication_modeled_startup_is_zero':all(r['modeled_noncommunication_startup_ms']==0 for r in rows),
            'interpretation':'Previous baseline omitted explicit kernel launch costs; strict local megakernel launch savings cannot be quantified from these inputs. Network-startup reductions are a separate hypothetical communication-runtime intervention.'},
        'verification':{'status':'passed','row_count':len(rows),'all_event_executions_match_saved_startup_crosscheck':True,
            'all_physical_invariants_preserved':True}})
    fields=['scope','case_id','architecture_id','startup_remaining_fraction','tp_a','tp_f','ratio','target_tpot_ms',
        'baseline_tpot_ms','tpot_ms','target_passed','status','communication_ms','communication_startup_ms',
        'communication_transfer_ms','attention_side_ms','moe_side_ms','peak_hbm_GB','modeled_noncommunication_startup_ms','result_path','config_path']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in public)
    save(out/'verification.json',{'status':'passed','row_count':len(rows),'source_sha256':hashes,
        'checks':['Physical work, dependencies, allocations, capacities unchanged in each graph',
                  'Event simulation agrees with independent saved-startup crosscheck',
                  'Communication startup scales exactly; latency/operator sums preserved',
                  'No modeled noncommunication kernel startup was silently rescaled']})
    print(json.dumps({'complete':True,'rows':len(rows),'passed_main':sum(r['target_passed'] for r in rows if r['scope']=='main'),
        'passed_ratio':sum(r['target_passed'] for r in rows if r['scope']=='ratio')}),flush=True)


if __name__=='__main__':
    main()

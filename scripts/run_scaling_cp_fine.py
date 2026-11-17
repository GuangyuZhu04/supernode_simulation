"""Execute fine operator graphs for the preserved SP/CP study."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.config import load_config
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import combine_breakdowns,window_breakdown
from supernode_sim.report import write_latency_csv,latency_markdown
from scripts.scaling_cp_fine_ops import build_fine_cp_layer

STUDY=ROOT/'outputs/scaling-10case-20261113'
FILES=['scripts/run_scaling_cp_fine.py','scripts/scaling_cp_fine_ops.py','scripts/cp_fine_trace_specs.py',
       'scripts/scaling_context_parallel.py','scripts/scaling_kimi_attention.py','scripts/scaling_v41_attention.py',
       'supernode_sim/config.py','supernode_sim/core.py','supernode_sim/engine.py','supernode_sim/topology.py',
       'supernode_sim/operator_trace.py','supernode_sim/latency_breakdown.py','supernode_sim/operator_costs.py']


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def fingerprint():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}


def export_specs(path,graph):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=['event','parent','component','kernel','original_kernel','dtype','flops','hbm_bytes','startup_us','matmul','implementation']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for event in graph.events:
            if not event.metadata.get('kernel') or not event.work:continue
            compute=next((k for k in event.work if '.compute' in k),None)
            writer.writerow({'event':event.id,'parent':event.metadata.get('trace_parent_event',''),
                'component':event.metadata.get('component','Unassigned'),'kernel':event.metadata['kernel'],
                'original_kernel':event.metadata.get('original_kernel',''),
                'dtype':'fp32' if compute and compute.endswith('.fp32') else 'fp8',
                'flops':event.work.get(compute,0),'hbm_bytes':sum(v for k,v in event.work.items() if k.endswith('.hbm')),
                'startup_us':event.startup_s*1e6,'matmul':json.dumps(event.metadata.get('matmul'),ensure_ascii=False),
                'implementation':json.dumps(event.metadata.get('implementation',{}),ensure_ascii=False)})


def execute(config,profile,base,launch_us,out,hashes,keep_evidence):
    alpha=base['startup_factor'];case_id=base['case_id']
    identifier=f'{case_id}-net{alpha*100:g}-kernel{launch_us:g}us'
    directory=out/'cases'/identifier
    directory.mkdir(parents=True,exist_ok=True)
    save(directory/'config.json',{'native_budget_config':config.model_dump(),'attention_profile':profile,
        'fine_cp_sidecar':{'network_startup_factor':alpha,'kernel_launch_us':launch_us,
                           'execution':'use scripts/run_scaling_cp_fine.py; native Config alone does not express CP'},
        'source_sha256':hashes})
    records=[];kinds={};peaks={};traffic=defaultdict(float);network_startups=defaultdict(float)
    started=time.perf_counter()
    for kind,count in profile['counts'].items():
        graph,audit=build_fine_cp_layer(config,profile,kind,alpha=alpha,launch_s=launch_us*1e-6)
        trace_config=audit.pop('trace_config',None)
        if trace_config is not None:save(directory/f'{kind}.kernel-traces.json',trace_config)
        simulation=simulate(graph,retain_event_payload=False)
        record=window_breakdown(graph,simulation,'infer.d0.token1') if simulation['status']=='feasible' else None
        if record:records.extend([record]*count)
        for key,value in simulation['memory_peaks_bytes'].items():peaks[key]=max(peaks.get(key,0),value)
        communications={event.metadata.get('operation') for event in graph.events if event.metadata.get('payload_bytes')}
        for event in graph.events:
            if event.metadata.get('payload_bytes'):
                network_startups[event.metadata['operation']]+=event.startup_s*count*1000
        for name,value in graph.metadata['cp_study']['transport_by_semantic_bytes'].items():traffic[name]+=value*count
        kinds[kind]={'layers':count,'status':simulation['status'],'layer_time_s':simulation['duration_s'],
            'layer_breakdown':record,'fine_audit':audit,'native_operator_trace':graph.metadata.get('operator_trace'),
            'diagnostics':simulation['diagnostics'],'event_count':len(graph.events),
            'communication_operators':sorted(communications),
            'kernel_trace_artifact':f'cases/{identifier}/{kind}.kernel-traces.json' if trace_config is not None else None}
        if keep_evidence:
            path=directory/f'{kind}.graph-and-simulation.json.gz'
            with gzip.open(path,'wt',encoding='utf-8',compresslevel=1) as f:
                json.dump({'scope':'one representative fine CP attention-kind layer with full-model resident state',
                    'graph':asdict(graph),'simulation':simulation},f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
            export_specs(directory/f'{kind}.operator_specs.csv',graph)
            kinds[kind]['evidence_artifact']=str(path.relative_to(out)).replace('\\','/')
        print(json.dumps({'case':identifier,'kind':kind,'status':simulation['status'],
                          'events':len(graph.events),'layer_ms':simulation['duration_s']*1000}),flush=True)
        del graph,simulation,trace_config,audit
    complete=all(row['status']=='feasible' for row in kinds.values())
    record=combine_breakdowns(records,divisor=1) if complete else None
    if record:record.update(replica=0,sequence=0,interval_count=1)
    seconds=record['total_s'] if record else None
    communication_names={name for detail in kinds.values() for name in detail['communication_operators']}
    operators=record['operator_breakdown']['operators'] if record else []
    comm_ms=sum(o['total_s']*1000 for o in operators if o['operator'] in communication_names) if record else None
    costs=record['operator_breakdown']['components_s'] if record else {}
    component_ms=defaultdict(float)
    for op in operators:
        name=str(op['component']).lower()
        component_ms['moe' if 'moe' in name else 'attention' if 'attention' in name else 'other']+=op['total_s']*1000
    row={'id':identifier,'case_id':case_id,'architecture_id':profile['architecture_id'],
        'hardware':base['hardware'],'cp':base['cp'],'ep':base['ep'],'tp':1,
        'network_startup_factor':alpha,'kernel_launch_us':launch_us,
        'coarse_tpot_ms':base['tpot_ms'],'fine_tpot_ms':seconds*1000 if seconds is not None else None,
        'delta_ms':seconds*1000-base['tpot_ms'] if seconds is not None else None,
        'target_ms':base['target_ms'],'target_satisfied':bool(seconds is not None and seconds*1000<=base['target_ms']),
        'capacity_status':'feasible' if complete else 'infeasible',
        'status':('feasible' if seconds*1000<=base['target_ms'] else 'slo_failed') if complete else 'infeasible',
        'communication_ms':comm_ms,'communication_startup_ms':sum(network_startups.values()) if record else None,
        'communication_transfer_ms':comm_ms-sum(network_startups.values()) if record else None,
        'kernel_launch_ms':costs.get('launch_s')*1000 if record else None,
        'attention_ms':component_ms['attention'] if record else None,'moe_ms':component_ms['moe'] if record else None,
        'other_component_ms':component_ms['other'] if record else None,
        'peak_hbm_GB':max(peaks.values(),default=0)/1e9,'latency_breakdown':record,
        'transport_by_semantic_bytes':dict(traffic),'network_startup_by_operator_ms':dict(network_startups),
        'kinds':kinds,'source_sha256':hashes,'wall_s':time.perf_counter()-started,
        'config_path':f'cases/{identifier}/config.json','result_path':f'cases/{identifier}/result.json',
        'fidelity':'analytical_operator_graph; parameter-budget matrices unresolved; no measured kernel/binary/tiling calibration'}
    if record:
        assert abs(sum(o['total_s'] for o in operators)-seconds)<1e-9
        assert abs(sum(component_ms.values())-seconds*1000)<1e-7
    save(directory/'result.json',row)
    wrapped={'status':row['status'],'config':config.model_dump(),'metrics':{
        'ttft_s':None,'tpot_s':seconds,'latency_breakdown':{'ttft':None,'tpot':record,
        'ttft_unavailable_reason':'SP/CP fine experiment is warm-decode-only'}}}
    write_latency_csv(wrapped,directory)
    lines=['# SP/CP细粒度算子仿真','',
        f"{case_id}；网络启动系数{alpha:g}；每kernel启动{launch_us:g}微秒（工程假设，未校准）。",'',
        f"聚合模型TPOT：{base['tpot_ms']:.6f} ms；细粒度TPOT：{row['fine_tpot_ms']} ms；状态：{row['status']}。",'',
        '使用原生kernel_traces展开并实际执行子算子。未知共享D矩阵保留预算标记；严格shape、参考代数shape和逻辑预算分别标注。'
        '细化的中间HBM访问、FP32点算与缓冲容量计入；不将更细粒度等同实机精度提升。',
        *latency_markdown(wrapped['metrics'])]
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--case',action='append')
    parser.add_argument('--skip-evidence',action='store_true')
    args=parser.parse_args();study=args.study
    out=args.output or study/'sp-megakernel/cp-fine-operators';out.mkdir(parents=True,exist_ok=True)
    baseline=read(study/'sp-megakernel/cp-owner-star/summary.json')
    manifest={r['id']:r for r in read(study/'manifest.json')['cases']}
    hashes=fingerprint();rows=[];gc.disable()
    save(out/'provenance.json',{'source_sha256':hashes,'python':platform.python_version(),'command':sys.argv,
        'scope':'12 fine zero-kernel-launch comparisons plus 4 explicit 1us-per-kernel engineering sensitivities'})
    for base in baseline['rows']:
        if args.case and base['case_id'] not in args.case:continue
        assert fingerprint()==hashes,'Fine numerical source changed during execution'
        meta=manifest[base['case_id']]
        config=load_config(study/meta['config_path']);profile=read(study/meta['attention_profile_path'])
        for launch_us in ([0.,1.] if base['startup_factor']==1. else [0.]):
            row=execute(config,profile,base,launch_us,out,hashes,
                        not args.skip_evidence and base['startup_factor']==1. and launch_us==0.)
            rows.append(row)
            save(out/'summary.json',{'complete':False,'rows':rows,'source_sha256':hashes})
            gc.collect()
    save(out/'summary.json',{'complete':len(rows)==16,'rows':rows,'source_sha256':hashes,
        'scope':'Same four CP models and network factors; analytical fine operators plus isolated kernel-launch scenario. Remaining16 model/hardware CP targets were already excluded by mandatory shared-weight read bounds and are not fabricated as executions.'})
    fields=['id','case_id','architecture_id','hardware','cp','ep','network_startup_factor','kernel_launch_us',
            'coarse_tpot_ms','fine_tpot_ms','delta_ms','target_ms','target_satisfied','capacity_status',
            'communication_ms','communication_startup_ms','communication_transfer_ms','kernel_launch_ms',
            'attention_ms','moe_ms','other_component_ms','peak_hbm_GB','config_path','result_path']
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({'complete':len(rows)==16,'cases':len(rows),'output':str(out)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

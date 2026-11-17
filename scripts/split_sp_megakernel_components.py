"""Attribute saved baseline/startup-reduction/CP windows to Attention and MoE.

This export reads saved critical-path ledgers only. It never changes a graph,
reruns numerical execution, or infers missing work from an efficiency factor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMM_ATTENTION={'tp_attention','tp_index_scores','cp_query_broadcast','cp_max_gather',
    'cp_max_broadcast','cp_numerator_denominator_gather','cp_index_candidates_gather',
    'cp_selected_indices_broadcast'}
COMM_MOE={'dispatch','combine','tp_ffn'}
MOE_KERNELS={'experts','afd_additional_partial_merge'}
ATTENTION_KERNELS={'attention_dense','kda','mla','kv_conversion','csa2_index_score',
    'csa2_index_select','csa2_topk_candidates','csa2_sparse_qk_av','csa2_softmax_rope',
    'kda_scalar','mla_scalar','swa','csa2_encoder','csa2_encoder_index','csa2_decoder',
    'csa2_decoder_full_index','csa2_decoder_candidate_index','cp_attention_core',
    'cp_cache_append','cp_exp_rescale','cp_global_max','cp_global_topk','cp_index_score',
    'cp_local_softmax_stats','cp_local_topk','cp_softmax_merge','kda_recurrent',
    'shared_projection_post','shared_projection_pre'}
COMM=COMM_ATTENTION|COMM_MOE
LABELS={'attention':'Attention侧（含共享稠密预算）','moe':'MoE侧（路由专家及其通信）','other':'其他/未归属'}
BEGIN='<!-- sp-megakernel-component-split:start -->'
END='<!-- sp-megakernel-component-split:end -->'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def owner(operator):
    if operator in COMM_ATTENTION|ATTENTION_KERNELS:return 'attention'
    if operator in COMM_MOE|MOE_KERNELS:return 'moe'
    raise ValueError('Unmapped nonzero operator: '+operator)


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:(';'.join(v) if isinstance(v,list) else v) for k,v in r.items()} for r in rows)


def collect(study):
    out=study/'sp-megakernel';manifest=load(study/'manifest.json')
    original_comm=load(study/'communication_breakdown.json')['rows']
    original_startup=defaultdict(lambda:defaultdict(float))
    for r in original_comm:
        if r['scope']=='main' and r['metric'].lower()=='tpot':
            original_startup[r['case_id']][r['operator']]+=r['startup_ms']
    windows=[];base_metadata={}
    for item in manifest['cases']:
        r=load(study/item['result_path'])
        extra={'hardware':item['hardware_label'],'model_T':r['config']['model']['total_params']/1e12}
        base_metadata[item['id']]=extra
        windows.append({'scope':'baseline','case_id':item['id'],'architecture_id':item['architecture_id'],
            **extra,
            'variant':'baseline','startup_remaining_fraction':1.,'metric':'tpot',
            'target_tpot_ms':item['target_tpot_ms'],'target_passed':r['metrics']['tpot_s']*1000<=item['target_tpot_ms'],
            'status':r['status'],'source_artifact':item['result_path'],
            'record':r['metrics']['latency_breakdown']['tpot'],'startup':dict(original_startup[item['id']]),
            'report_path':None})
    startup=load(out/'startup-reduction/summary.json');assert startup['complete']
    for r in startup['rows']:
        source='sp-megakernel/startup-reduction/'+r['result_path']
        windows.append({'scope':r['scope'],'case_id':r['case_id'],'architecture_id':r['architecture_id'],
            **base_metadata.get(r['case_id'],{'hardware':'970DT+LPU','model_T':10.}),
            'variant':r['variant'],'startup_remaining_fraction':r['startup_remaining_fraction'],'metric':'tpot',
            'target_tpot_ms':r['target_tpot_ms'],'target_passed':r['target_passed'],'status':r['status'],
            'tp_a':r.get('tp_a'),'tp_f':r.get('tp_f'),'source_artifact':source,
            'record':r['latency_breakdown'],'startup':r['communication_startup_by_operator_ms'],
            'report_path':str((study/source).with_name('report.md'))})
    cp=load(out/'cp-owner-star/summary.json');assert cp['complete']
    for r in cp['rows']:
        suffix=f"{r['case_id']}-alpha{r['startup_factor']:g}"
        source=f'sp-megakernel/cp-owner-star/cases/{suffix}.json'
        assert (study/source).exists(),source
        windows.append({'scope':'cp','case_id':r['case_id'],'architecture_id':r['architecture_id'],
            **base_metadata[r['case_id']],
            'variant':f"cp_startup_{r['startup_factor']*100:g}pct",'startup_remaining_fraction':r['startup_factor'],
            'metric':'tpot','cp':r['cp'],'tp':r['tp'],'ep':r['ep'],'target_tpot_ms':r['target_ms'],
            'target_passed':r['target_satisfied'],'status':r['status'],'source_artifact':source,
            'record':r['latency_breakdown'],'startup':r['startup_by_operator_ms'],
            'report_path':str(out/'cp-owner-star/component-reports'/(suffix+'.md'))})
    assert len(windows)==108,len(windows)
    return windows


def summarize(window):
    record=window['record'];assert record is not None,'All requested 108 windows must contain actual executed ledgers'
    total=record['total_s']*1000
    metadata={k:v for k,v in window.items() if k not in ('record','startup','report_path')}
    metadata['window_id']='/'.join(str(window[k]) for k in ('scope','case_id','variant','metric'))
    grouped={}
    fields=('total_ms','compute_ms','communication_ms','memory_ms','other_ms','communication_startup_ms',
        'communication_transfer_ms','communication_wait_other_ms','launch_ms','scheduling_wait_ms',
        'memory_wait_ms','synchronization_wait_ms','resource_contention_ms','unattributed_ms')
    for component,label in LABELS.items():
        grouped[component]={**metadata,'component':component,'component_label':label,
            'window_total_ms':total,**dict.fromkeys(fields,0.),'operators':[]}
    operator_totals=defaultdict(float)
    for op in record['operator_breakdown']['operators']:
        operator_totals[op['operator']]+=op['total_s']*1000
    evidence=[]
    for op in record['operator_breakdown']['operators']:
        name=op['operator'];amount=op['total_s']*1000
        if amount==0:continue
        component=owner(name);target=grouped[component];costs=op['components_s']
        target['total_ms']+=amount;target['operators'].append(name)
        row={'operator':name,'source_component':op['component'],'assigned_component':component,
             'communication_operator':name in COMM,'total_ms':amount,'source_components_s':costs}
        if name in COMM:
            if name not in window['startup']:raise ValueError('Missing recorded startup for '+name)
            start=window['startup'][name]*amount/operator_totals[name]
            wait=sum(costs.get(k,0) for k in ('scheduling_wait_s','memory_wait_s','synchronization_wait_s',
                    'launch_s','unattributed_service_s','other_s'))*1000
            transfer=amount-start-wait
            assert transfer>=-1e-8,(metadata['window_id'],name,amount,start,wait)
            target['communication_ms']+=amount
            target['communication_startup_ms']+=start
            target['communication_transfer_ms']+=max(0.,transfer)
            target['communication_wait_other_ms']+=wait
            row.update(communication_startup_ms=start,communication_transfer_ms=max(0.,transfer),
                       communication_wait_other_ms=wait)
        else:
            compute=costs.get('compute_s',0)*1000;memory=costs.get('memory_access_s',0)*1000
            target['compute_ms']+=compute;target['memory_ms']+=memory
            target['other_ms']+=amount-compute-memory
            for dest,source in [('launch_ms','launch_s'),('scheduling_wait_ms','scheduling_wait_s'),
                    ('memory_wait_ms','memory_wait_s'),('synchronization_wait_ms','synchronization_wait_s'),
                    ('resource_contention_ms','resource_contention_s')]:target[dest]+=costs.get(source,0)*1000
            target['unattributed_ms']+=sum(costs.get(k,0) for k in ('unattributed_service_s','other_s'))*1000
        evidence.append(row)
    rows=[]
    for r in grouped.values():
        r['operators']=sorted(set(r['operators']))
        r['window_share_pct']=100*r['total_ms']/total if total else None
        for field in ('compute','communication','memory','other'):
            r[field+'_pct']=100*r[field+'_ms']/r['total_ms'] if r['total_ms'] else None
            r[field+'_pct_of_window']=100*r[field+'_ms']/total if total else None
        r['communication_startup_pct_of_component']=100*r['communication_startup_ms']/r['total_ms'] if r['total_ms'] else None
        r['communication_transfer_pct_of_component']=100*r['communication_transfer_ms']/r['total_ms'] if r['total_ms'] else None
        r['communication_startup_pct_of_communication']=100*r['communication_startup_ms']/r['communication_ms'] if r['communication_ms'] else None
        r['communication_transfer_pct_of_communication']=100*r['communication_transfer_ms']/r['communication_ms'] if r['communication_ms'] else None
        assert math.isclose(sum(r[k+'_ms'] for k in ('compute','communication','memory','other')),r['total_ms'],abs_tol=1e-8)
        assert math.isclose(r['communication_startup_ms']+r['communication_transfer_ms']+r['communication_wait_other_ms'],r['communication_ms'],abs_tol=1e-8)
        rows.append(r)
    assert math.isclose(sum(r['total_ms'] for r in rows),total,abs_tol=1e-8)
    assert math.isclose(sum(r['communication_startup_ms'] for r in rows),sum(window['startup'].values()),abs_tol=1e-8)
    summary={**metadata,'tpot_ms':total,'window_total_ms':total,
        'communication_ms':sum(r['communication_ms'] for r in rows),
        'communication_startup_ms':sum(r['communication_startup_ms'] for r in rows),
        'communication_transfer_ms':sum(r['communication_transfer_ms'] for r in rows)}
    for r in rows:
        prefix=r['component']+'_'
        for k in ('total_ms','window_share_pct','compute_ms','communication_ms','memory_ms','other_ms',
                  'compute_pct','communication_pct','memory_pct','other_pct','communication_startup_ms','communication_transfer_ms'):
            summary[prefix+k]=r[k]
    return rows,summary,evidence


def report_block(rows):
    lines=[BEGIN,'','## Attention / MoE 时间与内部占比','',
        'Attention侧包含共享稠密参数预算；MoE侧指路由专家及其通信。外层分母为本窗口TPOT，内部分母为各组件自身时间。'
        '通信采用通信算子完整关键路径时间，启动与传输是通信内部子项，不重复相加。','',
        '| 组件 | 时间 ms | 占TPOT % | 计算 % | 通信 % | 访存 % | 其他 % |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[r[k] for k in ('total_ms','window_share_pct','compute_pct','communication_pct','memory_pct','other_pct')]
        lines.append('| '+r['component_label']+' | '+' | '.join('—' if v is None else f'{v:.6f}' for v in values)+' |')
    lines+=['','| 组件通信拆解 | 通信总 ms | 启动 ms | 传输 ms | 等待/其他 ms |','|---|---:|---:|---:|---:|']
    for r in rows[:2]:
        lines.append('| '+r['component_label']+' | '+' | '.join(f'{r[k]:.6f}' for k in
            ('communication_ms','communication_startup_ms','communication_transfer_ms','communication_wait_other_ms'))+' |')
    lines+=['','计算与访存按限制资源归因；0%不表示没有对应工作量。CP查询、索引、softmax及归并通信全部归Attention。','',END]
    return '\n'.join(lines)


def append_report(path,block):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=path.read_text(encoding='utf-8') if path.exists() else '# 本轮组件分解\n'
    if BEGIN in text:
        left,rest=text.split(BEGIN,1);_,right=rest.split(END,1);text=left+block+right
    else:text=text.rstrip()+'\n\n'+block+'\n'
    path.write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    args=parser.parse_args();out=args.study_root/'sp-megakernel'
    windows=collect(args.study_root);rows=[];summaries=[];evidence={}
    for window in windows:
        group,summary,ops=summarize(window);rows.extend(group);summaries.append(summary)
        evidence[summary['window_id']]={'source_artifact':window['source_artifact'],'operators':ops,
            'recorded_startup_by_operator_ms':window['startup']}
        if window['report_path']:append_report(window['report_path'],report_block(group))
    verify={'status':'passed','window_count':len(windows),'component_row_count':len(rows),
        'scope_counts':{s:sum(w['scope']==s for w in windows) for s in ('baseline','main','ratio','cp')},
        'all_nonzero_operators_mapped':True,'outer_components_add_to_window':True,
        'inner_compute_communication_memory_other_add_to_component':True,
        'startup_transfer_wait_add_to_communication':True}
    payload={'rows':rows,'mapping':{'attention_communication':sorted(COMM_ATTENTION),
        'moe_communication':sorted(COMM_MOE),'attention_kernels':sorted(ATTENTION_KERNELS),
        'moe_kernels':sorted(MOE_KERNELS),'unknown_nonzero_operator_policy':'raise_error'},
        'operator_evidence':evidence,'verification':verify,
        'method':'Saved critical-path operator ledgers; communication whole-event time; noncommunication compute/memory limiting-resource attribution. Recorded named startup subtotals split proportionally only among rows of the same operator name.',
        'attention_scope':'Attention side contains shared dense-budget projections, KDA state work, CP queries/index/softmax/merge, and their communication; not a pure tensor-weight attribution.',
        'source_files':{'original_baseline_manifest':'../manifest.json','startup':'startup-reduction/summary.json','cp':'cp-owner-star/summary.json'},
        'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out/'component_breakdown.json',payload);write_csv(out/'component_breakdown.csv',rows)
    save(out/'window_summary.json',{'rows':summaries,'verification':verify});write_csv(out/'window_summary.csv',summaries)
    save(out/'component-verification.json',verify)
    print(json.dumps(verify))


if __name__=='__main__':main()

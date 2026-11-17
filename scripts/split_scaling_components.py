"""Split saved critical-path operator ledgers into Attention-side and MoE.

Attention-side includes the shared dense-parameter budget. This is not a
tensor-by-tensor pure-Attention attribution. Communication is a contained
subtotal; startup/transfer are its subcomponents and are never added twice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
COMM_ATTENTION={'tp_attention','tp_index_scores'}
COMM_MOE={'dispatch','combine','tp_ffn'}
MOE_KERNELS={'experts','afd_additional_partial_merge'}
ATTENTION_KERNELS={'attention_dense','kda','mla','kv_conversion','csa2_index_score','csa2_index_select','csa2_topk_candidates',
                   'csa2_sparse_qk_av','csa2_softmax_rope','kda_scalar','mla_scalar','swa',
                   'csa2_encoder','csa2_encoder_index','csa2_decoder','csa2_decoder_full_index',
                   'csa2_decoder_candidate_index'}
LABELS={'attention':'Attention侧（含共享稠密预算）','moe':'MoE','other':'其他/未归属'}
BEGIN='<!-- component-split:start -->'
END='<!-- component-split:end -->'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def owner(operator):
    if operator in COMM_MOE|MOE_KERNELS:return 'moe'
    if operator in COMM_ATTENTION|ATTENTION_KERNELS:return 'attention'
    return 'other'


def key(row):return (row['scope'],row['case_id'],row.get('variant','baseline'),row.get('metric','tpot').lower())


def summarize(ledger,comm_rows):
    record=ledger['record']
    if record is None:
        rows=[]
        fields=('total_ms','window_total_ms','window_share_pct','communication_ms','communication_startup_ms',
                'communication_transfer_ms','communication_wait_other_ms','compute_ms','memory_ms','other_ms',
                'launch_ms','scheduling_wait_ms','memory_wait_ms','synchronization_wait_ms','resource_contention_ms',
                'unattributed_ms','communication_pct','compute_pct','memory_pct','other_pct')
        for component in LABELS:
            rows.append({**{k:v for k,v in ledger.items() if k!='record'},'metric':ledger['metric'].lower(),
                         'component':component,'component_label':LABELS[component],**dict.fromkeys(fields),
                         'communication_split_available':False,'operators':[],
                         'unavailable_reason':ledger.get('unavailable_reason','window_execution_incomplete')})
        return rows,[]
    window=record['total_s']*1000
    grouped={};evidence=[]
    for component in LABELS:
        grouped[component]={
            **{k:v for k,v in ledger.items() if k!='record'},'metric':ledger['metric'].lower(),
            'component':component,'component_label':LABELS[component],
            'window_total_ms':window,'total_ms':0.,'communication_ms':0.,
            'communication_startup_ms':0.,'communication_transfer_ms':0.,
            'communication_wait_other_ms':0.,'communication_split_available':True,
            'compute_ms':0.,'memory_ms':0.,'other_ms':0.,
            'launch_ms':0.,'scheduling_wait_ms':0.,'memory_wait_ms':0.,'synchronization_wait_ms':0.,
            'resource_contention_ms':0.,'unattributed_ms':0.,'operators':[]}
    for op in record['operator_breakdown']['operators']:
        name=op['operator'];component=owner(name);target=grouped[component]
        total=op['total_s']*1000;costs=op['components_s'];target['total_ms']+=total
        target['operators'].append(name)
        communication=name in COMM_ATTENTION|COMM_MOE
        if communication:
            target['communication_ms']+=total
            split=comm_rows.get((key(ledger),name))
            if split:
                # A communication operator may occur in several component rows;
                # allocate its independently obtained startup/transfer subtotal
                # by this row's share of that same named operator.
                fraction=total/split['communication_total_ms'] if split['communication_total_ms'] else 0.
                target['communication_startup_ms']+=split['startup_ms']*fraction
                target['communication_transfer_ms']+=split['transfer_ms']*fraction
                target['communication_wait_other_ms']+=sum(split[k] for k in
                    ('scheduling_wait_ms','memory_wait_ms','synchronization_wait_ms','other_ms'))*fraction
            else:target['communication_split_available']=False
        else:
            compute=costs.get('compute_s',0)*1000;memory=costs.get('memory_access_s',0)*1000
            target['compute_ms']+=compute;target['memory_ms']+=memory
            target['other_ms']+=total-compute-memory
            for field,source in [('launch_ms','launch_s'),('scheduling_wait_ms','scheduling_wait_s'),
                                 ('memory_wait_ms','memory_wait_s'),('synchronization_wait_ms','synchronization_wait_s'),
                                 ('resource_contention_ms','resource_contention_s')]:
                target[field]+=costs.get(source,0)*1000
            target['unattributed_ms']+=sum(costs.get(k,0) for k in ('unattributed_service_s','other_s'))*1000
        evidence.append({'operator':name,'source_component':op['component'],'assigned_component':component,
                         'communication_operator':communication,'total_ms':total,'source_components_s':costs})
    rows=[]
    for row in grouped.values():
        if not row['communication_split_available']:
            for field in ('communication_startup_ms','communication_transfer_ms','communication_wait_other_ms'):row[field]=None
        row['operators']=sorted(set(row['operators']))
        row['window_share_pct']=100*row['total_ms']/window if window else None
        for field in ('communication','compute','memory','other'):
            row[field+'_pct']=100*row[field+'_ms']/row['total_ms'] if row['total_ms'] else None
        if abs(sum(row[field+'_ms'] for field in ('communication','compute','memory','other'))-row['total_ms'])>1e-8:
            raise ValueError('Component internal breakdown does not reconcile')
        if row['communication_split_available'] and abs(row['communication_startup_ms']+row['communication_transfer_ms']+
                row['communication_wait_other_ms']-row['communication_ms'])>1e-8:
            raise ValueError('Communication split does not reconcile inside component')
        rows.append(row)
    if abs(sum(r['total_ms'] for r in rows)-window)>1e-8:
        raise ValueError('Attention+MoE+Other does not reconcile to window')
    return rows,evidence


def report_block(rows):
    metric=rows[0]['metric'].upper()
    if rows[0]['window_total_ms'] is None:
        return '\n'.join([BEGIN,'','## Attention 与 MoE 时延及内部开销','',
            f"{metric} 未执行完成，组件耗时与占比均不可用，不补零。原因：{rows[0].get('unavailable_reason','unknown')}。",'',END])
    lines=[BEGIN,'','## Attention 与 MoE 时延及内部开销','',
        'Attention侧包含共享稠密参数预算，不代表纯Attention张量计时。外层占比的分母为对应TTFT/TPOT窗口；'
        '组件内部比例的分母为该组件自身时间。通信的启动、传输已包含在通信总计中，不再次相加。','',
        f'### {metric}', '',
        '| 组件 | 时间 / ms | 占窗口 / % | 通信 / ms | 计算 / ms | 访存 / ms | 其它 / ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['component_label']+' | '+' | '.join(f'{row[k]:.6f}' if row[k] is not None else '—'
            for k in ('total_ms','window_share_pct','communication_ms','compute_ms','memory_ms','other_ms'))+' |')
    moe=next(r for r in rows if r['component']=='moe')
    lines+=['', '| MoE 内部分项 | 时间 / ms | 占 MoE / % |','|---|---:|---:|']
    for field,label in [('communication','通信（含启动、传输及通信自身争用）'),('compute','计算限制服务'),
                        ('memory','访存限制服务'),('other','启动/等待/未细分等其他开销')]:
        value=moe[field+'_pct']
        lines.append(f"| {label} | {moe[field+'_ms']:.6f} | {value:.6f} |" if value is not None else f"| {label} | 0 | — |")
    lines+=['','计算和访存按限制资源归因，不把重叠服务重复相加；完整算子映射和原始来源见 component_breakdown.json。','',END]
    return '\n'.join(lines)


def append(path,block):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    old=path.read_text(encoding='utf-8') if path.exists() else '# 组件开销报告\n'
    if BEGIN in old:
        left,rest=old.split(BEGIN,1);_,right=rest.split(END,1);out=left+block+right
    else:out=old.rstrip()+'\n\n'+block+'\n'
    path.write_text(out,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--additional-ledgers',type=Path,action='append',default=[],
        help='Additional saved records with scope/case_id/architecture_id/variant/metric/record/source_artifact fields')
    parser.add_argument('--additional-communication',type=Path,action='append',default=[])
    args=parser.parse_args();directory=args.output
    ledgers=read(directory/'communication_case_ledgers.json')
    communications=read(directory/'communication_breakdown.json')['rows']
    for path in args.additional_ledgers:ledgers.extend(read(path))
    for path in args.additional_communication:communications.extend(read(path)['rows'])
    split_index={(key(r),r['operator']):r for r in communications}
    all_rows=[];evidence={}
    manifest=read(directory/'manifest.json');items={r['id']:r for r in manifest['cases']}
    ratio_best=read(directory/'ratio-search/summary.json')['best_by_architecture']
    for ledger in ledgers:
        rows,source=summarize(ledger,split_index);all_rows+=rows
        identity='/'.join(key(ledger));evidence[identity]=source
        block=report_block(rows)
        if ledger['scope']=='main':
            append((directory/items[ledger['case_id']]['result_path']).with_name('report.md'),block)
        elif ledger['scope']=='ratio':
            append(directory/'ratio-search/reports'/(ledger['case_id']+'.md'),block)
            best=ratio_best[ledger['architecture_id']]['baseline']
            best_id=f"{ledger['architecture_id']}-a{best['tp_a']}-f{best['tp_f']}-baseline"
            if ledger['case_id']==best_id:
                append(directory/'ratio-search/best-evidence'/ledger['architecture_id']/'report.md',block)
        elif ledger['scope']=='prefill':append(directory/'prefill-study/cases'/ledger['case_id']/'report.md',block)
        else:append(directory/'component-reports'/ledger['scope']/(ledger['case_id']+'-'+ledger['variant']+'-'+ledger['metric']+'.md'),block)
    payload={'rows':all_rows,'mapping':{
        'attention_communication':sorted(COMM_ATTENTION),'moe_communication':sorted(COMM_MOE),
        'attention_kernels':sorted(ATTENTION_KERNELS),'moe_kernels':sorted(MOE_KERNELS),'unmapped':'other'},
        'operator_evidence':evidence,'method':'Saved critical-path operator ledger; explicit stable identity mapping; communication operators retain their complete wall time and internal startup/transfer are subcomponents.',
        'attention_scope':'Attention-side includes the complete shared dense-parameter budget; not a tensor-specific pure Attention claim.',
        'verification':{'status':'passed','window_count':len(ledgers),'component_row_count':len(all_rows),
                        'outer_attention_plus_moe_plus_other_adds_to_window':True,'inner_costs_add_to_component':True},
        'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(directory/'component_breakdown.json',payload)
    fields=list(dict.fromkeys(k for row in all_rows for k in row))
    with (directory/'component_breakdown.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows({k:(';'.join(v) if isinstance(v,list) else v) for k,v in r.items()} for r in all_rows)
    print(json.dumps(payload['verification']))


if __name__=='__main__':main()

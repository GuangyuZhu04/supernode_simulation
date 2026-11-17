"""Auditably split saved communication into startup, transfer and waits.

Main cases use recorded representative-event startup windows. Ratio cases use
events rebuilt by their frozen graph builder, without another simulation.
Sensitivity cases retain the same graph and apply their explicit link-latency
setting to the recorded startup windows. Main-minus-zero is only a cross-check.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.scaling_afd_ratios import build_layer

COMM={'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
WAIT_FIELDS={'scheduling_wait_ms':'scheduling_wait_s','memory_wait_ms':'memory_wait_s',
             'synchronization_wait_ms':'synchronization_wait_s'}
SPLIT_FIELDS=['startup_ms','transfer_ms',*WAIT_FIELDS,'other_ms']
BEGIN='<!-- communication-split:start -->'
END='<!-- communication-split:end -->'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def csv_write(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r if k!='operator_components_s'))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v)
                          for k,v in r.items() if k in columns} for r in rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(expected):
    for name,digest in expected.items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen source changed: '+name)


def main_startups(directory,item,result):
    profile=result['study']['attention_profile']
    sums=defaultdict(float)
    durations=defaultdict(float)
    evidence=[]
    for kind,count in profile['counts'].items():
        path=(directory/item['result_path']).parent/f'decode.{kind}.representative-layer.json.gz'
        with gzip.open(path,'rt',encoding='utf-8') as f:
            saved=json.load(f)
        trace=[t for t in saved['simulation']['trace'] if t.get('metadata',{}).get('operation') in COMM]
        trace.sort(key=lambda t:(t['start_s'],t['end_s'],t['id']))
        last_end=0.
        for event in trace:
            if event['start_s']<last_end-1e-12:
                raise ValueError('Communication stages overlap; need path-specific clipping')
            last_end=event['end_s']
            operator=event['metadata']['operation']
            clipped=min(event['startup_s'],event['end_s']-event['start_s'])
            sums[operator]+=clipped*count
            durations[operator]+=(event['duration_s']+event.get('slot_wait_s',0)+event.get('memory_wait_s',0))*count
            evidence.append({'kind':kind,'repetitions':count,'event_id':event['id'],'operator':operator,
                'start_s':event['start_s'],'end_s':event['end_s'],'startup_s':event['startup_s'],
                'clipped_startup_s':clipped,'weighted_startup_s':clipped*count,
                'artifact':str(path.relative_to(directory)).replace('\\','/')})
        del saved
    ledger=defaultdict(float)
    for op in result['metrics']['latency_breakdown']['tpot']['operator_breakdown']['operators']:
        if op['operator'] in COMM:ledger[op['operator']]+=op['total_s']
    for operator,total in ledger.items():
        if abs(durations[operator]-total)>1e-10:
            raise ValueError(f"Trace/ledger mismatch {item['id']} {operator}")
    return dict(sums),evidence


def ratio_startups(config,profile,tp_a,tp_f,*,zero=False):
    sums=defaultdict(float);evidence=[]
    for kind,count in profile['counts'].items():
        graph,_,_=build_layer(config,profile,kind,tp_a,tp_f,zero_startup=zero)
        for event in graph.events:
            operator=event.metadata.get('operation')
            if operator in COMM:
                sums[operator]+=event.startup_s*count
                evidence.append({'kind':kind,'repetitions':count,'event_id':event.id,'operator':operator,
                                 'startup_s':event.startup_s,'weighted_startup_s':event.startup_s*count,
                                 'artifact':'scripts/scaling_afd_ratios.py:build_layer'})
        del graph
    return dict(sums),evidence


def split_case(scope,case_id,arch,variant,record,startups,source,method,*,extra=None,metric='TPOT'):
    grouped={}
    for op in record['operator_breakdown']['operators']:
        name=op['operator']
        if name not in COMM:continue
        target=grouped.setdefault(name,{'total_s':0.,'components_s':defaultdict(float)})
        target['total_s']+=op['total_s']
        for key,value in op['components_s'].items():target['components_s'][key]+=value
    rows=[]
    for name,op in grouped.items():
        costs=op['components_s']
        row={'scope':scope,'case_id':case_id,'architecture_id':arch,'variant':variant,'metric':metric,
             'operator':name,'tpot_ms':record['total_s']*1000 if metric=='TPOT' else None,
             'ttft_ms':record['total_s']*1000 if metric=='TTFT' else None,
             'window_total_ms':record['total_s']*1000,'communication_total_ms':op['total_s']*1000,
             'startup_ms':startups.get(name,0)*1000,
             **{field:costs.get(source_field,0)*1000 for field,source_field in WAIT_FIELDS.items()},
             'other_ms':sum(costs.get(k,0) for k in ('other_s','unattributed_service_s','launch_s'))*1000,
             'source_artifact':source,'method':method,**(extra or {})}
        row['transfer_ms']=row['communication_total_ms']-sum(row[k] for k in SPLIT_FIELDS if k!='transfer_ms')
        if row['transfer_ms']<-1e-8:
            raise ValueError(f'Negative transfer component: {case_id} {name}')
        row['transfer_ms']=max(0.,row['transfer_ms'])
        row['operator_components_s']=dict(costs)
        proportions(row)
        rows.append(row)
    total={'scope':scope,'case_id':case_id,'architecture_id':arch,'variant':variant,'metric':metric,'operator':'ALL',
           'tpot_ms':record['total_s']*1000 if metric=='TPOT' else None,
           'ttft_ms':record['total_s']*1000 if metric=='TTFT' else None,
           'window_total_ms':record['total_s']*1000,'communication_total_ms':sum(r['communication_total_ms'] for r in rows),
           **{key:sum(r[key] for r in rows) for key in SPLIT_FIELDS},
           'source_artifact':source,'method':method,**(extra or {})}
    proportions(total)
    return rows,total


def proportions(row):
    total=row['communication_total_ms']
    window=row.get('window_total_ms',row['tpot_ms'])
    for field in SPLIT_FIELDS:
        row[field.removesuffix('_ms')+'_pct_of_comm']=100*row[field]/total if total else None
        row[field.removesuffix('_ms')+'_pct_of_tpot']=100*row[field]/row['tpot_ms'] if row['tpot_ms'] else None
        row[field.removesuffix('_ms')+'_pct_of_window']=100*row[field]/window if window else None
    row['communication_pct_of_tpot']=100*total/row['tpot_ms'] if row['tpot_ms'] else None
    row['communication_pct_of_window']=100*total/window if window else None
    if abs(sum(row[k] for k in SPLIT_FIELDS)-total)>1e-8:
        raise ValueError('Communication split does not sum to total')


def report_block(rows,total):
    lines=[BEGIN,'','## 通信启动与传输拆分','',
        '单位：ms。启动取保存事件的实际启动窗口（或冻结配置重建的同一通信事件）；传输包括网络/端点访存服务及共享带宽争用。'
        '显式等待与未细分项单列，不把主基线减零启动当作拆解证据。', '',
        '| 算子 | 通信总计 | 启动 | 传输 | 调度等待 | 内存等待 | 同步等待 | 其它 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in [*rows,total]:
        lines.append('| '+row['operator']+' | '+' | '.join(f'{row[k]:.6f}' for k in ['communication_total_ms',*SPLIT_FIELDS])+' |')
    lines+=['',f"启动占通信 {total['startup_pct_of_comm']:.3f}%；传输占通信 {total['transfer_pct_of_comm']:.3f}%。",
        '', '完整数据见研究根目录 communication_breakdown.json / communication_breakdown.csv。','',END]
    return '\n'.join(lines)


def append_report(path,block):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    old=path.read_text(encoding='utf-8') if path.exists() else '# 通信拆分报告\n'
    if BEGIN in old:
        left,rest=old.split(BEGIN,1)
        _,right=rest.split(END,1)
        text=left+block+right
    else:text=old.rstrip()+'\n\n'+block+'\n'
    path.write_text(text,encoding='utf-8')


def append_prefill(directory):
    source=read(directory/'prefill-study/summary.json')
    if not source.get('complete'):
        raise ValueError('Prefill study has not finished; no partial final export')
    rows=[];totals=[];evidence={}
    for item in source['rows']:
        record=item['latency_breakdown']
        if record is None:
            total={'scope':'prefill','case_id':item['case_id'],'architecture_id':item['architecture_id'],
                'variant':'baseline','metric':'TTFT','operator':'ALL','tpot_ms':None,'ttft_ms':None,
                'window_total_ms':None,'communication_total_ms':None,**dict.fromkeys(SPLIT_FIELDS),
                'source_artifact':item['source_artifact'],'method':'unavailable_execution_incomplete',
                'unavailable_reason':item['status']}
            for field in SPLIT_FIELDS:
                for suffix in ('pct_of_comm','pct_of_tpot','pct_of_window'):
                    total[field.removesuffix('_ms')+'_'+suffix]=None
            total['communication_pct_of_tpot']=total['communication_pct_of_window']=None
            case_rows=[]
            block='\n'.join([BEGIN,'','## 通信启动与传输拆分','',
                'Prefill未完成，TTFT通信耗时与分项均不可用，不填零。','',END])
        else:
            case_rows,total=split_case('prefill',item['case_id'],item['architecture_id'],'baseline',record,
                item['startup_by_operator_s'],item['source_artifact'],
                'recorded_prefill_event_startup_weighted_by_attention_kind',metric='TTFT')
            block=report_block(case_rows,total)
        rows.extend(case_rows);totals.append(total)
        evidence[item['case_id']]={'source_artifact':item['source_artifact'],
            'startup_by_operator_s':item.get('startup_by_operator_s'),
            'communication_by_operator':item.get('communication_by_operator'),
            'status':item['status']}
        append_report(directory/'prefill-study/cases'/item['case_id']/'report.md',block)
    payload={'rows':rows,'totals':totals,'evidence':evidence,'verification':{'status':'passed',
        'case_count':len(totals),'available_count':sum(r['communication_total_ms'] is not None for r in totals)},
        'exporter_sha256':sha(Path(__file__))}
    save(directory/'prefill-study/communication_breakdown.json',payload)
    csv_write(directory/'prefill-study/communication_breakdown.csv',rows+totals)
    existing=read(directory/'communication_breakdown.json')
    existing['rows']=[r for r in existing['rows'] if r['scope']!='prefill']+rows
    existing['totals']=[r for r in existing['totals'] if r['scope']!='prefill']+totals
    existing['evidence']['prefill']=evidence
    existing['verification']['case_count']=len(existing['totals'])
    existing['verification']['prefill']=payload['verification']
    existing['prefill_exporter_sha256']=payload['exporter_sha256']
    save(directory/'communication_breakdown.json',existing)
    csv_write(directory/'communication_breakdown.csv',existing['rows']+existing['totals'])
    print(json.dumps({'stage':'prefill_appended','case_count':len(totals),
                      'all_case_count':len(existing['totals'])}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    parser.add_argument('--append-prefill',action='store_true',help='Add completed saved Prefill results without re-reading all Decode evidence')
    args=parser.parse_args();directory=args.output
    if args.append_prefill:
        append_prefill(directory)
        return
    gc.disable()
    manifest=read(directory/'manifest.json');items={i['id']:i for i in manifest['cases']}
    all_rows=[];totals=[];evidence={};main_data={};main_starts={};case_ledgers=[]
    for item in manifest['cases']:
        result=read(directory/item['result_path'])
        starts,proof=main_startups(directory,item,result)
        case_id=item['id'];main_starts[case_id]=starts;main_data[case_id]=result
        evidence['main/'+case_id]=proof
        record=result['metrics']['latency_breakdown']['tpot']
        rows,total=split_case('main',case_id,item['architecture_id'],'baseline',record,starts,item['result_path'],
                              'recorded_trace_startup_windows_weighted_by_attention_kind')
        all_rows+=rows;totals.append(total)
        append_report((directory/item['result_path']).with_name('report.md'),report_block(rows,total))
        case_ledgers.append({'scope':'main','case_id':case_id,'architecture_id':item['architecture_id'],'variant':'baseline',
                             'metric':'TPOT','record':record,'source_artifact':item['result_path']})
        print(json.dumps({'stage':'main_split','case':case_id}),flush=True)
    ratios=read(directory/'ratio-search/summary.json')
    check_sources(ratios['source_sha256'])
    ratio_cache={}
    for row in ratios['rows']:
        candidate=read(directory/'ratio-search'/row['config_path'])
        config,profile=candidate['native_budget_config'],candidate['attention_profile']
        key=(row['architecture_id'],row['tp_a'],row['tp_f'])
        if key not in ratio_cache:
            ratio_cache[key]=ratio_startups(config,profile,row['tp_a'],row['tp_f'])
        starts,proof=ratio_cache[key]
        zero=row['variant']=='zero_network_startup'
        factor=0. if zero else 1.
        starts={k:v*factor for k,v in starts.items()}
        case_id=f"{row['architecture_id']}-a{row['tp_a']}-f{row['tp_f']}-{row['variant']}"
        evidence['ratio/'+case_id]={'configuration':'ratio-search/'+row['config_path'],
            'startup_setting_factor':factor,'source_sha256':ratios['source_sha256'],'events':proof}
        source='ratio-search/'+row['result_path']
        rows,total=split_case('ratio',case_id,row['architecture_id'],row['variant'],row['latency_breakdown'],starts,source,
             'frozen_ratio_graph_event_startup_weighted_by_kind_no_resimulation',extra={'tp_a':row['tp_a'],'tp_f':row['tp_f'],'ratio':row['ratio']})
        all_rows+=rows;totals.append(total)
        append_report(directory/'ratio-search'/'reports'/(case_id+'.md'),report_block(rows,total))
        best=ratios['best_by_architecture'][row['architecture_id']]['baseline']
        if row['variant']=='baseline' and row['tp_a']==best['tp_a'] and row['tp_f']==best['tp_f']:
            append_report(directory/'ratio-search/best-evidence'/row['architecture_id']/'report.md',report_block(rows,total))
        case_ledgers.append({'scope':'ratio','case_id':case_id,'architecture_id':row['architecture_id'],'variant':row['variant'],
            'metric':'TPOT','record':row['latency_breakdown'],'source_artifact':source,'tp_a':row['tp_a'],'tp_f':row['tp_f'],'ratio':row['ratio']})
        print(json.dumps({'stage':'ratio_split','case':case_id}),flush=True)
    sensitivity=read(directory/'sensitivity.json')
    checks=[]
    for item in sensitivity:
        case_id=item['case'];base=main_data[case_id];config=read(directory/item['config_path'])
        expected=copy.deepcopy(base['config'])
        expected['hardware']['scale_up']=copy.deepcopy(config['hardware']['scale_up'])
        if expected!=config:
            raise ValueError('Sensitivity changes fields other than the declared network: '+case_id)
        original_latency=base['config']['hardware']['scale_up']['latency_s']
        new_latency=config['hardware']['scale_up']['latency_s']
        factor=new_latency/original_latency
        starts={k:v*factor for k,v in main_starts[case_id].items()}
        source='sensitivity.json#'+case_id+'/'+item['variant']
        rows,total=split_case('sensitivity',case_id,items[case_id]['architecture_id'],item['variant'],item['latency_breakdown'],starts,source,
            'recorded_main_event_startup_scaled_by_only_changed_link_latency',extra={'config_path':item['config_path']})
        all_rows+=rows;totals.append(total)
        evidence['sensitivity/'+case_id+'/'+item['variant']]={'recorded_main_trace_case':case_id,
            'config_path':item['config_path'],'original_latency_s':original_latency,'new_latency_s':new_latency,'startup_factor':factor,
            'all_nonnetwork_configuration_fields_identical':True}
        if item['variant']=='zero_network_startup':
            main_total=next(r for r in totals if r['scope']=='main' and r['case_id']==case_id)
            # Diagnostic equality only, never used to construct startup_ms.
            difference_ms=base['metrics']['tpot_s']*1000-item['tpot_s']*1000
            checks.append({'case_id':case_id,'check':'main_minus_zero_diagnostic_only',
                           'observed_difference_ms':difference_ms,'trace_startup_ms':main_total['startup_ms'],
                           'absolute_error_ms':abs(difference_ms-main_total['startup_ms'])})
        append_report(directory/'communication-reports/sensitivity'/(case_id+'-'+item['variant']+'.md'),report_block(rows,total))
        case_ledgers.append({'scope':'sensitivity','case_id':case_id,'architecture_id':items[case_id]['architecture_id'],
            'variant':item['variant'],'metric':'TPOT','record':item['latency_breakdown'],'source_artifact':source})
    if len(totals)!=84:
        raise ValueError(f'Expected 20+36+28=84 cases, observed {len(totals)}')
    payload={'rows':all_rows,'totals':totals,'evidence':evidence,
             'method_note':'Startup from recorded windows or frozen event graph; transfer is execution excluding startup and explicit waits/other, including endpoint HBM and resource contention. No main-minus-zero subtraction used for the split.',
             'verification':{'status':'passed','case_count':len(totals),'all_components_add_to_communication':True,
                             'main_trace_durations_match_saved_operator_ledger':True,'diagnostic_crosschecks':checks},
             'exporter_sha256':sha(Path(__file__))}
    save(directory/'communication_breakdown.json',payload)
    csv_write(directory/'communication_breakdown.csv',all_rows+totals)
    save(directory/'communication_case_ledgers.json',case_ledgers)
    print(json.dumps({'stage':'complete','case_count':len(totals),'operator_rows':len(all_rows)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

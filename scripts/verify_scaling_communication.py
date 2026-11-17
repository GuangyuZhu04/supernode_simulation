"""Independently verify saved startup/transfer communication decomposition."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'outputs/scaling-10case-20261113'
PARTS=('startup_ms','transfer_ms','scheduling_wait_ms','memory_wait_ms','synchronization_wait_ms','other_ms')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(left,right):
    assert math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-8),(left,right)


def startup_by_operator(config,profile,tp,latency,*,prefill=False):
    layers=profile['layers']
    output={'dispatch':layers*latency*1000,'combine':layers*latency*1000}
    if tp>1:
        output.update(tp_attention=layers*2*(tp-1)*latency*1000,
                      tp_ffn=layers*2*(tp-1)*latency*1000)
        if profile['family']=='deepseek_v41':
            tiles=0
            for kind,count in profile['counts'].items():
                if 'INDEX' not in kind:
                    continue
                ratio=2 if 'ENCODER' in kind else 1
                if prefill:
                    q=config['workload']['input_length']
                    for begin in range(0,q,256):
                        keys=(begin+min(256,q-begin))//ratio
                        if 'CANDIDATE' in kind:keys=min(keys,profile['candidate_rows'])
                        tiles+=count*math.ceil(keys/profile['score_tile_rows'])
                else:
                    keys=(config['workload']['input_length']+1)//ratio
                    if 'CANDIDATE' in kind:
                        keys=min(keys,profile['candidate_rows'])
                    tiles+=count*math.ceil(keys/profile['score_tile_rows'])
            output['tp_index_scores']=tiles*2*(tp-1)*latency*1000
    return output


def main():
    directory=DIRECTORY
    data=read(directory/'communication_breakdown.json')
    manifest=read(directory/'manifest.json')
    items={i['id']:i for i in manifest['cases']}
    grouped=defaultdict(list)
    for row in data['rows']:
        grouped[(row['scope'],row['case_id'],row['variant'],row['metric'])].append(row)
        assert row['communication_total_ms']>=0
        assert all(row[k]>=-1e-9 for k in PARTS)
        close(sum(row[k] for k in PARTS),row['communication_total_ms'])
        close(sum(row['operator_components_s'].values())*1000,row['communication_total_ms'])
        assert abs(row['operator_components_s'].get('compute_s',0))<1e-12
        close(sum(row[k.removesuffix('_ms')+'_pct_of_comm'] for k in PARTS),100)
        window=row.get('window_total_ms',row['tpot_ms'])
        if row['tpot_ms']:
            close(row['startup_pct_of_tpot'],row['startup_ms']/row['tpot_ms']*100)
            close(row['transfer_pct_of_tpot'],row['transfer_ms']/row['tpot_ms']*100)
        else:
            assert row['startup_pct_of_tpot'] is None and row['transfer_pct_of_tpot'] is None
        if 'startup_pct_of_window' in row:
            close(row['startup_pct_of_window'],row['startup_ms']/window*100)
            close(row['transfer_pct_of_window'],row['transfer_ms']/window*100)
    counts=Counter(t['scope'] for t in data['totals'])
    assert counts['main']==20 and counts['ratio']==36 and counts['sensitivity']==28,counts
    if counts['prefill']:
        assert counts['prefill']==20
    startup_checks=[]
    for total in data['totals']:
        rows=grouped[(total['scope'],total['case_id'],total['variant'],total['metric'])]
        if total['communication_total_ms'] is None:
            assert total['scope']=='prefill' and not rows and all(total[k] is None for k in PARTS)
            continue
        for field in (*PARTS,'communication_total_ms'):
            close(sum(row[field] for row in rows),total[field])
        if total['scope']=='ratio':
            case=read(directory/total['source_artifact'])
            saved=read(directory/'ratio-search'/case['config_path'])
            config,profile=saved['native_budget_config'],saved['attention_profile']
            tp=case['tp_a']
            latency=0. if case['variant']=='zero_network_startup' else config['hardware']['scale_up']['latency_s']
            evidence_key='ratio/'+total['case_id']
        else:
            item=items[total['case_id']]
            config=read(directory/(total['config_path'] if total['scope']=='sensitivity' else item['config_path']))
            profile=read(directory/item['attention_profile_path'])
            tp=read(directory/item['afd_path'])['tp'] if item.get('afd_path') else config['execution']['tp']
            latency=config['hardware']['scale_up']['latency_s']
            evidence_key=total['scope']+'/'+total['case_id']+('/'+total['variant'] if total['scope']=='sensitivity' else '')
        expected=startup_by_operator(config,profile,tp,latency,prefill=total['metric']=='TTFT')
        for row in rows:
            close(row['startup_ms'],expected.get(row['operator'],0))
        close(total['startup_ms'],sum(expected.values()))
        evidence=data['evidence'].get(evidence_key,[])
        if isinstance(evidence,list):
            evidence_sums=defaultdict(float)
            for event in evidence:
                if 'weighted_startup_s' in event:
                    close(event['weighted_startup_s'],event.get('clipped_startup_s',event['startup_s'])*event['repetitions'])
                    evidence_sums[event['operator']]+=event['weighted_startup_s']*1000
            if evidence_sums:
                for row in rows:
                    close(row['startup_ms'],evidence_sums[row['operator']])
        startup_checks.append({'scope':total['scope'],'case_id':total['case_id'],'variant':total['variant'],
                               'independent_serial_round_startup_ms':sum(expected.values()),'passed':True})
    result={'status':'passed','scope':'Saved actual-event startup windows independently checked against serial collective rounds; no subtraction of zero-startup runs used for decomposition',
            'case_counts':dict(counts),'operator_rows':len(data['rows']),
            'all_parts_sum_to_saved_communication':True,'no_compute_time_hidden_in_transfer':True,
            'startup_round_checks':startup_checks}
    path=directory/'communication-independent-verification.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','case_counts':dict(counts),'operator_rows':len(data['rows'])}))


if __name__=='__main__':
    main()

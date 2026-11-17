"""Independent 108-window Attention/MoE audit from original operator ledgers."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/scaling-10case-20261113'

MOE={'dispatch','combine','experts','tp_ffn','afd_additional_partial_merge'}
NETWORK={'dispatch','combine','tp_attention','tp_ffn','tp_index_scores',
    'cp_query_broadcast','cp_max_gather','cp_max_broadcast','cp_numerator_denominator_gather',
    'cp_index_candidates_gather','cp_selected_indices_broadcast'}
ATTENTION={'tp_attention','tp_index_scores','kda','mla','attention_dense','kv_conversion',
    'swa','csa2_encoder','csa2_encoder_index','csa2_decoder','csa2_decoder_full_index',
    'csa2_decoder_candidate_index','csa2_index_score','csa2_softmax_rope','csa2_sparse_qk_av',
    'csa2_topk_candidates','csa2_index_select','kda_scalar','mla_scalar',
    'cp_attention_core','cp_cache_append','cp_exp_rescale','cp_global_max','cp_global_topk',
    'cp_index_score','cp_local_softmax_stats','cp_local_topk','cp_softmax_merge','kda_recurrent',
    'shared_projection_post','shared_projection_pre'} | (NETWORK-MOE)


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):
    assert a is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def source_windows():
    manifest=load(STUDY/'manifest.json')
    comm=load(STUDY/'communication_breakdown.json')['rows']
    sources={};files=[]
    for item in manifest['cases']:
        path=STUDY/item['result_path'];data=load(path);files.append(path)
        startup={r['operator']:r['startup_ms'] for r in comm if r['scope']=='main' and r['case_id']==item['id'] and r['metric']=='TPOT'}
        sources['baseline',item['id'],1.]={'record':data['metrics']['latency_breakdown']['tpot'],'startup':startup,'source':str(path.relative_to(STUDY))}
    path=STUDY/'sp-megakernel/startup-reduction/summary.json';data=load(path);files.append(path)
    assert data['complete'] and len(data['rows'])==76
    for row in data['rows']:
        sources[row['scope'],row['case_id'],float(row['startup_remaining_fraction'])]={
            'record':row['latency_breakdown'],'startup':row['communication_startup_by_operator_ms'],'source':str(path.relative_to(STUDY))}
    path=STUDY/'sp-megakernel/cp-owner-star/summary.json';data=load(path);files.append(path)
    assert data['complete'] and len(data['rows'])==12
    for row in data['rows']:
        sources['cp',row['case_id'],float(row['startup_factor'])]={
            'record':row['latency_breakdown'],'startup':row['startup_by_operator_ms'],'source':str(path.relative_to(STUDY))}
    assert len(sources)==108
    return sources,files


def key(row):
    alpha=row.get('startup_remaining_fraction')
    return row['scope'],row['case_id'],float(alpha if alpha is not None else 1.)


def main():
    directory=STUDY/'sp-megakernel'
    sources,files=source_windows()
    exported=load(directory/'component_breakdown.json')
    groups=defaultdict(dict)
    for row in exported['rows']:
        k=key(row);assert row['component'] not in groups[k],k
        groups[k][row['component']]=row
    assert len(exported['rows'])==324 and len(groups)==108
    assert set(groups)==set(sources),(set(groups)-set(sources),set(sources)-set(groups))
    checks=[];cp_names=set()
    for identity,source in sources.items():
        rec=source['record'];window=rec['total_s']*1000
        expected={c:defaultdict(float) for c in ('attention','moe','other')}
        expected_ops={c:set() for c in expected}
        for op in rec['operator_breakdown']['operators']:
            name=op['operator'];assert name in MOE|ATTENTION,name
            component='moe' if name in MOE else 'attention'
            expected_ops[component].add(name)
            amount=op['total_s']*1000;costs=op['components_s'];target=expected[component]
            close(sum(costs.values())*1000,amount)
            target['total_ms']+=amount
            if name in NETWORK:
                target['communication_ms']+=amount
            else:
                compute=costs.get('compute_s',0)*1000;memory=costs.get('memory_access_s',0)*1000
                target['compute_ms']+=compute;target['memory_ms']+=memory
                target['other_ms']+=amount-compute-memory
                assert costs.get('communication_s',0)==0,(identity,name)
            if identity[0]=='cp':
                cp_names.add(name)
                if name.startswith('cp_') or name in ('kda_recurrent','shared_projection_pre','shared_projection_post'):
                    assert component=='attention'
        close(sum(r['total_ms'] for r in expected.values()),window)
        assert set(groups[identity])=={'attention','moe','other'}
        for component,actual in groups[identity].items():
            target=expected[component]
            close(actual['window_total_ms'],window)
            for field in ('total_ms','communication_ms','compute_ms','memory_ms','other_ms'):
                close(actual[field],target[field])
            assert set(actual['operators'])==expected_ops[component]
            close(actual['window_share_pct'],target['total_ms']/window*100)
            for field in ('communication','compute','memory','other'):
                close(actual[field+'_pct_of_window'],target[field+'_ms']/window*100)
                if target['total_ms']:
                    close(actual[field+'_pct'],target[field+'_ms']/target['total_ms']*100)
                else:
                    assert actual[field+'_pct'] is None,(identity,component,field)
            startup=sum(v for name,v in source['startup'].items() if name in expected_ops[component] and name in NETWORK)
            close(actual['communication_startup_ms'],startup)
            wait_other=actual.get('communication_wait_other_ms',0) or 0
            close(actual['communication_transfer_ms'],target['communication_ms']-startup-wait_other)
            close(actual['communication_startup_ms']+actual['communication_transfer_ms']+wait_other,actual['communication_ms'])
            close(actual['communication_ms']+actual['compute_ms']+actual['memory_ms']+actual['other_ms'],actual['total_ms'])
            for field in ('startup','transfer'):
                for denominator,denom_field in [('component','total_ms'),('communication','communication_ms')]:
                    value=actual[f'communication_{field}_pct_of_{denominator}']
                    if actual[denom_field]:close(value,actual[f'communication_{field}_ms']/actual[denom_field]*100)
                    else:assert value is None
        close(sum(row['window_share_pct'] for row in groups[identity].values()),100)
        checks.append({'scope':identity[0],'case_id':identity[1],'startup_remaining_fraction':identity[2],
            'window_total_ms':window,'outer_and_inner_denominators':'passed','source_operators_reconciled':'passed',
            'network_startup_transfer_not_double_counted':'passed'})
    flat=load(directory/'window_summary.json')['rows']
    assert len(flat)==108 and {key(r) for r in flat}==set(groups)
    for row in flat:
        components=groups[key(row)]
        close(row['window_total_ms'],sources[key(row)]['record']['total_s']*1000)
        close(row['tpot_ms'],row['window_total_ms'])
        for component,source in components.items():
            for field in ('total_ms','window_share_pct','compute_ms','communication_ms','memory_ms','other_ms',
                          'compute_pct','communication_pct','memory_pct','other_pct','communication_startup_ms','communication_transfer_ms'):
                value=row[component+'_'+field]
                if source[field] is None:assert value is None
                else:close(value,source[field])
        for field in ('communication_ms','communication_startup_ms','communication_transfer_ms'):
            close(row[field],sum(c[field] for c in components.values()))
    # Source summaries must still match the earlier independent numerical audit.
    for path,field in [(directory/'startup-reduction/independent-verification.json','actual_summary_sha256'),
                       (directory/'cp-owner-star/independent-verification.json','summary_sha256')]:
        before=load(path)
        summary=path.with_name('summary.json')
        assert before[field]==hashlib.sha256(summary.read_bytes()).hexdigest(),str(summary)
    result={'status':'passed','window_count':108,'component_row_count':324,'checks':checks,
        'chart_window_rows_verified':len(flat),
        'cp_operator_names':sorted(cp_names),'cp_attention_names':sorted(cp_names&ATTENTION),
        'cp_moe_names':sorted(cp_names&MOE),'numerical_sources_unchanged':True,
        'source_sha256':{str(p.relative_to(STUDY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'exported_components_sha256':hashlib.sha256((directory/'component_breakdown.json').read_bytes()).hexdigest(),
        'semantics':'Outer denominator is the full TPOT window. Inner denominator is that component. Communication startup/transfer are contained subcomponents. Noncommunication compute/memory time follows saved limiting-resource attribution, not independent overlapping activity durations.'}
    (directory/'component-independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','windows':108,'components':324,'cp_operators':len(cp_names)},ensure_ascii=False))


if __name__=='__main__':main()

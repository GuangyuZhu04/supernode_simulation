"""Independently reconcile saved Attention/MoE component projections."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'outputs/scaling-10case-20261113'


def main():
    path=DIRECTORY/'component_breakdown.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    groups=defaultdict(list)
    available=unavailable=0
    for row in data['rows']:
        ident=(row['scope'],row['case_id'],row['variant'],row['metric'])
        groups[ident].append(row)
        if row['window_total_ms'] is None:
            assert row['total_ms'] is None and row['window_share_pct'] is None
            assert all(row[k] is None for k in ('communication_ms','compute_ms','memory_ms','other_ms'))
            continue
        total=row['total_ms']
        assert all(row[k]>=-1e-8 for k in ('communication_ms','compute_ms','memory_ms','other_ms'))
        assert math.isclose(sum(row[k] for k in ('communication_ms','compute_ms','memory_ms','other_ms')),total,rel_tol=1e-10,abs_tol=1e-8)
        assert math.isclose(row['window_share_pct'],total/row['window_total_ms']*100,rel_tol=1e-10,abs_tol=1e-8)
        if total:
            assert math.isclose(sum(row[k] for k in ('communication_pct','compute_pct','memory_pct','other_pct')),100,abs_tol=1e-8)
        else:
            assert all(row[k] is None for k in ('communication_pct','compute_pct','memory_pct','other_pct'))
        if row['communication_split_available']:
            assert math.isclose(row['communication_startup_ms']+row['communication_transfer_ms']+row['communication_wait_other_ms'],row['communication_ms'],rel_tol=1e-10,abs_tol=1e-8)
    mapping=data['mapping']
    for ident,rows in groups.items():
        assert {row['component'] for row in rows}=={'attention','moe','other'}
        if rows[0]['window_total_ms'] is None:
            unavailable+=1
            assert all(row['window_total_ms'] is None for row in rows)
            continue
        available+=1
        assert math.isclose(sum(row['total_ms'] for row in rows),rows[0]['window_total_ms'],rel_tol=1e-10,abs_tol=1e-8)
        evidence=data['operator_evidence']['/'.join(ident)]
        for row in rows:
            owned=[e for e in evidence if e['assigned_component']==row['component']]
            assert math.isclose(sum(e['total_ms'] for e in owned),row['total_ms'],rel_tol=1e-10,abs_tol=1e-8)
            assert math.isclose(sum(e['total_ms'] for e in owned if e['communication_operator']),row['communication_ms'],rel_tol=1e-10,abs_tol=1e-8)
        for event in evidence:
            name=event['operator']
            if name in ('tp_attention','tp_index_scores','csa2_index_score','csa2_index_select','csa2_topk_candidates','csa2_sparse_qk_av','csa2_softmax_rope'):
                assert event['assigned_component']=='attention'
            if name in ('dispatch','combine','tp_ffn','experts','afd_additional_partial_merge'):
                assert event['assigned_component']=='moe'
    payload={'status':'passed','scope':'Saved operator identities, outer window partition and inner limiting-resource cost fractions; no tensor-specific pure-Attention claim',
             'windows':len(groups),'available_windows':available,'unavailable_windows':unavailable,
             'window_counts':dict(Counter(i[0]+':'+i[3] for i in groups)),
             'outer_and_inner_additivity':True,'index_operators_not_omitted':True,'null_windows_not_zero_filled':True}
    (DIRECTORY/'component-independent-verification.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload))


if __name__=='__main__':main()

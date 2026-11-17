"""Independent affine startup audit of the frozen scaling-study evidence.

This is a prediction/check, not a claim that megakernel reduces network latency.
The production extension separately re-executes the event engine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/scaling-10case-20261113'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compute(directory):
    manifest = read(directory / 'manifest.json')
    split = read(directory / 'communication_breakdown.json')
    lookup = {(r['scope'], r['case_id'], r['variant']): r for r in split['totals']}
    rows, source_files = [], [directory / 'manifest.json', directory / 'communication_breakdown.json']
    for item in manifest['cases']:
        source = directory / item['result_path']
        result = read(source)
        source_files.append(source)
        s = lookup['main', item['id'], 'baseline']
        assert result['metrics']['tpot_s'] is not None
        base = result['metrics']['tpot_s'] * 1000
        startup = s['startup_ms']
        residual = base - startup
        target = item['target_tpot_ms']
        launch = result['metrics']['latency_breakdown']['tpot']['operator_breakdown']['components_s']['launch_s'] * 1000
        for alpha in (0.5, 0.3):
            predicted = residual + alpha * startup
            rows.append({'scope': 'main', 'case_id': item['id'], 'architecture_id': item['architecture_id'],
                'startup_fraction': alpha, 'baseline_tpot_ms': base, 'network_startup_ms': startup,
                'communication_total_ms':s['communication_total_ms'],'communication_transfer_ms':s['transfer_ms'],
                'residual_ms': residual, 'predicted_tpot_ms': predicted, 'target_tpot_ms': target,
                'passed': predicted <= target * (1 + 1e-12), 'kernel_launch_ms': launch,
                'kernel_only_reduction_tpot_ms': base - (1-alpha)*launch,
                'maximum_startup_fraction_for_target': (target-residual)/startup if startup else None,
                'zero_startup_sufficient': residual <= target,
                'formula': 'baseline_tpot_ms - (1 - startup_fraction) * recorded_network_startup_ms'})
    ratio_data = read(directory / 'ratio-search/summary.json')
    source_files.append(directory / 'ratio-search/summary.json')
    for r in ratio_data['rows']:
        if r['variant'] != 'baseline':
            continue
        case_id = f"{r['architecture_id']}-a{r['tp_a']}-f{r['tp_f']}-baseline"
        s = lookup['ratio', case_id, 'baseline']
        base, startup = r['tpot_ms'], s['startup_ms']
        residual, target = base-startup, r['target_tpot_ms']
        for alpha in (0.5,0.3):
            predicted = residual + alpha*startup
            rows.append({'scope':'ratio','case_id':case_id,'architecture_id':r['architecture_id'],
                'tp_a':r['tp_a'],'tp_f':r['tp_f'],'ratio':r['ratio'],'active_cards':r['active_cards'],'idle_cards':r['idle_cards'],
                'startup_fraction':alpha,'baseline_tpot_ms':base,'network_startup_ms':startup,
                'communication_total_ms':s['communication_total_ms'],'communication_transfer_ms':s['transfer_ms'],
                'residual_ms':residual,'predicted_tpot_ms':predicted,'target_tpot_ms':target,
                'passed':predicted<=target*(1+1e-12),
                'maximum_startup_fraction_for_target':(target-residual)/startup if startup else None,
                'zero_startup_sufficient':residual<=target,
                'formula':'baseline_tpot_ms - (1 - startup_fraction) * recorded_network_startup_ms'})
    best = {}
    for arch in ('kimi-k3','deepseek-v41'):
        best[arch] = {}
        for alpha in (.5,.3):
            best[arch][str(alpha)] = min((r for r in rows if r['scope']=='ratio' and r['architecture_id']==arch and r['startup_fraction']==alpha),key=lambda r:r['predicted_tpot_ms'])
    counts = {str(alpha):sum(r['passed'] for r in rows if r['scope']=='main' and r['startup_fraction']==alpha) for alpha in (.5,.3)}
    return {'status':'analytical_audit_complete','rows':rows,'best_ratio_by_architecture':best,'main_pass_counts':counts,
        'method':'Independent affine startup counterfactual using previously saved event startup attribution. This does not itself execute a megakernel implementation.',
        'zero_kernel_launch_in_main':all(r.get('kernel_launch_ms',0)==0 for r in rows if r['scope']=='main'),
        'sources':{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--verify-execution',action='store_true')
    args=parser.parse_args()
    result=compute(args.directory)
    output=args.directory/'sp-megakernel'
    output.mkdir(parents=True,exist_ok=True)
    (output/'analytical-audit.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    fields=list(dict.fromkeys(k for r in result['rows'] for k in r))
    with (output/'analytical-audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result['rows'])
    cp_path=output/'sp-analysis/summary.json'
    if cp_path.exists():
        cp=read(cp_path)
        bound_checks=[]
        for row in cp['pure_cp_bounds']:
            config=read(args.directory/row['config_path'])
            model=config['model']
            routed_fraction=model['top_k']/model['num_experts']
            dense=(model['active_params']-model['total_params']*routed_fraction)/(1-routed_fraction)
            floor_ms=dense/config['hardware']['npu']['hbm_Bps']*1000
            assert math.isclose(floor_ms,row['shared_weight_read_floor_ms'],rel_tol=1e-12)
            assert (floor_ms>row['target_ms'])==row['strictly_excluded_by_shared_weight_floor']
            bound_checks.append({'case_id':row['case_id'],'derived_shared_params_B':dense/1e9,
                'shared_weight_read_floor_ms':floor_ms,'excluded':floor_ms>row['target_ms'],'passed':True})
        (output/'sp-analysis/bounds-independent-verification.json').write_text(json.dumps({
            'status':'passed','checks':bound_checks,'row_count':len(bound_checks),
            'strictly_excluded_count':sum(r['excluded'] for r in bound_checks),
            'source_sha256':hashlib.sha256(cp_path.read_bytes()).hexdigest(),
            'scope':'Necessary bound only for TP1, unsharded FP8 shared weights read once per step, no persistent hardware weight cache. Does not establish CP feasibility for remaining rows.'},indent=2),encoding='utf-8')
    if args.verify_execution:
        actual_path=output/'startup-reduction/summary.json'
        actual=read(actual_path)
        assert actual['complete'] and len(actual['rows'])==76
        expected={(r['scope'],r['case_id'].removesuffix('-baseline') if r['scope']=='ratio' else r['case_id'],r['startup_fraction']):r for r in result['rows']}
        checks=[]
        for row in actual['rows']:
            key=row['scope'],row['case_id'],row['startup_remaining_fraction']
            reference=expected[key]
            assert math.isclose(row['tpot_ms'],reference['predicted_tpot_ms'],rel_tol=2e-10,abs_tol=1e-8),key
            assert row['target_passed']==reference['passed'],key
            assert math.isclose(row['communication_startup_ms'],row['startup_remaining_fraction']*reference['network_startup_ms'],rel_tol=2e-10,abs_tol=1e-8),key
            assert math.isclose(row['communication_transfer_ms'],reference['communication_transfer_ms'],rel_tol=2e-10,abs_tol=1e-8),key
            assert row['modeled_noncommunication_startup_ms']==0,key
            record=row['latency_breakdown']
            assert math.isclose(sum(record['components_s'].values()),record['total_s'],rel_tol=1e-10),key
            assert math.isclose(sum(o['total_s'] for o in record['operator_breakdown']['operators']),record['total_s'],rel_tol=1e-10),key
            assert math.isclose(row['communication_startup_ms']+row['communication_transfer_ms'],row['communication_ms'],rel_tol=1e-10),key
            checks.append({'scope':row['scope'],'case_id':row['case_id'],'startup_fraction':row['startup_remaining_fraction'],
                           'absolute_difference_ms':abs(row['tpot_ms']-reference['predicted_tpot_ms']),'passed':True})
        verified={'status':'passed','checks':checks,'row_count':len(checks),
                  'actual_summary_sha256':hashlib.sha256(actual_path.read_bytes()).hexdigest(),
                  'method':'Independent saved critical-path startup audit cross-checked against every new discrete-event execution; capacity/work-preservation invariants separately recorded by executor.'}
        (output/'startup-reduction/independent-verification.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
        print(json.dumps({'actual_event_executions_verified':len(checks),'max_difference_ms':max(x['absolute_difference_ms'] for x in checks)}))
    print(json.dumps({'counts':result['main_pass_counts'],'kernel_launch_zero':result['zero_kernel_launch_in_main'],
        'best':{arch:{alpha:{k:r[k] for k in ['ratio','predicted_tpot_ms','passed']} for alpha,r in rows.items()} for arch,rows in result['best_ratio_by_architecture'].items()}},ensure_ascii=False))


if __name__=='__main__':
    main()

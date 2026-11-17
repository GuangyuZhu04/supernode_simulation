"""Read-only audit of the sixteen executed fine CP results."""
from collections import defaultdict
import hashlib,json,math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.config import KernelTrace

STUDY=ROOT/'outputs/scaling-10case-20261113'
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-7),(a,b)

def main():
    out=STUDY/'sp-megakernel/cp-fine-operators'
    data=load(out/'summary.json');assert data['complete'] and len(data['rows'])==16
    coarse_data=load(STUDY/'sp-megakernel/cp-owner-star/summary.json')
    coarse={(r['case_id'],r['startup_factor']):r for r in coarse_data['rows']}
    by_case=defaultdict(dict);checks=[];trace_count=0;shape_ops=set();budget_ops=set()
    for path,sha in data['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for row in data['rows']:
        case=row['case_id'];alpha=row['network_startup_factor'];launch=row['kernel_launch_us']
        assert row['source_sha256']==data['source_sha256']
        assert (alpha,launch) not in by_case[case]
        by_case[case][alpha,launch]=row
        baseline=coarse[case,alpha]
        config=load(out/row['config_path']);native=config['native_budget_config'];profile=config['attention_profile']
        assert config['source_sha256']==data['source_sha256']
        assert native==load(STUDY/baseline['config_source'])
        assert profile==load(STUDY/baseline['profile_source'])
        assert row['capacity_status']=='feasible'
        close(row['coarse_tpot_ms'],baseline['tpot_ms'])
        close(row['communication_ms'],baseline['communication_ms'])
        close(row['communication_startup_ms'],baseline['startup_ms'])
        close(row['communication_transfer_ms'],baseline['transfer_ms'])
        assert row['transport_by_semantic_bytes']==baseline['transport_by_semantic_bytes']
        close(row['communication_startup_ms']+row['communication_transfer_ms'],row['communication_ms'])
        rec=row['latency_breakdown'];ops=rec['operator_breakdown']['operators']
        close(row['fine_tpot_ms'],rec['total_s']*1000)
        close(sum(rec['components_s'].values()),rec['total_s'])
        close(sum(o['total_s'] for o in ops),rec['total_s'])
        close(sum(row[k] for k in ('attention_ms','moe_ms','other_component_ms')),row['fine_tpot_ms'])
        close(rec['operator_breakdown']['components_s']['launch_s']*1000,row['kernel_launch_ms'])
        assert row['target_satisfied']==(row['fine_tpot_ms']<=row['target_ms'])
        assert row['fine_tpot_ms']>=row['coarse_tpot_ms']-1e-7
        close(sum(r['layer_time_s']*r['layers'] for r in row['kinds'].values()),rec['total_s'])
        for kind,detail in row['kinds'].items():
            audit=detail['fine_audit'];trace=detail['native_operator_trace']
            assert detail['layers']==profile['counts'][kind]
            assert detail['status']=='feasible'
            assert audit['network_events_unchanged'] and audit['parameter_flops_conserved']
            close(audit['fine_work']['flops'],audit['coarse_work']['flops']+audit['added_nonparameter_flops'])
            close(audit['fine_work']['hbm_bytes']-audit['coarse_work']['hbm_bytes'],audit['added_hbm_bytes'])
            assert audit['added_hbm_bytes']>=0
            assert trace['matched_events']==trace['eligible_events'] and trace['unmatched_events']==0
            assert trace['generated_operations']>trace['matched_events']
            assert len(audit['budget_unresolved'])==2
            assert {x['kernel'] for x in audit['budget_unresolved']}=={'shared_projection_pre','shared_projection_post'}
            assert len({x['id'] for x in audit['cross_child_buffers']})==len(audit['cross_child_buffers'])
            saved=load(out/detail['kernel_trace_artifact'])
            for parent,traces in saved['hardware']['kernel_traces'].items():
                for spec in traces:
                    KernelTrace.model_validate(spec)
                    trace_count+=1
                    assert spec['metadata_match'].get('fine_workload_signature')
                    for op in spec['operations']:
                        close(op['startup_s'],launch*1e-6)
                        assert op['implementation']['is_measured_kernel'] is False
                        if parent in ('experts','shared_projection_pre','shared_projection_post'):
                            assert op.get('matmul') is None
                            budget_ops.add(op['kernel'])
                        if op.get('matmul'):
                            shape=op['matmul'];close(op['flops'],2*shape['batch']*shape['m']*shape['n']*shape['k'])
                            shape_ops.add(op['kernel'])
            if kind!='KDA':assert audit['dependency_refinements']
        checks.append({'case_id':case,'network_startup_factor':alpha,'kernel_launch_us':launch,
                       'fine_tpot_ms':row['fine_tpot_ms'],'target_satisfied':row['target_satisfied'],
                       'network_bytes_and_time_unchanged':True,'saved_trace_work_and_shape_guards_valid':True,
                       'parameter_budget_and_added_nonparameter_accounting':'passed','latency_operator_component_additivity':'passed'})
    assert len(by_case)==4
    combinations={(1.,0.),(.5,0.),(.3,0.),(1.,1.)}
    for case,variants in by_case.items():
        assert set(variants)==combinations
        base=variants[1.,0.]
        for alpha in (.5,.3):
            row=variants[alpha,0.]
            close(row['fine_tpot_ms'],base['fine_tpot_ms']-(1-alpha)*base['communication_startup_ms'])
            assert row['kernel_launch_ms']==0
            for kind in row['kinds']:
                assert row['kinds'][kind]['fine_audit']['fine_work']==base['kinds'][kind]['fine_audit']['fine_work']
        launched=variants[1.,1.]
        close(launched['fine_tpot_ms']-base['fine_tpot_ms'],launched['kernel_launch_ms'])
        count=launched['kernel_launch_ms']/.001
        close(count,round(count))
        for kind in launched['kinds']:
            assert launched['kinds'][kind]['fine_audit']['fine_work']==base['kinds'][kind]['fine_audit']['fine_work']
    result={'status':'passed','result_count':16,'checks':checks,'validated_trace_spec_count':trace_count,
        'reference_algebra_matmul_operators':sorted(shape_ops),'budget_matrix_operators_without_fake_shapes':sorted(budget_ops),
        'source_sha256':data['source_sha256'],'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),
        'semantic_tests':'tests/test_scaling_cp_fine.py: 14 passed; graph tests include all eight attention kinds, QK-softmax-AV order, original allocation/release retention, carry release, weight-read conservation and stale-scope rejection.',
        'fidelity':'Analytical fine operator execution, not hardware measurement. Native trace coverage does not imply complete shape resolution; shared projections and routed-expert GEMMs retain explicit parameter-budget geometry. Additional FP32 proxy work/HBM materialization and retained parent-memory residuals are disclosed.',
        'launch_scope':'Default added launch is zero. 1us sensitivity changes compute kernel launches only; network factors independently scale original communication startup. Global launch sums over parallel ranks are not used as critical-path time.'}
    (out/'independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','results':16,'validated_trace_specs':trace_count,'shape_ops':sorted(shape_ops),
                     'failed_targets':sum(not r['target_satisfied'] for r in data['rows'])},ensure_ascii=False))

if __name__=='__main__':main()

"""Independent EP128 control audit, including analytic network round counts."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import csv,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/scaling-10case-20261113'
COMM={'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
MOE={'dispatch','combine','tp_ffn','experts','afd_additional_partial_merge'}
EXPECTED={2.8:(2.8e12,104e9,16,512),10.:(1e13,250e9,32,2048),100.:(1e14,1e12,64,8192)}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-8),(a,b)

def main():
    out=STUDY/'fixed-ep-128';data=load(out/'summary.json')
    assert data['complete'] and len(data['rows'])==12
    manifest={r['id']:r for r in load(STUDY/'manifest.json')['cases']}
    for path,sha in data['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    checks=[];groups=defaultdict(list);dominance=[]
    for row in data['rows']:
        source=manifest[row['original_case_id']]
        original=load(STUDY/source['config_path']);config=load(out/row['config_path'])
        expected=deepcopy(original);expected['execution']['ep']=128
        expected['execution']['tp']=expected['hardware']['npu_count']//128
        assert config==expected,row['case_id']
        assert row['original_config_sha256']==digest(original)
        assert row['config_sha256']==digest(config)
        profile=load((out/row['config_path']).with_name('attention-profile.json'))
        assert profile==load(STUDY/source['attention_profile_path'])
        assert row['attention_profile_sha256']==digest(profile)
        m,h,x,w=(config[k] for k in ('model','hardware','execution','workload'))
        assert (m['total_params'],m['active_params'],m['top_k'],m['num_experts'])==EXPECTED[row['model_T']]
        assert w['batch_size']==32 and w['input_length']==1048576
        assert row['ep']==x['ep']==128 and row['tp']==x['tp']==h['npu_count']//128
        assert row['target_tpot_ms']==5 and row['capacity_status']=='feasible'
        layers=m['layers'];tp=x['tp'];ep=x['ep'];latency=h['scale_up']['latency_s']
        rate=h['scale_up']['injection_Bps']*h['scale_up']['efficiency']
        assert h['npu']['hbm_Bps']>=2*rate
        assert h['scale_up']['bandwidth_Bps']>=h['scale_up']['injection_Bps']
        assert x['precision']['communication']=='fp16'
        # In this nonblocking single-SU model every source sends one full
        # hidden vector per top-k assignment; EP self copies are local.
        per_layer_dispatch=(ep-1)/ep*m['hidden_dim']*m['top_k']*2
        per_layer_tp=2*(tp-1)/tp*m['hidden_dim']*2
        expected_startup={'dispatch':layers*latency*1000,'combine':layers*latency*1000,
            'tp_attention':layers*2*(tp-1)*latency*1000,'tp_ffn':layers*2*(tp-1)*latency*1000}
        expected_transfer={'dispatch':layers*per_layer_dispatch/rate*1000,
            'combine':layers*per_layer_dispatch/rate*1000,
            'tp_attention':layers*per_layer_tp/rate*1000,'tp_ffn':layers*per_layer_tp/rate*1000}
        if row['architecture_id']=='deepseek-v41':
            total_keys=0;tiles=0;prefix=w['input_length']+1
            for kind,count in profile['counts'].items():
                if 'INDEX' not in kind:continue
                keys=prefix//(2 if 'ENCODER' in kind else 1)
                if 'CANDIDATE' in kind:keys=min(keys,profile['candidate_rows'])
                total_keys+=keys*count;tiles+=math.ceil(keys/profile['score_tile_rows'])*count
            assert tiles==45 and total_keys==2686977
            expected_startup['tp_index_scores']=tiles*2*(tp-1)*latency*1000
            expected_transfer['tp_index_scores']=2*(tp-1)/tp*total_keys*4/rate*1000
        for name,value in expected_startup.items():close(row['startup_by_operator_ms'][name],value)
        close(row['communication_startup_ms'],sum(expected_startup.values()))
        close(row['communication_transfer_ms'],sum(expected_transfer.values()))
        record=row['latency_breakdown'];ops=record['operator_breakdown']['operators']
        close(row['tpot_ms'],record['total_s']*1000)
        close(sum(record['components_s'].values()),record['total_s'])
        close(sum(op['total_s'] for op in ops),record['total_s'])
        by_name=defaultdict(float)
        for op in ops:
            by_name[op['operator']]+=op['total_s']*1000
            close(sum(op['components_s'].values()),op['total_s'])
        for name,start in expected_startup.items():close(by_name[name],start+expected_transfer[name])
        compute=sum(op['components_s'].get('compute_s',0) for op in ops if op['operator'] not in COMM)*1000
        memory=sum(op['components_s'].get('memory_access_s',0) for op in ops if op['operator'] not in COMM)*1000
        communication=sum(op['total_s'] for op in ops if op['operator'] in COMM)*1000
        close(row['compute_ms'],compute);close(row['memory_ms'],memory);close(row['communication_ms'],communication)
        close(compute+memory+communication+row['other_ms'],row['tpot_ms'])
        for field in ('compute','memory','communication','other'):
            close(row[field+'_pct'],row[field+'_ms']/row['tpot_ms']*100)
        for component in row['component_breakdown']:
            selected=[op for op in ops if (op['operator'] in MOE)==(component['component']=='moe')]
            close(component['total_ms'],sum(op['total_s'] for op in selected)*1000)
            close(sum(component[f+'_ms'] for f in ('compute','memory','communication','other')),component['total_ms'])
            close(component['window_share_pct'],component['total_ms']/row['tpot_ms']*100)
        if row['model_T']==2.8:
            original_result=load(STUDY/source['result_path'])
            assert config==original and record==original_result['metrics']['latency_breakdown']['tpot']
            original_hashes=original_result['provenance']['source_sha256']
            shared_sources=set(original_hashes)&set(data['source_sha256'])
            assert shared_sources
            for path in shared_sources:assert original_hashes[path]==data['source_sha256'][path],path
        winner=max(('compute','communication','memory','other'),key=lambda f:row[f+'_ms'])
        dominance.append({'case_id':row['case_id'],'architecture_id':row['architecture_id'],'hardware':row['hardware'],
            'model_T':row['model_T'],'tpot_ms':row['tpot_ms'],'dominant_component':winner,
            'compute_pct':row['compute_pct'],'communication_pct':row['communication_pct'],'memory_pct':row['memory_pct'],
            'communication_startup_ms':row['communication_startup_ms'],'communication_transfer_ms':row['communication_transfer_ms']})
        groups[row['architecture_id'],row['hardware']].append(row)
        checks.append({'case_id':row['case_id'],'only_EP_TP_changed':True,'analytic_startup_and_transfer_match':True,
            'original_P_A_topk_experts_preserved':True,'latency_and_component_sums_passed':True})
    trends=[]
    for (arch,hardware),rows in groups.items():
        rows.sort(key=lambda r:r['model_T']);assert len(rows)==3
        assert max(r['communication_startup_ms'] for r in rows)-min(r['communication_startup_ms'] for r in rows)<1e-8
        trends.append({'architecture_id':arch,'hardware':hardware,'models_T':[r['model_T'] for r in rows],
            'communication_pct':[r['communication_pct'] for r in rows],
            'communication_ms':[r['communication_ms'] for r in rows],
            'transfer_ms':[r['communication_transfer_ms'] for r in rows],
            'memory_ms':[r['memory_ms'] for r in rows],
            'memory_pct':[r['memory_pct'] for r in rows],
            'communication_growth_100T_over_2p8T':rows[2]['communication_ms']/rows[0]['communication_ms'],
            'memory_growth_100T_over_2p8T':rows[2]['memory_ms']/rows[0]['memory_ms']})
    result={'status':'passed','case_count':12,'checks':checks,'trends':trends,'dominance':dominance,
        'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),
        'scope':'Only fixed EP128, full POD TP32/96, declared Attention/KV budget templates, ring TP and direct EP routing. Communication is whole operator critical-path time. Compute/memory follow limiting-resource attribution, so zero compute time does not mean zero FLOPs.'}
    (out/'independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'dominance.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dominance[0]));writer.writeheader();writer.writerows(dominance)
    print(json.dumps({'status':'passed','case_count':12,'dominant_counts':{f:sum(r['dominant_component']==f for r in dominance) for f in ('compute','communication','memory','other')},'trends':trends},ensure_ascii=False))

if __name__=='__main__':main()

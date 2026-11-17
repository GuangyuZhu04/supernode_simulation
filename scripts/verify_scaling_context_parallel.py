"""Independent source/graph/result audit for the conditional CP experiment."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.config import load_config
from scripts.scaling_context_parallel import build_cp_layer


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-8),(a,b)


def main():
    study=ROOT/'outputs/scaling-10case-20261113'
    location=study/'sp-megakernel/cp-owner-star'
    data=load(location/'summary.json')
    assert data['complete'] and len(data['rows'])==12
    for path,sha in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    graph_checks=[];result_checks=[]
    by_case=defaultdict(dict)
    for row in data['rows']:by_case[row['case_id']][row['startup_factor']]=row
    assert len(by_case)==4
    for case,variants in by_case.items():
        baseline=variants[1.]
        config=load_config(study/baseline['config_source'])
        profile=load(study/baseline['profile_source'])
        m=config.model;w=config.workload;ep=config.execution.ep;cp=config.execution.tp
        ratio=m.top_k/m.num_experts
        dense=(m.active_params-m.total_params*ratio)/(1-ratio)
        expert=m.total_params-dense
        touched=-math.expm1(w.batch_size*math.log1p(-ratio))
        for kind,count in profile['counts'].items():
            graph=build_cp_layer(config,profile,kind,1.)
            metadata=graph.metadata['cp_study']
            initial=graph.initial_allocations
            close(sum(a.size_bytes for a in initial if a.category=='weight'),config.hardware.npu_count*(dense+expert/ep))
            owner_only=set(metadata['owner_only_categories'])
            for allocation in initial:
                if allocation.category in owner_only:
                    device=int(allocation.pool.split('.')[0].removeprefix('npu'))
                    assert device%cp==0 and device//cp<w.batch_size
            if profile['family']=='kimi_k3':
                assert 'recurrent_state' not in metadata['context_sharded_categories']
                assert 'convolution_state' not in metadata['context_sharded_categories']
            parameters=[e for e in graph.events if e.metadata.get('kernel') in ('shared_projection_pre','shared_projection_post','experts')]
            flops=sum(v*e.metadata['symmetry_multiplicity'] for e in parameters for r,v in e.work.items() if r.endswith('.compute'))
            close(flops,2*m.active_params*w.batch_size/m.layers)
            projection=[e for e in parameters if e.metadata['kernel'].startswith('shared_projection')]
            projection_reads=sum(v for e in projection for r,v in e.work.items() if r.endswith('.hbm'))-8*m.hidden_dim
            close(projection_reads,dense/m.layers)
            expert_reads=sum(v for e in parameters if e.metadata['kernel']=='experts' for r,v in e.work.items() if r.endswith('.hbm'))
            close(expert_reads,expert*touched/m.layers)
            expected_moe=w.batch_size*m.hidden_dim*m.top_k*2*(ep-1)/ep
            close(metadata['transport_by_semantic_bytes']['dispatch'],expected_moe)
            close(metadata['transport_by_semantic_bytes']['combine'],expected_moe)
            ids={e.id:e for e in graph.events}
            for event in graph.events:
                if event.metadata.get('payload_bytes'):
                    assert event.metadata.get('type')=='transfer'
                    assert event.metadata['startup_s_by_domain']
            if kind=='KDA':
                assert not any('cp_' in e.id for e in graph.events)
            else:
                appended=[e for e in graph.events if e.metadata.get('kernel')=='cp_cache_append']
                assert len(appended)==1
                assert appended[0].id.startswith(f'npu{w.input_length%cp}.')
                assert 'cp_max_gather' in ids and 'cp_max_broadcast' in ids and 'cp_numerator_denominator_gather' in ids
                assert ids['owner.cp_global_max'].deps==('cp_max_gather',)
                assert ids['cp_max_broadcast'].deps==('owner.cp_global_max',)
                for event in graph.events:
                    if event.metadata.get('kernel')=='cp_exp_rescale':assert event.deps==('cp_max_broadcast',)
                assert ids['owner.cp_softmax_merge'].deps==('cp_numerator_denominator_gather',)
            if profile['family']=='deepseek_v41' and 'INDEX' in kind:
                assert 'cp_index_candidates_gather' in ids
                assert ids['owner.cp_global_topk'].deps==('cp_index_candidates_gather',)
                assert ids['cp_selected_indices_broadcast'].deps==('owner.cp_global_topk',)
            graph_checks.append({'case_id':case,'kind':kind,'parameter_flops':flops,'weights_and_states':'passed','communication_dependencies':'passed'})
        for factor,row in variants.items():
            assert row['capacity_status']=='feasible'
            close(row['tpot_ms'],baseline['tpot_ms']-(1-factor)*baseline['startup_ms'])
            close(row['startup_ms'],baseline['startup_ms']*factor)
            close(row['transfer_ms'],baseline['transfer_ms'])
            close(row['communication_ms'],row['startup_ms']+row['transfer_ms'])
            rec=row['latency_breakdown']
            close(sum(rec['components_s'].values()),rec['total_s'])
            close(sum(o['total_s'] for o in rec['operator_breakdown']['operators']),rec['total_s'])
            communication=sum(value for key,value in rec['components_s'].items() if key.startswith('communication_'))
            # These owner-star kernels have no cross-kernel scheduling overlap;
            # coalesced transfer resources own their contention service.
            close(communication*1000,row['communication_ms'])
            assert row['target_satisfied']==(row['tpot_ms']<=row['target_ms'])
            assert row['tpot_ms']>=dense/config.hardware.npu.hbm_Bps*1000
            result_checks.append({'case_id':case,'startup_factor':factor,'tpot_ms':row['tpot_ms'],'target_satisfied':row['target_satisfied'],'passed':True})
    result={'status':'passed','graph_checks':graph_checks,'result_checks':result_checks,
        'source_sha256':data['source_sha256'],'summary_sha256':hashlib.sha256((location/'summary.json').read_bytes()).hexdigest(),
        'scope':'Only the declared owner/star CP budget model. Checks do not validate deployed parallel layouts, real kernel latency, quality, prefill or production serving.'}
    (location/'independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','graph_checks':len(graph_checks),'result_checks':len(result_checks)},ensure_ascii=False))


if __name__=='__main__':main()

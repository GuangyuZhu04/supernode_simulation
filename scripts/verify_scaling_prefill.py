"""Check Prefill template formulas, streamed staging and saved TTFT results."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.config import load_config, read_data
from supernode_sim.components import component_activations
from supernode_sim.case_studies.v41_fp4.target import index_geometry as native_index_geometry
from scripts.scaling_study_graph import build_layer
from scripts.scaling_kimi_attention import default_profile as kimi_profile
from scripts.scaling_v41_attention import default_profile as v41_profile
from scripts.scaling_prefill_study import (ORIGINAL_KIMI_WORK, kimi_prefill_work,
    v41_prefill_work, index_geometry, stream_transport, prefill_callbacks, STAGING_BYTES)

DIRECTORY=ROOT/'outputs/scaling-10case-20261113'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def formula_checks():
    model=read_data(ROOT/'examples/kimi_k3_ascend950_fp8.yaml')['model']
    kp=kimi_profile();vp=v41_profile();checks=[]
    n=0
    for tp in (1,2,3,4,6,8,12,16,24,32,96):
        for context in (1,255,256,1024,1048576):
            for kind,layer in (('KDA',0),('MLA',3)):
                expected=sum(r['elements']*2 for r in component_activations(model,layer,context,tp=tp,training=False) if r['component']==kind)
                native=ORIGINAL_KIMI_WORK(kp,kind,model,context,tp,'prefill')
                actual=kimi_prefill_work(kp,kind,model,context,tp,'prefill')
                close(actual['workspace_bytes']-native['workspace_bytes'],expected)
                close(actual['memory_bytes']-native['memory_bytes'],2*expected)
                close(actual['extra_compute_flops'],native['extra_compute_flops'])
                close(actual['cache_write_bytes'],native['cache_write_bytes'])
                n+=1
    checks.append({'check':'Kimi_prefill_workspace_and_HBM_against_native_component_activations','comparisons':n,'passed':True})
    n=0
    for q in (1,127,128,129,255,256,257,1024,1025,65536,1048576):
        for ratio in (1,2):
            for candidate in (False,True):
                actual=index_geometry(q,ratio,candidate)
                expected=native_index_geometry(q,0,ratio,candidate,True)
                assert actual==expected,(q,ratio,candidate)
                n+=1
    checks.append({'check':'V4_prefill_index_geometry_matches_native_tile_envelope','comparisons':n,'passed':True})
    n=0
    for q in (1,127,128,129,255,256,257,1024,1025):
        for kind in vp['counts']:
            actual=v41_prefill_work(vp,kind,model,q,8,'prefill')
            ratio=2 if 'ENCODER' in kind else 1
            expected=sum(min(i,128)+(min(512,i//ratio) if kind!='SWA' else 0) for i in range(1,q+1))
            close(actual['sparse_attention_pairs'],expected)
            close(actual['extra_compute_flops'],4*expected*vp['num_heads']/8*vp['head_dim'])
            n+=1
    checks.append({'check':'V4_sparse_pairs_vs_independent_per_token_causal_sum','comparisons':n,'passed':True})
    manifest=read(DIRECTORY/'manifest.json')
    for architecture,kind in (('kimi-k3','KDA'),('deepseek-v41','CSA2_DECODER_FULL_INDEX')):
        item=next(x for x in manifest['cases'] if x['id']==architecture+'-model-960DT-10T')
        config=load_config(DIRECTORY/item['config_path']);profile=read(DIRECTORY/item['attention_profile_path'])
        with prefill_callbacks():
            graph=build_layer(config,phase='prefill',attention_profile=profile,attention_kind=kind)
        before={e.id:{'work':dict(e.work),'deps':e.deps,'startup':e.startup_s,
                       'allocations':[(a.id,a.category,a.size_bytes) for a in e.allocs]} for e in graph.events}
        initial=[(a.id,a.size_bytes) for a in graph.initial_allocations]
        stream_transport(graph,profile,kind,config,None)
        assert initial==[(a.id,a.size_bytes) for a in graph.initial_allocations]
        reduced=0
        for e in graph.events:
            original=before[e.id]
            assert original['work']==e.work and original['deps']==e.deps and original['startup']==e.startup_s
            for allocation,(aid,category,size) in zip(e.allocs,original['allocations']):
                assert allocation.id==aid and allocation.category==category
                if category=='communication_buffer':
                    assert 0<allocation.size_bytes<=size
                    if e.metadata.get('operation')!='tp_index_scores':
                        assert allocation.size_bytes<=STAGING_BYTES
                    reduced+=allocation.size_bytes<size
                else:
                    assert allocation.size_bytes==size
        assert reduced>0
        checks.append({'check':architecture+'_streaming_changes_only_communication_scratch','reduced_allocations':reduced,'passed':True})
    return checks


def main():
    checks=formula_checks()
    directory=DIRECTORY/'prefill-study'
    summary=read(directory/'summary.json')
    current={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in summary['source_sha256']}
    assert current==summary['source_sha256']
    rows=summary['rows'];result_checks=[]
    for row in rows:
        result=read(DIRECTORY/row['source_artifact'])
        assert result['provenance']==current
        assert result['config']['workload']['batch_size']==32 and result['config']['workload']['input_length']==1048576
        record=row['latency_breakdown']
        if row['status']=='feasible':
            assert record is not None and row['ttft_s']>0
            close(row['ttft_s'],record['total_s'])
            close(sum(record['components_s'].values()),record['total_s'])
            close(sum(op['total_s'] for op in record['operator_breakdown']['operators']),record['total_s'])
            for name,comm in row['communication_by_operator'].items():
                close(comm['startup_s']+comm['transfer_s']+comm['other_wait_s'],comm['communication_total_s'])
                assert comm['transfer_s']>=0
            close(row['max_hbm_bytes'],max(result['memory_peaks_bytes'].values()))
            assert row['max_hbm_bytes']<=result['config']['hardware']['npu']['hbm_bytes']+1
        else:
            assert row['ttft_s'] is None and record is None and row['communication_by_operator'] is None
            assert row['diagnostics']
        assert result['metrics']['tpot_s'] is None
        result_checks.append({'case_id':row['case_id'],'status':row['status'],'passed':True})
    if summary['complete']:
        assert len(rows)==20
        by_id={row['case_id']:row for row in rows}
        for architecture in ('kimi-k3','deepseek-v41'):
            for hardware in ('960DT','970DT'):
                left=by_id[f'{architecture}-model-{hardware}-10T'];right=by_id[f'{architecture}-hardware-{hardware}-10T']
                assert left['ttft_s']==right['ttft_s'] and left['status']==right['status']
    payload={'status':'passed','complete_prefill_search_checked':summary['complete'],
             'formula_and_staging_checks':checks,'result_checks':result_checks,'source_sha256':current,
             'scope':'Attention/KV budget templates, Kimi native attention workspace, V4 native causal tile geometry, unchanged full payload with bounded streaming DMA scratch; V4 uses a declared conservative uniform index arena, not minimal per-kind workspace; no CED skipping or checkpoint reproduction'}
    (directory/'independent-verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','formula_groups':len(checks),'result_rows':len(rows),'complete':summary['complete']}))


if __name__=='__main__':main()

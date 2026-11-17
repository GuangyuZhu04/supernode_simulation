"""Auditable EP placement and weight-prefetch limits for the scaling study.

No unknown on-chip cache or overlap improvement is invented. Existing HBM
residency, per-token HBM service, and asynchronous overlap are distinct facts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    args=parser.parse_args();study=args.study
    manifest=read(study/'manifest.json')
    rows=[];sources={}
    for case in manifest['cases']:
        c=read(study/case['config_path']);m,h,x,w=(c[k] for k in ('model','hardware','execution','workload'))
        afd=read(study/case['afd_path']) if case.get('afd_path') else None
        ep=x['ep'];tp=afd['tp'] if afd else x['tp']
        route=m['top_k']/m['num_experts']
        dense=(m['active_params']-m['total_params']*route)/(1-route)
        expert=m['total_params']-dense
        assert math.isclose(dense,m['total_params']*(1-m['expert_parameter_fraction']),rel_tol=1e-12)
        touched=-math.expm1(w['batch_size']*math.log1p(-route))
        full_expert_shard=expert/ep
        shared_per_card=dense/tp
        expert_per_card=full_expert_shard/tp
        # A and F state are separated for AFD; homogeneous owner carries both.
        baseline_a=shared_per_card+(0 if afd else expert_per_card)
        cp_a=dense+(0 if afd else full_expert_shard)
        a_bandwidth=h['npu']['hbm_Bps']
        f_bandwidth=afd['lpu_hbm_Bps'] if afd else a_bandwidth
        target_s=case['target_tpot_ms']/1000
        runtime=h['npu']['runtime_bytes']+h['npu']['workspace_bytes']
        scale_block=x['features']['weight_scale_block'];scale_bytes=x['features']['scale_bytes']
        # Baseline resource ledger supplies a genuine sum-work/capacity bound,
        # including endpoint traffic, state, activations and kernel memory.
        resource_file=(study/case['result_path']).with_name('resources.csv')
        maximum_hbm_ms=0.;npu0_hbm_work=0.;max_hbm_resource=None
        with resource_file.open(encoding='utf-8-sig',newline='') as f:
            for resource in csv.DictReader(f):
                if resource['resource'].startswith('npu') and resource['resource'].endswith('.hbm'):
                    work=float(resource['work']);capacity=float(resource['capacity'])
                    if resource['resource']=='npu0.hbm':npu0_hbm_work=work
                    bound=work/capacity*1000
                    if bound>maximum_hbm_ms:maximum_hbm_ms=bound;max_hbm_resource=resource['resource']
        config_path=study/case['config_path']
        sources[str(config_path.relative_to(study))]=hashlib.sha256(config_path.read_bytes()).hexdigest()
        sources[str(resource_file.relative_to(study))]=hashlib.sha256(resource_file.read_bytes()).hexdigest()
        owner_known=dense+(0 if afd else full_expert_shard*touched)
        rows.append({'case_id':case['id'],'architecture_id':case['architecture_id'],'model_T':m['total_params']/1e12,
            'hardware':case['hardware_label'],'target_tpot_ms':case['target_tpot_ms'],'ep':ep,'baseline_tp':tp,'pure_cp_tp':1,'pure_cp_degree':tp,
            'shared_dense_B':dense/1e9,'routed_expert_total_T':expert/1e12,'expert_shard_per_EP_B':full_expert_shard/1e9,
            'expected_expert_touch_fraction':touched,
            'baseline_shared_weight_per_card_GB':shared_per_card/1e9,'baseline_expert_weight_per_card_GB':expert_per_card/1e9,
            'pure_cp_shared_weight_per_card_GB':dense/1e9,'pure_cp_expert_weight_per_card_GB':full_expert_shard/1e9,
            'baseline_attention_weight_footprint_GB':baseline_a/1e9,'pure_cp_attention_weight_footprint_GB':cp_a/1e9,
            'pure_cp_attention_weight_plus_scale_runtime_GB':(cp_a+math.ceil(cp_a/scale_block)*scale_bytes+runtime)/1e9,
            'baseline_expert_weight_read_per_step_GB':expert_per_card*touched/1e9,
            'pure_cp_expert_weight_read_per_step_GB':full_expert_shard*touched/1e9,
            'baseline_shared_weight_service_floor_ms':shared_per_card/a_bandwidth*1000,
            'pure_cp_shared_weight_service_floor_ms':dense/a_bandwidth*1000,
            'pure_cp_owner_known_weight_hbm_floor_ms':owner_known/a_bandwidth*1000,
            'pure_cp_expert_weight_service_floor_ms':full_expert_shard*touched/f_bandwidth*1000,
            'baseline_total_hbm_resource_floor_ms':maximum_hbm_ms,'baseline_total_hbm_resource':max_hbm_resource,
            'baseline_npu0_hbm_work_GB':npu0_hbm_work/1e9,
            'minimum_shared_weight_read_avoidance_GB':max(0,dense-a_bandwidth*target_s)/1e9,
            'minimum_shared_only_hbm_TBps':dense/target_s/1e12,
            'minimum_shared_only_TP_degree':max(1,math.ceil(dense/(a_bandwidth*target_s)-1e-12)),
            'shared_weight_bound_rejects_pure_cp':dense/a_bandwidth>target_s,
            'weights_preloaded_in_hbm':True,'cold_weight_loading_in_tpot':False,
            'onchip_weight_cache_capacity_supplied':False,'prefetch_speedup_quantified':False,
            'scope':'FP8 data bytes; metadata/runtime counted in footprints, excluded from mandatory read lower bound. EP already shards routed experts; shared D remains replicated over EP. AFD F service is separate from A.'})
    output=study/'sp-megakernel';output.mkdir(parents=True,exist_ok=True)
    document={'rows':rows,'status':'analysis_complete','sources':sources,
        'notes':[
            'EP already distributes routed experts; it does not partition the shared attention/projection budget D in this execution model.',
            'The baseline preloads model weights in HBM and excludes cold loading. HBM residency is different from on-chip residency.',
            'HBM-to-onchip asynchronous prefetch can overlap other work, but does not reduce per-owner per-step HBM byte demand at fixed weights/precision/cache policy.',
            'In a steady sequence of decode steps on the same owner, HBM byte demand divided by its service bandwidth remains a throughput lower bound, even with perfect scheduling overlap.',
            'A first-step measurement can hide a prefetched tile before the clock starts, but this changes the measured window and does not establish sustainable TPOT.',
            'Moving D or expert weights to a truly persistent on-chip cache would change HBM demand. Its size, placement, hit rate and startup warmup have not been supplied and are not assumed.',
            'minimum_shared_weight_read_avoidance_GB is an optimistic necessary amount of shared FP8 data whose HBM reads must be avoided to fit the target with TP1; it leaves zero time for all other work.',
            'minimum_shared_only_TP_degree and minimum_shared_only_hbm_TBps are necessary shared-weight bounds, not sufficient configurations.',
            'baseline_total_hbm_resource_floor_ms uses actual saved HBM resource work / capacity. It includes more than weight service and is not subtracted from measured TPOT as a supposed prefetch gain.',
            'Owner/star CP graphs read the complete shared D on the owner and a complete EP expert shard on its owner. If replicas distribute tokens or weights, that is an additional execution design, not pure CP substitution.',
            'Per-resource service demands may overlap. An ideal lower bound uses maxima across independent resources; adding or subtracting resource times to manufacture a new TPOT is not valid.' ],
        'verification':{'row_count':len(rows),'shared_budget_formula_reconciled':True,
            'pure_cp_shared_floor_rejections':sum(r['shared_weight_bound_rejects_pure_cp'] for r in rows)},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'prefetch_ep_analysis.json').write_text(json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    with (output/'prefetch_ep_analysis.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'pure_cp_shared_floor_rejections':document['verification']['pure_cp_shared_floor_rejections'],
        'tenT970':{k:v for k,v in next(r for r in rows if r['case_id']=='kimi-k3-hardware-970DT-10T').items() if k in ('shared_dense_B','expert_shard_per_EP_B','pure_cp_shared_weight_service_floor_ms','minimum_shared_weight_read_avoidance_GB','minimum_shared_only_TP_degree')}},ensure_ascii=False))


if __name__=='__main__':main()

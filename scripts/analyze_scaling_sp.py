"""Native-SP evidence and pure-context-parallel necessary bounds.

No context-parallel event execution is claimed by this analysis.  In the
TP=1 replacement, shared projection weights are unsharded and must be read
once per request step; that alone supplies a strict stage-latency lower bound
within the study's no-persistent-weight-cache, FP8 budget assumptions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supernode_sim.analysis import compile_graph, execution_metrics
from supernode_sim.config import Config
from supernode_sim.engine import simulate


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def native_config(tp, sp):
    return Config.model_validate({
        'model': {'name':'synthetic-native-SP-proof', 'layers':2, 'hidden_dim':64,
            'num_heads':8,'kv_heads':8,'head_dim':8,
            'architecture': {'vocab_size':64,'attention_by_layer':['MLA','KDA'],
                'ffn_by_layer':['Dense','MoE'],'dense_intermediate_dim':64,
                'mla':{'q_lora_rank':8,'kv_lora_rank':8,'qk_nope_head_dim':8,'qk_rope_head_dim':4,'v_head_dim':8},
                'kda':{'num_heads':8,'head_k_dim':8,'head_v_dim':8,'conv_kernel':4,'gate_low_rank_dim':8},
                'moe':{'num_experts':16,'top_k':2,'intermediate_dim':32,'num_shared_experts':1}}},
        'hardware': {'npu_count':tp*2,'su_size':tp*2,'npus_per_host':min(tp*2,8),
            'npu':{'hbm_bytes':1e9,'hbm_Bps':1e12,'flops':{'fp8':1e14,'fp16':5e13,'fp32':5e12},
                   'efficiency':1,'runtime_bytes':0,'workspace_bytes':0},
            'scale_up':{'bandwidth_Bps':1e11,'injection_Bps':1e11,'efficiency':1,'latency_s':2.1e-6}},
        'workload':{'scenario':'inference','batch_size':2,'input_length':128,'output_length':2},
        'execution':{'tp':tp,'ep':2,'sequence_parallel':sp,
            'precision':{'weight':'fp8','compute':'fp8','kv':'fp16','activation':'fp16','communication':'fp16','recurrent':'fp32'}},
        'sources':{'*':{'status':'assumed','description':'Small synthetic native-SP semantic check, not a scaled production prediction.'}}})


def native_evidence(directory):
    rows=[]; checks=[]
    state_categories={'kv_cache','kv_scales','page_table','recurrent_state','convolution_state'}
    for tp in (2,4,8):
        assets=[]
        for sp in (False,True):
            config=native_config(tp,sp); graph=compile_graph(config); sim=simulate(graph)
            metrics=execution_metrics(config,graph,sim)
            assert sim['status']=='feasible'
            events={e.id:e for e in graph.events}
            state={a.id:a.size_bytes for e in graph.events for a in e.allocs if a.category in state_categories}
            decode_events=[e for e in graph.events if e.id.startswith('infer.d0.s1.')]
            traffic=sum(e.metadata.get('payload_bytes',0) for e in decode_events)
            attention_ffn_traffic=sum(e.metadata.get('payload_bytes',0) for e in decode_events
                if any(fragment in e.id for fragment in ('.MLA.e', '.KDA.e', '.Dense.e', '.MoE.e')))
            row={'tp':tp,'sequence_parallel':sp,'tpot_ms':metrics['tpot_s']*1000,
                 'ttft_ms':metrics['ttft_s']*1000,'decode_transport_bytes':traffic,
                 'attention_ffn_boundary_bytes':attention_ffn_traffic,
                 'state_bytes':sum(state.values()),'status':sim['status']}
            save(directory/f'native-tp{tp}-sp{int(sp)}.json',{'config':config.model_dump(),'summary':row,'metrics':metrics})
            rows.append(row);assets.append((graph,events,state,row))
        base,current=assets
        assert base[2]==current[2]
        for layer,component in ((0,'MLA'),(0,'Dense'),(1,'KDA'),(1,'MoE')):
            event_id=f'infer.d0.s1.l{layer}.npu0.{component}'
            assert base[1][event_id].work==current[1][event_id].work
        ratio=current[3]['attention_ffn_boundary_bytes']/base[3]['attention_ffn_boundary_bytes']
        assert math.isclose(ratio,tp,rel_tol=1e-12)
        assert current[3]['tpot_ms']>=base[3]['tpot_ms']
        checks.append({'tp':tp,'cache_identical':True,'decode_attention_ffn_work_identical':True,
            'SP_boundary_traffic_over_TP':ratio,'SP_not_faster_in_native_execution':True})
    return {'scope':'Small native structured graph evidence, not a forecast for the exact-budget study.',
            'rows':rows,'checks':checks,'status':'passed'}


def cp_bounds(study):
    manifest=json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for case in manifest['cases']:
        c=json.loads((study/case['config_path']).read_text(encoding='utf-8'))
        profile=json.loads((study/case['attention_profile_path']).read_text(encoding='utf-8'))
        m,h,x=c['model'],c['hardware'],c['execution']
        afd=json.loads((study/case['afd_path']).read_text(encoding='utf-8')) if case['afd_path'] else None
        degree=afd['tp'] if afd else x['tp']
        dense=m['total_params']*(1-m['expert_parameter_fraction'])
        expert=m['total_params']-dense
        p_a=dense+(0 if afd else expert/x['ep'])
        p_f=expert/x['ep'] if afd else None
        base_memory=p_a+math.ceil(p_a/x['features']['weight_scale_block'])*x['features']['scale_bytes']
        base_memory+=h['npu']['runtime_bytes']+h['npu']['workspace_bytes']
        if profile['family']=='kimi_k3':
            from scripts.scaling_kimi_attention import attention_state
            categories=attention_state(profile,c['workload']['input_length']+1,1)['categories_bytes']
            shardable={'kv_cache','kv_scales','page_table'}
        else:
            from scripts.scaling_v41_attention import attention_state
            categories=attention_state(profile,c['workload']['input_length']+1,1)['categories_bytes']
            shardable=set(categories)-{'compressor_state'}
        cache_lower=sum(value/(degree if key in shardable else 1) for key,value in categories.items())
        memory_lower=base_memory+cache_lower
        dense_floor=dense/h['npu']['hbm_Bps']*1000
        weight_total_floor=(dense/h['npu']['hbm_Bps']+
            (expert/x['ep'])*(1-(1-m['top_k']/m['num_experts'])**c['workload']['batch_size'])/
            (afd['lpu_hbm_Bps'] if afd else h['npu']['hbm_Bps']))*1000
        # The routed-expert term assumes one full batch per CP replica. It is
        # NOT the universal lower bound if expert work is split across replicas.
        row={'case_id':case['id'],'architecture_id':case['architecture_id'],
            'model_T':m['total_params']/1e12,'hardware':case['hardware_label'],'target_ms':case['target_tpot_ms'],
            'former_tp':degree,'assumed_cp':degree,'new_tp':1,'ep':x['ep'],
            'shared_dense_params_B':dense/1e9,'shared_weight_read_floor_ms':dense_floor,
            'strictly_excluded_by_shared_weight_floor':dense_floor>case['target_tpot_ms'],
            'replicated_full_batch_weight_floor_ms':weight_total_floor,
            'attention_resident_lower_GB':memory_lower/1e9,'hbm_capacity_GB':h['npu']['hbm_bytes']/1e9,
            'capacity_excluded_even_with_ideal_cache_balance':memory_lower>h['npu']['hbm_bytes'],
            'unsharded_attention_weights_GB':p_a/1e9,'ideal_balanced_context_cache_GB':cache_lower/1e9,
            'ffn_weights_GB':p_f/1e9 if p_f is not None else None,
            'status':'necessary_bound_only','config_path':case['config_path'],'attention_profile_path':case['attention_profile_path']}
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'outputs/scaling-10case-20261113')
    args=parser.parse_args();out=args.study/'sp-megakernel/sp-analysis';out.mkdir(parents=True,exist_ok=True)
    native=native_evidence(out/'native-evidence')
    bounds=cp_bounds(args.study)
    payload={'native_sp':native,'pure_cp_bounds':bounds,
        'native_sp_notes':[
            'Megatron SP shares TP ranks and keeps attention/matrix TP; it does not replace tensor parallelism.',
            'At one decode query per EP owner, ceil(1/TP)=1 row per SP rank: pointwise workload is not reduced, and boundary traffic is padded.',
            'The attention/FFN boundary ring startup count stays 4*(TP-1) per layer; its payload increases by TP because padded full rows are gathered/scattered.',
            'MLA compressed KV remains replicated and KDA state remains head-sharded by TP. Native Config rejects SP on declared-budget models.' ],
        'pure_cp_notes':[
            'Bounds assume TP is entirely replaced by context parallelism and shared projection weights therefore remain unsharded FP8.',
            'Every decode step reads each shared weight once; persistent hardware caching of tens/hundreds of GB of weights is not assumed.',
            'The shared-weight bound excludes all context reads, expert service, softmax/index merging and communication, so exceeding an SLO is a necessary rejection only.',
            'Passing a bound does not establish feasibility. No full CP graph, algorithm runtime or deployable layout is claimed.',
            'KDA recurrent/conv state is not divided by context degree; it is sequence-length independent.',
            'Ideal cache balance is a capacity lower bound, before token/page rounding and transient communication/kernel buffers.',
            'The additional full-batch expert read bound applies only when FFN replicas each process the full batch; token-splitting expert replicas are a separate hybrid design.' ],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'supernode_sim/structured.py',ROOT/'skills/supernode-modeling/references/inference.md']}}
    save(out/'summary.json',payload)
    with (out/'pure-cp-bounds.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(bounds[0]));writer.writeheader();writer.writerows(bounds)
    lines=['# SP解释与纯CP必要下界','',
        '原生Megatron SP与TP共rank，不是把TP改名为SP；1M长度的MLA KV不会因此分片。',
        '小规模原生执行验证通过，decode每EP一条序列时，SP保留Attention/FFN计算与KV，但边界通信padding增加。','',
        '纯CP表仅为TP=1、共享权重不分片的必要下界，绝非完整CP仿真。通过下界不能宣称达到目标。','',
        '| 场景 | 原TP/假设CP | 目标ms | 共享权重读下界ms | 是否已排除达标 | 理想常驻下界GB |',
        '|---|---:|---:|---:|---|---:|']
    lines += [f'| {r["case_id"]} | {r["assumed_cp"]} | {r["target_ms"]} | {r["shared_weight_read_floor_ms"]:.6f} | {r["strictly_excluded_by_shared_weight_floor"]} | {r["attention_resident_lower_GB"]:.3f} |' for r in bounds]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'native_checks':native['status'],'bound_rows':len(bounds),
        'strict_rejections':sum(r['strictly_excluded_by_shared_weight_floor'] for r in bounds),'output':str(out)}))


if __name__=='__main__': main()

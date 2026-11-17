"""Unequal Attention/FFN TP ratio search on the frozen scaling-study templates.

Every active device has its own compute, HBM and full-duplex endpoint resources.
Uniform, stage-barrier traffic is represented by the exact resource-demand
envelope; unmodeled switches are not inferred. This is a budget-model study,
not a checkpoint deployment validator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from dataclasses import asdict
from fractions import Fraction
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import combine_breakdowns, window_breakdown
from supernode_sim.report import write_latency_csv, latency_markdown

DEFAULT = ROOT / 'outputs/scaling-10case-20261113'
CANDIDATES = [(a, 24-a) for a in (1, 2, 3, 4, 6, 8, 12, 16)] + [(16, 4)]
COMMUNICATION = {'dispatch', 'combine', 'tp_attention', 'tp_ffn', 'tp_index_scores'}
SOURCE_FILES = ['scripts/scaling_afd_ratios.py', 'scripts/scaling_kimi_attention.py',
                'scripts/scaling_v41_attention.py', 'supernode_sim/core.py',
                'supernode_sim/engine.py', 'supernode_sim/latency_breakdown.py',
                'supernode_sim/operator_costs.py']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def build_layer(config, profile, kind, tp_a, tp_f, *, zero_startup=False):
    m, h, w, x = (config[k] for k in ('model','hardware','workload','execution'))
    ep, batch, layers = x['ep'], w['batch_size'], m['layers']
    context = w['input_length']
    a_count, f_count = ep*tp_a, ep*tp_f
    if batch > ep or a_count+f_count > h['npu_count'] or (ep*tp_f) % tp_a:
        raise ValueError('Requires global batch <= EP, valid physical budget and uniform lane mapping')
    if profile['family'] == 'kimi_k3':
        from scripts.scaling_kimi_attention import attention_state, attention_work
    elif profile['family'] == 'deepseek_v41':
        from scripts.scaling_v41_attention import attention_state, attention_work
    else:
        raise ValueError('Unknown attention family')
    aw = attention_work(profile, kind, m, context, tp_a, 'decode')
    state = attention_state(profile, context+1, tp_a)
    dense = m['total_params']*(1-m['expert_parameter_fraction'])
    experts = m['total_params']-dense
    expert_active = m['active_params']-dense
    fraction = -math.expm1(batch*math.log1p(-m['top_k']/m['num_experts']))
    bandwidth = h['scale_up']['injection_Bps']*h['scale_up']['efficiency']
    startup = 0. if zero_startup else h['scale_up']['latency_s']
    g = Graph()
    pools = {'attention': [], 'ffn': [], 'idle': []}
    for i in range(h['npu_count']):
        rank = f'npu{i}'
        role = 'attention' if i < a_count else 'ffn' if i < a_count+f_count else 'idle'
        pools[role].append(rank)
        f_role = role == 'ffn'
        g.memory_caps[rank+'.hbm'] = h['npu']['hbm_bytes']
        g.resources[rank+'.hbm'] = Resource(rank+'.hbm', 144e12 if f_role else h['npu']['hbm_Bps'], 'byte/s')
        g.resources[rank+'.compute'] = Resource(rank+'.compute', (8e15 if f_role else h['npu']['flops']['fp8'])*h['npu']['efficiency'], 'FLOP/s','compute')
        g.resources[rank+'.compute.fp32'] = Resource(rank+'.compute.fp32', h['npu']['flops']['fp32']*h['npu']['efficiency'], 'FLOP/s','compute')
        for direction in ('tx','rx'):
            g.resources[rank+'.SU.'+direction] = Resource(rank+'.SU.'+direction, bandwidth, 'byte/s')
    activation = 2*m['hidden_dim']*2
    expert_buffer = batch*m['hidden_dim']*m['top_k']*2/ep
    expert_intermediate = batch*m['top_k']*m['intermediate_dim']*2/(tp_f*ep)
    resident_categories = defaultdict(float)

    def alloc(rank, suffix, size, category, dtype=None):
        return Allocation(rank+'.'+suffix, rank+'.hbm', float(size), category, dtype,
                          metadata={'component':'Unassigned','state':category})

    for role in ('attention','ffn'):
        for local_index, rank in enumerate(pools[role]):
            params = dense/tp_a if role=='attention' else experts/(ep*tp_f)
            items = [alloc(rank,'weights',params,'weight','fp8'),
                     alloc(rank,'weight_scales',math.ceil(params/128)*4,'quantization_metadata')]
            items += [alloc(rank, name, h['npu'][name+'_bytes'],name) for name in ('runtime','workspace')]
            if role=='attention' and local_index//tp_a < batch:
                for category, amount in state['categoriesbytes'].items():
                    dtype = ('fp32' if category in ('recurrent_state','compressor_state','shared_rope_table')
                             else None if category in ('page_table','kv_scales') else 'fp16')
                    items.append(alloc(rank,category,amount,category,dtype))
            g.initial_allocations.extend(items)
            for item in items:
                resident_categories[item.category] += item.size_bytes
    active_a = [f'npu{i}' for i in range(batch*tp_a)]
    dynamic = [alloc(rank,'activation',activation,'activation','fp16') for rank in active_a]
    for rank in pools['ffn']:
        dynamic += [alloc(rank,'expert_buffer',expert_buffer,'expert_buffer','fp16'),
                    alloc(rank,'expert_intermediate',expert_intermediate,'activation','fp16')]
    for rank in active_a:
        dynamic.append(alloc(rank,'attention_workspace',aw['workspace_bytes'],'attention_workspace'))
    g.events.append(Event('ready',allocs=tuple(dynamic),metadata={'phase':'decode','operation':'barrier'}))
    previous = 'ready'
    traffic = defaultdict(float)

    def event(name, work, *, component=kind, startup_s=0., communication=False, allocs=(), releases=()):
        nonlocal previous
        md = {'phase':'decode','operation':name,'kernel':name,'component':component}
        if communication:
            md.update(type='transfer',domains=['SU'],startup_s_by_domain={'SU':startup_s},
                      resource_domains={key:('SU' if '.SU.' in key else 'other') for key in work})
        g.events.append(Event(name,(previous,),dict(work),startup_s,
                              allocs=tuple(allocs),releases=tuple(releases),metadata=md))
        previous = name

    def kernel(name, flops, memory, *, fp32=False, role='attention'):
        rank = pools[role][0]
        event(name,{rank+('.compute.fp32' if fp32 else '.compute'):flops,
                    rank+'.hbm':memory},component=kind if role=='attention' else 'MoE')

    def ring(name, size, repeats=1):
        if tp_a==1 or size==0:
            return
        rounds = 2*(tp_a-1)
        per_card = size/tp_a*rounds
        work = {}
        # All Attention replicas use disjoint equal endpoint resources.
        for rank in active_a:
            work[rank+'.SU.tx'] = per_card
            work[rank+'.SU.rx'] = per_card
            work[rank+'.hbm'] = 2*per_card
        traffic[name] += size*rounds*batch
        temp = [alloc(rank,name+'.buffer',2*size/tp_a,'communication_buffer','fp16') for rank in active_a]
        event(name,work,component='TP',startup_s=startup*rounds*repeats,communication=True,
              allocs=temp,releases=[a.id for a in temp])

    # Parameter projection budget and architecture-dependent nonparameter work.
    v41 = profile['family']=='deepseek_v41'
    kernel(aw['kernel_name'],2*dense/layers/tp_a+(0 if v41 else aw['extra_compute_flops']),
           dense/layers/tp_a+2*activation+(0 if v41 else aw['memory_bytes']+aw['cache_write_bytes']))
    if v41:
        if aw['index_flops']:
            kernel('csa2_index_score',aw['index_flops'],aw['index_memory_bytes'])
            ring('tp_index_scores',aw['index_reduce_bytes'],aw['index_reduce_repeats'])
            kernel('csa2_index_select',aw['index_selection_flops'],aw['index_selection_memory_bytes'],fp32=True)
        kernel('csa2_sparse_qk_av',aw['extra_compute_flops'],aw['memory_bytes']+aw['cache_write_bytes'])
        kernel('csa2_softmax_rope',aw['extra_fp32_flops'],aw.get('fp32_memory_bytes',0),fp32=True)
    else:
        if aw.get('cache_conversion_flops'):
            kernel('kv_conversion',aw['cache_conversion_flops'],aw['cache_raw_bytes']+aw['cache_storage_bytes'])
        if aw['extra_fp32_flops']:
            kernel(kind.lower()+'_scalar',aw['extra_fp32_flops'],aw.get('fp32_memory_bytes',0),fp32=True)
    ring('tp_attention',m['hidden_dim']*2)

    def bridge(name):
        # A lane mapping (e_F*TP_F+t_F)%TP_A is uniform for these candidates.
        a_bytes = m['top_k']*m['hidden_dim']*2*tp_f/tp_a
        f_bytes = batch*m['top_k']*m['hidden_dim']*2/ep
        work, temporary = {}, []
        for rank in active_a:
            direction = 'tx' if name=='dispatch' else 'rx'
            work[rank+'.SU.'+direction] = a_bytes
            work[rank+'.hbm'] = a_bytes
            temporary.append(alloc(rank,name+'.buffer',a_bytes,'communication_buffer','fp16'))
        for rank in pools['ffn']:
            direction = 'rx' if name=='dispatch' else 'tx'
            work[rank+'.SU.'+direction] = f_bytes
            work[rank+'.hbm'] = f_bytes
            temporary.append(alloc(rank,name+'.buffer',f_bytes,'communication_buffer','fp16'))
        total = batch*m['top_k']*m['hidden_dim']*2*tp_f
        assert math.isclose(sum(work[r+'.SU.'+('tx' if name=='dispatch' else 'rx')] for r in active_a),total)
        assert math.isclose(sum(work[r+'.SU.'+('rx' if name=='dispatch' else 'tx')] for r in pools['ffn']),total)
        traffic[name] += total
        event(name,work,component='MoE',startup_s=startup,communication=True,
              allocs=temporary,releases=[a.id for a in temporary])

    bridge('dispatch')
    kernel('experts',2*expert_active/layers*batch/(tp_f*ep),
           experts/layers*fraction/(tp_f*ep),role='ffn')
    bridge('combine')
    excess = max(0.,tp_f/tp_a-1)
    if excess:
        # Additional local merges beyond the equal-TP baseline's fused combine.
        merge_elements = m['top_k']*m['hidden_dim']*excess
        kernel('afd_additional_partial_merge',merge_elements,merge_elements*6,fp32=True)
    ring('tp_ffn',m['hidden_dim']*2)
    g.events.append(Event('done',(previous,),releases=tuple(a.id for a in dynamic),
                          metadata={'phase':'decode','operation':'barrier'}))
    g.metadata.update(scenario='inference',topology={'npus_per_host':8},
        token_completion_events={'0':['done']},
        sequence_token_completion_events={'0':{str(i):['done'] for i in range(batch)}})
    g.metadata['ratio_study'] = {'tp_a':tp_a,'tp_f':tp_f,'ep':ep,'batch':batch,
        'active_cards':a_count+f_count,'idle_cards':h['npu_count']-a_count-f_count,
        'traffic_bytes_per_layer':dict(traffic),'attention_state_bytes_per_owned_rank':state,
        'attention_work_per_layer':aw,'resident_bytes_by_category':dict(resident_categories),
        'full_model_state_resident':True,'partial_merge_excess_factor':excess,
        'bridge_mapping':'tA=(eF*TP_F+tF)%TP_A',
        'compute_symmetry':'Representative per-role kernel; all equal replicas have disjoint resources and global stage barriers.',
        'network_scope':'All active physical endpoints, no oversubscribed switches; link bandwidth >= endpoint injection; endpoint envelope covers pair-link constraints.',
        'active_parameter_flops_total':2*batch*m['active_params'],
        'layout_validation':'Kimi A heads divisible by TP_A; FFN remains declared-budget sharding. V4.1 o_groups8 native layout not established.'}
    return g, a_count, f_count


def run_candidate(config, profile, tp_a, tp_f, *, zero_startup=False, evidence=None):
    records, kinds, traffic = [], {}, defaultdict(float)
    peak_a = peak_f = 0.
    status, diagnostics = 'feasible', []
    for kind, count in profile['counts'].items():
        g, a_count, f_count = build_layer(config,profile,kind,tp_a,tp_f,zero_startup=zero_startup)
        sim = simulate(g,retain_event_payload=False)
        if sim['status']!='feasible':
            status = sim['status']
            diagnostics.extend(sim['diagnostics'])
        record = window_breakdown(g,sim,'done')
        if record:
            records.append(combine_breakdowns([record]*count,divisor=1))
        peak_a = max(peak_a,max((sim['memory_peaks_bytes'].get(f'npu{i}.hbm',0) for i in range(a_count)),default=0))
        peak_f = max(peak_f,max((sim['memory_peaks_bytes'].get(f'npu{i}.hbm',0) for i in range(a_count,a_count+f_count)),default=0))
        for name,value in g.metadata['ratio_study']['traffic_bytes_per_layer'].items():
            traffic[name] += count*value
        kinds[kind] = {'count':count,'status':sim['status'],'layer_seconds':sim['duration_s'],
                       'total_seconds':count*sim['duration_s'] if sim['status']=='feasible' else None}
        if evidence is not None:
            path=Path(evidence)/f'{kind}.json.gz'
            path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(path,'wt',encoding='utf-8',compresslevel=1) as f:
                json.dump({'scope':'one_attention_kind_layer_with_all_model_state_resident',
                           'graph':asdict(g),'simulation':sim},f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        del g,sim
    full = combine_breakdowns(records,divisor=1) if status=='feasible' else None
    seconds = full['total_s'] if full else None
    comm = sum(o['total_s'] for o in full['operator_breakdown']['operators'] if o['operator'] in COMMUNICATION) if full else None
    fraction=Fraction(tp_a,tp_f)
    result = {'architecture_id':profile['architecture_id'],'tp_a':tp_a,'tp_f':tp_f,
        'ratio':f'{fraction.numerator}:{fraction.denominator}','attention_cards':tp_a*config['execution']['ep'],
        'ffn_cards':tp_f*config['execution']['ep'],'active_cards':(tp_a+tp_f)*config['execution']['ep'],
        'idle_cards':config['hardware']['npu_count']-(tp_a+tp_f)*config['execution']['ep'],
        'variant':'zero_network_startup' if zero_startup else 'baseline',
        'status':status if status!='feasible' else ('feasible' if seconds<=.001 else 'slo_failed'),
        'capacity_status':status,'target_tpot_ms':1.,'tpot_ms':seconds*1000 if seconds is not None else None,
        'comm_ms':comm*1000 if comm is not None else None,
        'comm_pct':100*comm/seconds if seconds else None,
        'peak_a_GB':peak_a/1e9,'peak_f_GB':peak_f/1e9,'hbm_capacity_GB':288.,
        'transport_bytes':sum(traffic.values()),'transport_by_semantic_bytes':dict(traffic),
        'latency_breakdown':full,'attention_kinds':kinds,'diagnostics':diagnostics}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--skip-best-evidence',action='store_true')
    args=parser.parse_args()
    directory=args.output/'ratio-search'
    directory.mkdir(parents=True,exist_ok=True)
    source_hashes=fingerprints()
    gc.disable()
    manifest=load(args.output/'manifest.json')
    bases={r['architecture_id']:r for r in manifest['cases'] if r.get('afd_path')}
    rows=[]
    checks=[]
    inputs={}
    for arch,item in bases.items():
        config=load(args.output/item['config_path'])
        profile=load(args.output/item['attention_profile_path'])
        inputs[arch]=(config,profile)
        base=load(args.output/item['result_path'])
        equal=run_candidate(config,profile,12,12)
        delta=equal['tpot_ms']-base['metrics']['tpot_s']*1000
        checks.append({'architecture':arch,'check':'equal_TP_12_12_baseline_regression','difference_ms':delta,
                       'passed':math.isclose(equal['tpot_ms'],base['metrics']['tpot_s']*1000,rel_tol=1e-10,abs_tol=1e-10)})
        if not checks[-1]['passed']:
            save(directory/'verification.json',{'status':'failed','checks':checks})
            raise AssertionError(checks[-1])
        print(json.dumps({'stage':'equal_ratio_verified','architecture':arch,'tpot_ms':equal['tpot_ms']}),flush=True)
    if args.verify_only:
        save(directory/'verification.json',{'status':'passed','checks':checks,'source_sha256':source_hashes})
        return
    for arch,(config,profile) in inputs.items():
        for variant in ('baseline','zero_network_startup'):
            for tp_a,tp_f in CANDIDATES:
                assert source_hashes==fingerprints(),'Numerical source changed during execution'
                started=time.perf_counter()
                row=run_candidate(config,profile,tp_a,tp_f,zero_startup=variant!='baseline')
                row['wall_s']=time.perf_counter()-started
                name=f'{arch}-a{tp_a}-f{tp_f}-{variant}'
                row['config_path']=f'configs/{name}.json'
                row['result_path']=f'cases/{name}.json'
                save(directory/row['config_path'],{'native_budget_config':config,'attention_profile':profile,
                    'ratio_sidecar':{'tp_a':tp_a,'tp_f':tp_f,'zero_network_startup':variant!='baseline'},
                    'source_sha256':source_hashes,
                    'rerun':'python scripts/scaling_afd_ratios.py --output outputs/scaling-10case-20261113'})
                save(directory/row['result_path'],row)
                rows.append(row)
                save(directory/'summary.json',{'rows':rows,'complete':False,'source_sha256':source_hashes})
                print(json.dumps({'stage':'candidate_complete','architecture':arch,'a':tp_a,'f':tp_f,
                                  'variant':variant,'tpot_ms':row['tpot_ms'],'wall_s':row['wall_s']}),flush=True)
    best={}
    for arch in inputs:
        best[arch]={}
        for variant in ('baseline','zero_network_startup'):
            valid=[r for r in rows if r['architecture_id']==arch and r['variant']==variant and r['capacity_status']=='feasible']
            selected=min(valid,key=lambda r:(r['tpot_ms'],r['active_cards']))
            best[arch][variant]={k:v for k,v in selected.items() if k not in ('latency_breakdown','attention_kinds','diagnostics')}
            if variant=='baseline' and not args.skip_best_evidence:
                config,profile=inputs[arch]
                evidence=directory/'best-evidence'/arch
                actual=run_candidate(config,profile,selected['tp_a'],selected['tp_f'],evidence=evidence)
                assert math.isclose(actual['tpot_ms'],selected['tpot_ms'],rel_tol=1e-12)
                result={'status':actual['status'],'metrics':{'ttft_s':None,'tpot_s':actual['tpot_ms']/1000,
                    'latency_breakdown':{'ttft':None,'tpot':actual['latency_breakdown'],'ttft_unavailable_reason':'warm_decode_only'}},'config':config}
                write_latency_csv(result,evidence)
                (evidence/'report.md').write_text('\n'.join(['# AFD配比候选最优结果','',
                    f"{arch}: 970DT:LPU={selected['ratio']}；A TP={selected['tp_a']}，F TP={selected['tp_f']}；TPOT={selected['tpot_ms']:.6f} ms。",
                    '', '仅限声明的9个候选、均匀路由、环形TP归约和当前Attention/KV预算模板；不是全局或实机最优。',
                    *latency_markdown(result['metrics'])]),encoding='utf-8')
    for arch in inputs:
        exact=next(r for r in rows if r['architecture_id']==arch and r['variant']=='baseline' and r['tp_a']==16 and r['tp_f']==4)
        checks.append({'architecture':arch,'check':'exact_4_to_1_idle_budget','passed':exact['idle_cards']==2048 and exact['ratio']=='4:1'})
        for row in (r for r in rows if r['architecture_id']==arch):
            config,profile=inputs[arch]
            expected=config['workload']['batch_size']*config['model']['top_k']*config['model']['hidden_dim']*2*row['tp_f']*profile['layers']
            for semantic in ('dispatch','combine'):
                assert math.isclose(row['transport_by_semantic_bytes'][semantic],expected,rel_tol=1e-12)
            if row['latency_breakdown']:
                rec=row['latency_breakdown']
                assert math.isclose(sum(rec['components_s'].values()),rec['total_s'],rel_tol=1e-12)
                assert math.isclose(sum(o['total_s'] for o in rec['operator_breakdown']['operators']),rec['total_s'],rel_tol=1e-12)
    checks.append({'check':'all_36_traffic_and_additive_latency','passed':True})
    summary={'rows':rows,'complete':True,'best_by_architecture':best,'source_sha256':source_hashes,
             'candidate_count_per_architecture_per_variant':len(CANDIDATES),
             'objective':'Minimum warm decode TPOT among explicitly evaluated capacity-feasible candidates; total POD <=12288, EP512, whole POD batch32.',
             'assumptions':['Uniform fractional expert routing and nonblocking SU with identical per-direction endpoints.',
                'A hidden is fully reduced before dispatch; each F TP shard receives from one mapped A lane. F partials return to A followed by sum/broadcast.',
                'Additional partial merge beyond equal-TP baseline charges FP32 adds and 2-read/1-write FP16 traffic when TP_F>TP_A.',
                'Pair-link bandwidth is no smaller than endpoint injection, so its demand bound is included by endpoint constraints.',
                'Kimi Attention head TP divisibility is preserved; declared FFN-budget sharding and V4.1 native grouped layout remain unvalidated.',
                'Zero startup is a diagnostic limit and not a realized platform; no TTFT, service queue or P99 simulation.']}
    save(directory/'summary.json',summary)
    fields=['architecture_id','variant','tp_a','tp_f','ratio','attention_cards','ffn_cards','active_cards','idle_cards','status','capacity_status',
            'target_tpot_ms','tpot_ms','comm_ms','comm_pct','peak_a_GB','peak_f_GB','transport_bytes','config_path','result_path']
    with (directory/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    save(directory/'verification.json',{'status':'passed','checks':checks,'source_sha256':source_hashes})
    print(json.dumps({'stage':'complete','rows':len(rows),'best_by_architecture':best},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

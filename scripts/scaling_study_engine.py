"""Reproducible ten-case 2026-11-13 MoE scaling study.

This experiment adapter keeps the user's exact parameter budgets and executes
the repository's discrete-event engine on a uniform representative layer.
All model layers' resident weights and KV are present in every memory check.
The native configuration is intentionally supplemented by a sidecar for AFD;
AFD is not advertised as a native Config capability.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim.config import Config, load_config
from supernode_sim.analysis import graph_estimate, evaluate_targets
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import combine_breakdowns, latency_breakdown_metadata, window_breakdown
from supernode_sim.report import markdown_report, write_latency_csv, _csv

DEFAULT_OUTPUT = ROOT / 'outputs/scaling-10case-20261113'
MODELS = [
    {'label': '2.8T', 'params': 2.8e12, 'active': 104e9, 'experts': 512, 'topk': 16, 'ep': 128, 'hidden': 6144, 'heads': 96},
    {'label': '10T', 'params': 10e12, 'active': 250e9, 'experts': 2048, 'topk': 32, 'ep': 512, 'hidden': 12288, 'heads': 192},
    {'label': '100T', 'params': 100e12, 'active': 1e12, 'experts': 8192, 'topk': 64, 'ep': 2048, 'hidden': 24576, 'heads': 384},
]
HARDWARE = {
    '950DT': {'cards': 1024, 'flops': 1e15, 'network': 2e12, 'hbm': 144e9, 'hbm_Bps': 4e12},
    '960DT': {'cards': 4096, 'flops': 2e15, 'network': 2.2e12, 'hbm': 288e9, 'hbm_Bps': 9.6e12},
    '970DT': {'cards': 12288, 'flops': 4e15, 'network': 4e12, 'hbm': 288e9, 'hbm_Bps': 14.4e12},
}
AFD = {'attention_count': 6144, 'ffn_count': 6144, 'tp': 12, 'ep': 512,
       'lpu_flops': 8e15, 'lpu_hbm_Bps': 144e12, 'lpu_hbm_bytes': 288e9}
SOURCE_FILES = ['scripts/scaling_study_engine.py','scripts/scaling_study_graph.py',
               'scripts/scaling_kimi_attention.py','scripts/scaling_v41_attention.py',
               'supernode_sim/core.py','supernode_sim/config.py','supernode_sim/engine.py',
               'supernode_sim/topology.py','supernode_sim/analysis.py','supernode_sim/latency_breakdown.py',
               'supernode_sim/operator_costs.py','supernode_sim/moe_wire.py',
               'supernode_sim/inference.py','supernode_sim/sparse_attention.py']
# Capture bytes once, when this process loads. Never label a later on-disk
# revision as the source of an already-imported implementation.
SOURCE_SNAPSHOT = {name:(ROOT/name).read_bytes() for name in SOURCE_FILES if (ROOT/name).exists()}
SOURCE_HASHES = {name:hashlib.sha256(content).hexdigest() for name,content in SOURCE_SNAPSHOT.items()}


def save(path: Path, value, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2)


def provenance(attention_profile=None):
    changed=[name for name,digest in SOURCE_HASHES.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed:
        raise RuntimeError('Numerical source changed after process startup: '+', '.join(changed))
    result={'python':platform.python_version(),'python_executable':sys.executable,
            'command':[sys.executable,*sys.argv],
            'source_sha256':dict(SOURCE_HASHES),'source_snapshot_scope':'process_startup_bytes_verified_unchanged_before_export'}
    for key,args in [('git_head',['git','rev-parse','HEAD']),('git_status',['git','status','--short'])]:
        try: result[key]=subprocess.check_output(args,cwd=ROOT,text=True,encoding='utf-8').strip()
        except (OSError,subprocess.SubprocessError): result[key]=None
    result['source_identity_note']='Source bytes captured at process startup; every export checks files remained unchanged. git HEAD alone excludes uncommitted changes.'
    if attention_profile:
        result['attention_profile_sha256']=hashlib.sha256(json.dumps(attention_profile,sort_keys=True).encode()).hexdigest()
    return result


def make_config(model, hardware, target_ms, *, afd=None, attention_profile=None):
    h = HARDWARE[hardware]
    fraction = (1 - model['active'] / model['params']) / (1 - model['topk'] / model['experts'])
    tp = h['cards'] // model['ep']
    data = {
        'schema_version': '1.0',
        'model': {'name': f"synthetic-exact-budget-{model['label']}-MoE", 'total_params': model['params'],
                  'active_params': model['active'], 'layers': 96, 'hidden_dim': model['hidden'],
                  'intermediate_dim': model['hidden'] * 4, 'num_heads': model['heads'],
                  'kv_heads': model['heads'], 'head_dim': 64, 'attention_kind': 'latent',
                  'kv_latent_dim': 512, 'kv_rope_dim': 64, 'num_experts': model['experts'],
                  'top_k': model['topk'], 'moe_layer_fraction': 1.0,
                  'expert_parameter_fraction': fraction},
        'hardware': {'name': hardware + (' + FFN LPU (AFD sidecar)' if afd else ''),
                     'npu_count': h['cards'], 'su_size': h['cards'], 'npus_per_host': 8,
                     'topology': 'full_mesh', 'npu': {'hbm_bytes': h['hbm'], 'hbm_Bps': h['hbm_Bps'],
                     'flops': {'fp8': h['flops'], 'fp16': h['flops']/2, 'bf16': h['flops']/2, 'fp32': h['flops']/20},
                     'efficiency': 0.5, 'runtime_bytes': 8e9, 'workspace_bytes': 256e6},
                     'scale_up': {'bandwidth_Bps': h['network'], 'injection_Bps': h['network'], 'efficiency': 0.8, 'latency_s': 2.1e-6}},
        'workload': {'scenario': 'inference', 'batch_size': 32, 'input_length': 1048576, 'output_length': 2},
        'execution': {'tp': tp, 'ep': model['ep'], 'pp': 1, 'dp': 1, 'fsdp': 1,
                      'all_to_all_algorithm': 'direct', 'sequence_parallel': False,
                      'precision': {'weight': 'fp8', 'compute': 'fp8', 'kv': 'fp16',
                                    'activation': 'fp16', 'communication': 'fp16', 'weight_transfer': 'fp8'},
                      'features': {'weight_scale_block': 128, 'scale_bytes': 4, 'kv_offload': 'none'}},
        'targets': {'tpot_s': target_ms / 1000},
        'sources': {
            'model.parameter_budgets_and_moe': {'status': 'user', 'description': 'Exact requested total/active parameters, expert count, top-k and EP; no parameter rounding.'},
            'hardware.nominal': {'status': 'user', 'description': 'Requested card count, FP8 peak FLOPs, memory capacity/bandwidth and nominal interconnect; 1K=1024,4K=4096,12K=12288.'},
            'workload.batch_size': {'status': 'user', 'description': '32 requests total over the entire POD, confirmed by user.'},
            '*': {'status': 'assumed', 'description': '96 uniform all-latent-attention MoE layers; synthetic architecture dimensions and uniform routing. Analytical uncalibrated performance.'},
            'hardware.scale_up': {'status': 'assumed', 'description': 'Requested network rate interpreted per card per direction; logical nonblocking full-mesh with independent endpoints, efficiency 0.8, 2.1us path startup. Directionality not supplied by user.'},
            'hardware.npu.efficiency': {'status': 'assumed', 'description': '0.5 compute efficiency, 8GB runtime and 256MB workspace inherited from repository 950DT example; HBM uses nominal rate.'},
        },
        'assumptions': [
            'Synthetic architecture: 96 uniform MoE layers, hidden widths 6144/12288/24576 and heads 96/192/384 with head_dim64; exact user parameter budgets retained independently of these workload dimensions.',
            'All 96 attention layers use a 512+64-element compressed latent KV cache, FP16, replicated across TP; no KDA/sparse attention/context parallelism is assumed.',
            '1M means 1048576 tokens. Fixed batch32 is the total POD request count; sequences assigned round-robin to EP lanes.',
            'Shared parameter budget D=(active-total*topk/experts)/(1-topk/experts), routed budget=total-D; no arbitrary fixed expert fraction.',
            'FP8 weights and compute, FP16 KV/activation/communication; uniform fractional expert routing and expected distinct-expert weight reads.',
            'Direct concurrent all-to-all over a logical nonblocking POD; results depend on the collective algorithm and supplied startup assumption.',
            'One representative layer runs in the actual discrete-event engine with all96layers resident. Uniform serial layer service is repeated96times; no inter-layer overlap.',
            'Independent warm-cache decode has TTFT unavailable when prefill is infeasible; output_length2 measures one first decode interval, not steady multi-token or P99 latency.',
            'TPOT is a target, not a forced measured latency. Failure of capacity or SLO is retained.',
        ],
    }
    if afd:
        data['assumptions'].append('AFD sidecar partitions the12288devices into6144Attention970DT plus6144FFN LPU, TP12 EP512 per pool. Native Config tp24 only validates the physical POD envelope; execute with this experiment runner and sidecar, not native simulate.')
        data['sources']['AFD'] = {'status': 'user', 'description': 'User-confirmed12Ktotal devices, equal6KAttention+6KFFN pools; LPU8PFLOPS FP8 and144TB/s HBM, network4TB/s. LPU HBM capacity288GB is assumed.'}
    if attention_profile:
        data['model']['layers'] = attention_profile['layers']
        data['model']['num_heads'] = attention_profile.get('num_heads', 96)
        data['model']['kv_heads'] = data['model']['num_heads']
        data['model']['name'] += '-' + attention_profile['family'] + '-attention-template'
        data['sources']['attention_template'] = {'status':'assumed','description':attention_profile['scope']}
        data['sources']['*']['description'] = 'Exact user parameter budgets plus separately sourced Attention/KV template. Projection weights are budget-scaled, not shape-faithful checkpoint weights. Analytical uncalibrated performance.'
        data['assumptions'] = [row for row in data['assumptions'] if not row.startswith(('Synthetic architecture:', 'All 96 attention', 'One representative layer'))]
        data['assumptions'] += [attention_profile['scope'],
            f"Attention/KV template has {attention_profile['layers']} layers with counts {attention_profile['counts']}; representative layers execute with all template states resident and are summed by these counts.",
            'Attention-core head/cache dimensions follow the reference template; dense/shared projection weights are rescaled to exact user budgets, so this is not a shape-faithful checkpoint reconstruction.',
            'Three model scales retain synthetic MoE hidden widths6144/12288/24576 for dispatch/combine; no weight-shape or quality equivalence is claimed.',
            'Main experiment is warm-cache decode only. TTFT is undefined; no successful prefill or end-to-end deployment is implied.',
        ]
    return Config.model_validate(data)


def prepare(directory: Path):
    from scripts.scaling_kimi_attention import default_profile as kimi_profile
    from scripts.scaling_v41_attention import default_profile as v41_profile
    cases = []
    for hw in ('960DT', '970DT'):
        for model in MODELS:
            cases.append((f"model-{hw}-{model['label'].replace('.', 'p')}", 'model_scaling', model, hw, 5, None))
    for hw, target in [('950DT', 10), ('960DT', 5), ('970DT', 1)]:
        cases.append((f'hardware-{hw}-10T', 'hardware_scaling', MODELS[1], hw, target, None))
    cases.append(('hardware-970DT-LPU-10T', 'hardware_scaling', MODELS[1], '970DT', 1, AFD))
    manifest = {'study_date': '2026-11-13', 'method': 'exact-budget-uniform-layer-discrete-event-study', 'cases': []}
    for profile in (kimi_profile(),v41_profile()):
        architecture_id=profile['architecture_id']
        profile_path=f'profiles/{architecture_id}.json'
        save(directory/profile_path,profile)
        for base_id, group, model, hw, target, afd in cases:
            case_id=architecture_id+'-'+base_id
            config = make_config(model, hw, target, afd=afd, attention_profile=profile)
            case_dir = directory / 'cases' / case_id
            save(case_dir / 'config.json', config.model_dump())
            if afd:
                save(case_dir / 'afd.json', afd)
            row = {'id': case_id, 'group': group, 'model_label': model['label'], 'hardware_label': hw + ('+LPU' if afd else ''),
                   'model': model['label'], 'hardware': hw + ('+LPU' if afd else ''), 'target_tpot_ms': target,
                   'architecture_id':architecture_id,'architecture_label':profile['display_label'],
                   'display_label':profile['display_label'], 'attention_profile_path':profile_path,
                   'config_path': f'cases/{case_id}/config.json', 'result_path': f'cases/{case_id}/result.json',
                   'afd_path': f'cases/{case_id}/afd.json' if afd else None}
            manifest['cases'].append(row)
    save(directory / 'manifest.json', manifest)
    manifest['provenance']=provenance()
    for name,content in SOURCE_SNAPSHOT.items():
        target=directory/'source-snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    save(directory/'source-snapshot/manifest.json',manifest['provenance'])
    save(directory / 'manifest.json', manifest)
    return manifest


def _scaled_communication(study, layers, fallback):
    semantics = study.get('per_semantic_bytes_total', fallback['payload_by_semantic_bytes'])
    physical = study.get('physical_bytes_by_domain_total', study.get('physical_bytes_total', sum(fallback['physical_bytes_by_domain'].values())))
    if isinstance(physical, dict):
        physical_domains = {k: v * layers for k, v in physical.items()}
    else:
        physical_domains = {'scale_up': physical * layers}
    return {'transport_payload_bytes': study.get('transport_payload_bytes_total', fallback['transport_payload_bytes']) * layers,
            'physical_bytes_by_domain': physical_domains,
            'payload_by_semantic_bytes': {k: v * layers for k, v in semantics.items()},
            'payload_by_collective_bytes': {},
            'classification_note': 'Full POD demand totals from graph multiplicities; semantics and collective types are alternative views.'}


def _run_kind(config: Config, *, phase='decode', afd=None, output=None, attention_profile=None, attention_kind=None, repetitions=None):
    from scripts.scaling_study_graph import build_layer, attribution_graph
    # Match the repository's large-batch runners: graph references are IDs,
    # and repeated cycle scans over large acyclic event dictionaries are wasteful.
    gc.disable()
    started = time.perf_counter()
    graph = build_layer(config, phase=phase, afd=afd, attention_profile=attention_profile, attention_kind=attention_kind)
    print(json.dumps({'stage':'graph_built','phase':phase,'events':len(graph.events),'wall_s':time.perf_counter()-started}),flush=True)
    estimate = graph_estimate(graph)
    estimate.pop('optimistic_event_completion_s', None)
    print(json.dumps({'stage':'estimate_complete','phase':phase,'wall_s':time.perf_counter()-started}),flush=True)
    simulation = simulate(graph, retain_event_payload=False)
    print(json.dumps({'stage':'simulation_complete','phase':phase,'status':simulation['status'],'wall_s':time.perf_counter()-started}),flush=True)
    layer_count = repetitions if repetitions is not None else config.model.layers
    record = window_breakdown(attribution_graph(graph), simulation, 'infer.d0.token1')
    full_record = combine_breakdowns([record] * layer_count, divisor=1) if record else None
    if full_record:
        full_record.update(replica=0, sequence=0, interval_count=1)
    study = graph.metadata.get('study', {})
    communication = _scaled_communication(study, layer_count, estimate['communication'])
    stage = {'status': simulation['status'], 'duration_s': full_record['total_s'] if full_record else None,
             'layer_duration_s': simulation['duration_s'], 'layer_count': layer_count,
             'latency_breakdown': full_record, 'communication': communication,
             'memory_peaks_bytes': simulation['memory_peaks_bytes'],
             'memory_categories_at_peak_bytes': simulation['memory_categories_at_peak_bytes'],
             'diagnostics': simulation['diagnostics'], 'graph_metadata': study,
             'representative_layer_events': simulation['total_events']}
    stage['resident_memory_bytes'] = estimate['resident_memory_bytes']
    if full_record:
        service = sum(v for k,v in full_record['components_s'].items() if k.startswith('communication_'))
        communication_operators = {'dispatch','combine','tp_attention','tp_ffn','tp_index_scores'}
        comm_time = sum(row['total_s'] for row in full_record['operator_breakdown']['operators'] if row['operator'] in communication_operators)
        stage.update(communication_service_s=service, communication_operator_s=comm_time,
                     communication_service_share_pct=100*service/full_record['total_s'],
                     communication_operator_share_pct=100*comm_time/full_record['total_s'])
    if output:
        # Exact representative execution, explicitly not a full96layer event replay.
        representative_path=Path(output)/f'{phase}.{attention_kind or "uniform"}.representative-layer.json.gz'
        representative_path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(representative_path,'wb',compresslevel=1) as handle:
            handle.write(json.dumps({'scope':'representative_attention_kind_layer_with_all_model_state_resident',
                         'graph_metadata':graph.metadata,'simulation':simulation},ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))
    estimate['duration_lower_bound_s'] *= layer_count
    estimate['dependency_path_lower_bound_s'] *= layer_count
    for field in ('resource_time_lower_bounds_s','resource_work'):
        estimate[field] = {k:v*layer_count for k,v in estimate[field].items()}
    estimate['communication']=communication
    simulation=copy.copy(simulation)
    simulation.update(duration_s=stage['duration_s'],trace=[],memory_log=[])
    for field in ('resource_work','resource_busy_s'):
        simulation[field]={k:v*layer_count for k,v in simulation[field].items()}
    simulation['resources']={k:{**v,'work':v['work']*layer_count,'busy_s':v['busy_s']*layer_count} for k,v in simulation['resources'].items()}
    return stage, graph, estimate, simulation


def run_phase(config:Config, *, phase='decode',afd=None,output=None,attention_profile=None):
    kinds=attention_profile['counts'] if attention_profile else {'uniform':config.model.layers}
    assets=[]
    for kind,count in kinds.items():
        assets.append(_run_kind(config,phase=phase,afd=afd,output=output,attention_profile=attention_profile,
                                attention_kind=None if kind=='uniform' else kind,repetitions=count))
    stage,graph,estimate,simulation=assets[0]
    if len(assets)==1:
        return stage,graph,estimate,simulation
    stage=copy.deepcopy(stage)
    estimate=copy.deepcopy(estimate)
    simulation=copy.deepcopy(simulation)
    records=[row[0]['latency_breakdown'] for row in assets]
    valid=all(row[0]['status']=='feasible' for row in assets)
    stage['status']='feasible' if valid else next(row[0]['status'] for row in assets if row[0]['status']!='feasible')
    stage['latency_breakdown']=combine_breakdowns(records,divisor=1) if valid else None
    stage['duration_s']=stage['latency_breakdown']['total_s'] if valid else None
    stage['layer_count']=config.model.layers
    stage['attention_kinds']={kind:{'count':count,'status':row[0]['status'],'duration_s':row[0]['duration_s'],
                                   'layer_duration_s':row[0]['layer_duration_s']} for (kind,count),row in zip(kinds.items(),assets)}
    stage['diagnostics']=[item for row in assets for item in row[0]['diagnostics']]
    stage['representative_layer_events']=sum(row[0]['representative_layer_events'] for row in assets)
    for other,_,other_estimate,other_sim in assets[1:]:
        for field in ('communication_service_s','communication_operator_s'):
            if field in stage and field in other: stage[field]+=other[field]
        stage['communication']['transport_payload_bytes']+=other['communication']['transport_payload_bytes']
        for field in ('physical_bytes_by_domain','payload_by_semantic_bytes'):
            for key,value in other['communication'][field].items():
                stage['communication'][field][key]=stage['communication'][field].get(key,0)+value
        for key,value in other['memory_peaks_bytes'].items():
            if value>stage['memory_peaks_bytes'].get(key,0):
                stage['memory_peaks_bytes'][key]=value
                stage['memory_categories_at_peak_bytes'][key]=other['memory_categories_at_peak_bytes'].get(key,{})
        for field in ('duration_lower_bound_s','dependency_path_lower_bound_s'):
            estimate[field]+=other_estimate[field]
        for field in ('resource_time_lower_bounds_s','resource_work'):
            for key,value in other_estimate[field].items(): estimate[field][key]=estimate[field].get(key,0)+value
        for field in ('resource_work','resource_busy_s'):
            for key,value in other_sim[field].items(): simulation[field][key]=simulation[field].get(key,0)+value
        for key,row in other_sim['resources'].items():
            if key not in simulation['resources']: simulation['resources'][key]=copy.copy(row)
            else:
                simulation['resources'][key]['work']+=row['work']
                simulation['resources'][key]['busy_s']+=row['busy_s']
    if valid:
        stage['communication_service_share_pct']=100*stage['communication_service_s']/stage['duration_s']
        stage['communication_operator_share_pct']=100*stage['communication_operator_s']/stage['duration_s']
    simulation.update(status=stage['status'],duration_s=stage['duration_s'],memory_peaks_bytes=stage['memory_peaks_bytes'],
                      memory_categories_at_peak_bytes=stage['memory_categories_at_peak_bytes'],diagnostics=stage['diagnostics'])
    estimate['communication']=stage['communication']
    # Later kinds may add FP32/selection resources. Union their definitions.
    for _,other_graph,_,_ in assets[1:]: graph.resources.update(other_graph.resources)
    return stage,graph,estimate,simulation


def export_result(result, directory):
    """Keep standard report/CSV detail without duplicating it into huge JSON."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    save(directory/'config.resolved.json',result['config'])
    resources=result['simulation']['resources']
    _csv(directory/'resources.csv', ['resource','unit','work_unit','capacity','work','utilization','busy_s',
                                   'average_rate_at_lower_bound','required_average_rate_lower_bound'],
         ({'resource':key,**item,**resources.get(key,{})} for key,item in result['requirements']['per_resource'].items()))
    resident=result['estimate']['resident_memory_bytes']
    optimistic=result['estimate']['optimistic_schedule_memory_peak_bytes']
    peaks=result['simulation']['memory_peaks_bytes']
    _csv(directory/'memory.csv',['pool','capacity_bytes','resident_bytes','optimistic_peak_bytes','simulated_peak_bytes'],
         ({'pool':pool,'capacity_bytes':result['memory_capacities_bytes'].get(pool),
           'resident_bytes':resident.get(pool,0),'optimistic_peak_bytes':optimistic.get(pool,0),
           'simulated_peak_bytes':peaks.get(pool)} for pool in sorted(set(resident)|set(optimistic)|set(peaks))))
    write_latency_csv(result,directory)
    report=markdown_report(result)
    compact=dict(result)
    compact['simulation']={key:value for key,value in result['simulation'].items()
                           if key not in ('resources','resource_work','resource_busy_s','trace','memory_log')}
    compact['estimate']={key:value for key,value in result['estimate'].items()
                         if key not in ('resource_work','resource_time_lower_bounds_s','resident_memory_categories_bytes')}
    compact['requirements']={'target_window_s':None,'per_resource_artifact':'resources.csv'}
    def compact_stage(stage):
        if stage is None:return None
        item={key:value for key,value in stage.items() if key not in
              ('memory_peaks_bytes','memory_categories_at_peak_bytes','resident_memory_bytes')}
        item.update(max_hbm_peak_bytes=max((value for pool,value in stage['memory_peaks_bytes'].items() if pool.startswith('npu')),default=0),
                    max_resident_hbm_bytes=max((value for pool,value in stage['resident_memory_bytes'].items() if pool.startswith('npu')),default=0),
                    per_pool_memory_artifact='memory.csv')
        return item
    compact['study']={key:value for key,value in result['study'].items() if key not in ('stages','decode','prefill')}
    compact['study']['decode']=compact_stage(result['study']['decode'])
    compact['study']['prefill']=compact_stage(result['study']['prefill'])
    compact['study']['stages']={key:compact_stage(value) for key,value in result['study']['stages'].items()}
    compact['artifact_export']={'detail_mode':'study_summary_with_complete_csv_and_compressed_layer_evidence',
        'resource_detail':'resources.csv','per_pool_memory':'memory.csv',
        'representative_simulations':[path.name for path in sorted(directory.glob('*.representative-layer.json.gz'))],
        'full_model_trace_available':False,
        'note':'Numerical metrics, capacities and critical-path/operator breakdowns are retained. Repeated resource ledgers are written once in CSV; complete representative engine results are gzip JSON.'}
    save(directory/'result.json',compact,compact=True)
    save(directory/'summary.json',compact,compact=True)
    (directory/'report.md').write_text(report,encoding='utf-8')
    return compact


def run_study(config: Config, *, afd=None, output=None, phases=('decode',),attention_profile=None):
    started = time.perf_counter()
    stages = {}
    decode_assets = None
    for phase in phases:
        print(json.dumps({'stage': 'phase_started', 'phase': phase, 'model': config.model.name, 'hardware': config.hardware.name}), flush=True)
        assets = run_phase(config, phase=phase, afd=afd, output=output,attention_profile=attention_profile)
        stages[phase] = assets[0]
        if phase == 'decode':
            decode_assets = assets
        print(json.dumps({'stage': 'phase_complete', 'phase': phase, 'status': assets[0]['status'], 'duration_s': assets[0]['duration_s'], 'wall_s': time.perf_counter()-started}), flush=True)
    if decode_assets is None:
        raise ValueError('decode phase required for this TPOT study')
    decode, graph, estimate, simulation = decode_assets
    prefill = stages.get('prefill')
    breakdown = latency_breakdown_metadata()
    breakdown.update(ttft=prefill['latency_breakdown'] if prefill else None, tpot=decode['latency_breakdown'],
                     ttft_unavailable_reason=None if prefill and prefill['status']=='feasible' else 'prefill_infeasible_or_not_executed',
                     tpot_unavailable_reason=None if decode['status']=='feasible' else 'decode_resident_or_execution_infeasible')
    breakdown['notes'].append(f'Representative attention-kind windows are summed over {config.model.layers} layers after execution with full-model resident memory; graph symmetry assumptions are recorded in study metadata.')
    tpot, ttft = decode['duration_s'], prefill['duration_s'] if prefill else None
    metrics = {'duration_s': ttft + tpot if ttft is not None and tpot is not None else tpot,
               'ttft_s': ttft, 'tpot_s': tpot, 'tpop_s': tpot, 'latency_breakdown': breakdown,
               'decode_throughput_tokens_s': config.workload.batch_size/tpot if tpot else None,
               'standalone_decode': ttft is None}
    # Normal result schema, with no fabricated expanded trace. The exact
    # representative trace is a separately named and scoped artifact.
    simulation = copy.copy(simulation)
    simulation.update(duration_s=tpot, trace=[], memory_log=[], trace_scope='representative_layer_saved_separately; no_full_model_event_trace')
    estimate['communication'] = decode['communication']
    slo = evaluate_targets(config, metrics, decode['status']=='feasible')
    status = decode['status'] if decode['status']!='feasible' else ('feasible' if slo['satisfied'] else 'slo_failed')
    snapshot = config.model_dump()
    result = {'schema_version': '1.0', 'engine_version': '0.1.0+scaling-study-adapter', 'mode': 'simulate', 'status': status,
              'config': snapshot, 'config_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
              'sources': snapshot['sources'], 'assumptions': snapshot['assumptions'],
              'units': {'memory':'byte','bandwidth':'byte/s','time':'second','compute':'FLOP'},
              'model_metadata': {**graph.metadata, 'fidelity': {'parameter_capacity':'declared_budget','hardware_performance':'uncalibrated'}},
              'memory_capacities_bytes': dict(graph.memory_caps), 'estimate': estimate,
              'simulation': simulation, 'metrics': metrics, 'slo': slo,
              'requirements': {'target_window_s': None, 'per_resource': {
                  k: {'work': amount, 'unit': graph.resources[k].unit, 'capacity': graph.resources[k].capacity,
                      'work_unit': 'FLOP' if graph.resources[k].unit=='FLOP/s' else 'byte',
                      'average_rate_at_lower_bound': amount/estimate['duration_lower_bound_s'] if estimate['duration_lower_bound_s'] else 0,
                      'required_average_rate_lower_bound': None}
                  for k,amount in estimate['resource_work'].items()}},
              'provenance':provenance(attention_profile),
              'study': {'method':'attention-kind-layer-repetition-with-full-resident-state', 'afd':afd,'attention_profile':attention_profile,
                        'stages': stages, 'prefill': prefill, 'decode': decode,
                        'wall_time_s': time.perf_counter()-started,
                        'resource_scope': 'Representative symmetry classes; full POD traffic is in stage.communication.',
                        'full_model_event_trace_available': False}}
    if output:
        compact=export_result(result,output)
        path = Path(output)/'report.md'
        existing = path.read_text(encoding='utf-8')
        afd_note = f'本AFD适配器实际每池TP={afd["tp"]}、EP={afd["ep"]}；config的TP={config.execution.tp}只描述总POD容量包络。' if afd else ''
        path.write_text(f'# 本研究执行口径\n\n本文件使用原引擎报告格式，数值来自独立预算模型实验适配器。完整模型常驻状态下按Attention种类实际执行代表层，再按模板次数累加到{config.model.layers}层；通信采用显式记录的对称约简与direct算法。{afd_note}未导出完整模型事件回放，逐事件证据保存在 `*.representative-layer.json.gz`。资源逐项详表在resources.csv，容量详表在memory.csv；result.json保留全部汇总、配置、时延与算子分解。\n\n' + existing, encoding='utf-8')
        return compact
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--decode-only', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--sensitivity-variant',action='append',choices=['zero_network_startup','half_network_bandwidth'])
    args = parser.parse_args()
    manifest = prepare(args.output) if args.prepare else json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
    if args.run:
        for row in manifest['cases']:
            if args.case and row['id'] not in args.case:
                continue
            config = load_config(args.output/row['config_path'])
            afd = json.loads((args.output/row['afd_path']).read_text(encoding='utf-8')) if row['afd_path'] else None
            output = args.output/'cases'/row['id']
            profile=json.loads((args.output/row['attention_profile_path']).read_text(encoding='utf-8')) if row.get('attention_profile_path') else None
            run_study(config, afd=afd, output=output, phases=('decode',),attention_profile=profile)
            gc.collect()
    if args.sensitivity:
        sensitivity_file=args.output/'sensitivity.json'
        rows = json.loads(sensitivity_file.read_text(encoding='utf-8')) if sensitivity_file.exists() else []
        for item in manifest['cases']:
            if args.case and item['id'] not in args.case:
                continue
            original = load_config(args.output/item['config_path'])
            afd = json.loads((args.output/item['afd_path']).read_text(encoding='utf-8')) if item['afd_path'] else None
            profile=json.loads((args.output/item['attention_profile_path']).read_text(encoding='utf-8')) if item.get('attention_profile_path') else None
            for variant in (args.sensitivity_variant or ('zero_network_startup','half_network_bandwidth')):
                data = original.model_dump()
                network = data['hardware']['scale_up']
                if variant=='zero_network_startup':
                    network['latency_s']=0
                else:
                    for key in ('bandwidth_Bps','injection_Bps','uplink_Bps'):
                        if network.get(key): network[key]/=2
                config = Config.model_validate(data)
                config_path=f'sensitivity-configs/{item["id"]}/{variant}.json'
                save(args.output/config_path,config.model_dump())
                result = run_study(config, afd=afd, phases=('decode',),attention_profile=profile)
                rows=[row for row in rows if (row['case'],row['variant'])!=(item['id'],variant)]
                rows.append({'case':item['id'],'variant':variant,'status':result['status'],
                             'tpot_s':result['metrics']['tpot_s'], 'latency_breakdown':result['metrics']['latency_breakdown']['tpot'],
                             'config_path':config_path,'provenance':result['provenance']})
                save(args.output/'sensitivity.json', rows)
        with (args.output/'sensitivity.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['case','variant','status','tpot_s'])
            writer.writeheader()
            writer.writerows({k:row[k] for k in writer.fieldnames} for row in rows)


if __name__ == '__main__':
    main()

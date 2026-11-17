"""Export CP fine semantic kernel time and non-additive resource demands.

Reads existing graph/trace evidence. Uses the native latest-finishing dependency
path (lexically smallest ID on ties) and native event-time attribution; no graph
is simulated or changed. FLOPs/capacity and bytes/bandwidth are concurrent solo
service demands, not independently observed portions of kernel elapsed time.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supernode_sim.core import Event,Graph,Resource
from supernode_sim.latency_breakdown import _WindowAttributor,_empty_components
from supernode_sim.operator_costs import empty_costs,operator_identity

DEFAULT=ROOT/'outputs/scaling-10case-20261113/sp-megakernel/cp-fine-operators'
LABELS={'matrix':'矩阵/GEMM类算子','pointwise_state':'向量、归约与状态算子',
        'cache_io':'缓存追加/转换算子','communication':'通信算子'}


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
def csv_save(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:(';'.join(v) if isinstance(v,list) else v) for k,v in r.items()} for r in rows)


def category(name,communications):
    if name in communications:return 'communication'
    if name=='atomic.cp_cache_append':return 'cache_io'
    if ('budget_matmul' in name or 'budget_gemm' in name or
            name in ('attention.qk_dot','attention.av_dot','index.score_dot')):return 'matrix'
    return 'pointwise_state'


def resource_max(event,resources,predicate):
    values=[(amount/resources[key].capacity,key,amount,resources[key].capacity)
            for key,amount in event.work.items() if amount>0 and predicate(key,resources[key])]
    return min(values,key=lambda v:(-v[0],v[1])) if values else (0.,None,0.,None)


def semantic_rows(rows):
    output=[];mapping={};checks=[]
    for row in rows:
        rec=row['latency_breakdown'];assert rec is not None
        comm={name for info in row['kinds'].values() for name in info['communication_operators']}
        groups={name:defaultdict(float) for name in LABELS};names=defaultdict(set)
        for op in rec['operator_breakdown']['operators']:
            kind=category(op['operator'],comm);mapping[op['operator']]=kind;g=groups[kind]
            g['total_ms']+=op['total_s']*1000;names[kind].add(op['operator'])
            for key,value in op['components_s'].items():g[key.removesuffix('_s')+'_attributed_ms']+=value*1000
        for kind,g in groups.items():
            output.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
                'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
                'kernel_launch_us':row['kernel_launch_us'],'category':kind,'category_label':LABELS[kind],
                'total_ms':g['total_ms'],'window_total_ms':row['fine_tpot_ms'],
                'window_share_pct':100*g['total_ms']/row['fine_tpot_ms'],
                'operators':sorted(names[kind]),**{k:v for k,v in g.items() if k!='total_ms'},
                'source_artifact':row['result_path']})
        assert math.isclose(sum(g['total_ms'] for g in groups.values()),row['fine_tpot_ms'],rel_tol=1e-10,abs_tol=1e-8)
        checks.append({'id':row['id'],'semantic_total_reconciles':True})
    return output,mapping,checks


def hydrate(raw):
    # Attribution needs observed timings plus work/resources/dependencies only.
    # It does not replay memory admission or reinterpret the stored trace.
    return Graph(resources={k:Resource(**v) for k,v in raw['resources'].items()},
        memory_caps=raw['memory_caps'],metadata=raw['metadata'],
        events=[Event(id=e['id'],deps=tuple(e['deps']),work=e['work'],startup_s=e['startup_s'],
                      slots=tuple(e['slots']),metadata=e['metadata']) for e in raw['events']])


def demand_rows(row,directory):
    groups={};path_evidence=[];per_kind=[]
    for kind,info in row['kinds'].items():
        artifact=info.get('evidence_artifact')
        if not artifact:raise ValueError('Missing actual graph/trace evidence for '+row['id']+'/'+kind)
        with gzip.open(directory/artifact,'rt',encoding='utf-8') as f:payload=json.load(f)
        graph=hydrate(payload['graph']);sim=payload['simulation'];assert sim['status']=='feasible'
        attr=_WindowAttributor(graph,sim);native=attr.window('infer.d0.token1')
        assert native is not None
        saved=info['layer_breakdown'];assert math.isclose(native['total_s'],saved['total_s'],abs_tol=1e-12)
        end=attr.rows['infer.d0.token1']['end_s'];current='infer.d0.token1';count=info['layers'];ids=[]
        native_ops={(o['phase'],o['component'],o['operator']):o['total_s']*1000 for o in native['operator_breakdown']['operators']}
        independently_grouped=defaultdict(float)
        while current is not None:
            event,timing=attr.events[current],attr.rows[current]
            if timing['end_s']<=0:break
            predecessor=min(event.deps,key=lambda d:(-attr.rows[d]['end_s'],d)) if event.deps else None
            ready=attr.rows[predecessor]['end_s'] if predecessor else 0.
            costs=empty_costs();parts=_empty_components()
            attr._add_event(event,timing,ready,0.,end,parts,costs)
            if any(costs.values()):
                ident=operator_identity(event.metadata);actual=math.fsum(costs.values())*1000
                independently_grouped[ident]+=actual;ids.append(current)
                if ident not in groups:
                    groups[ident]={'phase':ident[0],'component':ident[1],'operator':ident[2],
                        'actual_total_ms':0.,'compute_service_demand_ms':0.,'hbm_service_demand_ms':0.,
                        'network_service_demand_ms':0.,'other_service_demand_ms':0.,'roofline_solo_demand_ms':0.,
                        'compute_work_flops_selected_resource':0.,'hbm_bytes_selected_resource':0.,
                        'weighted_critical_path_event_count':0,'unweighted_selected_event_count':0,
                        'attributed_compute_ms':0.,'attributed_memory_ms':0.,'attributed_communication_ms':0.,
                        'attributed_launch_ms':0.,'attributed_contention_ms':0.,'attributed_wait_other_ms':0.,
                        'compute_resources':set(),'hbm_resources':set(),'source_artifacts':set()}
                g=groups[ident]
                comp=resource_max(event,graph.resources,lambda key,r:r.kind in ('compute','iops') or '.compute' in key)
                mem=resource_max(event,graph.resources,lambda key,r:key.endswith('.hbm'))
                net=resource_max(event,graph.resources,lambda key,r:
                    event.metadata.get('type')=='transfer' and not key.endswith(('.hbm','.dram','.memory')) and r.kind not in ('compute','iops'))
                other=resource_max(event,graph.resources,lambda key,r:
                    not(key.endswith('.hbm') or r.kind in ('compute','iops') or '.compute' in key or
                        (event.metadata.get('type')=='transfer' and not key.endswith(('.hbm','.dram','.memory')))))
                solo=resource_max(event,graph.resources,lambda key,r:True)
                g['actual_total_ms']+=actual*count
                for prefix,maximum in [('compute',comp),('hbm',mem),('network',net),('other',other)]:
                    g[prefix+'_service_demand_ms']+=maximum[0]*1000*count
                g['roofline_solo_demand_ms']+=solo[0]*1000*count
                g['compute_work_flops_selected_resource']+=comp[2]*count
                g['hbm_bytes_selected_resource']+=mem[2]*count
                g['weighted_critical_path_event_count']+=count;g['unweighted_selected_event_count']+=1
                if comp[1]:g['compute_resources'].add(comp[1])
                if mem[1]:g['hbm_resources'].add(mem[1])
                g['source_artifacts'].add(artifact)
                for out,cost in [('compute','compute_s'),('memory','memory_access_s'),('communication','communication_s'),
                                 ('launch','launch_s'),('contention','resource_contention_s')]:
                    g['attributed_'+out+'_ms']+=costs[cost]*1000*count
                g['attributed_wait_other_ms']+=sum(costs[k] for k in ('synchronization_wait_s','scheduling_wait_s',
                    'memory_wait_s','unattributed_service_s','other_s'))*1000*count
                path_evidence.append({'id':row['id'],'kind':kind,'layers':count,'event_id':current,
                    'predecessor_id':predecessor,'phase':ident[0],'component':ident[1],'operator':ident[2],
                    'event_start_s':timing['start_s'],'event_end_s':timing['end_s'],'dependency_ready_s':ready,
                    'actual_attributed_ms_per_layer':actual,'compute_demand_ms_per_layer':comp[0]*1000,
                    'compute_resource':comp[1],'compute_capacity':comp[3],
                    'hbm_demand_ms_per_layer':mem[0]*1000,'hbm_resource':mem[1],'hbm_capacity_Bps':mem[3],
                    'network_demand_ms_per_layer':net[0]*1000,'network_bottleneck_resource':net[1],
                    'all_resource_roofline_ms_per_layer':solo[0]*1000,'source_artifact':artifact})
            current=predecessor
        assert set(independently_grouped)==set(native_ops),(row['id'],kind,'operator identity mismatch')
        for ident,value in independently_grouped.items():
            assert math.isclose(value,native_ops[ident],rel_tol=1e-10,abs_tol=1e-9),(row['id'],kind,ident,value,native_ops[ident])
        per_kind.append({'kind':kind,'layers':count,'selected_positive_time_event_count':len(ids),
            'native_operator_rows':len(native_ops),'native_critical_path_time_reconciles':True,
            'source_artifact':artifact})
        del graph,sim,payload,attr;gc.collect()
    expected={(o['phase'],o['component'],o['operator']):o for o in row['latency_breakdown']['operator_breakdown']['operators']}
    assert set(expected)==set(groups)
    communications={n for info in row['kinds'].values() for n in info['communication_operators']}
    results=[]
    for ident,g in groups.items():
        expected_ms=expected[ident]['total_s']*1000
        assert math.isclose(g['actual_total_ms'],expected_ms,rel_tol=1e-10,abs_tol=1e-8),(ident,g['actual_total_ms'],expected_ms)
        g.update({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],'hardware':row['hardware'],
            'network_startup_factor':row['network_startup_factor'],'kernel_launch_us':row['kernel_launch_us'],
            'semantic_category':category(ident[2],communications),'window_total_ms':row['fine_tpot_ms'],
            'window_share_pct':expected[ident]['share_pct'],'source_ledger_artifact':row['result_path']})
        for name in ('compute_resources','hbm_resources','source_artifacts'):g[name]=sorted(g[name])
        g['compute_to_hbm_demand_ratio']=g['compute_service_demand_ms']/g['hbm_service_demand_ms'] if g['hbm_service_demand_ms'] else None
        g['demand_additivity']='non_additive_overlapping_resource_demands; never sum compute+HBM as elapsed'
        g.update(actual_ms=g['actual_total_ms'],compute_service_ms=g['compute_service_demand_ms'],
                 hbm_service_ms=g['hbm_service_demand_ms'],network_service_ms=g['network_service_demand_ms'])
        results.append(g)
    results.sort(key=lambda r:-r['actual_total_ms'])
    assert math.isclose(sum(r['actual_total_ms'] for r in results),row['fine_tpot_ms'],rel_tol=1e-10,abs_tol=1e-8)
    return results,path_evidence,per_kind


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    args=p.parse_args();directory=args.output;summary=load(directory/'summary.json')
    assert summary['complete'] and len(summary['rows'])==16
    original_sha=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest()
    primary=[r for r in summary['rows'] if r['network_startup_factor']==1 and r['kernel_launch_us']==0]
    assert len(primary)==4
    demands=[];evidence=[];checks=[]
    for row in primary:
        actual,paths,kinds=demand_rows(row,directory);demands+=actual;evidence+=paths
        checks.append({'id':row['id'],'all_saved_operator_totals_reconciled':True,'kinds':kinds})
        print(json.dumps({'case':row['case_id'],'operators':len(actual),'path_events':len(paths)}),flush=True)
    semantic,mapping,semantic_checks=semantic_rows(summary['rows'])
    semantic_windows=[];semantic_operators=[]
    for row in summary['rows']:
        parts={r['category']:r['total_ms'] for r in semantic if r['id']==row['id']}
        semantic_windows.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
            'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
            'kernel_launch_us':row['kernel_launch_us'],'matrix_ms':parts['matrix'],'pointwise_ms':parts['pointwise_state'],
            'cache_ms':parts['cache_io'],'communication_ms':parts['communication'],'other_ms':0.,
            'total_ms':row['fine_tpot_ms'],'source_artifact':row['result_path']})
        comm={n for info in row['kinds'].values() for n in info['communication_operators']}
        for op in row['latency_breakdown']['operator_breakdown']['operators']:
            semantic_operators.append({'id':row['id'],'case_id':row['case_id'],'architecture_id':row['architecture_id'],
                'hardware':row['hardware'],'network_startup_factor':row['network_startup_factor'],
                'kernel_launch_us':row['kernel_launch_us'],'operator':op['operator'],'component':op['component'],
                'category':category(op['operator'],comm),'actual_ms':op['total_s']*1000,'window_share_pct':op['share_pct'],
                'attributed_compute_ms':op['components_s'].get('compute_s',0)*1000,
                'attributed_hbm_ms':op['components_s'].get('memory_access_s',0)*1000,
                'attributed_kernel_launch_ms':op['components_s'].get('launch_s',0)*1000,
                'source_artifact':row['result_path']})
    verification={'status':'passed','resource_demand_primary_windows':4,'resource_demand_operator_rows':len(demands),
        'semantic_windows':16,'semantic_category_rows':len(semantic),'native_critical_path_rule':'latest completion; lexical minimum event ID on ties',
        'parallel_rank_or_endpoint_times_not_summed':True,'all_actual_operator_totals_match_saved_ledger':True,
        'numerical_summary_unchanged':original_sha==hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),
        'primary_checks':checks,'semantic_checks':semantic_checks}
    assert verification['numerical_summary_unchanged']
    method=('Compute solo service demand is the maximum FLOPs/capacity over compute resources per selected event; '
        'HBM demand is the maximum bytes/bandwidth over its HBM endpoints. Network and all-resource demands also use max, '
        'never the sum of parallel endpoints. These independently necessary service demands overlap and must not be added '
        'as a measured time partition. Per-event values sum only along the native selected dependency path and explicit layer counts.')
    save(directory/'resource_service_demands.json',{'rows':demands,'critical_path_event_evidence':evidence,
        'method':method,'verification':verification,'source_summary_sha256':original_sha,
        'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    csv_save(directory/'resource_service_demands.csv',demands)
    save(directory/'semantic_kernel_categories.json',{'rows':semantic,'mapping':mapping,'method':
        'Semantic class describes the executed operator, independently of which resource limits its service. Matrix-kernel time can be memory-attributed.',
        'source_summary_sha256':original_sha,'verification':{'status':'passed','window_count':16,'checks':semantic_checks}})
    csv_save(directory/'semantic_kernel_categories.csv',semantic)
    save(directory/'operator_semantics.json',{'windows':semantic_windows,'rows':semantic_operators,'mapping':mapping,
        'source_summary_sha256':original_sha,'verification':{'status':'passed','window_count':16,'checks':semantic_checks}})
    csv_save(directory/'operator_semantics.csv',semantic_operators)
    csv_save(directory/'semantic_window_summary.csv',semantic_windows)
    save(directory/'resource-service-verification.json',verification)
    print(json.dumps({'status':'passed','primary_windows':4,'demand_rows':len(demands),'semantic_windows':16,'semantic_rows':len(semantic)}),flush=True)


if __name__=='__main__':main()

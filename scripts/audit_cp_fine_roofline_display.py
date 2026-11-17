"""Explain operator wall time versus limiting-resource chart attribution."""
from collections import defaultdict
import csv,gzip,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/scaling-10case-20261113/sp-megakernel/cp-fine-operators'
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)

def main():
    data=load(OUT/'summary.json');assert data['complete'] and len(data['rows'])==16
    sample=next(r for r in data['rows'] if r['case_id']=='kimi-k3-model-960DT-2p8T' and r['network_startup_factor']==1 and r['kernel_launch_us']==0)
    examples=[];aggregates=defaultdict(lambda:defaultdict(float))
    for kind,detail in sample['kinds'].items():
        path=OUT/detail['evidence_artifact']
        with gzip.open(path,'rt',encoding='utf-8') as f: evidence=json.load(f)
        graph=evidence['graph'];resources=graph['resources'];trace={r['id']:r for r in evidence['simulation']['trace']}
        for event in graph['events']:
            kernel=event['metadata'].get('kernel')
            if kernel not in ('shared_projection_pre.budget_matmul','shared_projection_post.budget_matmul'):continue
            compute_key=next(k for k in event['work'] if '.compute' in k)
            memory_key=next(k for k in event['work'] if k.endswith('.hbm'))
            flops=event['work'][compute_key];memory=event['work'][memory_key]
            compute_capacity=resources[compute_key]['capacity'];hbm_capacity=resources[memory_key]['capacity']
            compute_seconds=flops/compute_capacity;memory_seconds=memory/hbm_capacity
            observed=trace[event['id']]['duration_s'];layers=detail['layers']
            assert memory_seconds>compute_seconds
            close(observed,max(compute_seconds,memory_seconds)+event['startup_s'])
            row={'case_id':sample['id'],'kind':kind,'layers':layers,'operator':kernel,'event':event['id'],
                 'flops_per_layer':flops,'hbm_bytes_per_layer':memory,
                 'effective_compute_flops_s':compute_capacity,'hbm_Bps':hbm_capacity,
                 'compute_demand_per_layer_s':compute_seconds,'hbm_demand_per_layer_s':memory_seconds,
                 'recorded_per_layer_s':observed,'weighted_compute_demand_ms':compute_seconds*layers*1000,
                 'weighted_hbm_demand_ms':memory_seconds*layers*1000,'weighted_recorded_ms':observed*layers*1000,
                 'arithmetic_intensity_flops_per_byte':flops/memory,
                 'machine_balance_flops_per_byte':compute_capacity/hbm_capacity,
                 'symmetry_multiplicity_not_used_as_latency_multiplier':event['metadata'].get('symmetry_multiplicity'),
                 'source_artifact':detail['evidence_artifact'],
                 'source_fields':'graph.events[].work; graph.resources[].capacity; simulation.trace[].duration_s; summary.rows[].kinds[].layers'}
            examples.append(row)
            for key,value in [('flops',flops*layers),('hbm_bytes',memory*layers),
                              ('compute_demand_ms',compute_seconds*layers*1000),('hbm_demand_ms',memory_seconds*layers*1000),
                              ('recorded_ms',observed*layers*1000)]:aggregates[kernel][key]+=value
        del evidence,graph,trace
    ledger=sample['latency_breakdown']['operator_breakdown']['operators']
    for kernel,row in aggregates.items():
        op=next(o for o in ledger if o['operator']==kernel)
        close(row['recorded_ms'],op['total_s']*1000)
        close(row['hbm_demand_ms'],op['components_s']['memory_access_s']*1000)
        assert op['components_s']['compute_s']==0
        row['recorded_compute_limited_ms']=op['components_s']['compute_s']*1000
        row['recorded_memory_limited_ms']=op['components_s']['memory_access_s']*1000
    with (OUT/'operator_details.csv').open(encoding='utf-8-sig',newline='') as f:
        exported={(r['id'],r['source_component'],r['operator']):r for r in csv.DictReader(f)}
    component_export={(r['id'],r['component']):r for r in load(OUT/'component_costs.json')['rows']}
    checks=[]
    for row in data['rows']:
        expected={k:defaultdict(float) for k in ('attention','moe','other')}
        communications={op for kind in row['kinds'].values() for op in kind['communication_operators']}
        for op in row['latency_breakdown']['operator_breakdown']['operators']:
            e=exported[row['id'],op['component'],op['operator']]
            total=op['total_s']*1000;cost=op['components_s']
            component='moe' if 'moe' in op['component'].lower() else 'attention' if 'attention' in op['component'].lower() else 'other'
            values={'communication':total,'compute':0.,'memory':0.,'launch':0.,'other':0.} if op['operator'] in communications else {
                'communication':0.,'compute':cost['compute_s']*1000,'memory':cost['memory_access_s']*1000,
                'launch':cost['launch_s']*1000,'other':max(0.,total-(cost['compute_s']+cost['memory_access_s']+cost['launch_s'])*1000)}
            close(float(e['total_ms']),total)
            close(sum(values.values()),total)
            expected[component]['total_ms']+=total
            for k,v in values.items():
                close(float(e[k+'_ms']),v)
                expected[component][k+'_ms']+=v
        for component,values in expected.items():
            e=component_export[row['id'],component]
            for k,v in values.items():close(e[k],v)
            if e['total_ms']:
                for k in ('compute','communication','memory','launch','other'):close(e[k+'_pct'],e[k+'_ms']/e['total_ms']*100)
        checks.append({'case_id':row['id'],'all_source_operators_mapped':True,'chart_fields_match_recorded_limiting_resource_costs':True,
                       'compute_limited_ms':sum(x['compute_ms'] for x in expected.values()),
                       'memory_limited_ms':sum(x['memory_ms'] for x in expected.values())})
    result={'status':'passed','window_count':16,'checks':checks,'representative_kernel_evidence':examples,
        'kind_weighted_kernel_aggregates':dict(aggregates),
        'interpretation':['Operator total time is grouped by function (for example GEMM); compute/memory columns group that same time by the limiting resource.',
            'The engine uses max(FLOPs/effective_compute_rate, HBM_bytes/HBM_rate) for simultaneous demands, plus explicit launch/waits. Their standalone service demands are not additive elapsed times.',
            'A GEMM can have large recorded wall time and zero compute-limited attribution while retaining nonzero FLOPs. This is not a missing-compute chart mapping.',
            'All results remain analytical roofline-style predictions. Effective shape-specific arithmetic throughput, nonlinear instruction costs, caching and overlap need actual kernel evidence; the display does not measure physical utilization.'],
        'source_summary_sha256':hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
        'numerical_execution_modified':False}
    (OUT/'roofline-display-independent-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'representative_roofline_demands.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(examples[0]));writer.writeheader();writer.writerows(examples)
    print(json.dumps({'status':'passed','windows':16,'kernel_aggregates':dict(aggregates),
                     'compute_limited_ms_range':[min(r['compute_limited_ms'] for r in checks),max(r['compute_limited_ms'] for r in checks)]},ensure_ascii=False))

if __name__=='__main__':main()

"""Operator-trace execution of the preserved V4.1 FP4/POD study.

Launch costs are explicit engineering assumptions, NOT Ascend measurements.
The original rank-envelope, fusion, memory reservation and network schedules
remain assumptions. Dense projections and grouped expert GEMMs are expanded
from tensor shapes; activation is scheduled before each MLP down projection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import copy
import hashlib
import json
import math
from pathlib import Path

from supernode_sim.config import KernelTrace
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.operator_trace import apply_kernel_traces
from . import model, target

PROFILE_PATH = Path(__file__).with_name('operator_launch_assumptions.json')
PROFILE = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
REMAP = {'NPU.fp8': 'npu0.compute', 'NPU.fp32': 'npu0.compute.fp32',
         'NPU.hbm': 'npu0.hbm'}


def startup_kind(name):
    if name.startswith('gemm.'):
        if '.expert.' in name: return 'expert_grouped_gemm'
        if name.endswith('q_a'): return 'low_rank_projection'
        if name.endswith('q_b'): return 'head_projection'
        if name.endswith('o_a'): return 'grouped_output_projection'
        if name.endswith('router'): return 'router_projection'
        return 'dense_gemm'
    return PROFILE['operators'][name]


def launch_seconds(name, multiplier=1.0):
    return PROFILE['startup_us'][startup_kind(name)] * 1e-6 * multiplier


def _key(row):
    return row['name'], row['dtype'], tuple(sorted(row['work'].items()))


def _op(name, flops, memory, matmul=None):
    result = dict(kernel=name, flops=flops, memory_bytes=memory)
    if matmul: result['matmul'] = matmul
    return result


@lru_cache(maxsize=8)
def get_hardware(chip, cards):
    return model.hardware(chip, cards)


class FinePhase(model.Phase):
    def __init__(self, *args, launch_multiplier=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipes = {}
        self.launch_multiplier = launch_multiplier

    def local_matrix(self, layer, selected, token_counts, component, *, divisor=1):
        start = len(self.ops)
        super().local_matrix(layer, selected, token_counts, component, divisor=divisor)
        tensors = [t for t in target.BY_LAYER[layer] if selected(t)]
        m = max(math.ceil(n / divisor) for n in token_counts)
        for row in self.ops[start:]:
            ops = []
            for tensor in tensors:
                if tensor.dtype != row['dtype']: continue
                shape = tensor.local_shape(self.l.tp, self.l.ep)
                groups = math.prod(shape[:-2]) if len(shape) >= 2 else 1
                io = (groups * (shape[-1] + shape[-2] * (4 if tensor.dtype == 'fp32' else 1))
                      if len(shape) >= 2 else 0)
                shape_spec = (dict(m=m, n=shape[-2], k=shape[-1], batch=groups)
                              if len(shape) >= 2 and m else None)
                # A scalar router bias is retained because it was in the
                # original parent budget; do not invent a GEMM shape for it.
                suffix = '.'.join(tensor.name.split('.')[2:])
                ops.append(_op('gemm.' + suffix,
                               2*m*tensor.local_params(self.l.tp, self.l.ep),
                               sum(tensor.bytes(self.l.tp, self.l.ep)) + m*io if m else 0,
                               shape_spec))
            self.recipes[_key(row)] = ops

    def operations(self, row):
        if _key(row) in self.recipes:
            return copy.deepcopy(self.recipes[_key(row)])
        f = row['work'].get('NPU.' + row['dtype'], 0)
        mem = row['work'].get('NPU.hbm', 0)
        name = row['name']
        h, i, t = model.MODEL['dim'], model.MODEL['moe_inter_dim'], self.l.tp
        if name == 'expert_GEMMs':
            # Three separate grouped GEMMs, with explicit materialized inputs.
            # Original per-rank maxima are retained; additional HBM reads pay
            # for separate gate/up launches. This is not a native FP4 GEMM.
            a = f / (6*h*i/t)
            extra = a * max(0, h-i/t)
            return [_op('gemm.expert.'+part, f/3, (mem+extra)/3)
                    for part in ('gate', 'up', 'down')]
        if name == 'noise_QKV':
            qd, kd = h/t, model.DRAFT['kv_heads']*model.DRAFT['head_dim']/t
            n = f/(2*h*(qd+2*kd))
            return [_op('gemm.draft.'+part, 2*n*h*width,
                        h*width*(1+1/1024)+n*(h+width),
                        dict(m=int(n), n=int(width), k=h, batch=1))
                    for part, width in [('q',qd), ('k',kd), ('v',kd)]]
        if name == 'gate_up':
            width = model.DRAFT['intermediate']/t
            n = f/(4*h*width)
            return [_op('gemm.draft.'+part, f/2, mem/2+n*h/2,
                        dict(m=int(n), n=int(width), k=h, batch=1))
                    for part in ('gate', 'up')]
        if name == 'shared_SwiGLU_and_combine':
            n = f/(i/t*8+h*model.MODEL['n_activated_experts']*2)
            activation_f, activation_m = n*i/t*8, n*i/t*5
            return [_op('shared_swiglu', activation_f, activation_m),
                    _op('shared_expert_combine', f-activation_f, mem-activation_m)]
        return [_op(name, f, mem)]

    def run(self, keep_stages=False, keep_graph=False):
        hw = self.hw
        memory = self.memory()
        rates = {'npu0.compute': hw['npu']['flops']['fp8']*hw['npu']['efficiency'],
                 'npu0.compute.fp32': hw['npu']['flops']['fp32']*hw['npu']['efficiency'],
                 'npu0.hbm': hw['npu']['hbm_Bps'],
                 'SU.tx': min(hw['scale_up']['bandwidth_Bps'], hw['scale_up']['injection_Bps'])*hw['scale_up']['efficiency'],
                 'HOST': hw['host_link']['injection_Bps']*hw['host_link']['efficiency'],
                 'CPU.dram': hw['cpu']['dram_Bps'],
                 'CPU.fp32': hw['cpu']['flops']*hw['cpu']['efficiency'],
                 'CPU.iops': hw['cpu']['iops']}
        rates['SU.rx'] = rates['SU.tx']
        graph = Graph(resources={k:Resource(k,v,'unit/s') for k,v in rates.items()},
                      memory_caps={'npu0.hbm':hw['npu']['hbm_bytes'],'DRAM':hw['cpu']['dram_bytes']},
                      initial_allocations=[Allocation('hbm','npu0.hbm',memory['hbm_peak_bytes']),
                                           Allocation('host','DRAM',memory['host_peak_bytes'])])
        traces = {}
        previous = ()
        parent_metadata = {}
        original_work = defaultdict(float)
        for index, row in enumerate(self.ops):
            eid = str(index)
            compute = any(k in row['work'] for k in ('NPU.fp8','NPU.fp32'))
            scope = dict(layer=row['layer'], component=row['component'], phase=self.phase,
                         tp=self.l.tp,ep=self.l.ep,sp=self.l.sp,
                         model_shape='v41-64L-7168-1792E-top24-fp4-v1',
                         workload_shape=dict(batch=self.b,context_tokens=self.context,block=self.block_size),
                         parent_index=index)
            md = {**scope, 'original_name':row['name'], 'totals':row['totals']}
            startup = row['startup_s']
            if compute:
                kernel = row['name']+'@'+eid
                md['kernel'] = kernel
                ops = self.operations(row)
                for op in ops:
                    op['startup_s'] = launch_seconds(op['kernel'], self.launch_multiplier)
                trace = dict(id=kernel,source=PROFILE['source'],hardware_name=hw['name'],
                             phase=self.phase,dtype=row['dtype'],
                             flops=row['work']['NPU.'+row['dtype']],memory_bytes=row['work']['NPU.hbm'],
                             metadata_match=scope,operations=ops)
                traces[kernel] = [KernelTrace.model_validate(trace).model_dump()]
            elif 'SU.tx' not in row['work'] and 'HOST' not in row['work']:
                startup += launch_seconds(row['name'], self.launch_multiplier)
                md['host_launch'] = True
            work = {REMAP.get(k,k):v for k,v in row['work'].items()}
            graph.events.append(Event(eid,previous,work,startup,
                                      slots=('npu0.compute_stream',) if compute else (),metadata=md))
            parent_metadata[eid] = md
            previous = (eid,)
            for k,v in work.items(): original_work[k] += v
        config = {'hardware':{'name':hw['name'],'kernel_traces':traces},
                  'execution':{'precision':{'compute':'fp8'}}}
        apply_kernel_traces(graph,config)
        summary = graph.metadata.get('operator_trace', {})
        if traces:
            assert summary['matched_events'] == summary['eligible_events'] == len(traces)
        # Restore true MLP dependencies within the barrier schedule. All phase
        # memory is reserved up front; completion nodes own no allocations.
        groups = defaultdict(list)
        for event in graph.events:
            parent = event.metadata.get('trace_parent_event',event.id)
            groups[parent].append(event)
        ordered = []
        index = 0
        while index < len(self.ops):
            row = self.ops[index]
            group = groups[str(index)]
            nextrow = self.ops[index+1] if index+1<len(self.ops) else {}
            if row['name'] == 'expert_GEMMs' and nextrow.get('name') == 'SwiGLU':
                ordered.extend(group[:2])
                ordered.extend(groups[str(index+1)])
                ordered.extend(group[2:])
                index += 2
            elif row['name'] == 'MoE_shared.matrix.fp8' and nextrow.get('name') == 'shared_SwiGLU_and_combine':
                second = groups[str(index+1)]
                ordered.extend(group[:2])
                ordered.append(second[0])
                ordered.extend(group[2:])
                ordered.extend(second[1:])
                index += 2
            else:
                ordered.extend(group)
                index += 1
        previous = ()
        for event in ordered:
            event.deps = previous
            previous = (event.id,)
        graph.events = ordered
        run = simulate(graph,retain_event_payload=False)
        ok = run['status'] == 'feasible'
        breakdown, components, operator_counts = defaultdict(float),defaultdict(float),Counter()
        stages, seed_time, service_without_launch, added_critical_hbm = [],0.,0.,0.
        if ok:
            timings = {r['id']:r for r in run['trace']}
            for event in graph.events:
                if event.metadata.get('operator_trace_completion'): continue
                demands = {k:v/rates[k] for k,v in event.work.items() if v}
                service = max(demands.values(),default=0.)
                bottleneck = max(demands,key=demands.get) if demands else None
                if 'SU.tx' in event.work: kind='su_transfer_s'
                elif 'HOST' in event.work: kind='host_transfer_s'
                elif bottleneck=='npu0.hbm': kind='hbm_bound_s'
                elif bottleneck=='CPU.fp32': kind='cpu_compute_s'
                elif bottleneck and bottleneck.startswith('CPU.'): kind='host_memory_s'
                else: kind='compute_bound_s'
                breakdown[kind] += service
                launch_key = ('communication_startup_s' if 'SU.tx' in event.work or 'HOST' in event.work
                              else 'operator_launch_s')
                breakdown[launch_key] += event.startup_s
                duration = timings[event.id]['duration_s']
                assert math.isclose(duration,service+event.startup_s,rel_tol=2e-8,abs_tol=1e-7)
                component = event.metadata['component']
                components[component] += duration
                if component == 'DFlash_KV_seed': seed_time += duration
                name = event.metadata.get('kernel',event.metadata['original_name'])
                operator_counts[name] += 1
                service_without_launch += service + (event.startup_s if launch_key=='communication_startup_s' else 0)
                if keep_stages:
                    stages.append(dict(id=event.id,name=name,component=component,layer=event.metadata['layer'],
                                       work=event.work,startup_s=event.startup_s,duration_s=duration,
                                       bottleneck=bottleneck,phase=self.phase,
                                       start_s=timings[event.id]['start_s'],end_s=timings[event.id]['end_s'],
                                       matmul=event.metadata.get('matmul'),deps=list(event.deps)))
            assert math.isclose(sum(breakdown.values()),run['duration_s'],rel_tol=1e-9,abs_tol=1e-7)
            new_work = defaultdict(float)
            for event in graph.events:
                for k,v in event.work.items(): new_work[k] += v
            for k,v in original_work.items():
                if k != 'npu0.hbm': assert math.isclose(new_work[k],v,rel_tol=1e-9,abs_tol=1e-5), k
            added_critical_hbm = new_work['npu0.hbm']-original_work['npu0.hbm']
            assert added_critical_hbm >= -1e-5
        # Do not store thousands of IDs for every matrix candidate.
        trace_summary = {k:v for k,v in summary.items() if k not in ('matched_trace_ids','policy')}
        result = dict(status=run['status'],phase=self.phase,duration_s=run['duration_s'] if ok else None,
                      breakdown_s=dict(breakdown) if ok else None,
                      component_latency_s=dict(components) if ok else None,memory=memory,
                      native_completed_events=run['completed_events'],native_diagnostics=run['diagnostics'],
                      seed_s=seed_time if ok else None,operator_trace=trace_summary,
                      operator_counts=dict(operator_counts),added_envelope_hbm_bytes=added_critical_hbm if ok else None,
                      zero_operator_launch_s=service_without_launch if ok else None,
                      stages=stages,physical_work=dict(self.totals) if ok else None,
                      physical_work_scope='original parent ledger; extra HBM separately reported as rank envelope')
        if keep_graph:
            result['trace_config'] = config
        return result


def run_phase(chip,tp,ep,sp,batch,context,block,phase,committed=1,
              keep_stages=False,launch_multiplier=1.0,keep_graph=False):
    cards=max(16,tp*ep)
    p=FinePhase(get_hardware(chip,cards),model.Layout(tp,ep,cards,sp),batch,context,block,phase,
                launch_multiplier=launch_multiplier)
    memory=p.memory()
    if memory['hbm_peak_ratio']>1 or memory['host_peak_ratio']>1:
        result=p.run(keep_stages,keep_graph)
        result['capacity_check']='persistent_lower_bound'
        assert result['status']=='infeasible'
        return result
    if tp*ep<8 or tp*ep%8:
        raise ValueError('Feasible replicas require disjoint whole 8-NPU hosts')
    result=p.build(committed).run(keep_stages,keep_graph)
    result['capacity_check']='full_reserved_phase'
    return result

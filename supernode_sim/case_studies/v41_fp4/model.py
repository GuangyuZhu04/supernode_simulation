"""Conditional DFlash resource adapter for the preserved synthetic V4.1 design.

No downloaded reference code is executed. The current repository engine snapshot
executes every feasible phase. Main-model operators are inherited unchanged.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import copy, hashlib, json, math, sys

HERE = Path(__file__).resolve().parent
from supernode_sim.config import HardwareSpec
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from . import target as base

MODEL = base.MODEL
CONTEXTS = {'1k':1024, '4k':4096, '16k':16384, '32k':32768,
            '64k':65536, '128k':131072, '256k':262144, '512k':524288,
            '1M':1048576, '4M':4194304}
BATCHES = [1,4,8,16,32,64]
EPS = [2,4,8,16,32,64,128,256]
CARDS = [16,32,64,128,256,512]
PAIRS = [(3,4),(6,8),(9,12),(12,16)]
PRECISION_VARIANT = 'routed_fp4_other_unchanged'
PRECISION_NOTE = ('Only target routed expert gate/up/down weights use FP4 storage. '
    'FP4 scales follow the preserved model native_storage layout: one byte per 32 last-axis values. '
    'Other weights, FP8 GEMM compute, activation, KV, communication, DFlash and hardware are unchanged. '
    'Unfused FP4-to-FP8 conversion follows current engine structured.weight_conversion: '
    '2 scalar operations/element, packed weight+scale reads, FP8 buffer writes, then FP8 GEMM buffer reads. '
    'The conversion buffer coexists with the expert workspace and is reused between layers.')
DRAFT = dict(kind='synthetic_DFlash_dense_full_context', layers=5,
             hidden=7168, intermediate=21504, heads=56, kv_heads=8,
             head_dim=128, target_layers=[1,7,13,19,25], dtype='fp8',
             scale_block=32, source='sources/dflash-qwen3-8b-config.json',
             note='5-layer full-attention DFlash mechanism; hidden scaled to target, '
                  'MLP width=3h, 56 Q heads, 8 KV heads. Encoder-only feature taps '
                  'preserve CED; no trained 10T drafter or quality claim.')


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def hardware(chip, cards):
    hw=copy.deepcopy(json.loads((HERE/'hardware_reference.json').read_text())['platforms']['950dt'])
    hw.update(name=f'{chip.upper()} user capacity / {cards}-card logical POD',
              npu_count=cards,su_size=cards)
    hw['npu']['hbm_bytes']={'950pr':128e9,'950dt':96e9}[chip]
    hw['npu']['hbm_Bps']={'950pr':1.6e12,'950dt':4e12}[chip]
    # Same 950 compute die assumption, not a measured 96GB SKU profile.
    return HardwareSpec.model_validate(hw).model_dump()


@dataclass(frozen=True)
class Layout:
    tp:int
    ep:int
    cards:int
    sp:int=1
    @property
    def n(self): return self.tp*self.ep
    @property
    def dp(self): return self.cards//self.n
    @property
    def name(self): return f'tp{self.tp}-ep{self.ep}-dp{self.dp}-sp{self.sp}'


@lru_cache(maxsize=64)
def draft_weights(tp):
    h,i,k=DRAFT['hidden'],DRAFT['intermediate'],DRAFT['kv_heads']*DRAFT['head_dim']
    rows=[]
    # Replicated fusion sum enables online accumulation of five encoder taps.
    shapes=[('feature_fusion',(-1,), (5*h,h))]
    for layer in range(DRAFT['layers']):
        for name,shape in [('q',(h//tp,h)),('k',(k//tp,h)),('v',(k//tp,h)),
                           ('o',(h,h//tp)),('gate',(i//tp,h)),
                           ('up',(i//tp,h)),('down',(h,i//tp))]:
            shapes.append((name,(layer,),shape))
    for name,layer,shape in shapes:
        count=math.prod(shape)
        rows.append(dict(tensor='DFlash.'+name,layer=layer[0],shape=shape,
                         weight_bytes_per_card=count,
                         scale_bytes_per_card=math.ceil(shape[0]/32)*math.ceil(shape[1]/32)))
    rows.append(dict(tensor='DFlash.norms',layer=-1,shape=(11*h,),
                     weight_bytes_per_card=11*h*4,scale_bytes_per_card=0))
    return rows


@lru_cache(maxsize=64)
def target_weight_ledger(tp,ep):
    rows=[]
    for t in base.TENSORS:
        if t.component.startswith('Draft'):continue
        value,scale=t.bytes(tp,ep,scheme='native_storage' if t.expert_axis else 'fp8_comparison')
        rows.append(dict(layer=t.layer,component=t.component,tensor=t.name,
                         storage_dtype='fp4' if t.expert_axis else t.dtype,
                         weight_bytes_per_card=value,scale_bytes_per_card=scale))
    return rows


def persistent_state(context,block,tp):
    rows=base.kv_ledger(context+block-1,False)
    state=64*128*512*(1+1/32)+3*2*2*512*4
    rows.append(dict(component='Target_prefix_snapshot_versions',bytes=(block+1)*state))
    kv=2*DRAFT['layers']*(context+block)*DRAFT['kv_heads']*DRAFT['head_dim']/tp*(1+1/32)
    rows.append(dict(component='DFlash_full_context_KV',bytes=kv))
    rows.append(dict(component='DFlash_KV_page_table',bytes=math.ceil((context+block)/16)*8*DRAFT['layers']))
    return rows


class Phase(base.Phase):
    def __init__(self,hw,layout,batch,context,block,phase):
        assert hw['npu_count']==hw['su_size']==layout.cards
        assert hw['topology']=='full_mesh' and layout.cards%layout.n==0
        assert MODEL['n_routed_experts']%layout.ep==0
        assert layout.tp in (1,2,4,8)
        assert layout.sp in (1,layout.tp)
        self.hw,self.l,self.b,self.context=hw,layout,batch,context
        self.dspark,self.phase,self.stable=False,phase,False
        self.seq=base.local_sequences(batch,layout.ep)
        self.ops=[]; self.arena=[0.]*layout.ep; self.arena_cause=['none']*layout.ep
        self.totals=defaultdict(float); self.hosts=max(1,layout.n//8)
        self.host_arena=0.; self.c=MODEL; self.block_size=block
        self.target_only_ops=None
        self._moe_templates={}; self._reserve_record=None
        self._conversion_buffer=[0.]*layout.ep
        self._arena_conversion=[0.]*layout.ep
        self._max_conversion_buffer=[0.]*layout.ep
        self._moe_q=None

    def reserve(self,values,name):
        if name=='MoE':
            values=[v+c for v,c in zip(values,self._conversion_buffer)]
            name='MoE_with_FP4_conversion_buffer'
        for i,value in enumerate(values):
            if value>self.arena[i]:
                self._arena_conversion[i]=self._conversion_buffer[i] if name=='MoE_with_FP4_conversion_buffer' else 0.
        if self._reserve_record is not None:
            self._reserve_record.append((tuple(values),name))
        super().reserve(values,name)

    def moe(self,layer,q,*,draft=False):
        # All 64 target layers have exactly the same MoE tensor shapes. Cache
        # operator construction only; every layer still executes and is billed.
        assert not draft
        self._moe_q=q
        if q not in self._moe_templates:
            start=len(self.ops); self._reserve_record=[]
            super().moe(layer,q,draft=False)
            reservations=self._reserve_record; self._reserve_record=None
            self._moe_templates[q]=(copy.deepcopy(self.ops[start:]),reservations,tuple(self._conversion_buffer))
        else:
            ops,reservations,buffer=self._moe_templates[q]
            self._conversion_buffer=list(buffer)
            for row in ops:
                self.op(row['name'],row['component'],row['work'],row['totals'],
                        row['startup_s'],row['dtype'],layer)
            for values,name in reservations:self.reserve(values,name)

    def kernel(self,name,component,flops,hbm,*,dtype='fp8',layer=-1,copies=None):
        if name=='expert_GEMMs' and component=='MoE_experts':
            h,i,t=MODEL['dim'],MODEL['moe_inter_dim'],self.l.tp
            _,_,unique=base.routing(self.b,self._moe_q,self.l.ep,MODEL['n_routed_experts'],MODEL['n_activated_experts'])
            ep_params=3*h*i/t
            fp8_scale=3*math.ceil(h/32)*math.ceil(i/t/32)
            # gate/up: [intermediate/TP, hidden]; down: [hidden, intermediate/TP]
            fp4_scale=2*(i/t)*math.ceil(h/32)+h*math.ceil(i/t/32)
            elements=[u*ep_params for u in unique]
            self._conversion_buffer=elements
            self._max_conversion_buffer=[max(a,b) for a,b in zip(self._max_conversion_buffer,elements)]
            packed=[e*.5+u*fp4_scale for e,u in zip(elements,unique)]
            base.Phase.kernel(self,'routed_weight_conversion','MoE_weight_conversion',
                              [2*e for e in elements],[a+e for a,e in zip(packed,elements)],dtype='fp32',layer=layer)
            extra={'fp4_packed_weight_read_bytes':sum(e*.5 for e in elements)*t,
                   'fp4_scale_read_bytes':sum(unique)*fp4_scale*t,
                   'fp8_conversion_buffer_write_bytes':sum(elements)*t,
                   'fp8_conversion_buffer_read_bytes':sum(elements)*t,
                   'weight_conversion_scalar_operations':sum(elements)*2*t}
            self.ops[-1]['totals'].update(extra)
            for k,v in extra.items():self.totals[k]+=v
            # The existing GEMM reads the converted byte/element buffer; scale
            # data were consumed by conversion, not read a second time.
            hbm=[value-u*fp8_scale for value,u in zip(hbm,unique)]
        base.Phase.kernel(self,name,component,flops,hbm,dtype=dtype,layer=layer,copies=copies)

    def capture(self,layer,q):
        if layer not in DRAFT['target_layers']: return
        h=DRAFT['hidden']; counts=[b*q for b in self.seq]
        if self.l.sp>1:
            self.tp_collective([n*h for n in counts],'all_gather','DFlash_feature_gather',layer)
        # Linear projection may be summed by input slice; raw five-layer
        # hidden tensors need not coexist. Fusion result lives until KV seed.
        self.kernel('tap_projection_accumulate','DFlash_feature_fusion',
                    [2*n*h*h for n in counts],
                    [(h*h*(1+1/1024)+3*n*h) if n else 0 for n in counts],layer=layer)

    def block(self,layer,q,**kwargs):
        super().block(layer,q,**kwargs)
        self.capture(layer,q)

    def draft_seed(self,q,component='DFlash_KV_seed'):
        h=DRAFT['hidden']; kd=DRAFT['kv_heads']*DRAFT['head_dim']/self.l.tp
        counts=[b*q for b in self.seq]
        self.kernel('feature_norm',component,[n*h*8 for n in counts],
                    [2*n*h for n in counts],dtype='fp32')
        for layer in range(DRAFT['layers']):
            p=2*h*kd
            self.kernel('context_KV_projection',component,[2*n*p for n in counts],
                        [(p*(1+1/1024)+n*(h+2*kd*(1+1/32))) if n else 0 for n in counts],layer=64+layer)

    def dense_matrix(self,name,counts,m,k,n,layer):
        p=m*k
        self.kernel(name,'DFlash_dense',[2*x*p for x in counts],
                    [(p*(1+1/1024)+x*n) if x else 0 for x in counts],layer=layer)

    def draft(self):
        q=self.block_size; h=DRAFT['hidden']; i=DRAFT['intermediate']; t=self.l.tp
        kd=DRAFT['kv_heads']*DRAFT['head_dim']/t; qd=h/t
        counts=[b*q for b in self.seq]
        self.kernel('mask_embedding','DFlash_embedding',[n*h for n in counts],[2*n*h for n in counts])
        self.tp_collective([n*h for n in counts],'all_reduce','DFlash_embedding')
        for layer in range(5):
            self.kernel('norms_and_residual','DFlash_norm',[n*h*20 for n in counts],
                        [n*h*8 for n in counts],dtype='fp32',layer=64+layer)
            self.dense_matrix('noise_QKV',counts,qd+2*kd,h,h+qd+2*kd,64+layer)
            pairs=q*(self.context+q)
            # A fused attention tile reuses prefix KV across this <=16 block.
            self.kernel('full_context_block_attention','DFlash_attention',
                        [b*4*pairs*qd for b in self.seq],
                        [b*(2*(self.context+q)*kd*(1+1/32)+q*qd*2) for b in self.seq],layer=64+layer)
            self.kernel('softmax_rope','DFlash_attention',
                        [b*(pairs*DRAFT['heads']/t*5+q*qd*10) for b in self.seq],
                        [b*q*qd*2 for b in self.seq],dtype='fp32',layer=64+layer)
            self.dense_matrix('attention_output',counts,h,qd,h+qd,64+layer)
            self.tp_collective([n*h for n in counts],'all_reduce','DFlash_TP',64+layer)
            self.dense_matrix('gate_up',counts,2*i/t,h,h+2*i/t,64+layer)
            self.kernel('swiglu','DFlash_dense',[n*i/t*8 for n in counts],
                        [n*i/t*5 for n in counts],dtype='fp32',layer=64+layer)
            self.dense_matrix('down',counts,h,i/t,h+i/t,64+layer)
            self.tp_collective([n*h for n in counts],'all_reduce','DFlash_TP',64+layer)
        # Block contains a known prefix token and q-1 masked proposals.
        counts=[b*(q-1) for b in self.seq]; v=MODEL['vocab_size']; p=v*h/t
        self.kernel('shared_target_head','DFlash_output',[2*n*p for n in counts],
                    [(p*(1+1/1024)+n*(h+v/t*4)) if n else 0 for n in counts])
        self.tp_collective([n*v*4 for n in counts],'all_gather','DFlash_output')
        self.kernel('sample','DFlash_output',[n*v*8 for n in counts],
                    [n*v*4 for n in counts],dtype='fp32')
        self.reserve([b*(q*(6*h+4*i/t)+(q-1)*v*4*(1+2/t)) for b in self.seq],'DFlash_block_workspace')

    def build(self,committed=1):
        q=self.context
        if self.phase=='prefill':
            super().build()
            self.target_only_ops=len(self.ops)
            self.draft_seed(q)
        elif self.phase=='verify':
            q=self.block_size
            self.embedding(q)
            for layer in range(64): self.block(layer,q)
            self.head(q)
        elif self.phase=='draft': self.draft()
        elif self.phase=='commit':
            q=self.block_size; state=64*128*512*(1+1/32)+3*2*2*512*4
            self.kernel('snapshot_prefix_restore','Target_state_commit',
                        [b*q*64*512*4 for b in self.seq],
                        [b*(2*state*(q+2)+committed*64*512*2) for b in self.seq],dtype='fp32')
            self.tp_collective([b*q*8 for b in self.seq],'all_reduce','DFlash_accept')
            self.kernel('token_compare_and_truncate','DFlash_accept',[b*q*8 for b in self.seq],
                        [b*q*24 for b in self.seq],dtype='fp32')
            self.draft_seed(committed,'DFlash_KV_catchup')
        else: raise ValueError(self.phase)
        return self

    def memory(self):
        rows=target_weight_ledger(self.l.tp,self.l.ep)
        dr=draft_weights(self.l.tp)
        weight=sum(r['weight_bytes_per_card'] for r in rows+dr)
        scale=sum(r['scale_bytes_per_card'] for r in rows+dr)
        static=weight+scale+self.hw['npu']['runtime_bytes']+self.hw['npu']['workspace_bytes']
        state=sum(r['bytes'] for r in persistent_state(self.context,self.block_size,self.l.tp))
        # Specified reservation policy: full final KV + fused feature buffer
        # + maximum reusable component arena. No chunked-prefill optimization.
        fused_q=self.context if self.phase=='prefill' else self.block_size
        fused=[b*fused_q*DRAFT['hidden'] for b in self.seq]
        peaks=[static+b*state+a+f for b,a,f in zip(self.seq,self.arena,fused)]
        lane=max(range(self.l.ep),key=peaks.__getitem__)
        host_table=MODEL['ngram_params']*(1+1/32)
        host_peak=host_table/self.hosts+self.host_arena
        return dict(hbm_peak_bytes=max(peaks),host_peak_bytes=host_peak,
                    static_bytes_per_card=static,weight_bytes_per_card=weight,
                    weight_scale_bytes_per_card=scale,state_bytes_per_sequence_per_tp_rank=state,
                    hbm_peak_ratio=max(peaks)/self.hw['npu']['hbm_bytes'],
                    host_peak_ratio=host_peak/self.hw['cpu']['dram_bytes'],
                    peak_lane=lane,peak_components_bytes=dict(weight=weight,weight_scales=scale,
                    runtime=self.hw['npu']['runtime_bytes'],workspace=self.hw['npu']['workspace_bytes'],
                    KV_and_state=self.seq[lane]*state,activation_arena=self.arena[lane]-self._arena_conversion[lane],
                    weight_conversion_buffer=self._arena_conversion[lane],
                    fused_target_features=fused[lane]),arena_winning_component=self.arena_cause[lane],
                    lane_sequences=self.seq,host_table_replica_bytes=host_table,
                    max_conversion_buffer_bytes_per_card=max(self._max_conversion_buffer))

    def run(self,keep_stages=False):
        hw=self.hw; memory=self.memory()
        rates={f'NPU.{d}':hw['npu']['flops'][d]*hw['npu']['efficiency'] for d in ('fp8','fp32')}
        rates.update({'NPU.hbm':hw['npu']['hbm_Bps'],
                      'SU.tx':min(hw['scale_up']['bandwidth_Bps'],hw['scale_up']['injection_Bps'])*hw['scale_up']['efficiency'],
                      'HOST':hw['host_link']['injection_Bps']*hw['host_link']['efficiency'],
                      'CPU.dram':hw['cpu']['dram_Bps'],'CPU.fp32':hw['cpu']['flops']*hw['cpu']['efficiency'],
                      'CPU.iops':hw['cpu']['iops']})
        rates['SU.rx']=rates['SU.tx']
        assert rates['NPU.hbm']>=2*rates['SU.tx']
        assert hw['cpu']['dram_Bps']>=rates['HOST']
        g=Graph(resources={k:Resource(k,v,'unit/s') for k,v in rates.items()},
                memory_caps={'HBM':hw['npu']['hbm_bytes'],'DRAM':hw['cpu']['dram_bytes']},
                initial_allocations=[Allocation('hbm','HBM',memory['hbm_peak_bytes']),
                                     Allocation('host','DRAM',memory['host_peak_bytes'])])
        previous=()
        for i,row in enumerate(self.ops):
            eid=str(i);g.events.append(Event(eid,previous,row['work'],row['startup_s']))
            previous=(eid,)
        r=simulate(g,retain_event_payload=False)
        ok=r['status']=='feasible'; breakdown=defaultdict(float); components=defaultdict(float)
        stages=[]; seed_time=0.
        if ok:
            assert len(r['trace'])==len(self.ops)
            for trace,op in zip(r['trace'],self.ops):
                demands={k:v/rates[k] for k,v in op['work'].items() if v}
                bottleneck=max(demands,key=demands.get) if demands else None
                service=max(demands.values(),default=0.)
                assert math.isclose(trace['duration_s'],service+op['startup_s'],rel_tol=2e-8,abs_tol=1e-8)
                if 'SU.tx' in op['work']: key='su_transfer_s'
                elif 'HOST' in op['work']: key='host_transfer_s'
                elif bottleneck=='NPU.hbm': key='hbm_bound_s'
                elif bottleneck=='CPU.fp32': key='cpu_compute_s'
                elif bottleneck and bottleneck.startswith('CPU.'): key='host_memory_s'
                else: key='compute_bound_s'
                breakdown[key]+=service
                breakdown['communication_startup_s']+=op['startup_s']
                components[op['component']]+=trace['duration_s']
                if op['component']=='DFlash_KV_seed':seed_time+=trace['duration_s']
                if keep_stages: stages.append({**op,'duration_s':trace['duration_s'],'bottleneck':bottleneck})
            assert math.isclose(sum(breakdown.values()),r['duration_s'],rel_tol=1e-9,abs_tol=1e-7)
        return dict(status=r['status'],phase=self.phase,duration_s=r['duration_s'] if ok else None,
                    breakdown_s=dict(breakdown) if ok else None,
                    component_latency_s=dict(components) if ok else None,memory=memory,
                    physical_work=dict(self.totals) if ok else None,
                    native_completed_events=r['completed_events'],native_diagnostics=r['diagnostics'],
                    seed_s=seed_time if ok else None,stages=stages)


def run_phase(chip,tp,ep,sp,batch,context,block,phase,committed=1,keep_stages=False):
    # One disjoint replica suffices for latency; caller separately accounts DP.
    cards=max(16,tp*ep); layout=Layout(tp,ep,cards,sp); hw=hardware(chip,cards)
    p=Phase(hw,layout,batch,context,block,phase)
    m=p.memory()
    if m['hbm_peak_ratio']>1 or m['host_peak_ratio']>1:
        # Native admission checks the proven persistent lower bound; no pretend
        # full arena peak or latency is reported for this early rejection.
        r=p.run(keep_stages)
        r['capacity_check']='persistent_lower_bound'
        assert r['status']=='infeasible'
        return r
    assert layout.n>=8 and layout.n%8==0  # replicas use disjoint whole hosts
    r=p.build(committed).run(keep_stages)
    r['capacity_check']='full_reserved_phase'
    return r


def aggregate(results):
    ok=all(x['status']=='feasible' for x in results)
    out=dict(status='feasible' if ok else 'infeasible',
             duration_s=sum(x['duration_s'] for x in results) if ok else None,
             breakdown_s=None,component_latency_s=None,physical_work=None,
             memory=copy.deepcopy(max(results,key=lambda x:x['memory']['hbm_peak_bytes'])['memory']))
    out['memory']['host_peak_bytes']=max(x['memory']['host_peak_bytes'] for x in results)
    out['memory']['host_peak_ratio']=max(x['memory']['host_peak_ratio'] for x in results)
    if ok:
        for field in ('breakdown_s','component_latency_s','physical_work'):
            total=defaultdict(float)
            for x in results:
                for k,v in x[field].items(): total[k]+=v
            out[field]=dict(total)
    return out



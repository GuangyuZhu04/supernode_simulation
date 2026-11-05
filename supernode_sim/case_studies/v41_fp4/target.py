"""V4.1-specific analytical graph extension, using the frozen native scheduler.

This is not the generic streaming API, nor a measured DSpark implementation.
Every stage is a barrier. Independent, equal-capacity rank operations are
represented by their resource envelope; separate physical ledgers retain sums.
Ring/pairwise rounds are serialized, and their bottleneck bytes/startups add.
The quotient is valid only under the guards in validate_environment().
"""
from __future__ import annotations
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.config import load_config
from .architecture import synthetic_config, inventory

MODEL = synthetic_config()
TENSORS = inventory(MODEL)
BY_LAYER = defaultdict(list)
for _tensor in TENSORS:
    BY_LAYER[_tensor.layer].append(_tensor)
CONTEXTS = {"128k": 131072, "256k": 262144, "512k": 524288, "1M": 1048576}
BATCHES = (1, 2, 4, 8, 16, 32)
KIB, MIB = 1024, 1024**2


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")), encoding="utf-8")



@dataclass(frozen=True)
class Layout:
    tp: int
    ep: int
    sp: int

    @property
    def n(self):
        return self.tp * self.ep

    @property
    def dp(self):
        return 1024 // self.n

    @property
    def name(self):
        return f"tp{self.tp}-ep{self.ep}-dp{self.dp}-sp{self.sp}"

    @property
    def reason(self):
        if self.tp > MODEL["o_groups"] or MODEL["o_groups"] % self.tp:
            return "incompatible_output_groups_8"
        return None


def layouts(include_sp_off=True):
    for tp in (1, 2, 4, 8, 16, 32):
        for ep in (1, 2, 4, 8, 16, 32, 64, 128):
            if 32 <= tp * ep <= 1024:
                yield Layout(tp, ep, tp)
                if include_sp_off and 1 < tp <= 8:
                    yield Layout(tp, ep, 1)


def validate_environment(hw, layout):
    assert hw["npu_count"] == hw["su_size"] == 1024
    assert hw["topology"] == "full_mesh"
    assert hw["npus_per_host"] == 8 and layout.n % 8 == 0
    assert 1024 % layout.n == 0 and layout.sp in (1, layout.tp)
    assert MODEL["n_routed_experts"] % layout.ep == 0
    assert MODEL["dspark_n_routed_experts"] % layout.ep == 0
    assert hw["scale_up"]["bandwidth_Bps"] >= hw["scale_up"]["injection_Bps"]
    su = hw["scale_up"]["injection_Bps"] * hw["scale_up"]["efficiency"]
    assert hw["npu"]["hbm_Bps"] >= 2 * su
    host = hw["host_link"]["injection_Bps"] * hw["host_link"]["efficiency"]
    assert hw["cpu"]["dram_Bps"] >= host
    assert not hw["remote"]["enabled"] and not hw["ssd"]["enabled"]


def local_sequences(batch, ep):
    return [batch // ep + (i < batch % ep) for i in range(ep)]


@lru_cache(maxsize=256)
def weight_ledger(tp, ep, dspark):
    rows = []
    for t in TENSORS:
        if not dspark and t.component.startswith("Draft"):
            continue
        value, scale = t.bytes(tp, ep)
        rows.append({"layer": t.layer, "component": t.component, "tensor": t.name,
                     "weight_bytes_per_card": value, "scale_bytes_per_card": scale})
    return rows


def kv_ledger(context, dspark):
    """Per sequence, per TP rank. KV owners only; no cross-layer copies."""
    c, rows = MODEL, []
    length = context + (6 if dspark else 1)
    for layer in range(c["n_layers"]):
        rows.append({"layer": layer, "component": "SWA", "bytes": c["window_size"] * (c["head_dim"] + c["head_dim"]/32)})
        if layer in c["kv_source_layers"]:
            ratio = c["compress_ratios"][layer]
            slots = math.ceil(math.ceil(length / ratio) / 16) * 16
            rows.append({"layer": layer, "component": "CSA2_shared_KV",
                         "bytes": slots * (c["head_dim"] + c["head_dim"]/32)})
            rows.append({"layer": layer, "component": "CSA2_shared_index",
                         "bytes": slots * (c["index_head_dim"] + c["index_head_dim"]/32)})
            rows.append({"layer": layer, "component": "KV_page_table", "bytes": slots / 16 * 8 * 2})
            if ratio > 1:
                rows.append({"layer": layer, "component": "Compressor_partial_state",
                             "bytes": 2 * ratio * c["head_dim"] * 4})
    if dspark:
        rows.append({"layer": 64, "component": "DSpark_window",
                     "bytes": 3 * c["window_size"] * (c["head_dim"] + c["head_dim"]/32)})
        # Six candidate prefix versions; append-only global pools truncate.
        snapshot = 64 * c["window_size"] * (c["head_dim"] + c["head_dim"]/32)
        snapshot += 3 * 2 * 2 * c["head_dim"] * 4
        rows.append({"layer": -1, "component": "DSpark_snapshots_and_prefix_states",
                     "bytes": snapshot * 7 + 6 * 3 * c["dim"]})
    rows.append({"layer": -1, "component": "Engram_hash_history", "bytes": 4 * 8})
    # Reused RoPE sin/cos table per sequence owner, across layers.
    rows.append({"layer": -1, "component": "RoPE_shared_table", "bytes": length * c["rope_head_dim"] * 4})
    return rows


def count_mod(start, count, mod):
    q, r = divmod(count, mod)
    row = [q] * mod
    for j in range(r):
        row[(start + j) % mod] += 1
    return row


@lru_cache(maxsize=2048)
def routing(batch, q, ep, experts, topk, stable=False):
    """Integer cyclic routes, interleaved expert placement; explicit reuse.

    Main: consecutive top-k sets for consecutive query tokens. No within-
    round query reuse before wrapping the expert pool. Uniform load is an
    assumption, not an estimate of an unseen learned router.
    """
    seq = local_sequences(batch, ep)
    matrix, expert_counts, offset = [], [0] * experts, 0
    for b in seq:
        if stable:
            base = count_mod(offset * topk, b * topk, experts)
            local = [x * q for x in base]
            offset += b
        else:
            local = count_mod(offset * topk, b * q * topk, experts)
            offset += b * q
        row = [0] * ep
        for ex, count in enumerate(local):
            row[ex % ep] += count
            expert_counts[ex] += count
        matrix.append(row)
    assignments = [sum(row[j] for row in matrix) for j in range(ep)]
    unique = [sum(expert_counts[j] > 0 for j in range(lane, experts, ep)) for lane in range(ep)]
    assert sum(assignments) == batch * q * topk
    return tuple(map(tuple, matrix)), tuple(assignments), tuple(unique)


@lru_cache(maxsize=128)
def index_geometry(q, context, ratio, candidate, bulk, q_tile=256, key_tile=65536):
    """Streaming causal score tiles. No quadratic score tensor is allocated.

    Scores are reduced across TP *before* top-k. Later decoder indexers gather
    only the candidate list, an explicit optimized implementation hypothesis.
    Full candidate-source scores are still charged.
    """
    lengths, pairs, tiles, key_reads = [], 0, 0, 0
    for begin in range(0, q, q_tile):
        n = min(q_tile, q - begin)
        visible = (begin + n) // ratio if bulk else (context + q) // ratio
        keys = min(visible, 2048 * 8) if candidate else visible
        pairs += n * keys
        tiles += math.ceil(keys / key_tile) if keys else 0
        key_reads += keys
        lengths.append((n, keys))
    return {"pairs": pairs, "tiles": tiles, "key_reads": key_reads,
            "max_score_tile_elements": max((n * min(k, key_tile) for n,k in lengths), default=0)}


class Phase:
    def __init__(self, hw, layout, batch, context, dspark, phase, *, stable=False):
        validate_environment(hw, layout)
        self.hw, self.l, self.b, self.context = hw, layout, batch, context
        self.dspark, self.phase, self.stable = dspark, phase, stable
        self.seq = local_sequences(batch, layout.ep)
        self.ops = []
        self.arena = [0.] * layout.ep
        self.arena_cause = ["none"] * layout.ep
        self.totals = defaultdict(float)
        self.hosts = layout.n // 8
        self.host_arena = 0.
        self.c = MODEL

    def op(self, name, component, work, totals=None, startup=0., dtype="fp8", layer=-1):
        if not any(work.values()) and not startup:
            return
        row = {"name": name, "component": component, "layer": layer, "work": dict(work),
               "totals": dict(totals or {}), "startup_s": startup, "dtype": dtype}
        self.ops.append(row)
        for k, v in row["totals"].items():
            self.totals[k] += v

    def reserve(self, values, name):
        for i, value in enumerate(values):
            if value > self.arena[i]:
                self.arena[i], self.arena_cause[i] = value, name

    def kernel(self, name, component, flops, hbm, *, dtype="fp8", layer=-1, copies=None):
        copies = self.l.tp if copies is None else copies
        self.op(name, component, {f"NPU.{dtype}": max(flops, default=0),
                                 "NPU.hbm": max(hbm, default=0)},
                {f"flops_{dtype}": sum(flops) * copies, "hbm_bytes": sum(hbm) * copies},
                dtype=dtype, layer=layer)

    def local_matrix(self, layer, selected, token_counts, component, *, divisor=1):
        ts = [t for t in BY_LAYER[layer] if selected(t)]
        if not ts:
            return
        for dtype in ("fp8", "fp32"):
            subset = [t for t in ts if t.dtype == dtype]
            if not subset:
                continue
            params = sum(t.local_params(self.l.tp, self.l.ep) for t in subset)
            wb = sum(sum(t.bytes(self.l.tp, self.l.ep)) for t in subset)
            # Boundary IO per matrix (output/input); matrix fusion is not free.
            # Independent block-diagonal O_A groups have distinct inputs.
            # Matrix inputs use the controlled FP8 activation boundary;
            # FP32 router scores are written as FP32 before top-k.
            io = sum(math.prod(t.local_shape(self.l.tp,self.l.ep)[:-2]) *
                     (t.local_shape(self.l.tp,self.l.ep)[-1] +
                      t.local_shape(self.l.tp,self.l.ep)[-2]*(4 if dtype=="fp32" else 1))
                     for t in subset if len(t.shape)>=2)
            # SP pointwise kernels pad their local token count; dummy routes
            # are masked before expert dispatch and never produce extra tokens.
            local = [math.ceil(n/divisor) for n in token_counts]
            f = [2 * n * params for n in local]
            m = [wb + n * io if n else 0 for n in local]
            self.kernel(f"{component}.matrix.{dtype}", component, f, m, dtype=dtype, layer=layer)

    def tp_collective(self, sizes, kind, component, layer=-1, repeats=1):
        t = self.l.tp
        if t == 1 or not any(sizes) or not repeats:
            return
        rounds = (2 if kind == "all_reduce" else 1) * (t - 1)
        critical = max(sizes) / t * rounds
        payload = sum(sizes) * rounds
        self.op(kind, component, {"SU.tx": critical, "SU.rx": critical, "NPU.hbm": 2 * critical},
                {"su_payload_bytes": payload, "su_physical_bytes": payload, "hbm_bytes": 2 * payload,
                 "collective_rounds": rounds * repeats},
                startup=rounds * repeats * self.hw["scale_up"]["latency_s"], layer=layer)

    def pairwise(self, matrix, unit, component, *, layer=-1, copies=None):
        n = len(matrix)
        copies = self.l.tp if copies is None else copies
        critical, rounds, payload = 0., 0, 0.
        for step in range(1, n):
            amounts = [matrix[i][(i + step) % n] * unit for i in range(n)]
            if not any(amounts):
                continue
            critical += max(amounts)
            payload += sum(amounts) * copies
            rounds += 1
        self.op("pairwise", component, {"SU.tx": critical, "SU.rx": critical, "NPU.hbm": 2 * critical},
                {"su_payload_bytes": payload, "su_physical_bytes": payload, "hbm_bytes": 2 * payload,
                 "collective_rounds": rounds},
                startup=rounds * self.hw["scale_up"]["latency_s"], layer=layer)

    def embedding(self, q, component="Embedding"):
        h, c = self.c["dim"], self.c
        counts = [b * q for b in self.seq]
        self.kernel("row_lookup", component, [n * h for n in counts],
                    [n * h * (1 + c["hc_mult"]) for n in counts])
        self.tp_collective([n * h for n in counts], "all_reduce", component)

    def mhc(self, layer, counts, side, draft=False):
        c, t, sp = self.c, self.l.tp, self.l.sp
        h, hc = c["dim"], c["hc_mult"]
        local = [math.ceil(n / sp) for n in counts]
        comp = "Draft_mHC" if draft else "mHC"
        fn = 24 * hc * h
        ops = 2 * fn + 5 * hc * h + 2 * hc * hc * h + 4 * hc * h
        ops += c["hc_sinkhorn_iters"] * 4 * hc * hc + 8 * h
        self.kernel(side, comp, [n * ops for n in local],
                    [2 * n * hc * h + n * h * 2 + (fn * 4 if n else 0) for n in local],
                    dtype="fp32", layer=layer)
        # The pre/post operations are tiled in the reserved 256 MB workspace.
        if sp > 1:
            self.tp_collective([n * h for n in counts], "all_gather", "SP_gather", layer)

    def return_residual(self, layer, counts):
        h = self.c["dim"]
        self.tp_collective([n * h for n in counts],
                           "reduce_scatter" if self.l.sp > 1 else "all_reduce",
                           "TP_output", layer)

    def engram(self, layer, q):
        c, t, sp = self.c, self.l.tp, self.l.sp
        counts = [b * q for b in self.seq]
        queries = 24
        record = c["engram_head_dim"] * (1 + 1/32)
        nh = self.hosts
        request_matrix = [[0] * nh for _ in range(nh)]
        offset = 0
        for lane, n in enumerate(counts):
            owner = lane * t // 8
            values = count_mod(offset, n * queries, nh)
            offset += n * queries
            for j, value in enumerate(values):
                request_matrix[owner][j] += value
        src = [sum(r) for r in request_matrix]
        dst = [sum(r[j] for r in request_matrix) for j in range(nh)]
        self.op("ngram_hash", "Engram_host", {"CPU.fp32": max(src, default=0) * 64,
                 "CPU.dram": max(src, default=0) * 8},
                {"cpu_flops": sum(src) * 64, "dram_bytes": sum(src) * 8}, layer=layer)
        def host_transfer(values, unit, name):
            amount = max(values, default=0) * unit
            total = sum(values) * unit
            self.op(name, "Engram_host_transfer", {"HOST": amount, "CPU.dram": amount, "NPU.hbm": amount},
                    {"host_payload_bytes": total, "host_physical_bytes": total,
                     "dram_bytes": total, "hbm_bytes": total},
                    startup=self.hw["host_link"]["latency_s"] if total else 0, layer=layer)
        host_transfer(src, 8, "address_H2D")
        self.pairwise(request_matrix, 8, "Engram_address_SU", layer=layer, copies=1)
        host_transfer(dst, 8, "address_D2H")
        self.op("table_row_read", "Engram_host", {"CPU.iops": max(dst, default=0),
                 "CPU.dram": max(dst, default=0) * record},
                {"host_lookup_ops": sum(dst), "dram_bytes": sum(dst) * record}, layer=layer)
        host_transfer(dst, record, "row_H2D")
        self.pairwise(list(zip(*request_matrix)), record, "Engram_rows_SU", layer=layer, copies=1)
        # Return records first reach the originating host leader. Deliver to
        # the appropriate EP-lane leader when several TP groups share a host.
        for offset_in_host in range(1,8//t):
            selected=[n*queries*record for lane,n in enumerate(counts)
                      if lane % (8//t)==offset_in_host]
            critical,total=max(selected,default=0),sum(selected)
            self.op("host_leader_to_lane","Engram_rows_SU",
                    {"SU.tx":critical,"SU.rx":critical,"NPU.hbm":2*critical},
                    {"su_payload_bytes":total,"su_physical_bytes":total,"hbm_bytes":2*total},
                    startup=self.hw["scale_up"]["latency_s"] if total else 0,layer=layer)
        # All lookup bytes pass through the TP leader; scatter for SP, chain
        # broadcast for replicated pointwise execution. Explicit schedule.
        if t > 1:
            sizes = [n * queries * record for n in counts]
            factor = (t-1)/t if sp > 1 else t-1
            critical, total = max(sizes) * factor, sum(sizes) * factor
            self.op("row_scatter" if sp > 1 else "row_broadcast", "Engram_rows_TP",
                    {"SU.tx": critical, "SU.rx": critical, "NPU.hbm": 2 * critical},
                    {"su_payload_bytes": total, "su_physical_bytes": total, "hbm_bytes": 2*total},
                    startup=(t-1)*self.hw["scale_up"]["latency_s"], layer=layer)
        self.local_matrix(layer, lambda x: x.component == "EngramProjection" and x.role == "matrix",
                          counts, "Engram_projection", divisor=sp)
        self.kernel("gate_and_mix", "Engram_projection",
                    [math.ceil(n/sp) * 12 * c["hc_mult"] * c["dim"] for n in counts],
                    [math.ceil(n/sp) * (2*c["hc_mult"]*c["dim"] + queries*record) for n in counts],
                    dtype="fp32", layer=layer)
        self.host_arena = max(self.host_arena, max(dst, default=0)*(record+8))
        self.reserve([2*n/sp*c["hc_mult"]*c["dim"] + n/sp*queries*record +
                      n/sp*(c["hc_mult"]+1)*c["dim"] for n in counts], "Engram_projection")
        self.reserve([2*n/sp*c["hc_mult"]*c["dim"] +
                      (src[lane*t//8]+dst[lane*t//8])*(record+8)
                      for lane,n in enumerate(counts)],"Engram_transport")

    def global_projection(self, layer, q, *, skip_index=False):
        counts = [b*q for b in self.seq]
        self.local_matrix(layer, lambda x: x.component == "CSA2Compressor", counts, "CSA2_compressor")
        ratio = self.c["compress_ratios"][layer]
        self.local_matrix(layer, lambda x: ".index.k" in x.name and x.role == "matrix",
                          [n/ratio for n in counts], "CSA2_index_key")
        d = self.c["head_dim"]
        self.kernel("quantize_cache", "KV_write",
                    [n * d * 12 for n in counts],
                    [n / ratio * (d+128) * (1+1/32) + n*d*2 for n in counts],
                    dtype="fp32", layer=layer)

    def attention(self, layer, q, *, bulk=False, draft=False, skip_global=False):
        c, t, sp = self.c, self.l.tp, self.l.sp
        h, d = c["dim"], c["head_dim"]
        heads = c["draft_heads"] if draft else c["heads_by_layer"][layer]
        counts = [b*q for b in self.seq]
        comp = "Draft_attention" if draft else "CSA2_attention"
        self.local_matrix(layer, lambda x: ".attn." in x.name and x.role == "matrix"
                          and ".o_" not in x.name, counts, comp+"_QKV")
        if not draft and layer in c["kv_source_layers"] and not skip_global:
            self.global_projection(layer, q)
        if not draft and layer in c["index_source_layers"]:
            ratio = c["compress_ratios"][layer]
            replay=self.phase=="prefill" and not bulk
            geo = index_geometry(q, self.context-q if replay else self.context,
                                 ratio, layer > c["candidate_source_layer"], bulk)
            self.local_matrix(layer, lambda x: x.component == "CSA2Indexer" and
                              ".index.k" not in x.name and x.role == "matrix", counts, "CSA2_index_query")
            pairs = geo["pairs"]
            # Fused head accumulation avoids materializing per-head score cubes.
            f = [b * pairs * 2 * c["index_n_heads"]/t * c["index_head_dim"] for b in self.seq]
            mem = [b * (geo["key_reads"]*128*(1+1/32) + pairs*4 + q*32/t*128) for b in self.seq]
            self.kernel("score_dot", "CSA2_index_score", f, mem, layer=layer)
            self.tp_collective([b * pairs * 4 for b in self.seq], "all_reduce", "CSA2_index_reduce",
                               layer, repeats=geo["tiles"])
            self.kernel("topk_and_candidates", "CSA2_index_select",
                        [b*pairs*(3*c["index_n_heads"]/t + math.ceil(math.log2(512)) + 2)
                         for b in self.seq],
                        [b*(pairs*8+q*512*4+q*2048*4) for b in self.seq], dtype="fp32", layer=layer)
            assert geo["max_score_tile_elements"] * 4 + 65536*128*(1+1/32) + 256*32*128 < self.hw["npu"]["workspace_bytes"]
        # Shared latent key/value; head matrix work is TP-sharded, KV reads
        # replicate over TP. No hardware cache or cross-query hit is assumed.
        ratio = 0 if draft else c["compress_ratios"][layer]
        if bulk:
            window_pairs = q*(q+1)/2 if q < 128 else 128*q - 128*127/2
            global_pairs = (sum(min(512, pos//ratio) for pos in range(1, min(q,1024)+1)) +
                            max(0,q-1024)*512) if ratio else 0
        else:
            window_pairs = q*(q+1)/2 if self.phase=="prefill" else q*(128 + (5 if draft else 0))
            global_pairs = q*min(512,(self.context+q)//ratio) if ratio else 0
        pairs = window_pairs + global_pairs
        self.kernel("sparse_QK_AV", comp,
                    [b*4*pairs*heads/t*d for b in self.seq],
                    [b*(2*pairs*d*(1+1/32)+q*heads/t*d*2) for b in self.seq], layer=layer)
        self.kernel("softmax_rope_quant", comp,
                    [b*(pairs*heads/t*5 + q*(heads/t+1)*d*10) for b in self.seq],
                    [b*q*(heads/t+1)*d*2 for b in self.seq], dtype="fp32", layer=layer)
        self.local_matrix(layer, lambda x: ".attn.o_" in x.name and x.role == "matrix", counts, comp+"_output")
        self.return_residual(layer, counts)
        self.reserve([2*n/sp*h*c["hc_mult"] + 2*n*h + n*(c["q_lora_rank"]+2*heads/t*d) +
                      n*512*4 + (n*2048*4 if not draft and layer >= 32 else 0)
                      for n in counts], comp)

    def moe(self, layer, q, *, draft=False):
        c,t,sp = self.c,self.l.tp,self.l.sp
        h,i = c["dim"],c["moe_inter_dim"]
        e = c["dspark_n_routed_experts"] if draft else c["n_routed_experts"]
        k = c["dspark_n_activated_experts"] if draft else c["n_activated_experts"]
        comp = "Draft_MoE" if draft else "MoE"
        counts = [b*q for b in self.seq]
        self.local_matrix(layer, lambda x: x.role=="router", counts, comp+"_router", divisor=sp)
        self.kernel("route_topk", comp+"_router",
                    [math.ceil(n/sp)*e*(5+math.ceil(math.log2(k))) for n in counts],
                    [math.ceil(n/sp)*(e*8+k*8) for n in counts], dtype="fp32", layer=layer)
        self.reserve([2*n/sp*h*c["hc_mult"] + 2*n*h +
                      math.ceil(n/sp)*e*4 + n*k*8 for n in counts],comp+"_router_scores")
        if sp > 1:
            self.tp_collective([n*k*8 for n in counts], "all_gather", comp+"_routes", layer)
        matrix, assigns, unique = routing(self.b,q,self.l.ep,e,k,self.stable and not draft)
        unit = h + 8  # quantized hidden plus per-copy token/route indices
        self.pairwise(matrix, unit, comp+"_dispatch", layer=layer)
        # A destination groups tokens by expert, reads each used expert once
        # per phase, and retains dispatched inputs until expert GEMMs finish.
        expert_params = 3*h*i/t
        expert_scale = 3*math.ceil(h/32)*math.ceil(i/t/32)
        self.kernel("expert_GEMMs", comp+"_experts",
                    [2*a*expert_params for a in assigns],
                    [u*(expert_params+expert_scale)+a*(2*h+4*i/t) for a,u in zip(assigns,unique)],
                    layer=layer)
        self.kernel("SwiGLU", comp+"_experts", [a*i/t*8 for a in assigns],
                    [a*i/t*5 for a in assigns], dtype="fp32", layer=layer)
        # TP sum each routed expert output before returning it to its origin.
        self.tp_collective([a*h for a in assigns], "all_reduce", comp+"_expert_TP", layer)
        self.pairwise(list(zip(*matrix)), h, comp+"_combine", layer=layer)
        self.local_matrix(layer, lambda x: ".shared." in x.name, counts, comp+"_shared")
        self.kernel("shared_SwiGLU_and_combine", comp+"_shared",
                    [n*(i/t*8+h*k*2) for n in counts],
                    [n*(i/t*5+h*(k+1)) for n in counts], dtype="fp32", layer=layer)
        self.return_residual(layer, counts)
        # Retained dispatch inputs and expert outputs, tiled GEMM temporaries,
        # source route indices, SP residual and gathered input; one reusable arena.
        staging = []
        for s in range(self.l.ep):
            staging.append(max((matrix[s][(s+j)%self.l.ep]*unit +
                                matrix[(s-j)%self.l.ep][s]*unit for j in range(1,self.l.ep)),default=0))
        self.reserve([2*n/sp*h*c["hc_mult"] + n*h*2 + n*k*8 + a*(2*h+4*i/t) + z
                      for n,a,z in zip(counts,assigns,staging)], comp)

    def block(self, layer, q, *, bulk=False, draft=False, skip_global=False):
        counts = [b*q for b in self.seq]
        if self.dspark and not draft and layer in self.c["dspark_target_layer_ids"]:
            self.kernel("capture_target_hidden","DSpark_capture",
                        [n*self.c["hc_mult"]*self.c["dim"] for n in counts],
                        [n*(self.c["hc_mult"]+1)*self.c["dim"] for n in counts],
                        dtype="fp32",layer=layer)
        if not draft and layer in self.c["engram_layer_ids"]:
            self.engram(layer,q)
        self.mhc(layer,counts,"attention_prepost",draft)
        self.attention(layer,q,bulk=bulk,draft=draft,skip_global=skip_global)
        self.mhc(layer,counts,"ffn_prepost",draft)
        self.moe(layer,q,draft=draft)

    def head(self,q,component="Output"):
        c,t = self.c,self.l.tp
        counts = [b*q for b in self.seq]
        # Final mHC collapse/norm consumes the distributed residual; the
        # vocabulary-sharded head needs all valid query vectors on every TP rank.
        self.kernel("final_collapse_norm",component,
                    [math.ceil(n/self.l.sp)*c["dim"]*(c["hc_mult"]+5) for n in counts],
                    [math.ceil(n/self.l.sp)*c["dim"]*(c["hc_mult"]+1) for n in counts],
                    dtype="fp32")
        if self.l.sp>1:
            self.tp_collective([n*c["dim"] for n in counts],"all_gather",component)
        p = c["vocab_size"]*c["dim"]/t
        self.kernel("shared_output_head",component,[n*2*p for n in counts],
                    [(p*(1+1/1024)+n*(c["dim"]+4*c["vocab_size"]/t)) if n else 0 for n in counts])
        # Logits FP32; distributed all-gather before argmax/sample.
        self.tp_collective([n*c["vocab_size"]*4 for n in counts],"all_gather",component)
        self.kernel("sampling",component,[n*c["vocab_size"]*8 for n in counts],
                    [n*c["vocab_size"]*4 for n in counts],dtype="fp32")
        logits_factor=1+2/t if t>1 else 1
        self.reserve([2*n/self.l.sp*c["hc_mult"]*c["dim"]+n*c["dim"]+
                      n*c["vocab_size"]*4*logits_factor for n in counts],component+"_logits")

    def draft_seed(self, q, component="DSpark_seed"):
        counts=[b*q for b in self.seq]
        self.local_matrix(64,lambda x:x.component=="DraftInput" and x.role=="matrix",counts,component)
        for layer in range(64,67):
            self.local_matrix(layer,lambda x:".attn.window_kv" in x.name,counts,component)
        self.kernel("normalize_seed",component,[n*self.c["dim"]*12 for n in counts],
                    [n*(3*self.c["dim"]+3*self.c["head_dim"]*(1+1/32)) for n in counts],dtype="fp32")

    def build(self, committed=1):
        c=self.c
        if self.phase=="prefill":
            q=self.context
            self.embedding(q)
            for layer in range(32):
                self.block(layer,q,bulk=True)
            # Decoder global KV is computed from final encoder states once.
            self.global_projection(32,q)
            # Seed every decoder SWA and obtain the first output with bounded
            # replay. Its global projection must not run a second time.
            for layer in range(32,64):
                self.block(layer,min(128,q),skip_global=True)
            self.head(1)
            if self.dspark:
                self.draft_seed(min(128,q))
        elif self.phase in ("decode","verify"):
            q=1 if self.phase=="decode" else 6
            self.embedding(q)
            for layer in range(64):
                self.block(layer,q)
            self.head(q)
        elif self.phase=="draft":
            self.embedding(5,"Draft_embedding")
            # Main hidden projection/window update is performed in seed/commit.
            for layer in range(64,67):
                self.block(layer,5,draft=True)
            self.head(5,"Draft_output")
            r=c["dspark_markov_rank"]
            counts=[b for b in self.seq]
            for step in range(5):
                self.kernel("markov_embed_head","Draft_markov",
                            [n*2*c["vocab_size"]*r/self.l.tp for n in counts],
                            [(c["vocab_size"]*r/self.l.tp + n*(r+c["vocab_size"]/self.l.tp)*4)
                             if n else 0 for n in counts])
                self.tp_collective([n*r for n in counts],"all_reduce","Draft_markov")
                self.tp_collective([n*c["vocab_size"]*4 for n in counts],"all_gather","Draft_markov")
            self.kernel("confidence","Draft_confidence",
                        [b*5*2*(c["dim"]+r) for b in self.seq],
                        [b*5*(c["dim"]+r)*4 for b in self.seq],dtype="fp32")
            # Five target-head logit rows remain live while one Markov bias
            # and its all-gather staging are produced at a time.
            self.reserve([b*((6+(2/self.l.tp if self.l.tp>1 else 0))*c["vocab_size"]*4+
                              5*(c["dim"]+r)*4) for b in self.seq],"Draft_Markov_logits")
        elif self.phase=="commit":
            # Versioned SWA/pooling state, and append-only global cache prefixes.
            state = 64*128*512*(1+1/32) + 3*2*2*512*4
            reads_writes=2*state*(1+6+1)  # initial snapshot + six versions + restore
            counts=[b for b in self.seq]
            self.kernel("prefix_snapshot_restore","DSpark_state",
                        [b*6*64*512*4 for b in counts],
                        [b*(reads_writes+committed*(64*512+3*c["dim"])*2) for b in counts],
                        dtype="fp32")
            self.tp_collective([b*6*8 for b in counts],"all_reduce","DSpark_accept")
            self.kernel("accept_and_truncate","DSpark_accept",
                        [b*6*c["vocab_size"]*4 for b in counts],
                        [b*6*c["vocab_size"]*8 for b in counts],dtype="fp32")
            # At most C verified input hidden states become committed prefix.
            # The bonus/replacement token is the next round's unprocessed input.
            self.draft_seed(committed,"DSpark_catchup")
        else:
            raise ValueError(self.phase)
        return self

    def run(self):
        hw,l=self.hw,self.l
        weights=weight_ledger(l.tp,l.ep,self.dspark)
        weight=sum(r["weight_bytes_per_card"] for r in weights)
        scales=sum(r["scale_bytes_per_card"] for r in weights)
        static=weight+scales+hw["npu"]["runtime_bytes"]+hw["npu"]["workspace_bytes"]
        state_rows=kv_ledger(self.context,self.dspark)
        state=sum(r["bytes"] for r in state_rows)
        resident=[static+b*state for b in self.seq]
        peaks=[x+y for x,y in zip(resident,self.arena)]
        maxlane=max(range(l.ep),key=peaks.__getitem__)
        table=MODEL["ngram_params"]
        table_scale=table/32
        host_resident=(table+table_scale)/self.hosts
        memory={"weight_bytes_per_card":weight,"weight_scale_bytes_per_card":scales,
                "runtime_bytes_per_card":hw["npu"]["runtime_bytes"],
                "workspace_bytes_per_card":hw["npu"]["workspace_bytes"],
                "static_bytes_per_card":static,"state_bytes_per_sequence_per_tp_rank":state,
                "hbm_peak_bytes":max(peaks),"hbm_peak_ratio":max(peaks)/hw["npu"]["hbm_bytes"],
                "replica_hbm_reserved_bytes":sum(peaks)*l.tp,
                "replica_hbm_persistent_bytes":sum(resident)*l.tp,
                "peak_lane":maxlane,"peak_components_bytes":{"weight":weight,"weight_scales":scales,
                 "runtime":hw["npu"]["runtime_bytes"],"workspace":hw["npu"]["workspace_bytes"],
                 "KV_and_state":state*self.seq[maxlane],"activation_arena":self.arena[maxlane]},
                "arena_winning_component":self.arena_cause[maxlane],
                "host_table_replica_bytes":table+table_scale,
                "host_peak_bytes":host_resident+self.host_arena,
                "host_peak_ratio":(host_resident+self.host_arena)/hw["cpu"]["dram_bytes"],
                "lane_sequences":self.seq,"lane_arena_bytes":self.arena,
                "lane_peak_bytes":peaks}
        resources={}
        for dtype in ("fp8","fp32"):
            resources["NPU."+dtype]=Resource("NPU."+dtype,
                hw["npu"]["flops"][dtype]*hw["npu"]["efficiency"],"FLOP/s","compute")
        for key,value,unit in (
            ("NPU.hbm",hw["npu"]["hbm_Bps"],"byte/s"),
            ("SU.tx",hw["scale_up"]["injection_Bps"]*hw["scale_up"]["efficiency"],"byte/s"),
            ("SU.rx",hw["scale_up"]["injection_Bps"]*hw["scale_up"]["efficiency"],"byte/s"),
            ("HOST",hw["host_link"]["injection_Bps"]*hw["host_link"]["efficiency"],"byte/s"),
            ("CPU.dram",hw["cpu"]["dram_Bps"],"byte/s"),
            ("CPU.fp32",hw["cpu"]["flops"]*hw["cpu"]["efficiency"],"FLOP/s"),
            ("CPU.iops",hw["cpu"]["iops"],"operation/s")):
            resources[key]=Resource(key,value,unit)
        # All arenas are reserved for the phase; the maximum card/host is a
        # sufficient and exact capacity test for this specified policy.
        graph=Graph(resources=resources,
                    memory_caps={"HBM":hw["npu"]["hbm_bytes"],"DRAM":hw["cpu"]["dram_bytes"]},
                    initial_allocations=[Allocation("hbm_reserved","HBM",max(peaks),"reserved"),
                                         Allocation("host_reserved","DRAM",memory["host_peak_bytes"],"ngram")])
        previous=()
        for i,row in enumerate(self.ops):
            eid=f"{i:04}.{row['layer']}.{row['name']}"
            graph.events.append(Event(eid,previous,row["work"],row["startup_s"],
                                      metadata={"component":row["component"],"layer":row["layer"]}))
            previous=(eid,)
        result=simulate(graph,retain_event_payload=False)
        status=result["status"]
        breakdown=defaultdict(float)
        components=defaultdict(float)
        stages=[]
        if status=="feasible":
            assert len(result["trace"])==len(self.ops)
            for trace,row in zip(result["trace"],self.ops):
                duration=trace["duration_s"]
                demands={key:v/resources[key].capacity for key,v in row["work"].items() if v}
                bottleneck=max(demands,key=demands.get) if demands else None
                service=max(demands.values(),default=0.)
                assert math.isclose(duration,row["startup_s"]+service,rel_tol=2e-8,abs_tol=1e-9)
                if "SU.tx" in row["work"] or "HOST" in row["work"]:
                    label="host_transfer_s" if "HOST" in row["work"] else "su_transfer_s"
                elif bottleneck=="NPU.hbm":
                    label="hbm_bound_s"
                elif bottleneck and bottleneck.startswith("CPU."):
                    label="host_lookup_hash_s"
                else:
                    label="compute_bound_s"
                breakdown[label]+=service
                breakdown["communication_startup_s"]+=row["startup_s"]
                components[row["component"]]+=duration
                stages.append({**row,"duration_s":duration,"bottleneck":bottleneck})
            assert math.isclose(sum(breakdown.values()),result["duration_s"],rel_tol=1e-9,abs_tol=1e-8)
        return {"phase":self.phase,"status":status,
                "duration_s":result["duration_s"] if status=="feasible" else None,
                "memory":memory,"physical_work":dict(self.totals),"breakdown_s":dict(breakdown),
                "component_latency_s":dict(components),"stages":stages,
                "native_diagnostics":result["diagnostics"],"native_completed_events":result["completed_events"]}


def run_phase(hw,layout,batch,context,dspark,phase,committed=1,stable=False):
    return Phase(hw,layout,batch,context,dspark,phase,stable=stable).build(committed).run()


def merge_phases(results):
    feasible=all(r["status"]=="feasible" for r in results)
    out={"status":"feasible" if feasible else "infeasible","duration_s":
         sum(r["duration_s"] for r in results) if feasible else None,
         "breakdown_s":{},"component_latency_s":{},"physical_work":{},
         "memory":copy.deepcopy(max(results,key=lambda r:r["memory"]["hbm_peak_bytes"])["memory"])}
    for field in ("breakdown_s","component_latency_s","physical_work"):
        sums=defaultdict(float)
        for result in results:
            for k,v in result[field].items():
                sums[k]+=v
        out[field]=dict(sums)
    # Phases reuse the same arenas, no arithmetic addition of their peaks.
    return out

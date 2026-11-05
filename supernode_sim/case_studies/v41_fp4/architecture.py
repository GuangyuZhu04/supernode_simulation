"""Tensor-shape audit of a synthetic V4.1-Pro-10T, not an official checkpoint.

Reference code is retained as read-only evidence, never imported/executed.
The extrapolation keeps CED/CSA2/mHC/Engram mechanisms and explicitly widens
decoder attention to reconcile the requested ~76B/~155B phase-active anchors.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import defaultdict
from math import ceil, prod
from pathlib import Path
import copy, csv, hashlib, json

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Tensor:
    name: str
    layer: int
    component: str
    role: str
    shape: tuple[int, ...]
    tp_axis: int | None = None
    expert_axis: bool = False
    active_fraction: float = 1.0
    dtype: str = "fp8"
    source: str = "official-reference-shape"

    @property
    def params(self):
        return prod(self.shape)

    def local_shape(self, tp=1, ep=1):
        dims = list(self.shape)
        if self.expert_axis:
            assert dims[0] % ep == 0
            dims[0] //= ep
        if self.tp_axis is not None:
            assert dims[self.tp_axis] % tp == 0
            dims[self.tp_axis] //= tp
        return tuple(dims)

    def local_params(self, tp=1, ep=1):
        return prod(self.local_shape(tp, ep))

    def bytes(self, tp=1, ep=1, scheme="fp8_comparison"):
        shape = self.local_shape(tp, ep)
        count = prod(shape)
        dtype = self.dtype
        if scheme == "native_storage" and self.expert_axis:
            dtype = "fp4"
        size = count * {"fp4": .5, "fp8": 1, "bf16": 2, "fp32": 4}[dtype]
        scales = 0
        if dtype == "fp8" and self.role == "matrix":
            scales = prod(shape[:-2]) * ceil(shape[-2] / 32) * ceil(shape[-1] / 32)
        elif dtype == "fp4":
            scales = prod(shape[:-1]) * ceil(shape[-1] / 32)
        elif dtype == "fp8" and self.role == "lookup":
            scales = prod(shape[:-1]) * ceil(shape[-1] / 32)
        return size, scales


def flash_config():
    c = json.loads((HERE / "reference_config.json").read_text(encoding="utf-8"))
    c["encoder_layers"] = c["n_layers"] // 2
    c["heads_by_layer"] = [c["n_heads"]] * c["n_layers"]
    c["draft_heads"] = c["n_heads"]
    c["ngram_params"] = sum(c["engram_num_embeddings"]) * c["engram_head_dim"]
    return c


def synthetic_config():
    original = flash_config()
    c = copy.deepcopy(original)
    c.update(name="DeepSeek-V4.1-Pro-10T-synthetic", n_layers=64, encoder_layers=32,
             dim=7168, n_routed_experts=1792, moe_inter_dim=4096, n_activated_experts=24,
             n_mtp_layers=3, ngram_params=3_439_800_000_000,
             reported_backbone_params=10_000_000_000_000,
             reported_prefill_active_params=76_000_000_000,
             reported_decode_active_params=155_000_000_000)
    # 64 encoder heads retain the Flash attention width; 128 decoder heads
    # enlarge decode capacity without adding bulk encoder/prefill work.
    c["heads_by_layer"] = [64] * 32 + [128] * 32
    c["draft_heads"] = 64
    c["n_heads"] = 64
    scale = c["n_layers"] / original["n_layers"]
    for key in ("kv_source_layers", "index_source_layers", "engram_layer_ids"):
        c[key] = [round(i * scale) for i in original[key]]
    c["candidate_source_layer"] = c["encoder_layers"]
    c["compress_ratios"] = [0] * c["kv_source_layers"][0] + [2] * (
        c["encoder_layers"] - c["kv_source_layers"][0]) + [1] * (
        c["n_layers"] - c["encoder_layers"]) + [0] * c["n_mtp_layers"]
    c["dspark_target_layer_ids"] = list(range(c["n_layers"]-3, c["n_layers"]))
    assert c["ngram_params"] % (2 * c["engram_head_dim"]) == 0
    c["engram_num_embeddings"] = [c["ngram_params"] // (2 * c["engram_head_dim"])] * 2
    c["assumptions"] = [
        "Synthetic 64-layer,7168-hidden extrapolation; no released/trained Pro-10T checkpoint or quality equivalence is asserted.",
        "Encoder32 layers/64 heads; decoder32 layers/128 heads. Decoder-only attention widening reconciles the user phase-active anchors without an artificial latency multiplier.",
        "KV/index/Engram source positions are rounded relative-depth mappings from Flash40 to64; sharing quality has not been evaluated.",
        "Exact aggregate ngram table size3439.8B is split across two tables. Row counts are a storage/addressing model, not reconstructed prime hash sizes of a trained checkpoint.",
        "Text backbone only, excluding vision and projector. Three-layer DSpark is accounted separately; its experts128/top3 and five-position draft block follow the Flash reference.",
        "Main controlled experiment uses FP8 matrix storage/compute/activation/communication/KV for comparison with Kimi; FP32 mHC/scalar state remains explicit. Native FP4 storage requires separate conversion assumptions.",
        "o_groups=8 is retained: group-parallel TP>8 is incompatible and must be reported rather than silently reshaped.",
    ]
    return c


def inventory(c, include_draft=True):
    result = []
    h, d, q, o, groups = c["dim"], c["head_dim"], c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc = c["hc_mult"]

    def add(name, layer, component, role, shape, tp=None, expert=False, active=1., dtype="fp8"):
        result.append(Tensor(name, layer, component, role, tuple(shape), tp, expert, active, dtype))

    add("embed.weight", -1, "WordEmbedding", "lookup", (c["vocab_size"], h), tp=0)
    add("head.weight", c["n_layers"], "Output", "matrix", (c["vocab_size"], h), tp=0)
    add("norm.weight", c["n_layers"], "Output", "norm", (h,), dtype="fp32")
    for layer in range(c["n_layers"] + (c["n_mtp_layers"] if include_draft else 0)):
        draft = layer >= c["n_layers"]
        heads = c["draft_heads"] if draft else c["heads_by_layer"][layer]
        prefix = f"{'draft' if draft else 'layers'}.{layer}"
        component = "DraftAttention" if draft else "CSA2Attention"
        add(prefix+".attn.q_a", layer, component, "matrix", (q, h))
        add(prefix+".attn.q_norm", layer, component, "norm", (q,), dtype="fp32")
        add(prefix+".attn.q_b", layer, component, "matrix", (heads*d, q), tp=0)
        add(prefix+".attn.window_kv", layer, component, "matrix", (d, h))
        add(prefix+".attn.kv_norm", layer, component, "norm", (d,), dtype="fp32")
        add(prefix+".attn.o_a", layer, component, "matrix", (groups, o, heads*d//groups), tp=0)
        add(prefix+".attn.o_b", layer, component, "matrix", (h, groups*o), tp=1)
        add(prefix+".attn.sink", layer, component, "scalar", (heads,), tp=0, dtype="fp32")
        if not draft and layer in c["kv_source_layers"]:
            ratio = c["compress_ratios"][layer]
            add(prefix+".compress.kv", layer, "CSA2Compressor", "matrix", (d, h))
            if ratio > 1:
                add(prefix+".compress.gate", layer, "CSA2Compressor", "matrix", (d, h))
            add(prefix+".compress.norm", layer, "CSA2Compressor", "norm", (d,), dtype="fp32")
        if not draft and layer in c["index_source_layers"]:
            ih, ide = c["index_n_heads"], c["index_head_dim"]
            add(prefix+".index.q", layer, "CSA2Indexer", "matrix", (ih*ide, q), tp=0)
            add(prefix+".index.weights", layer, "CSA2Indexer", "matrix", (ih, h), tp=0)
            if layer in c["kv_source_layers"]:
                add(prefix+".index.k", layer, "CSA2Indexer", "matrix", (ide, d))
                add(prefix+".index.k_norm", layer, "CSA2Indexer", "norm", (ide,), dtype="fp32")
        e = c["dspark_n_routed_experts"] if draft else c["n_routed_experts"]
        k = c["dspark_n_activated_experts"] if draft else c["n_activated_experts"]
        comp = "DraftMoE" if draft else "MoE"
        add(prefix+".router", layer, comp, "router", (e, h), dtype="fp32")
        add(prefix+".router.bias", layer, comp, "scalar", (e,), dtype="fp32")
        for name, shape, axis in (("gate", (e,c["moe_inter_dim"],h),1),
                                  ("up", (e,c["moe_inter_dim"],h),1),
                                  ("down", (e,h,c["moe_inter_dim"]),2)):
            add(prefix+".experts."+name, layer, comp, "matrix", shape, tp=axis, expert=True, active=k/e)
        for name, shape, axis in (("gate", (c["moe_inter_dim"],h),0),
                                  ("up", (c["moe_inter_dim"],h),0),
                                  ("down", (h,c["moe_inter_dim"]),1)):
            add(prefix+".shared."+name, layer, comp, "matrix", shape, tp=axis)
        for side in ("attn", "ffn"):
            add(prefix+"."+side+"_norm", layer, "DraftNorm" if draft else "Norm", "norm", (h,), dtype="fp32")
            mix = (2+hc)*hc
            hc_comp = "DraftMHC" if draft else "mHC"
            add(prefix+".hc_"+side+".fn", layer, hc_comp, "matrix", (mix,hc*h), dtype="fp32")
            add(prefix+".hc_"+side+".base", layer, hc_comp, "scalar", (mix,), dtype="fp32")
            add(prefix+".hc_"+side+".scale", layer, hc_comp, "scalar", (3,), dtype="fp32")
        if not draft and layer in c["engram_layer_ids"]:
            lookup_width = (c["engram_max_ngram_size"]-1)*c["engram_n_heads"]*c["engram_head_dim"]
            add(prefix+".engram.wkv", layer, "EngramProjection", "matrix", ((hc+1)*h,lookup_width))
            for side in ("q", "k"):
                add(prefix+".engram."+side, layer, "EngramProjection", "norm", (hc,h), dtype="fp32")
    if include_draft:
        layer = c["n_layers"]
        add("draft.main_proj", layer, "DraftInput", "matrix", (h, len(c["dspark_target_layer_ids"])*h))
        add("draft.main_norm", layer, "DraftInput", "norm", (h,), dtype="fp32")
        last = c["n_layers"] + c["n_mtp_layers"] - 1
        add("draft.final_norm", last, "DraftOutput", "norm", (h,), dtype="fp32")
        r = c["dspark_markov_rank"]
        add("draft.markov.embed", last, "DraftMarkov", "lookup", (c["vocab_size"],r), tp=0)
        add("draft.markov.head", last, "DraftMarkov", "matrix", (c["vocab_size"],r), tp=0)
        add("draft.confidence", last, "DraftConfidence", "matrix", (1,h+r), dtype="fp32")
    return result


def summarize(c):
    tensors = inventory(c)
    main = [t for t in tensors if not t.component.startswith("Draft")]
    draft = [t for t in tensors if t.component.startswith("Draft")]
    active = lambda ts: sum(t.params*t.active_fraction for t in ts if t.role != "lookup")
    # Bulk prompt tokens bypass decoder blocks; final head and bounded replay
    # remain actual workload costs, not folded into the asymptotic active count.
    enc = [t for t in main if 0 <= t.layer < c["encoder_layers"]]
    decoder_kv = [t for t in main if t.layer >= c["encoder_layers"]
                  and (t.component == "CSA2Compressor" or ".index.k" in t.name)]
    return {
        "backbone_params": sum(t.params for t in main),
        "ngram_params": c["ngram_params"],
        "dspark_additional_params": sum(t.params for t in draft),
        "resident_logical_params_with_dspark": sum(t.params for t in tensors)+c["ngram_params"],
        "prefill_bulk_active_params": active(enc)+active(decoder_kv),
        "decode_active_params": active(main),
        "draft_active_params_excluding_shared_embed_output": active(draft),
        "active_count_definition": "Unique non-lookup parameters used in bulk phase; excludes giant lookup tables and input embedding row lookup; prefill excludes O(window) decoder replay and one-position output head. Those costs are still explicitly executed.",
        "components": dict((comp, sum(t.params for t in tensors if t.component==comp))
                           for comp in sorted({t.component for t in tensors})),
    }



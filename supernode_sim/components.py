"""Shape-derived architecture inventory, independent of placement and precision.

The optional ``model.architecture`` is an explicit architectural hypothesis, not
a model-name lookup.  Budget-only profiles deliberately remain ``Unassigned``.
KDA here denotes the declared projection/gating layout documented in formulas;
it must not be interpreted as a reconstruction of an unpublished Kimi model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, prod
from typing import Any


@dataclass(frozen=True)
class ParameterTensor:
    name: str
    component: str
    layer: int
    shape: tuple[int, ...]
    role: str = "matrix"
    tp_shard: bool = True
    ep_shard: bool = False
    active_fraction: float = 1.0
    alias_of: str | None = None
    formula: str = "product(shape)"
    tp_max_shards: int | None = None
    budget_elements: float | None = None

    @property
    def elements(self) -> float:
        return self.budget_elements if self.budget_elements is not None else prod(self.shape)

    @property
    def accounted_elements(self) -> float:
        return 0.0 if self.alias_of else self.elements

    @property
    def active_elements(self) -> float:
        return self.accounted_elements * self.active_fraction

    def local_elements(self, tp: int = 1, ep: int = 1, fsdp: int = 1) -> float:
        """Resident elements per rank, including declared TP replication."""
        if min(tp, ep, fsdp) < 1:
            raise ValueError("parallel factors must be positive")
        shards = min(tp, self.tp_max_shards or tp) if self.tp_shard else 1
        return self.accounted_elements / shards / (ep if self.ep_shard else 1) / fsdp

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "elements": self.elements,
                "accounted_elements": self.accounted_elements,
                "active_elements": self.active_elements}


def _arch(model: dict) -> dict:
    return model.get("architecture") or {}


def _kind(model: dict, layer: int, field: str) -> str:
    arch = _arch(model)
    if not arch:
        if field == "attention_by_layer":
            return "MLA" if model.get("attention_kind") == "latent" else model.get("attention_kind", "gqa").upper()
        return "MoE" if model.get("moe_layer_fraction", 0) else "Dense"
    return arch[field][layer]


def parameter_inventory(model: dict) -> list[ParameterTensor]:
    """Return individually named parameter tensors; aliases carry zero ownership.

    Shapes omit unspecified biases by default (``linear_bias=False`` and
    ``norm_bias=False``). Gated FFNs have gate/up/down matrices; the nonlinear
    gate function (e.g. SiTU or SiLU) does not change their shapes. A final norm
    is included by default and belongs to PostNorm on the final layer.
    """
    arch = _arch(model)
    if not arch:
        return [ParameterTensor("unassigned_parameter_budget", "Unassigned", -1, (),
                    role="unknown", budget_elements=model["total_params"],
                    active_fraction=(model.get("active_params") or model["total_params"]) / model["total_params"],
                    formula="user total_params; component structure not supplied")]
    h, layers = model["hidden_dim"], model["layers"]
    n, head = model.get("num_heads", 32), model.get("head_dim", 128)
    if len(arch["attention_by_layer"]) != layers or len(arch["ffn_by_layer"]) != layers:
        raise ValueError("architecture attention_by_layer and ffn_by_layer must match model.layers")
    vocab = arch["vocab_size"]
    tensors: list[ParameterTensor] = []

    def add(name, component, layer, shape, *, role="matrix", tp_shard=True,
            ep_shard=False, active_fraction=1.0, alias_of=None, formula="product(shape)", tp_max_shards=None):
        if any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError(f"{name}: tensor dimensions must be positive integers")
        tensors.append(ParameterTensor(name, component, layer, tuple(shape), role,
            tp_shard, ep_shard, active_fraction, alias_of, formula, tp_max_shards))

    def linear(name, component, layer, inputs, outputs, **kwargs):
        add(name + ".weight", component, layer, (outputs, inputs), **kwargs)
        if arch.get("linear_bias", False):
            bias_kwargs = {key: value for key, value in kwargs.items() if key != "formula"}
            add(name + ".bias", component, layer, (outputs,), role="vector", **bias_kwargs)

    def norm(name, component, layer, width):
        add(name + ".weight", component, layer, (width,), role="norm", tp_shard=False,
            formula="normalization scale width")
        if arch.get("norm_bias", False):
            add(name + ".bias", component, layer, (width,), role="norm", tp_shard=False)

    add("word_embeddings.weight", "WordEmbedding", 0, (vocab, h), role="embedding",
        active_fraction=1.0 if arch.get("tie_embeddings", False) else 1.0 / vocab,
        formula="vocab_size * hidden_dim; per-token lookup activates one row (full matrix if tied output)")
    for layer in range(layers):
        stem = f"layers.{layer}"
        norm(stem + ".pre_norm", "PreNorm", layer, h)
        norm(stem + ".post_norm", "PostNorm", layer, h)
        if arch.get("attnres_block_size"):
            for path in ("attention", "ffn"):
                norm(stem + ".attnres." + path + ".norm", "AttnRes", layer, h)
                linear(stem + ".attnres." + path + ".projection", "AttnRes", layer, h, 1,
                       tp_shard=False, formula="hidden_dim to scalar residual attention score")
        kind = _kind(model, layer, "attention_by_layer").upper()
        component = kind
        if kind == "MLA":
            a = arch.get("mla") or {}
            rq = a.get("q_lora_rank", 0)
            rkv = a.get("kv_lora_rank", model.get("kv_latent_dim", 512))
            nope = a.get("qk_nope_head_dim", head)
            rope = a.get("qk_rope_head_dim", model.get("kv_rope_dim", 64))
            value = a.get("v_head_dim", head)
            if rq:
                linear(stem + ".mla.q_down", component, layer, h, rq, tp_shard=False,
                       formula="hidden_dim * q_lora_rank; replicated latent projection")
                norm(stem + ".mla.q_norm", component, layer, rq)
                linear(stem + ".mla.q_up", component, layer, rq, n * (nope + rope))
            else:
                linear(stem + ".mla.q", component, layer, h, n * (nope + rope))
            linear(stem + ".mla.kv_down", component, layer, h, rkv + rope,
                   tp_shard=False, formula="hidden_dim * (kv_lora_rank + qk_rope_head_dim)")
            norm(stem + ".mla.kv_norm", component, layer, rkv)
            linear(stem + ".mla.kv_up", component, layer, rkv, n * (nope + value))
            linear(stem + ".mla.output", component, layer, n * value, h)
            if a.get("use_output_gate", False):
                linear(stem + ".mla.output_gate", component, layer, h, n * value)
        elif kind in ("CSA", "HCA"):
            from .sparse_attention import sparse_spec
            _, a = sparse_spec(model, layer)
            d, rq, groups, rank = a["head_dim"], a["q_lora_rank"], a["o_groups"], a["o_lora_rank"]
            prefix = stem + "." + kind.lower()
            linear(prefix + ".q_down", kind, layer, h, rq, tp_shard=False)
            norm(prefix + ".q_norm", kind, layer, rq)
            linear(prefix + ".q_up", kind, layer, rq, n * d)
            linear(prefix + ".kv", kind, layer, h, d, tp_shard=False)
            norm(prefix + ".kv_norm", kind, layer, d)
            add(prefix + ".output_down.weight", kind, layer, (groups, rank, n * d // groups),
                formula="grouped low-rank output: groups * rank * (heads*head_dim/groups)")
            linear(prefix + ".output_up", kind, layer, groups * rank, h)
            add(prefix + ".attention_sink", kind, layer, (n,), role="vector")

            def compressor(name, width):
                ratio = a["compress_ratio"]
                coff = 2 if ratio == 4 else 1
                for projection in ("kv", "gate"):
                    linear(name + "." + projection, kind, layer, h, coff * width, tp_shard=False)
                add(name + ".position_bias", kind, layer, (ratio, coff * width),
                    role="vector", tp_shard=False, formula="learned compression-position gate bias")
                norm(name + ".norm", kind, layer, width)

            compressor(prefix + ".compressor", d)
            if kind == "CSA":
                ih, index_d = a["index_num_heads"], a["index_head_dim"]
                linear(prefix + ".indexer.q_up", kind, layer, rq, ih * index_d)
                linear(prefix + ".indexer.head_weights", kind, layer, h, ih)
                compressor(prefix + ".indexer.compressor", index_d)
        elif kind == "KDA":
            a = arch.get("kda") or {}
            nh = a.get("num_heads", n)
            dk, dv = a.get("head_k_dim", head), a.get("head_v_dim", head)
            kernel = a.get("conv_kernel", 4)
            for projection, width in (("q", nh * dk), ("k", nh * dk), ("v", nh * dv)):
                linear(stem + f".kda.{projection}", component, layer, h, width)
                add(stem + f".kda.{projection}_conv.weight", component, layer,
                    (width, kernel), role="convolution", formula="channels * depthwise_conv_kernel")
            # The gate layout is explicit: low-rank per-key decay, scalar beta
            # per head, and per-value output gate. The user can set gate rank.
            gate_rank = a.get("gate_low_rank_dim", dk)
            linear(stem + ".kda.decay_down", component, layer, h, gate_rank, tp_shard=False)
            linear(stem + ".kda.decay_up", component, layer, gate_rank, nh * dk)
            linear(stem + ".kda.beta", component, layer, h, nh)
            if a.get("use_full_rank_gate", False):
                linear(stem + ".kda.output_gate", component, layer, h, nh * dv)
            else:
                linear(stem + ".kda.output_gate_down", component, layer, h, gate_rank, tp_shard=False)
                linear(stem + ".kda.output_gate_up", component, layer, gate_rank, nh * dv)
            add(stem + ".kda.decay_log", component, layer, (nh,), role="vector")
            add(stem + ".kda.decay_bias", component, layer, (nh * dk,), role="vector")
            # Shared per-head value norm. It is not multiplied by head count.
            norm(stem + ".kda.output_norm", component, layer, dv)
            linear(stem + ".kda.output", component, layer, nh * dv, h)
        elif kind in ("MHA", "GQA", "MQA"):
            kh = n if kind == "MHA" else (1 if kind == "MQA" else model.get("kv_heads", n))
            linear(stem + ".attention.q", component, layer, h, n * head)
            for projection in ("k", "v"):
                linear(stem + f".attention.{projection}", component, layer, h, kh * head,
                       tp_max_shards=kh)
            linear(stem + ".attention.output", component, layer, n * head, h)
        else:
            raise ValueError(f"unsupported attention component {kind!r}")
        ffn = _kind(model, layer, "ffn_by_layer").lower()
        if ffn == "dense":
            width = arch.get("dense_intermediate_dim") or model.get("intermediate_dim") or 4 * h
            for projection in ("gate", "up"):
                linear(stem + ".dense." + projection, "Dense", layer, h, width)
            linear(stem + ".dense.down", "Dense", layer, width, h)
        elif ffn == "moe":
            a = arch.get("moe") or {}
            experts = a.get("num_experts", model.get("num_experts", 1))
            top_k = a.get("top_k", model.get("top_k", 1))
            if top_k > experts:
                raise ValueError("MoE top_k must not exceed num_experts")
            width = a.get("intermediate_dim") or model.get("intermediate_dim") or 4 * h
            shared = a.get("num_shared_experts", 0)
            shared_width = a.get("shared_intermediate_dim") or width
            latent = a.get("latent_dim") or h
            if latent != h:
                linear(stem + ".moe.latent_down", "MoE", layer, h, latent, tp_shard=False,
                       formula="hidden_dim * routed_latent_dim; replicated routed-token projection")
                linear(stem + ".moe.latent_up", "MoE", layer, latent, h, tp_shard=False)
                if a.get("latent_norm", False):
                    norm(stem + ".moe.latent_norm", "MoE", layer, latent)
            # Router is replicated: every EP source can route its token batch.
            linear(stem + ".moe.router", "MoE", layer, h, experts, tp_shard=False)
            if a.get("router_bias", False):
                add(stem + ".moe.router_correction_bias", "MoE", layer, (experts,),
                    role="vector", tp_shard=False, formula="one expert routing correction bias per expert")
            for projection in ("gate", "up", "down"):
                shape = (experts, width, latent) if projection != "down" else (experts, latent, width)
                add(stem + ".moe.routed." + projection + ".weight", "MoE", layer, shape,
                    ep_shard=True, active_fraction=top_k / experts,
                    formula="num_experts * routed_latent_dim * expert_intermediate_dim")
                if arch.get("linear_bias", False):
                    add(stem + ".moe.routed." + projection + ".bias", "MoE", layer,
                        (experts, latent if projection == "down" else width), role="vector",
                        ep_shard=True, active_fraction=top_k / experts)
            if shared:
                shared_input = a.get("shared_input_dim") or h
                for projection in ("gate", "up", "down"):
                    shape = ((shared, shared_input, shared_width) if projection == "down" else
                             (shared, shared_width, shared_input))
                    add(stem + ".moe.shared." + projection + ".weight", "MoE", layer, shape,
                        formula="num_shared_experts * shared_input_dim * shared_intermediate_dim")
                    if arch.get("linear_bias", False):
                        add(stem + ".moe.shared." + projection + ".bias", "MoE", layer,
                            (shared, shared_input if projection == "down" else shared_width), role="vector")
        else:
            raise ValueError(f"unsupported FFN component {ffn!r}")
    if arch.get("final_norm", True):
        norm("final_norm", "PostNorm", layers - 1, h)
    if arch.get("attnres_block_size"):
        norm("attnres.output.norm", "AttnRes", layers - 1, h)
        linear("attnres.output.projection", "AttnRes", layers - 1, h, 1, tp_shard=False)
    add("output.weight", "Output", layers - 1, (vocab, h), role="output",
        alias_of="word_embeddings.weight" if arch.get("tie_embeddings", False) else None,
        formula="tied alias of word_embeddings.weight" if arch.get("tie_embeddings", False)
                else "vocab_size * hidden_dim")
    return tensors


def layer_inventory(model: dict, layer: int) -> list[ParameterTensor]:
    return [tensor for tensor in parameter_inventory(model) if tensor.layer == layer]


def inventory_summary(model: dict) -> dict:
    tensors = parameter_inventory(model)
    components: dict[str, dict] = {}
    for tensor in tensors:
        entry = components.setdefault(tensor.component, {"params": 0.0, "active_params": 0.0,
            "tensor_count": 0, "aliases": [], "layers": []})
        entry["params"] += tensor.accounted_elements
        entry["active_params"] += tensor.active_elements
        entry["tensor_count"] += not bool(tensor.alias_of)
        if tensor.alias_of:
            entry["aliases"].append({"name": tensor.name, "alias_of": tensor.alias_of})
        if tensor.layer not in entry["layers"]:
            entry["layers"].append(tensor.layer)
    return {"mode": "shape_derived" if _arch(model) else "parameter_budget",
            "total_params": sum(t.accounted_elements for t in tensors),
            "active_params": sum(t.active_elements for t in tensors),
            "expert_params": sum(t.accounted_elements for t in tensors if t.ep_shard),
            "components": components,
            "tensors": [t.to_dict() for t in tensors]}


def optimizer_states(tensor: ParameterTensor, optimizer: str = "adam", master: bool = True) -> list[str]:
    """State names per owned parameter tensor; Muon uses Adam for fallback groups.

    Batched expert matrices use independent per-expert matrix updates. The
    tensor shapes alone do not predict Muon's Newton-Schulz runtime/workspace.
    """
    if optimizer not in ("adam", "muon"):
        raise ValueError("optimizer must be 'adam' or 'muon'")
    if tensor.alias_of:
        return []
    if optimizer == "muon" and tensor.role == "unknown":
        raise ValueError("Muon parameter grouping requires model.architecture")
    states = ["master_weight"] if master else []
    if optimizer == "muon" and tensor.role == "matrix":
        return states + ["muon_momentum"]
    return states + ["optimizer_momentum", "optimizer_variance"]


def optimizer_state_inventory(tensor: ParameterTensor, optimizer: str = "adam", *,
                              tp: int = 1, ep: int = 1, fsdp: int = 1,
                              master_dtype: str = "fp32", state_dtype: str = "fp32") -> list[dict]:
    """Named optimizer storage descriptors; precision is independent of weights."""
    from .core import DTYPE_BYTES
    result = []
    for state in optimizer_states(tensor, optimizer):
        dtype = master_dtype if state == "master_weight" else state_dtype
        elements = tensor.local_elements(tp, ep, fsdp)
        result.append({"name": tensor.name + "." + state, "component": tensor.component,
                       "layer": tensor.layer, "state": state, "shape": tensor.shape,
                       "logical_elements": tensor.accounted_elements, "elements": elements,
                       "dtype": dtype, "size_bytes": elements * DTYPE_BYTES[dtype],
                       "formula": "owned parameter elements / declared TP / EP / FSDP shards"})
    return result


def component_activations(model: dict, layer: int, tokens: float, *, tp: int = 1,
                          ep: int = 1, training: bool = True,
                          checkpointing: bool = False) -> list[dict]:
    """Declared fused-kernel working/saved tensors, in elements per rank.

    This is a layout model, not allocator-measured peak memory. It assumes
    FlashAttention (no saved quadratic attention matrix), fused RMSNorm and
    a gated FFN, and balanced routed experts. Lifetimes belong to the caller.
    ``tokens`` is the local EP source batch, so received MoE assignments have
    expected size ``tokens * top_k``; EP is not divided a second time.
    """
    if min(tp, ep) < 1 or tokens < 0:
        raise ValueError("positive parallel factors and nonnegative tokens required")
    h, arch = model["hidden_dim"], _arch(model)
    rows: list[dict] = []
    def add(component, name, elements, formula):
        if elements:
            rows.append({"component": component, "name": name, "elements": elements,
                         "layer": layer, "formula": formula,
                         "lifetime": "saved_for_backward" if training else "operator_working_set"})
    if not arch:
        add("Unassigned", "activation_budget", tokens * (2 * h + model.get("intermediate_dim", 4 * h) / tp),
            "legacy coarse activation estimate; no component attribution")
        return rows
    if training and checkpointing:
        add("PreNorm", "checkpoint_input", tokens * h, "tokens * hidden_dim; retained layer input only")
        return rows
    add("PreNorm", "input", tokens * h, "tokens * hidden_dim")
    add("PreNorm", "inv_rms", tokens, "one saved inverse RMS per token")
    add("PostNorm", "input", tokens * h, "tokens * hidden_dim")
    add("PostNorm", "inv_rms", tokens, "one saved inverse RMS per token")
    kind = _kind(model, layer, "attention_by_layer").upper()
    n, head = model.get("num_heads", 32), model.get("head_dim", 128)
    if kind == "MLA":
        a = arch.get("mla") or {}
        rq, rkv = a.get("q_lora_rank", 0), a.get("kv_lora_rank", model.get("kv_latent_dim", 512))
        nope, rope = a.get("qk_nope_head_dim", head), a.get("qk_rope_head_dim", model.get("kv_rope_dim", 64))
        dv = a.get("v_head_dim", head)
        add(kind, "q_latent", tokens * rq, "tokens * q_lora_rank; replicated")
        add(kind, "compressed_kv", tokens * (rkv + rope), "tokens * (kv_lora_rank + rope_dim); replicated")
        add(kind, "qkv_and_output", tokens * n * (2 * (nope + rope) + 2 * dv) / tp,
            "tokens * heads * (Q_width + K_width + V_width + output_width) / TP")
        add(kind, "softmax_lse", tokens * n / tp, "FlashAttention logsumexp per query head")
        if a.get("use_output_gate", False):
            add(kind, "output_gate", tokens * n * dv / tp, "tokens * heads * v_head_dim / TP")
    elif kind in ("CSA", "HCA"):
        from .sparse_attention import sparse_spec
        _, a = sparse_spec(model, layer)
        d, r = a["head_dim"], a["compress_ratio"]
        coff = 2 if r == 4 else 1
        add(kind, "q_latent", tokens * a["q_lora_rank"], "replicated tokens * q_lora_rank")
        add(kind, "latent_kv", tokens * d, "shared K/V rows; replicated across TP")
        add(kind, "queries_and_output", 2 * tokens * n * d / tp, "tokens * heads * head_dim * 2 / TP")
        add(kind, "output_latent", tokens * a["o_groups"] * a["o_lora_rank"] / tp,
            "tokens * output_groups * output_rank / TP")
        add(kind, "softmax_lse", tokens * n / tp, "one saved attention LSE per query head")
        # Compressor gate/value projections must be retained for its backward,
        # independently of final compressed pools owned by attention_state.
        widths = d + (a["index_head_dim"] if kind == "CSA" else 0)
        add(kind, "compression_values_and_gates", 2 * tokens * coff * widths,
            "two FP32 compressor projections per input; main plus optional index compressor")
        if tokens:
            rows[-1]["dtype"] = "fp32"
        if kind == "CSA":
            add(kind, "index_queries", tokens * a["index_num_heads"] * a["index_head_dim"] / tp,
                "tokens * index_heads * index_head_dim / TP")
            add(kind, "index_head_weights", tokens * a["index_num_heads"] / tp,
                "tokens * index_heads / TP")
    elif kind == "KDA":
        a = arch.get("kda") or {}
        nh = a.get("num_heads", n)
        dk, dv = a.get("head_k_dim", head), a.get("head_v_dim", head)
        add(kind, "qkv_gates_output", tokens * nh * (3 * dk + 3 * dv + 1) / tp,
            "tokens * heads * (Q + K + V + decay + output_gate + output + beta) / TP")
        add(kind, "decay_latent", tokens * a.get("gate_low_rank_dim", dk), "tokens * gate_low_rank_dim")
        if not a.get("use_full_rank_gate", False):
            add(kind, "output_gate_latent", tokens * a.get("gate_low_rank_dim", dk), "tokens * gate_low_rank_dim")
    else:
        kh = n if kind == "MHA" else (1 if kind == "MQA" else model.get("kv_heads", n))
        add(kind, "qkv_output", tokens * head * (2 * n / tp + 2 * max(kh / tp, 1)),
            "Q/output TP heads plus K/V sharded-or-replicated heads")
        add(kind, "softmax_lse", tokens * n / tp, "FlashAttention logsumexp per query head")
    if _kind(model, layer, "ffn_by_layer").lower() == "moe":
        a = arch.get("moe") or {}
        top = a.get("top_k", model.get("top_k", 1))
        width = a.get("intermediate_dim") or model.get("intermediate_dim") or 4 * h
        shared = a.get("num_shared_experts", 0)
        shared_width = a.get("shared_intermediate_dim") or width
        if a.get("latent_dim") and a["latent_dim"] != h:
            add("MoE", "routed_latent", tokens * a["latent_dim"], "tokens * routed_latent_dim")
        add("MoE", "routed_gate_up", 2 * tokens * top * width / tp,
            "2 * local_source_tokens * top_k * intermediate_dim / TP; balanced EP")
        add("MoE", "routing_weights", tokens * top, "tokens * top_k")
        add("MoE", "shared_gate_up", 2 * tokens * shared * shared_width / tp,
            "2 * tokens * shared_experts * shared_intermediate_dim / TP")
    else:
        width = arch.get("dense_intermediate_dim") or model.get("intermediate_dim") or 4 * h
        add("Dense", "gate_up", 2 * tokens * width / tp, "2 * tokens * intermediate_dim / TP")
    if layer == 0:
        add("WordEmbedding", "embedding_output", tokens * h, "tokens * hidden_dim")
    if layer == model["layers"] - 1:
        add("Output", "logits", tokens * arch["vocab_size"] / tp,
            "tokens * vocab_size / TP; unfused logits materialization assumption")
    if arch.get("attnres_block_size") and tokens:
        scratch = attnres_state(model, layer, tokens)["workspace_elements"]
        add("AttnRes", "residual_attention_workspace", scratch,
            "two materialized FP32 residual blocks plus score/probability vectors; unfused reference estimate")
        rows[-1]["dtype"] = "fp32"
        rows[-1]["lifetime"] = "operator_working_set"
    return rows


def attnres_state(model: dict, layer: int, tokens: float) -> dict:
    """Residual block tensors shared across layers; caller owns their lifetime.

    Reference K3 appends the block input at zero-based layer multiples of the
    block size. These buffers must travel with hidden states across PP stages.
    The FP32 scratch estimate describes the unfused reference operation; it is
    explicitly separate from retained activation-precision block tensors.
    """
    block = _arch(model).get("attnres_block_size")
    if not block:
        return {"component": "AttnRes", "blocks": 0, "stored_elements": 0,
                "new_elements": 0, "workspace_elements": 0}
    if block < 1 or tokens < 0 or layer < 0 or layer >= model["layers"]:
        raise ValueError("valid block size, tokens and layer required")
    count = layer // block + 1
    hidden = model["hidden_dim"]
    return {"component": "AttnRes", "blocks": count,
            "stored_elements": tokens * hidden * count,
            "new_elements": tokens * hidden if layer % block == 0 else 0,
            "workspace_elements": 2 * tokens * hidden * (count + 1) + 3 * tokens * (count + 1),
            "formula": "tokens * hidden_dim * (floor(layer/block_size)+1); shared retained blocks",
            "workspace_dtype": "fp32"}


def attention_state(model: dict, layer: int, batch: int, sequence_length: int,
                    *, tp: int = 1) -> dict:
    """Logical cache/state elements per rank, excluding dtype/paging metadata.

    KDA recurrent state is sequence-length independent. Convolution state
    allocation defaults to kernel samples (conservative runtime layout); the
    mathematical minimum is kernel-1 and may be explicitly configured after
    measuring the runtime. MLA latent cache is replicated across TP.
    """
    if min(batch, sequence_length) < 0 or tp < 1:
        raise ValueError("nonnegative batch/sequence length and positive TP required")
    arch = _arch(model)
    kind = _kind(model, layer, "attention_by_layer").upper()
    if kind in ("CSA", "HCA"):
        from .sparse_attention import sparse_attention_state
        return sparse_attention_state(model, layer, [sequence_length] * batch, tp=tp)
    if kind == "KDA":
        a = arch.get("kda") or {}
        nh = a.get("num_heads", model.get("num_heads", 32))
        dk = a.get("head_k_dim", model.get("head_dim", 128))
        dv = a.get("head_v_dim", model.get("head_dim", 128))
        kernel = a.get("conv_kernel", 4)
        history = a.get("conv_state_tokens")
        history = kernel if history is None else history
        if history < kernel - 1:
            raise ValueError("KDA conv_state_tokens must retain at least conv_kernel-1 samples")
        recurrent = batch * nh * dk * dv / tp
        convolution = batch * nh * (2 * dk + dv) * history / tp
        return {"component": kind, "kind": "recurrent", "elements": recurrent + convolution,
                "recurrent_elements": recurrent, "convolution_elements": convolution,
                "elements_per_token": 0, "formula": "batch * heads / TP * (dk*dv + (2*dk+dv)*conv_state_tokens)",
                "conv_state_tokens": history,
                "convolution_layout_assumption": "kernel-sized allocated state unless an explicit calibrated layout is provided",
                "recurrent_dtype": a.get("state_dtype", "fp32")}
    if kind == "MLA":
        a = arch.get("mla") or {}
        if a.get('cache_layout', 'compressed') == 'expanded':
            width = model.get('num_heads', 32) / tp * (a.get('qk_nope_head_dim', model.get('head_dim', 128)) + a.get('qk_rope_head_dim', model.get('kv_rope_dim', 64)) + a.get('v_head_dim', model.get('head_dim', 128)))
            formula = 'batch * sequence_length * heads/TP * (qk_nope_dim + qk_rope_dim + v_dim); expanded reference K/V cache'
        else:
            width = a.get("kv_lora_rank", model.get("kv_latent_dim", 512)) + a.get("qk_rope_head_dim", model.get("kv_rope_dim", 64))
            formula = "batch * sequence_length * (kv_lora_rank + qk_rope_head_dim); compressed latent replicated across TP"
    else:
        kh = model.get("num_heads", 32) if kind == "MHA" else (1 if kind == "MQA" else model.get("kv_heads", 8))
        width = 2 * max(kh / tp, 1) * model.get("head_dim", 128)
        formula = "batch * sequence_length * 2 * max(kv_heads/TP,1) * head_dim"
    return {"component": kind, "kind": "kv", "elements": batch * sequence_length * width,
            "elements_per_token": width, "formula": formula}


def attention_training_state(model: dict, layer: int, batch: int,
                             sequence_length: int, *, tp: int = 1) -> dict:
    """Optional chunk-boundary KDA state saved for backward, not inference KV.

    Assumes each chunk-start recurrent matrix is retained, using a declared
    chunk size (64 by default). A recomputing kernel can override this working
    layout; checkpointing releases these states after each forward layer.
    """
    if min(batch, sequence_length) < 0 or tp < 1:
        raise ValueError("nonnegative batch/sequence length and positive TP required")
    kind = _kind(model, layer, "attention_by_layer").upper()
    if kind in ("CSA", "HCA"):
        from .sparse_attention import sparse_spec
        _, a = sparse_spec(model, layer)
        pools = batch * (sequence_length // a["compress_ratio"])
        width = a["head_dim"] + (a["index_head_dim"] if kind == "CSA" else 0)
        return {"component": kind, "recurrent_chunk_elements": 0, "chunks": 0,
                "compressed_pool_elements": pools * width,
                "selection_index_elements": batch * sequence_length * min(a["index_topk"], sequence_length // a["compress_ratio"]) if kind == "CSA" else 0,
                "formula": "saved compressed rows and (CSA) selected row indices; projection activations accounted separately",
                "assumption": "analytical attention backward storage; top-k is nondifferentiable, no indexer auxiliary training loss"}
    if kind != "KDA":
        return {"component": _kind(model, layer, "attention_by_layer"),
                "recurrent_chunk_elements": 0, "chunks": 0,
                "formula": "not a recurrent attention layer"}
    arch = _arch(model).get("kda") or {}
    chunk_size = arch.get("chunk_size", 64)
    if chunk_size < 1:
        raise ValueError("kda.chunk_size must be positive")
    chunks = ceil(sequence_length / chunk_size)
    state = attention_state(model, layer, batch, sequence_length, tp=tp)
    return {"component": "KDA", "recurrent_chunk_elements": state["recurrent_elements"] * chunks,
            "chunks": chunks, "chunk_size": chunk_size,
            "formula": "batch * ceil(sequence_length/chunk_size) * heads * dk * dv / TP",
            "assumption": "save one FP32 recurrent matrix per chunk boundary for backward; kernel-dependent"}

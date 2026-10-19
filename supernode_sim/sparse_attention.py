"""Analytical CSA/HCA attention costs; not a complete DeepSeek V4 model.

The layout follows deepseek-ai/DeepSeek-V4-Pro/inference/model.py: a shared
latent K/V row, sliding window, learned gated compression, and (CSA only) an
independent compressed index cache. Hyper-connections and their training loss
are outside this module. FLOPs are forward work, not measured kernel timings.
"""
from __future__ import annotations

from math import ceil, log2


def sparse_spec(model: dict, layer: int) -> tuple[str, dict]:
    arch = model.get("architecture") or {}
    kind = arch["attention_by_layer"][layer].upper()
    if kind not in ("CSA", "HCA"):
        raise ValueError("sparse attention requires an explicitly selected CSA/HCA layer")
    spec = arch.get(kind.lower())
    if not spec:
        raise ValueError(f"architecture.{kind.lower()} is required")
    return kind, spec


def _lengths(lengths):
    result = list(lengths)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in result):
        raise ValueError("lengths must be nonnegative integers")
    return result


def _floor_sum(n: int, ratio: int) -> int:
    """sum(floor(t / ratio), t=1..n), without iterating long contexts."""
    q = n // ratio
    return q * (n + 1) - ratio * q * (q + 1) // 2


def _capped_floor_sum(n: int, ratio: int, cap: int) -> int:
    end = min(n, ratio * cap - 1)
    return _floor_sum(end, ratio) + max(0, n - end) * cap


def _window_sum(n: int, window: int) -> int:
    end = min(n, window)
    return end * (end + 1) // 2 + (n - end) * window


def position_bias_rows(lengths, queries, ratio: int) -> int:
    """Distinct positional-table rows touched by a batch, without expanding tokens."""
    intervals = []
    for length, query in zip(lengths, queries):
        if query >= ratio:
            return ratio
        if query <= 0:
            continue
        start = (length - query) % ratio
        end = start + query
        intervals.append((start, min(end, ratio)))
        if end > ratio:
            intervals.append((0, end - ratio))
    total, previous = 0, 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous))
        previous = max(previous, end)
    return total


def sparse_attention_state(model: dict, layer: int, lengths, *, tp: int = 1) -> dict:
    """Per-rank storage with replicated latent pools and FP32 compressor state.

    Window and compressor buffers reserve their full per-request capacity.
    Compressed pools grow with committed rows; this is demand allocation, not
    the reference implementation's max_batch_size/max_seq_len preallocation.
    Precision/paging metadata are caller-owned; elements_per_token=0 explicitly
    rejects treating this nonlinear state as an ordinary dense KV cache.
    """
    if tp < 1:
        raise ValueError("tp must be positive")
    lengths = _lengths(lengths)
    kind, a = sparse_spec(model, layer)
    ratio, d, win = a["compress_ratio"], a["head_dim"], a["window_size"]
    coff = 2 if ratio == 4 else 1
    rows = sum(length // ratio for length in lengths)
    batch = len(lengths)
    window = batch * win * d
    pool = rows * d
    index_pool = rows * a["index_head_dim"] if kind == "CSA" else 0
    compression = batch * 2 * coff * coff * ratio * d
    index_state = batch * 2 * coff * coff * ratio * a["index_head_dim"] if kind == "CSA" else 0
    return {
        "component": kind, "kind": "compressed_sparse", "elements_per_token": 0,
        "window_elements": window, "window_live_elements": sum(min(x, win) for x in lengths) * d,
        "compressed_pool_elements": pool, "indexer_pool_elements": index_pool,
        "cache_elements": window + pool + index_pool,
        "compression_state_elements": compression, "indexer_compression_state_elements": index_state,
        "recurrent_elements": compression + index_state, "recurrent_dtype": "fp32",
        "elements": window + pool + index_pool + compression + index_state,
        "compressed_rows": rows, "window_capacity_tokens": batch * win,
        "storage_categories": {"kv_dtype": ["window_elements", "compressed_pool_elements", "indexer_pool_elements"],
                               "fp32": ["compression_state_elements", "indexer_compression_state_elements"]},
        "formula": "replicated latent window + floor(length/ratio) pools + 2*coff^2*ratio*width FP32 state per compressor",
        "allocation_assumption": "fixed window/state per request; demand-grown compressed pools, not max-length reservation",
        "scope": "analytical CSA/HCA attention only; excludes mHC",
    }


def sparse_attention_work(model: dict, layer: int, lengths, *, phase: str = "prefill",
                          tp: int = 1, query_lengths=None) -> dict:
    """Forward non-projection work and logical cache accesses for current queries.

    lengths include this round's queries. Default queries are L for prefill,
    one for decode; verify requires explicit query_lengths. Matrix projection
    FLOPs belong to parameter_inventory and MUST NOT be added again here.
    The reference prefill indexer scores all L//ratio rows before masking;
    multi-token verify is an analytical extension with causal visible rows.
    Attention reads assume ideal reuse across local query heads, not L2 hits.
    """
    if phase not in ("prefill", "decode", "verify") or tp < 1:
        raise ValueError("phase must be prefill/decode/verify and tp positive")
    lengths = _lengths(lengths)
    if query_lengths is None:
        if phase == "verify":
            raise ValueError("verify requires explicit query_lengths")
        queries = lengths if phase == "prefill" else [min(1, x) for x in lengths]
    else:
        queries = _lengths(query_lengths)
    if len(queries) != len(lengths) or any(q > length for q, length in zip(queries, lengths)):
        raise ValueError("query_lengths must match lengths and lie in [0, length]")
    kind, a = sparse_spec(model, layer)
    r, d, win = a["compress_ratio"], a["head_dim"], a["window_size"]
    coff = 2 if r == 4 else 1
    heads = model["num_heads"] / tp
    tokens = sum(queries)
    window_pairs = compressed_pairs = indexer_pairs = emitted = 0
    for length, query in zip(lengths, queries):
        before = length - query
        window_pairs += _window_sum(length, win) - _window_sum(before, win)
        emitted += length // r - before // r
        if kind == "CSA":
            topk = a["index_topk"]
            compressed_pairs += _capped_floor_sum(length, r, topk) - _capped_floor_sum(before, r, topk)
            # Full prefill einsum in the reference implementation, then mask.
            indexer_pairs += query * (length // r) if phase == "prefill" else _floor_sum(length, r) - _floor_sum(before, r)
        else:
            compressed_pairs += _floor_sum(length, r) - _floor_sum(before, r)
    pairs = window_pairs + compressed_pairs
    # QK and weighted V: the compressed latent row is shared as K and V.
    attention_flops = heads * pairs * (4 * d + 5) + tokens * heads * (5 * d + 6 * a["rope_head_dim"])
    indexer_flops = indexer_reads = selection_ops = score_workspace = 0
    index_d = a.get("index_head_dim", 0) if kind == "CSA" else 0
    if kind == "CSA":
        ih = a["index_num_heads"] / tp
        indexer_flops = indexer_pairs * ih * (2 * index_d + 3)
        indexer_flops += tokens * ih * index_d * ceil(log2(max(index_d, 2)))
        indexer_reads = indexer_pairs * index_d
        selection_ops = indexer_pairs * ceil(log2(max(2, a["index_topk"])))
        score_workspace = indexer_pairs * (ih + 1)
    widths = [d] + ([index_d] if kind == "CSA" else [])
    # Gate position addition for each input; softmax, weighting, reduction,
    # RMSNorm/RoPE at completed compression boundaries. Projection GEMMs omitted.
    compression_flops = sum(tokens * coff * width + emitted * (6 * coff * r * width + 5 * width + 3 * a["rope_head_dim"])
                            for width in widths)
    if kind == "CSA":
        compression_flops += emitted * index_d * ceil(log2(max(index_d, 2)))
    state_read = state_write = compression_read = window_writes = 0
    for length, query in zip(lengths, queries):
        rows = length // r - (length - query) // r
        if phase == "prefill" and query == length:
            # Bulk prefill pools projected activation rows, retaining only the
            # previous block (overlap) and incomplete tail for future decode.
            retained = length % r + (r if coff == 2 and length >= r else 0)
            state_write += retained * sum(2 * coff * width for width in widths)
            compression_read += rows * sum(2 * coff * r * width for width in widths)
            window_writes += min(length, win)
        else:
            state_read += rows * sum(2 * coff * r * width for width in widths)
            state_write += query * sum(2 * coff * width for width in widths)
            if coff == 2:
                copy_elements = rows * sum(2 * r * coff * width for width in widths)
                state_read += copy_elements
                state_write += copy_elements
            window_writes += query
    return {
        "component": kind, "phase": phase, "query_tokens": tokens,
        "window_pairs": window_pairs, "compressed_pairs": compressed_pairs,
        "indexer_pairs": indexer_pairs, "compressed_rows_emitted": emitted,
        "attention_flops": attention_flops, "indexer_flops": indexer_flops,
        "compression_flops": compression_flops,
        "extra_flops": attention_flops + indexer_flops + compression_flops,
        "selection_operations": selection_ops,
        "indexer_score_workspace_elements": score_workspace,
        "indexer_reduce_elements": indexer_pairs if kind == "CSA" and tp > 1 else 0,
        "attention_read_elements": pairs * d, "indexer_read_elements": indexer_reads,
        "state_read_elements": state_read, "state_write_elements": state_write,
        "compression_read_elements": compression_read,
        "cache_write_elements": window_writes * d + emitted * sum(widths),
        "scope": "analytical forward non-projection work; no whole-layer compression speedup; top-k comparisons separate from FLOPs",
        "access_assumption": "logical row reads with ideal sharing across local heads; no assumed hardware cache hit rate",
    }

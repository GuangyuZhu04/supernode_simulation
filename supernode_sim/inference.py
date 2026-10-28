"""Compile ragged, fixed-batch prefill/decode into the shared execution ledger.

The model is deliberately structural: it assumes uniform expert routing and
per-layer parameter distribution, and never infers undocumented architecture
from a model name.  Precision, paging and residency are separate concerns.
"""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from .core import Allocation, DTYPE_BYTES, Event, Graph, Resource
from .moe_wire import moe_payload_matrix, moe_wire_metadata, wire_payload_parts

if TYPE_CHECKING:
    from .topology import Topology


def kv_layout_bytes(
    model: dict[str, Any], precision: str, lengths: list[int], *,
    layers: int = 1, tp: int = 1, page_tokens: int = 16,
    scale_block: int = 128, scale_bytes: int = 4, page_entry_bytes: int = 8,
) -> dict[str, float]:
    """Stored KV footprint, including page padding and quantization metadata.

    Standard KV shards heads over TP; small KV head counts are replicated.
    Latent attention stores one shared compressed latent plus RoPE vector per
    token on each TP rank, not expanded per-head K/V and not a second ratio.
    """
    if model["attention_kind"] == "latent":
        width = model["kv_latent_dim"] + model["kv_rope_dim"]
    else:
        width = 2 * math.ceil(model["kv_heads"] / tp) * model["head_dim"]
    pages = sum(math.ceil(length / page_tokens) for length in lengths)
    elements = pages * page_tokens * width * layers
    data = elements * DTYPE_BYTES[precision]
    scales = 0
    if precision in ("fp8", "int8"):
        # Every layer/sequence page owns its own scale blocks.
        scales = pages * layers * math.ceil(page_tokens * width / scale_block) * scale_bytes
    table = pages * layers * page_entry_bytes
    return {"data_bytes": data, "scale_bytes": scales,
            "page_table_bytes": table, "total_bytes": data + scales + table,
            "padded_tokens": pages * page_tokens, "elements_per_token": width}


def build_inference(config: dict[str, Any], topology: "Topology") -> Graph:
    """Build a deterministic graph for every DP replica of a fixed batch."""
    m, h, w, x = (config[k] for k in ("model", "hardware", "workload", "execution"))
    p, f = x["precision"], x["features"]
    tp, pp, ep, dp = (x[k] for k in ("tp", "pp", "ep", "dp"))
    order = x.get("rank_order") or list(range(h["npu_count"]))
    graph = Graph()
    topology.populate(graph)
    metadata = graph.metadata
    metadata.update({
        "scenario": "inference", "input_tokens_per_replica": sum(w["input_lengths"]),
        "output_tokens_per_replica": sum(w["output_lengths"]),
        "input_tokens": sum(w["input_lengths"]) * dp,
        "output_tokens": sum(w["output_lengths"]) * dp,
        "replicas": dp, "batch_size": w["batch_size"],
        "finished_sequence_count": w["batch_size"] * dp,
        "token_completion_events": {}, "sequence_token_completion_events": {},
        "assumptions": [
            "Fixed ragged batch; finished sequences leave decode but no new requests arrive.",
            "EP lanes own round-robin sequences; uniform expert routing and balanced expert compute.",
            "Dense parameters are replicated over EP; expert parameters are sharded over EP.",
            "Expert HBM reads use expected distinct experts touched by uniform top-k routing; active_params controls FLOPs.",
            "Per-layer parameters are uniform within dense/expert layer categories.",
            "Standard KV shards heads (ceil); latent KV is replicated across TP ranks.",
            "TP boundary hidden states are replicated; PP and MoE messages carry full hidden vectors per TP lane.",
            "Weights are preloaded; cold-start weight loading and request queueing are excluded.",
            "Attention is fused/streaming: quadratic score matrices are not resident.",
            "KV read precision conversion is fused into attention; new KV storage conversion has an explicit kernel and buffer.",
        ],
    })

    def rank(d: int, stage: int, e: int, t: int) -> str:
        idx = (((d * pp + stage) * ep + e) * tp + t)
        physical = order[idx]
        return physical if isinstance(physical, str) else f"npu{physical}"

    def host(device: str) -> str:
        return topology.host_for(int(device.removeprefix("npu")))

    def pool(node: str) -> str:
        return node + (".hbm" if node.startswith("npu") else ".dram" if node.startswith("cpu") else ".memory")

    def add(event: Event) -> str:
        graph.events.append(event)
        return event.id

    def alloc(aid: str, node: str, size: float, category: str, dtype: str | None = None,
              **extra: Any) -> Allocation:
        return Allocation(aid, pool(node), max(0.0, size), category, dtype,
                          metadata=extra)

    def layout(lengths: list[int], tensor_parallel: int = tp, count: int = 1) -> dict[str, float]:
        return kv_layout_bytes(m, p["kv"], lengths, layers=count, tp=tensor_parallel,
                               page_tokens=f["kv_page_tokens"], scale_block=f["kv_scale_block"],
                               scale_bytes=f["scale_bytes"], page_entry_bytes=f.get("kv_page_entry_bytes", 8))

    def collective(prefix: str, kind: str, ranks: list[str], size: float,
                   deps: tuple[str, ...], payload_matrix=None, metadata=None) -> tuple[str, ...]:
        """Pack/unpack explicitly when wire and activation precision differ."""
        metadata = dict(metadata or {})
        wire = metadata.get('wire_dtype', p['communication'])
        moe = metadata.get('operation') in ('dispatch', 'combine')
        algorithm = x.get('all_to_all_algorithm', 'pairwise')
        outer = moe
        if moe:
            selection = topology.select_collective_timing_scope(kind, ranks, size, payload_matrix, metadata, algorithm)
            metadata['collective_scope_selection'] = selection
            outer = selection['selected_scope'] == 'activation_collective'

        def finish(terminals):
            if not outer:
                return tuple(terminals)
            return topology.complete_collective_profile(graph, prefix, kind, ranks, size,
                deps=tuple(deps), terminals=tuple(terminals), payload_matrix=payload_matrix,
                metadata=metadata, algorithm=algorithm, timing_scope='activation_collective')

        if p["activation"] == wire or len(ranks) == 1:
            return finish(topology.collective(graph, prefix, kind, ranks, size, deps,
                payload_matrix=payload_matrix, metadata=metadata, algorithm=algorithm, apply_profiles=not outer))
        packed, buffers = [], []
        for index, device in enumerate(ranks):
            send_bytes = sum(payload_matrix[index]) if payload_matrix is not None else size
            elements, extra = wire_payload_parts(send_bytes, wire, metadata)
            accesses = elements * DTYPE_BYTES[p['activation']] + send_bytes + extra
            aid = f"{prefix}.{device}.packed"
            buffers.append(aid)
            packed.append(add(Event(f"{prefix}.{device}.pack", deps,
                {device + ".compute": elements,
                 device + ".hbm": accesses},
                slots=(device + ".compute_stream",),
                allocs=(alloc(aid, device, send_bytes, "conversion_buffer", wire),),
                metadata={**(metadata or {}), "operation": "precision_conversion", "conversion": "communication_pack",
                          "kernel": "communication_pack", "working_set_bytes": accesses,
                          "tensor_elements": elements, "wire_metadata_bytes": extra,
                          "source_precision": p['activation'], "destination_precision": wire})))
        network = topology.collective(graph, prefix, kind, ranks, size, tuple(packed),
            payload_matrix=payload_matrix, metadata=metadata, algorithm=algorithm, apply_profiles=not outer)
        unpacked = []
        for index, device in enumerate(ranks):
            recv_bytes = sum(row[index] for row in payload_matrix) if payload_matrix is not None else size
            elements, extra = wire_payload_parts(recv_bytes, wire, metadata)
            accesses = elements * DTYPE_BYTES[p['activation']] + recv_bytes + extra
            unpacked.append(add(Event(f"{prefix}.{device}.unpack", tuple(network),
                {device + ".compute": elements,
                 device + ".hbm": accesses},
                slots=(device + ".compute_stream",), releases=(buffers[index],),
                metadata={**(metadata or {}), "operation": "precision_conversion", "conversion": "communication_unpack",
                          "kernel": "communication_unpack", "working_set_bytes": accesses,
                          "tensor_elements": elements, "wire_metadata_bytes": extra,
                          "source_precision": wire, "destination_precision": p['activation']})))
        return finish(unpacked)

    dense_params = m["total_params"] * (1 - m["expert_parameter_fraction"])
    expert_params = m["total_params"] - dense_params
    expert_active = max(0.0, m["active_params"] - dense_params)
    num_moe_layers = max(1, round(m["layers"] * m["moe_layer_fraction"])) if expert_params else 0
    moe_layers = set(range(m["layers"] - num_moe_layers, m["layers"]))
    dense_per_layer = dense_params / m["layers"]
    expert_per_layer = expert_params / num_moe_layers if num_moe_layers else 0
    active_per_moe_layer = expert_active / num_moe_layers if num_moe_layers else 0
    owned = {e: [i for i in range(w["batch_size"]) if i % ep == e] for e in range(ep)}
    engram = f["engram"]
    offload = f["kv_offload"]
    offload_fraction = f["kv_offload_fraction"] if offload != "none" else 0.0

    def storage_for(device: str) -> str:
        return host(device) if offload == "host" else "remote0"

    def query_store(device: str) -> str:
        return host(device) if engram["location"] == "host" else "remote0"

    # Resident tensors, per physical rank. Host tables are explicitly replicas;
    # the remote table is a single shared table.
    table_nodes: set[str] = set()
    resident_total = 0.0
    for d in range(dp):
        for stage in range(pp):
            layers = [l for l in range(m["layers"]) if l * pp // m["layers"] == stage]
            stage_params = dense_per_layer * len(layers) / tp
            stage_params += expert_per_layer * len(set(layers) & moe_layers) / (tp * ep)
            for e in range(ep):
                for t in range(tp):
                    device = rank(d, stage, e, t)
                    graph.initial_allocations.append(alloc(f"{device}.weights", device,
                        stage_params * DTYPE_BYTES[p["weight"]], "weight", p["weight"],
                        shard={"tp": tp, "ep": ep, "pp": pp}, replica=d))
                    resident_total += stage_params * DTYPE_BYTES[p["weight"]]
                    if p["weight"] in ("fp8", "int8"):
                        graph.initial_allocations.append(alloc(f"{device}.weight_scales", device,
                            math.ceil(stage_params / f["weight_scale_block"]) * f["scale_bytes"],
                            "quantization_metadata"))
                    if f.get("extra_weight_copy_dtype"):
                        dtype = f["extra_weight_copy_dtype"]
                        graph.initial_allocations.append(alloc(f"{device}.weight_copy", device,
                            stage_params * DTYPE_BYTES[dtype], "extra_weight_copy", dtype))
                    for category, field in (("runtime", "runtime_bytes"), ("workspace", "workspace_bytes")):
                        if h["npu"].get(field, 0):
                            graph.initial_allocations.append(alloc(f"{device}.{category}", device,
                                h["npu"][field], category))
                    if engram["enabled"] and any(l in engram["layers"] for l in layers):
                        table_nodes.add(query_store(device))
                        if engram.get("cache_bytes", 0):
                            graph.initial_allocations.append(alloc(f"{device}.engram_cache", device,
                                engram["cache_bytes"], "engram_cache"))
    if engram["enabled"]:
        for node in sorted(table_nodes):
            graph.initial_allocations.append(alloc(f"{node}.engram_table", node,
                                                   engram["table_bytes"], "engram_table"))
            rid = node + ".engram_iops"
            graph.resources[rid] = Resource(rid, engram["iops"], "queries/s", "iops")
        if engram["location"] == "host":
            metadata["assumptions"].append("Engram table is replicated on each participating host; cache_bytes is per participating NPU.")

    end_lengths = [i + o - 1 for i, o in zip(w["input_lengths"], w["output_lengths"])]
    logical_kv = layout(end_lengths, 1, m["layers"])
    metadata["logical_memory_bytes"] = {
        "weight": m["total_params"] * DTYPE_BYTES[p["weight"]],
        "kv_data": logical_kv["data_bytes"], "kv_scales": logical_kv["scale_bytes"],
        "kv_page_tables": logical_kv["page_table_bytes"],
        "engram_table": engram["table_bytes"] if engram["enabled"] else 0,
    }
    metadata["resident_weight_bytes_all_ranks"] = resident_total
    metadata["kv_layout_at_final_length"] = logical_kv

    # Every layer keeps its own growing cache. Allocations are increments, so
    # no fictitious old+new full-cache overlap occurs during page growth.
    kv_committed: dict[tuple[str, int], float] = {}
    for d in range(dp):
        token_barriers: list[str] = []
        per_sequence: dict[str, list[str]] = {str(i): [] for i in range(w["batch_size"])}
        previous_token: tuple[str, ...] = ()
        for step in range(max(w["output_lengths"])):
            phase = "prefill" if step == 0 else "decode"
            active = [i for i, out in enumerate(w["output_lengths"]) if out > step]
            local_ids = {e: [i for i in owned[e] if i in active] for e in range(ep)}
            local_tokens = {e: sum(w["input_lengths"][i] if step == 0 else 1 for i in local_ids[e]) for e in range(ep)}
            total_tokens = sum(local_tokens.values())
            touched_expert_fraction = (1 - (1 - m["top_k"] / m["num_experts"]) ** total_tokens)
            layer_deps = previous_token
            receive_buffers: dict[str, str] = {}
            for layer in range(m["layers"]):
                stage = layer * pp // m["layers"]
                prefix = f"infer.d{d}.s{step}.l{layer}"
                is_moe = layer in moe_layers
                dense_done: dict[tuple[int, int], str] = {}
                cleanup: dict[str, list[str]] = {}
                cache_writes: list[str] = []
                active_bytes: dict[str, float] = {}
                for e in range(ep):
                    current_lengths = [w["input_lengths"][i] + min(step, w["output_lengths"][i] - 1) for i in owned[e]]
                    # Decode Q/K/V for the current input is computed locally;
                    # only the already-existing prefix is fetched from memory.
                    fetched_lengths = [w["input_lengths"][i] + max(0, step - 1) for i in local_ids[e]]
                    for t in range(tp):
                        device = rank(d, stage, e, t)
                        base = f"{prefix}.{device}"
                        tokens = local_tokens[e]
                        cache = layout(current_lengths)
                        stored = cache["total_bytes"]
                        old_stored = kv_committed.get((device, layer), 0.0)
                        growth = max(0.0, stored - old_stored)
                        kv_committed[device, layer] = stored
                        temporary: list[str] = []
                        allocations: list[Allocation] = []
                        for fraction, location, suffix in ((1 - offload_fraction, device, "local"),
                                                          (offload_fraction, storage_for(device), "offloaded")):
                            if growth * fraction:
                                allocations.append(alloc(base + ".kv." + suffix, location, growth * fraction,
                                    "kv", p["kv"], layer=layer, precision=p["kv"]))
                        # Residual/output hidden states are replicated at TP
                        # boundaries; only the FFN intermediate is TP-sharded.
                        dense_intermediate = 0 if is_moe else m["intermediate_dim"] / tp
                        activation = tokens * (2 * m["hidden_dim"] + dense_intermediate) * DTYPE_BYTES[p["activation"]]
                        active_bytes[device] = activation
                        if activation:
                            aid = base + ".activation"
                            allocations.append(alloc(aid, device, activation, "activation", p["activation"]))
                            temporary.append(aid)
                        if device in receive_buffers:
                            temporary.append(receive_buffers.pop(device))
                        ready = add(Event(base + ".ready", layer_deps, allocs=tuple(allocations),
                                          metadata={"phase": phase, "layer": layer}))
                        inputs: list[str] = [ready]
                        # Fetch only caches of still-active sequences. Paged data
                        # is transferred, but the page table remains in its pool.
                        fetched = layout(fetched_lengths)
                        read_bytes = fetched["data_bytes"] + fetched["scale_bytes"]
                        kv_buffer = base + ".kv_fetch_buffer"
                        if step > 0 and offload_fraction and read_bytes:
                            size = read_bytes * offload_fraction
                            begin = add(Event(base + ".kv_fetch_alloc", (ready,),
                                allocs=(alloc(kv_buffer, device, size, "kv_fetch_buffer", p["kv"]),)))
                            event = topology.transfer(base + ".kv_read", storage_for(device), device,
                                size, (begin,), {"phase": phase, "feature": "kv_offload", "operation": "kv_read", "layer": layer})
                            inputs.append(add(event))
                            temporary.append(kv_buffer)
                        # Engram random reads and KV transfers share real DDR,
                        # H2D and remote paths and may start concurrently.
                        if engram["enabled"] and layer in engram["layers"] and tokens:
                            queries = tokens * engram["queries_per_token"] / tp
                            misses = queries * (1 - engram["hit_rate"])
                            query_bytes = misses * engram["record_bytes"]
                            if query_bytes:
                                aid = base + ".engram_result"
                                begin = add(Event(base + ".engram_alloc", (ready,),
                                    allocs=(alloc(aid, device, query_bytes, "engram_result"),)))
                                event = topology.transfer(base + ".engram_read", query_store(device), device,
                                    query_bytes, (begin,), {"phase": phase, "feature": "engram", "operation": "lookup", "layer": layer})
                                event.work[query_store(device) + ".engram_iops"] = misses
                                source_iops = query_store(device) + ".iops"
                                if source_iops in graph.resources:
                                    event.work[source_iops] = misses
                                event.startup_s += engram["latency_s"]
                                inputs.append(add(event))
                                temporary.append(aid)
                            hits = queries * engram["hit_rate"] * engram["record_bytes"]
                            if hits:
                                inputs.append(add(Event(base + ".engram_hit", (ready,),
                                    {device + ".hbm": hits}, metadata={"feature": "engram", "operation": "cache_hit"})))
                        layer_params = (dense_per_layer / tp if tokens else 0)
                        if is_moe:
                            layer_params += expert_per_layer * touched_expert_fraction / (tp * ep)
                        if p["weight"] != p["compute"] and f.get("extra_weight_copy_dtype") != p["compute"]:
                            aid = base + ".weight_conversion_buffer"
                            size = layer_params * DTYPE_BYTES[p["compute"]]
                            inputs.append(add(Event(base + ".convert_weight", (ready,),
                                {device + ".hbm": layer_params * DTYPE_BYTES[p["weight"]] + size,
                                 device + ".compute": layer_params},
                                slots=(device + ".compute_stream",),
                                allocs=(alloc(aid, device, size, "conversion_buffer", p["compute"]),),
                                metadata={"phase": phase, "operation": "precision_conversion", "kernel": "weight_conversion",
                                          "working_set_bytes": layer_params * DTYPE_BYTES[p["weight"]] + size})))
                            temporary.append(aid)
                        if step == 0:
                            pairs = sum(w["input_lengths"][i] * (w["input_lengths"][i] + 1) / 2 for i in local_ids[e])
                        else:
                            pairs = sum(w["input_lengths"][i] + step for i in local_ids[e])
                        attention_flops = 4 * m["num_heads"] * m["head_dim"] * pairs / tp
                        flops = 2 * dense_per_layer * tokens / tp + attention_flops
                        weight_reads = dense_per_layer * DTYPE_BYTES[p["compute"]] / tp if tokens else 0
                        hbm_reads = weight_reads + 2 * activation + (read_bytes if step else 0)
                        width = fetched["elements_per_token"]
                        new_kv = tokens * width * DTYPE_BYTES[p["kv"]]
                        if p["kv"] in ("fp8", "int8"):
                            new_kv += math.ceil(tokens * width / f["kv_scale_block"]) * f["scale_bytes"]
                        write_buffer = base + ".kv_write_buffer"
                        write_allocs = ()
                        convert_kv = p["kv"] != p["compute"] and new_kv > 0
                        raw_kv = tokens * width * DTYPE_BYTES[p["compute"]]
                        raw_buffer = base + ".kv_raw_buffer"
                        if convert_kv:
                            write_allocs = (alloc(raw_buffer, device, raw_kv, "kv_conversion_buffer", p["compute"]),)
                            if step:
                                flops += read_bytes / DTYPE_BYTES[p["kv"]]
                        elif offload_fraction and new_kv:
                            write_allocs = (alloc(write_buffer, device, new_kv * offload_fraction,
                                                   "kv_write_buffer", p["kv"]),)
                        done = add(Event(base + ".dense", tuple(inputs),
                            {device + ".compute": flops, device + ".hbm": hbm_reads + (raw_kv if convert_kv else new_kv)},
                            slots=(device + ".compute_stream",), allocs=write_allocs,
                            metadata={"phase": phase, "layer": layer, "operation": "attention_dense", "flops": flops,
                                      "kernel": "attention_dense", "working_set_bytes": weight_reads + activation + read_bytes + new_kv}))
                        if convert_kv:
                            final_write_allocs = ((alloc(write_buffer, device, new_kv * offload_fraction,
                                "kv_write_buffer", p["kv"]),) if offload_fraction else ())
                            done = add(Event(base + ".kv_convert", (done,),
                                {device + ".compute": tokens * width, device + ".hbm": raw_kv + new_kv},
                                slots=(device + ".compute_stream",), allocs=final_write_allocs,
                                releases=(raw_buffer,),
                                metadata={"phase": phase, "layer": layer, "operation": "precision_conversion",
                                          "kernel": "kv_conversion", "working_set_bytes": raw_kv + new_kv}))
                        dense_done[e, t] = done
                        cleanup[device] = temporary
                        if offload_fraction and new_kv:
                            event = topology.transfer(base + ".kv_write", device, storage_for(device),
                                new_kv * offload_fraction, (done,),
                                {"phase": phase, "feature": "kv_offload", "operation": "kv_write", "layer": layer})
                            event.releases = (write_buffer,)
                            cache_writes.append(add(event))

                layer_outputs: list[str] = []
                if is_moe:
                    for t in range(tp):
                        ranks = [rank(d, stage, e, t) for e in range(ep)]
                        source_tokens = [local_tokens[e] for e in range(ep)]
                        matrices = moe_payload_matrix(p, 'dispatch', source_tokens, m['hidden_dim'], m['top_k'])
                        dispatch_wire = moe_wire_metadata(p, 'dispatch', m['hidden_dim'], top_k=m['top_k'], num_experts=m['num_experts'])
                        combine_wire = moe_wire_metadata(p, 'combine', m['hidden_dim'], top_k=m['top_k'], num_experts=m['num_experts'])
                        expert_allocs = []
                        for device in ranks:
                            aid = f"{prefix}.{device}.expert_buffer"
                            size = total_tokens * m["hidden_dim"] * m["top_k"] * max(
                                DTYPE_BYTES[dispatch_wire['wire_dtype']], DTYPE_BYTES[combine_wire['wire_dtype']],
                                DTYPE_BYTES[p["activation"]]) / ep
                            expert_allocs.append(alloc(aid, device, size, "expert_buffer", p["activation"]))
                            cleanup[device].append(aid)
                            intermediate_id = f"{prefix}.{device}.expert_intermediate"
                            intermediate_size = total_tokens * m["top_k"] * m["intermediate_dim"] * DTYPE_BYTES[p["activation"]] / (tp * ep)
                            expert_allocs.append(alloc(intermediate_id, device, intermediate_size, "activation", p["activation"]))
                            cleanup[device].append(intermediate_id)
                        prepared = add(Event(f"{prefix}.t{t}.dispatch_alloc",
                            tuple(dense_done[e, t] for e in range(ep)), allocs=tuple(expert_allocs)))
                        dispatch = collective(f"{prefix}.t{t}.dispatch", "all_to_all_v",
                            ranks, 0, (prepared,), payload_matrix=matrices,
                            metadata={"phase": phase, "operation": "dispatch", "layer": layer, **dispatch_wire})
                        experts = []
                        for device in ranks:
                            flops = 2 * active_per_moe_layer * total_tokens / (tp * ep)
                            bytes_ = expert_per_layer * touched_expert_fraction * DTYPE_BYTES[p["compute"]] / (tp * ep)
                            experts.append(add(Event(f"{prefix}.{device}.experts", tuple(dispatch),
                                {device + ".compute": flops, device + ".hbm": bytes_},
                                slots=(device + ".compute_stream",),
                                metadata={"phase": phase, "layer": layer, "operation": "experts", "flops": flops,
                                          "kernel": "experts", "working_set_bytes": bytes_ + total_tokens * m["top_k"] * (m["hidden_dim"] + m["intermediate_dim"] / tp) * DTYPE_BYTES[p["activation"]] / ep})))
                        combined = collective(f"{prefix}.t{t}.combine", "all_to_all_v",
                            ranks, 0, tuple(experts), payload_matrix=moe_payload_matrix(
                                p, 'combine', source_tokens, m['hidden_dim'], m['top_k']),
                            metadata={"phase": phase, "operation": "combine", "layer": layer, **combine_wire})
                        layer_outputs.extend(combined)
                else:
                    layer_outputs.extend(dense_done.values())
                if tp > 1:
                    reductions = []
                    for e in range(ep):
                        ranks = [rank(d, stage, e, t) for t in range(tp)]
                        size = local_tokens[e] * m["hidden_dim"] * DTYPE_BYTES[p["communication"]]
                        first = collective(f"{prefix}.e{e}.tp_attention", "all_reduce",
                            ranks, size, tuple(layer_outputs), metadata={"phase": phase, "operation": "tp", "layer": layer})
                        reductions.extend(collective(f"{prefix}.e{e}.tp_ffn", "all_reduce",
                            ranks, size, tuple(first), metadata={"phase": phase, "operation": "tp", "layer": layer}))
                    layer_outputs = reductions
                # Pipeline transfers retain the source activation until the
                # transfer completes and the destination receive through use.
                next_stage = (layer + 1) * pp // m["layers"] if layer + 1 < m["layers"] else stage
                if next_stage != stage:
                    transfers = []
                    for e in range(ep):
                        for t in range(tp):
                            src, dst = rank(d, stage, e, t), rank(d, next_stage, e, t)
                            size = local_tokens[e] * m["hidden_dim"] * DTYPE_BYTES[p["communication"]]
                            eid = f"{prefix}.{src}.pp"
                            send_deps = tuple(layer_outputs)
                            pack_buffer = None
                            elements = local_tokens[e] * m["hidden_dim"]
                            conversion_work = elements * (DTYPE_BYTES[p["activation"]] + DTYPE_BYTES[p["communication"]])
                            if p["activation"] != p["communication"]:
                                pack_buffer = eid + ".packed"
                                send_deps = (add(Event(eid + ".pack", send_deps,
                                    {src + ".compute": elements, src + ".hbm": conversion_work},
                                    slots=(src + ".compute_stream",),
                                    allocs=(alloc(pack_buffer, src, size, "conversion_buffer", p["communication"]),),
                                    metadata={"phase": phase, "operation": "precision_conversion", "conversion": "pipeline_pack",
                                              "kernel": "pipeline_pack", "working_set_bytes": conversion_work})),)
                            event = topology.transfer(eid, src, dst, size, send_deps,
                                {"phase": phase, "operation": "pipeline", "layer": layer})
                            aid = eid + ".receive"
                            event.allocs = (alloc(aid, dst, size, "pipeline_buffer", p["communication"]),)
                            if pack_buffer:
                                event.releases = (pack_buffer,)
                            transferred = add(event)
                            if pack_buffer:
                                decoded = eid + ".decoded"
                                transferred = add(Event(eid + ".unpack", (transferred,),
                                    {dst + ".compute": elements, dst + ".hbm": conversion_work},
                                    slots=(dst + ".compute_stream",),
                                    allocs=(alloc(decoded, dst, elements * DTYPE_BYTES[p["activation"]], "pipeline_buffer", p["activation"]),),
                                    releases=(aid,),
                                    metadata={"phase": phase, "operation": "precision_conversion", "conversion": "pipeline_unpack",
                                              "kernel": "pipeline_unpack", "working_set_bytes": conversion_work}))
                                aid = decoded
                            receive_buffers[dst] = aid
                            transfers.append(transferred)
                    layer_outputs = transfers
                releases = tuple(aid for ids in cleanup.values() for aid in ids)
                layer_deps = (add(Event(prefix + ".done", tuple(layer_outputs + cache_writes),
                    releases=releases, metadata={"phase": phase, "layer": layer})),)
            barrier = add(Event(f"infer.d{d}.token{step}", layer_deps,
                metadata={"phase": phase, "token_index": step, "replica": d,
                          "completed_sequence_ids": active, "token_count": len(active)}))
            token_barriers.append(barrier)
            for i in active:
                per_sequence[str(i)].append(barrier)
            previous_token = (barrier,)
        metadata["token_completion_events"][str(d)] = token_barriers
        metadata["sequence_token_completion_events"][str(d)] = per_sequence
    metadata["assumptions"].append("TP attention and FFN reductions are serialized at layer end; this is a coarse operator schedule.")
    return graph

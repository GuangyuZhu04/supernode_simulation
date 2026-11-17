"""Auditable one-layer, symmetry-reduced graph for the 2026-11 scaling study.

The reduction is exact for this declared surrogate: one nonblocking full-mesh
SU, equal per-lane injection, global phase barriers, uniform expert routing,
batch <= EP, DP=PP=1, and identical layers. It does not model physical switch
oversubscription or predict measured Ascend performance. Communication volume
is expanded separately; graph event timings are never multiplied by TP.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy, copy
import math

from supernode_sim.core import Allocation, DTYPE_BYTES, Event, Graph, Resource
from supernode_sim.inference import kv_layout_bytes
from supernode_sim.moe_wire import moe_payload_matrix
from supernode_sim.topology import Topology


def attribution_graph(graph):
    """Use representative per-flow demand with actual aggregate elapsed time.

    This preserves native service-versus-contention attribution while the
    execution graph coalesces congruent flows solely to reduce host overhead.
    """
    result = copy(graph)
    result.events = []
    for event in graph.events:
        representative = event.metadata.get("representative_flow")
        if representative:
            replaced = copy(event)
            replaced.work = representative["work"]
            replaced.metadata = representative["metadata"]
            result.events.append(replaced)
        else:
            result.events.append(event)
    return result


def build_layer(config, *, phase="decode", context_length=None, afd=None,
                attention_profile=None, attention_kind=None):
    """Return a full-capacity graph with one representative layer's execution.

    AFD is an experiment adapter, not a new native Config feature. In that case
    config.execution.tp describes each half-pool; native config may retain its
    full-POD TP, since the explicitly supplied afd.tp is used here.
    """
    c = config.model_dump() if hasattr(config, "model_dump") else deepcopy(config)
    m, h, w, x = (c[k] for k in ("model", "hardware", "workload", "execution"))
    p, f = x["precision"], x["features"]
    if phase not in ("decode", "prefill"):
        raise ValueError("phase must be decode or prefill")
    if h["topology"] != "full_mesh" or h["su_size"] != h["npu_count"]:
        raise ValueError("Symmetry reduction requires one nonblocking full_mesh SU")
    if h.get("io_groups") or h.get("collective_profiles") or h.get("collective_models") or h.get("kernel_profiles") or h.get("kernel_traces"):
        raise ValueError("Shared IO groups and calibration profiles are outside this adapter")
    if x["dp"] != 1 or x["pp"] != 1 or x.get("sequence_parallel") or x.get("rank_order"):
        raise ValueError("Adapter requires DP=PP=1, no SP, and canonical rank order")
    if x.get("all_to_all_algorithm", "pairwise") != "direct":
        raise ValueError("This study adapter requires direct all-to-all scheduling")
    if f.get("kv_offload", "none") != "none" or f.get("engram", {}).get("enabled"):
        raise ValueError("Offload and Engram are outside this adapter")
    if p["weight"] != p["compute"] or p["activation"] != p["communication"]:
        raise ValueError("This adapter requires equal weight/compute and activation/wire dtypes")
    if p.get("moe_dispatch") not in (None, p["activation"]) or p.get("moe_combine") not in (None, p["activation"]):
        raise ValueError("Wire conversion is outside this adapter")
    if p.get("moe_dispatch_metadata_bytes_per_token", 0) or p.get("moe_combine_metadata_bytes_per_token", 0):
        raise ValueError("Additional wire metadata is outside this adapter")
    if m.get("architecture") or not math.isclose(m["moe_layer_fraction"], 1.0):
        raise ValueError("Adapter requires budget model with identical MoE layers")
    ep = int(afd.get("ep", x["ep"]) if afd else x["ep"])
    tp = int(afd.get("tp", x["tp"]) if afd else x["tp"])
    batch, layers = w["batch_size"], m["layers"]
    if not 0 < batch <= ep:
        raise ValueError("Symmetry reduction requires 0 < global batch <= EP")
    if len(set(w["input_lengths"])) != 1:
        raise ValueError("Identical source lengths are required")
    context = int(context_length if context_length is not None else w["input_length"])
    profile_work = profile_state = None
    if attention_profile:
        if attention_profile['family'] == 'kimi_k3':
            from scripts.scaling_kimi_attention import attention_work, attention_state
        elif attention_profile['family'] == 'deepseek_v41':
            from scripts.scaling_v41_attention import attention_work, attention_state
        else:
            raise ValueError('Unknown attention template family')
        profile_work = attention_work(attention_profile, attention_kind, m, context, tp, phase)
        profile_state = attention_state(attention_profile, context + (phase == 'decode'), tp)
    a_count = int(afd.get("attention_count", tp * ep) if afd else h["npu_count"])
    f_count = int(afd.get("ffn_count", tp * ep) if afd else h["npu_count"])
    if a_count != tp * ep or f_count != tp * ep:
        raise ValueError("Each pool must have TP * EP devices")
    if h["npu_count"] != (a_count + f_count if afd else a_count):
        raise ValueError("AFD pool counts must sum to the specified physical POD")
    topo = Topology(h, p)
    if afd:
        for index in range(a_count, a_count + f_count):
            rank = f"npu{index}"
            topo.resources[rank + ".compute"] = Resource(rank + ".compute", afd["lpu_flops"] * h["npu"]["efficiency"], "FLOP/s", "compute")
            topo.resources[rank + ".hbm"] = Resource(rank + ".hbm", afd["lpu_hbm_Bps"], "byte/s")
            topo.memory_caps[rank + ".hbm"] = afd["lpu_hbm_bytes"]
    if profile_work:
        for index in range(a_count):
            key = f'npu{index}.compute.fp32'
            topo.resources[key] = Resource(key, h['npu']['flops']['fp32'] * h['npu']['efficiency'], 'FLOP/s', 'compute')
    g = Graph()
    topo.populate(g)
    dense = m["total_params"] * (1 - m["expert_parameter_fraction"])
    expert = m["total_params"] - dense
    expert_active = max(0.0, m["active_params"] - dense)
    tokens_per_source = context if phase == "prefill" else 1
    total_tokens = batch * tokens_per_source
    sources = [tokens_per_source if e < batch else 0 for e in range(ep)]
    touched = (1.0 if m["top_k"] == m["num_experts"] else
               -math.expm1(total_tokens * math.log1p(-m["top_k"] / m["num_experts"])))
    final_context = context if phase == "prefill" else context + 1
    precision_bytes = DTYPE_BYTES[p["activation"]]
    semantic = defaultdict(float)
    physical = defaultdict(float)
    expanded_work = defaultdict(float)
    dynamic_ids = []

    def a_rank(e, t=0):
        return f"npu{e * tp + t}"

    def f_rank(e, t=0):
        return f"npu{(a_count if afd else 0) + e * tp + t}"

    def allocation(name, rank, size, category, dtype=None, **metadata):
        return Allocation(name, rank + ".hbm", float(size), category, dtype,
                          metadata={"component": "Unassigned", "state": category, **metadata})

    def add(event, multiplicity=1):
        event.metadata.setdefault("phase", phase)
        event.metadata.setdefault("layer", 0)
        event.metadata["symmetry_multiplicity"] = multiplicity
        for key, value in event.work.items():
            expanded_work[key.partition(".")[2] if key.startswith("npu") and "." in key else "fabric_link"] += value * multiplicity
        g.events.append(event)
        return event.id

    def barrier(name, deps, allocs=(), releases=()):
        return add(Event(name, tuple(deps), allocs=tuple(allocs), releases=tuple(releases)))

    def transfer(name, src, dst, size, deps, operation, multiplicity):
        event = topo.transfer(name, src, dst, size, tuple(deps),
                              {"phase": phase, "layer": 0, "operation": operation,
                               "component": "MoE" if operation in ("dispatch", "combine") else "TP",
                               "collective": "all_to_all_v" if operation in ("dispatch", "combine") else "all_reduce",
                               "algorithm": "direct", "afd_bridge": bool(afd and operation in ("dispatch", "combine"))})
        event.allocs = tuple(allocation(name + "." + role, rank, size, "communication_buffer", p["communication"])
                             for role, rank in (("send", src), ("recv", dst)))
        event.releases = tuple(item.id for item in event.allocs)
        semantic[operation] += size * multiplicity
        for domain, value in event.metadata["physical_bytes_by_domain"].items():
            physical[domain] += value * multiplicity
        return add(event, multiplicity)

    def coalesce_transfers(name, ids):
        if not ids:
            return []
        flows = g.events[-len(ids):]
        if [event.id for event in flows] != ids:
            raise AssertionError("Only a contiguous uniform collective may be coalesced")
        representative = min(flows, key=lambda event: event.id)
        if any(event.startup_s != representative.startup_s or event.deps != representative.deps or
               event.metadata["payload_bytes"] != representative.metadata["payload_bytes"] for event in flows):
            raise ValueError("Coalescing requires congruent same-start uniform flows")
        work = defaultdict(float)
        allocs, releases = [], []
        for event in flows:
            for key, amount in event.work.items():
                work[key] += amount
            allocs.extend(event.allocs)
            releases.extend(event.releases)
        metadata = {**representative.metadata, "payload_bytes": sum(event.metadata["payload_bytes"] for event in flows),
                    "coalesced_uniform_flow_count": len(flows),
                    "representative_flow": {"work": dict(representative.work), "metadata": dict(representative.metadata)},
                    "physical_bytes": sum(event.metadata["physical_bytes"] for event in flows),
                    "physical_bytes_by_domain": {"SU": sum(event.metadata["physical_bytes"] for event in flows)},
                    "coalescing_scope": "Uniform direct flows advance and finish together; summed shared-resource demands preserve fluid elapsed time."}
        del g.events[-len(ids):]
        g.events.append(Event(name, representative.deps, dict(work), representative.startup_s,
                              allocs=tuple(allocs), releases=tuple(releases), metadata=metadata))
        return [name]

    def ring_reduce(operation, size, previous, repeats=1):
        if tp == 1 or not size:
            return previous
        ends = [transfer(f'{operation}.r{t}', a_rank(0, t), a_rank(0, (t + 1) % tp),
                         size / tp, (previous,), operation, batch) for t in range(tp)]
        macro_id = coalesce_transfers(operation + '.ring_envelope', ends)[0]
        macro = g.events[-1]
        rounds = 2 * (tp - 1)
        # Each round has disjoint ring links and identical endpoint demand.
        # Repeat bytes once per logical round; tiled index scans repeat startup
        # per score tile but partition, rather than duplicate, total bytes.
        for key in macro.work:
            extra = macro.work[key] * (rounds - 1) * batch
            expanded_work[key.partition('.')[2] if key.startswith('npu') and '.' in key else 'fabric_link'] += extra
            macro.work[key] *= rounds
        macro.startup_s *= rounds * repeats
        macro.metadata['startup_s_by_domain'] = {k: v * rounds * repeats for k, v in macro.metadata['startup_s_by_domain'].items()}
        macro.metadata['payload_bytes'] *= rounds
        macro.metadata['physical_bytes'] *= rounds
        macro.metadata['physical_bytes_by_domain'] = {k: v * rounds for k, v in macro.metadata['physical_bytes_by_domain'].items()}
        macro.metadata['ring_serial_rounds'] = rounds * repeats
        representative = macro.metadata['representative_flow']
        representative['work'] = {k: v * rounds for k, v in representative['work'].items()}
        representative['metadata']['startup_s_by_domain'] = {
            k: v * rounds * repeats for k, v in representative['metadata']['startup_s_by_domain'].items()}
        representative['metadata']['ring_serial_rounds'] = rounds * repeats
        semantic[operation] += size * batch * (rounds - 1)
        physical['SU'] += size * batch * (rounds - 1)
        return macro_id

    def kv(length, count=layers):
        return kv_layout_bytes(m, p["kv"], [length], layers=count, tp=tp,
                               page_tokens=f["kv_page_tokens"], scale_block=f["kv_scale_block"],
                               scale_bytes=f["scale_bytes"], page_entry_bytes=f.get("kv_page_entry_bytes", 8))

    resident = defaultdict(float)
    weight_splits = [(a_rank(e, t), dense / tp + (0 if afd else expert / (tp * ep)), e < batch, "attention")
                     for e in range(ep) for t in range(tp)]
    if afd:
        weight_splits += [(f_rank(e, t), expert / (tp * ep), False, "ffn") for e in range(ep) for t in range(tp)]
    for rank, params, owns_kv, role in weight_splits:
        items = [allocation(rank + ".weights", rank, params * DTYPE_BYTES[p["weight"]], "weight", p["weight"], device_role=role)]
        if p["weight"] in ("fp8", "int8"):
            items.append(allocation(rank + ".weight_scales", rank, math.ceil(params / f["weight_scale_block"]) * f["scale_bytes"], "quantization_metadata", device_role=role))
        for category in ("runtime", "workspace"):
            items.append(allocation(rank + "." + category, rank, h["npu"].get(category + "_bytes", 0), category, device_role=role))
        if owns_kv:
            if profile_state:
                for category, amount in profile_state['categoriesbytes'].items():
                    state_dtype = ('fp32' if category in ('recurrent_state', 'compressor_state', 'shared_rope_table')
                                   else None if category in ('page_table', 'kv_scales') else p['kv'])
                    items.append(allocation(rank + '.' + category, rank, amount, category, state_dtype, device_role=role))
            else:
                items.append(allocation(rank + ".kv", rank, kv(final_context)["total_bytes"], "kv", p["kv"], device_role=role))
        for item in items:
            resident[item.category] += item.size_bytes
        g.initial_allocations.extend(items)
    active_allocs = []
    activation = tokens_per_source * 2 * m["hidden_dim"] * precision_bytes
    for e in range(batch):
        for t in range(tp):
            rank = a_rank(e, t)
            item = allocation(rank + ".activation", rank, activation, "activation", p["activation"])
            active_allocs.append(item)
            dynamic_ids.append(item.id)
    ready = barrier("layer.ready", (), active_allocs)
    attention_ends = []
    fetched = kv(context, 1)
    width = fetched["elements_per_token"]
    read_bytes = 0 if phase == "prefill" else fetched["data_bytes"] + fetched["scale_bytes"]
    pairs = context * (context + 1) / 2 if phase == "prefill" else context + 1
    for e in range(batch):
        rank = a_rank(e)
        if profile_work:
            is_v41 = attention_profile['family'] == 'deepseek_v41'
            flops = 2 * dense / layers * tokens_per_source / tp + (0 if is_v41 else profile_work['extra_compute_flops'])
            weight_bytes = dense / layers * DTYPE_BYTES[p['compute']] / tp
            memory_bytes = weight_bytes + 2 * activation + (0 if is_v41 else profile_work['memory_bytes'] + profile_work['cache_write_bytes'])
            workspace = allocation(rank + '.attention_workspace', rank, profile_work['workspace_bytes'], 'attention_workspace', None)
            last = add(Event(rank + '.attention', (ready,),
                {rank + '.compute': flops, rank + '.hbm': memory_bytes}, slots=(rank + '.compute_stream',),
                allocs=(workspace,), metadata={'component': attention_kind, 'operation': 'attention_dense',
                'kernel': profile_work['kernel_name'], 'flops': flops, 'working_set_bytes': memory_bytes}), tp)
            if is_v41:
                if profile_work['index_flops']:
                    last = add(Event(rank + '.index_score', (last,),
                        {rank + '.compute': profile_work['index_flops'], rank + '.hbm': profile_work['index_memory_bytes']},
                        slots=(rank + '.compute_stream',), metadata={'component': attention_kind,
                        'operation': 'index_score', 'kernel': 'csa2_index_score'}), tp)
                attention_ends.append(last)
                continue
            if profile_work.get('cache_conversion_flops'):
                last = add(Event(rank + '.kv_convert', (last,),
                    {rank + '.compute': profile_work['cache_conversion_flops'],
                     rank + '.hbm': profile_work['cache_raw_bytes'] + profile_work['cache_storage_bytes']},
                    slots=(rank + '.compute_stream',), metadata={'component': attention_kind, 'operation': 'precision_conversion', 'kernel': 'kv_conversion'}), tp)
            if profile_work['extra_fp32_flops']:
                last = add(Event(rank + '.attention_fp32', (last,),
                    {rank + '.compute.fp32': profile_work['extra_fp32_flops'], rank + '.hbm': profile_work.get('fp32_memory_bytes', 0)},
                    slots=(rank + '.compute_stream',), metadata={'component': attention_kind,
                    'operation': 'attention_scalar', 'kernel': attention_kind.lower() + '_scalar'}), tp)
            last = barrier(rank + '.attention_done', (last,), releases=(workspace.id,))
            attention_ends.append(last)
            continue
        flops = 2 * dense / layers * tokens_per_source / tp + 4 * m["num_heads"] * m["head_dim"] * pairs / tp
        weight_bytes = dense / layers * DTYPE_BYTES[p["compute"]] / tp
        new_kv = tokens_per_source * width * DTYPE_BYTES[p["kv"]]
        raw_kv = tokens_per_source * width * DTYPE_BYTES[p["compute"]]
        if p["kv"] in ("fp8", "int8"):
            new_kv += math.ceil(tokens_per_source * width / f["kv_scale_block"]) * f["scale_bytes"]
        convert = p["kv"] != p["compute"]
        if convert and phase == "decode":
            flops += read_bytes / DTYPE_BYTES[p["kv"]]
        memory_bytes = weight_bytes + 2 * activation + read_bytes + (raw_kv if convert else new_kv)
        raw_id = rank + ".kv_raw"
        last = add(Event(rank + ".attention", (ready,), {rank + ".compute": flops, rank + ".hbm": memory_bytes},
                         slots=(rank + ".compute_stream",),
                         allocs=(allocation(raw_id, rank, raw_kv, "kv_conversion_buffer", p["compute"]),) if convert else (),
                         metadata={"component": "Unassigned", "operation": "attention_dense", "kernel": "attention_dense", "flops": flops,
                                   "working_set_bytes": weight_bytes + activation + read_bytes + new_kv}), tp)
        if convert:
            last = add(Event(rank + ".kv_convert", (last,), {rank + ".compute": tokens_per_source * width, rank + ".hbm": raw_kv + new_kv},
                             slots=(rank + ".compute_stream",), releases=(raw_id,),
                             metadata={"component": "Unassigned", "operation": "precision_conversion", "kernel": "kv_conversion",
                                       "working_set_bytes": raw_kv + new_kv}), tp)
        attention_ends.append(last)
    if profile_work and profile_work['index_reduce_bytes']:
        ready_index = barrier('attention.index_ready', attention_ends)
        index_end = ring_reduce('tp_index_scores', profile_work['index_reduce_bytes'], ready_index,
                                profile_work['index_reduce_repeats'])
        attention_ends = [index_end]
    if profile_work and attention_profile['family'] == 'deepseek_v41':
        predecessors = tuple(attention_ends)
        attention_ends = []
        for e in range(batch):
            rank = a_rank(e)
            last = predecessors
            if profile_work['index_selection_flops']:
                last = (add(Event(rank + '.index_select', last,
                    {rank + '.compute.fp32': profile_work['index_selection_flops'], rank + '.hbm': profile_work['index_selection_memory_bytes']},
                    slots=(rank + '.compute_stream',), metadata={'component': attention_kind,
                    'operation': 'index_selection', 'kernel': 'csa2_topk_candidates'}), tp),)
            last = add(Event(rank + '.sparse_attention', last,
                {rank + '.compute': profile_work['extra_compute_flops'],
                 rank + '.hbm': profile_work['memory_bytes'] + profile_work['cache_write_bytes']},
                slots=(rank + '.compute_stream',), metadata={'component': attention_kind,
                'operation': 'sparse_attention', 'kernel': 'csa2_sparse_qk_av'}), tp)
            last = add(Event(rank + '.attention_fp32', (last,),
                {rank + '.compute.fp32': profile_work['extra_fp32_flops'], rank + '.hbm': profile_work.get('fp32_memory_bytes', 0)},
                slots=(rank + '.compute_stream',), metadata={'component': attention_kind,
                'operation': 'attention_scalar', 'kernel': 'csa2_softmax_rope'}), tp)
            last = barrier(rank + '.attention_done', (last,), releases=(rank + '.attention_workspace',))
            attention_ends.append(last)
    expert_allocs = []
    for e in range(ep):
        for t in range(tp):
            rank = f_rank(e, t)
            for suffix, size, category in (
                    ("expert_buffer", total_tokens * m["hidden_dim"] * m["top_k"] * precision_bytes / ep, "expert_buffer"),
                    ("expert_intermediate", total_tokens * m["top_k"] * m["intermediate_dim"] * precision_bytes / (tp * ep), "activation")):
                item = allocation(rank + "." + suffix, rank, size, category, p["activation"])
                expert_allocs.append(item)
                dynamic_ids.append(item.id)
    dispatch_start = barrier("dispatch.ready", attention_ends, expert_allocs)
    dispatch = []
    matrix = moe_payload_matrix(p, "dispatch", sources, m["hidden_dim"], m["top_k"])
    for src_e in range(batch):
        for dst_e in range(ep):
            if afd or src_e != dst_e:
                dispatch.append(transfer(f"dispatch.e{src_e}.e{dst_e}", a_rank(src_e), f_rank(dst_e), matrix[src_e][dst_e], (dispatch_start,), "dispatch", tp))
    dispatch = coalesce_transfers("dispatch.uniform_flows", dispatch)
    dispatch_end = barrier("dispatch.done", dispatch or [dispatch_start])
    experts = []
    for e in range(ep):
        rank = f_rank(e)
        flops = 2 * expert_active / layers * total_tokens / (tp * ep)
        memory_bytes = expert / layers * touched * DTYPE_BYTES[p["compute"]] / (tp * ep)
        experts.append(add(Event(rank + ".experts", (dispatch_end,), {rank + ".compute": flops, rank + ".hbm": memory_bytes},
                                 slots=(rank + ".compute_stream",),
                                 metadata={"component": "Unassigned", "operation": "experts", "kernel": "experts", "flops": flops,
                                           "working_set_bytes": memory_bytes + total_tokens * m["top_k"] * (m["hidden_dim"] + m["intermediate_dim"] / tp) * precision_bytes / ep}), tp))
    expert_end = barrier("experts.done", experts)
    combine = []
    matrix = moe_payload_matrix(p, "combine", sources, m["hidden_dim"], m["top_k"])
    for src_e in range(ep):
        for dst_e in range(batch):
            if afd or src_e != dst_e:
                combine.append(transfer(f"combine.e{src_e}.e{dst_e}", f_rank(src_e), a_rank(dst_e), matrix[src_e][dst_e], (expert_end,), "combine", tp))
    combine = coalesce_transfers("combine.uniform_flows", combine)
    previous = barrier("combine.done", combine or [expert_end])
    # Preserve the native budget compiler's coarse layer-end ordering.
    for operation in ("tp_attention", "tp_ffn"):
        previous = ring_reduce(operation, tokens_per_source * m['hidden_dim'] * DTYPE_BYTES[p['communication']], previous)
    final = barrier("infer.d0.token1", (previous,), releases=dynamic_ids)
    g.metadata.update(scenario="inference", assumptions=[
        "Synthetic budget-model graph: uniform layer parameters and uniform fractional expert routing; no measured Ascend calibration.",
        "Full model weights and architecture-specific cached context are resident; warm decode-only results do not include prefill/TTFT.",
        "One full-mesh nonblocking SU, per-direction bandwidth, no shared uplinks; each TP lane has independent endpoint resources.",
        "Symmetry reduction keeps one TP lane for attention/MoE and one active EP group for each TP ring; all full-POD permanent and activation allocations remain explicit.",
        "Reported transport volumes are expanded by recorded event multiplicities; critical-path time is never multiplied by TP or active EP replicas.",
        "TP attention and FFN reductions occur after combine, matching the existing coarse budget compiler.",
        "AFD splits dense/attention and routed expert state into physical pools; every A-F transfer, including same EP index, crosses the fabric." if afd else "Homogeneous same-EP dispatch/combine copies are local and do not traverse the fabric.",
        "Uniform direct-flow groups and identical ring rounds are executed as summed-resource envelopes with identical elapsed time; representative-flow attribution is an explicitly reconstructed service/contending split, not a saved micro-flow trace.",
    ], token_completion_events={"0": [final]},
        sequence_token_completion_events={"0": {str(i): [final] for i in range(batch)}})
    if attention_profile:
        g.metadata['assumptions'].append(attention_profile['scope'])
        if attention_profile.get('layout_status'):
            g.metadata['assumptions'].append(attention_profile['layout_status'])
    g.metadata["study"] = {
        "adapter": "symmetry_reduced_budget_layer_v1", "phase": phase, "context_length": context,
        "layer_count": layers, "tp": tp, "ep": ep, "batch_size": batch,
        "attention_count": a_count, "ffn_count": f_count, "afd": deepcopy(afd),
        "end_event": final, "transport_payload_bytes_total": sum(semantic.values()),
        "physical_bytes_total": sum(physical.values()), "per_semantic_bytes_total": dict(semantic),
        "physical_bytes_by_domain_total": dict(physical), "expanded_resource_work_by_kind": dict(expanded_work),
        "resident_bytes_by_category": dict(resident),
        "resident_weight_bytes_total": resident["weight"],
        "compression": {"moe_and_compute_multiplicity": tp, "tp_ring_multiplicity": batch,
                        "omitted_components_are_disjoint_congruent_resource_graphs": True,
                        "communication_buffer_peak_equivalence": "For mirrored TP lanes use the representative t=0 peak; nonrepresentative raw memory traces omit mirrored transient staging."},
        "dense_params": dense, "expert_params": expert, "expert_active_params": expert_active,
        "touched_expert_fraction": touched,
        "attention_profile": deepcopy(attention_profile), "attention_kind": attention_kind,
        "attention_work": profile_work, "attention_state": profile_state,
    }
    return g

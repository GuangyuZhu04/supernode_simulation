"""Training tensor ledger and operator graph for a declared parameter budget.

This is an architecture model, not a reconstruction of an unpublished model.
Parameter and token work are uniform over layers/experts; those assumptions are
included in every result. Dense EP replicas are deliberately *not* divided by EP
when calculating resident parameters.
"""
from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from .core import Allocation, DTYPE_BYTES, Event, Graph, Resource
from .moe_wire import moe_token_bytes, moe_wire_metadata, wire_payload_parts

if TYPE_CHECKING:
    from .topology import Topology


def pipeline_schedule(stage: int, stages: int, microbatches: int) -> list[tuple[str, int]]:
    """Non-interleaved 1F1B: warmup, alternating steady state, cooldown."""
    warmup = min(stages - stage - 1, microbatches)
    schedule = [("forward", m) for m in range(warmup)]
    for m in range(microbatches - warmup):
        schedule.extend((("forward", m + warmup), ("backward", m)))
    schedule.extend(("backward", m) for m in range(microbatches - warmup, microbatches))
    return schedule


class _TrainingBuilder:
    def __init__(self, config: dict, topology: Topology):
        self.cfg, self.topo = config, topology
        self.model, self.hw = config["model"], config["hardware"]
        self.exe, self.wl = config["execution"], config["workload"]
        self.precision, self.features = self.exe["precision"], self.exe["features"]
        self.tp, self.pp, self.ep, self.dp = (self.exe[x] for x in ("tp", "pp", "ep", "dp"))
        self.fsdp = self.exe["fsdp"]
        self.layers = self.model["layers"]
        self.microbatches = self.wl["gradient_accumulation"]
        self.tokens = self.wl["batch_size"] * self.wl["sequence_length"]
        self.offload = self.features["optimizer_offload"]
        self.checkpoint = self.features["activation_checkpointing"]
        self.reshard = self.exe["reshard_after_forward"]
        self.graph = Graph()
        self.topo.populate(self.graph)
        order = self.exe.get("rank_order") or list(range(self.hw["npu_count"]))
        coords = product(range(self.dp), range(self.pp), range(self.ep), range(self.tp))
        self.ranks = {coord: f"npu{order[i]}" for i, coord in enumerate(coords)}
        self.reverse = {rank: coord for coord, rank in self.ranks.items()}
        self.stage_ranks = {s: [r for c, r in self.ranks.items() if c[1] == s] for s in range(self.pp)}
        self.stage_layers = {s: list(range(s * self.layers // self.pp, (s + 1) * self.layers // self.pp))
                             for s in range(self.pp)}
        fraction = self.model["expert_parameter_fraction"]
        self.dense_per_layer = self.model["total_params"] * (1 - fraction) / self.layers / self.tp
        self.expert_per_layer = self.model["total_params"] * fraction / self.layers / self.tp / self.ep
        self.params_per_layer = self.dense_per_layer + self.expert_per_layer
        self.shard_params = self.params_per_layer / self.fsdp
        self.ends: list[str] = []

    def size(self, field: str, elements: float) -> float:
        return elements * DTYPE_BYTES[self.precision[field]]

    def scale_bytes(self, elements: float, dtype: str) -> float:
        # Fractional blocks are a continuous lower bound for the aggregate
        # parameter-budget model; profiles can calibrate packing overhead.
        if dtype != "fp8":
            return 0.0
        return elements / self.features["weight_scale_block"] * self.features["scale_bytes"]

    def pool(self, rank: str) -> str:
        return rank + (".hbm" if rank.startswith("npu") else ".dram")

    def scalar_resource(self, rank: str) -> str:
        """Scalar updates use the FP32 profile and share the compute stream."""
        if not rank.startswith("npu") or "fp32" not in self.hw["npu"]["flops"]:
            return rank + ".compute"
        name = rank + ".compute.fp32"
        self.graph.resources[name] = Resource(name,
            self.hw["npu"]["flops"]["fp32"] * self.hw["npu"]["efficiency"], "FLOP/s", "compute")
        return name

    def add(self, event: Event) -> str:
        self.graph.events.append(event)
        return event.id

    def barrier(self, name: str, deps=(), *, allocs=(), releases=(), metadata=None) -> str:
        return self.add(Event(name, tuple(dict.fromkeys(deps)), allocs=tuple(allocs),
                              releases=tuple(releases), metadata=metadata or {}))

    def allocation(self, name: str, rank: str, amount: float, category: str,
                   dtype: str | None = None, **metadata) -> Allocation:
        shape = ()
        if dtype:
            elements = amount / DTYPE_BYTES[dtype]
            metadata = {"logical_elements": elements, "shape_model": "flat_parameter_budget", **metadata}
            if elements.is_integer():
                shape = (int(elements),)
        return Allocation(name, self.pool(rank), amount, category, dtype, shape=shape, metadata=metadata)

    def initial_state(self) -> None:
        for rank in self.reverse:
            stage = self.reverse[rank][1]
            host = self.topo.host_for(int(rank[3:]))
            optimizer_rank = host if self.offload else rank
            for layer in self.stage_layers[stage]:
                prefix = f"state.{rank}.l{layer}"
                info = {"layer": layer, "shard_factor": self.fsdp,
                        "tensor_parallel_shards": self.tp, "expert_parallel_shards": self.ep,
                        "dense_replicated_across_ep": True}
                for field, category, copies, device in (
                    ("weight", "weights", 1, rank), ("gradient", "gradients", 1, rank),
                    ("master", "master_weights", 1, optimizer_rank),
                    ("optimizer", "optimizer_states", 2, optimizer_rank),
                ):
                    self.graph.initial_allocations.append(self.allocation(
                        f"{prefix}.{field}", device, self.size(field, self.shard_params) * copies,
                        category, self.precision[field], **info))
                scales = self.scale_bytes(self.shard_params, self.precision["weight"])
                if scales:
                    self.graph.initial_allocations.append(self.allocation(
                        prefix + ".weight_scales", rank, scales, "quantization_metadata", **info))
                extra = self.features.get("extra_weight_copy_dtype")
                if extra:
                    self.graph.initial_allocations.append(self.allocation(
                        prefix + ".weight_extra", rank, self.shard_params * DTYPE_BYTES[extra],
                        "extra_weight_copy", extra, **info))
            for field, category in (("runtime_bytes", "runtime_reserve"), ("workspace_bytes", "workspace")):
                amount = self.hw["npu"].get(field, 0)
                if amount:
                    self.graph.initial_allocations.append(self.allocation(
                        f"state.{rank}.{field}", rank, amount, category))

    def groups(self, stage: int, axis: str) -> list[list[str]]:
        groups: list[list[str]] = []
        if axis == "fsdp":
            for start, e, t in product(range(0, self.dp, self.fsdp), range(self.ep), range(self.tp)):
                groups.append([self.ranks[d, stage, e, t] for d in range(start, start + self.fsdp)])
        elif axis == "replica":
            for offset, e, t in product(range(self.fsdp), range(self.ep), range(self.tp)):
                groups.append([self.ranks[d, stage, e, t] for d in range(offset, self.dp, self.fsdp)])
        elif axis == "ep":
            for d, t in product(range(self.dp), range(self.tp)):
                groups.append([self.ranks[d, stage, e, t] for e in range(self.ep)])
        elif axis == "tp":
            for d, e in product(range(self.dp), range(self.ep)):
                groups.append([self.ranks[d, stage, e, t] for t in range(self.tp)])
        return groups

    def collective(self, prefix: str, stage: int, axis: str, kind: str, size: float,
                   deps, metadata: dict) -> list[str]:
        ends: list[str] = []
        wire = metadata.get('wire_dtype', self.precision['communication'])
        moe = metadata.get('semantic') in ('dispatch', 'combine')
        original_deps = tuple(deps)
        algorithm = self.exe.get('all_to_all_algorithm', 'pairwise')
        for i, ranks in enumerate(self.groups(stage, axis)):
            group_prefix = f"{prefix}.g{i}"
            matrix = None
            if kind == "all_to_all_v":
                # Equal expert loads: size is each source's total routed token
                # payload, including the local share (which is not transmitted).
                matrix = [[0 if a == b else size / len(ranks) for b in range(len(ranks))]
                          for a in range(len(ranks))]
            outer = moe
            if moe:
                selection = self.topo.select_collective_timing_scope(kind, ranks, size, matrix, metadata, algorithm)
                metadata = {**metadata, 'collective_scope_selection': selection}
                outer = selection['selected_scope'] == 'activation_collective'
            activation_wire = metadata.get("semantic") in ("dispatch", "combine") or metadata.get("category") == "tensor_parallel"
            convert = (activation_wire and len(ranks) > 1 and
                       self.precision["activation"] != wire)
            network_deps, buffers = list(deps), []
            if convert:
                network_deps = []
                for index, rank in enumerate(ranks):
                    payload = sum(matrix[index]) if matrix is not None else size
                    elements, extra = wire_payload_parts(payload, wire, metadata)
                    accesses = self.size("activation", elements) + payload + extra
                    buffer_id = group_prefix + f".{rank}.packed"
                    buffers.append(buffer_id)
                    network_deps.append(self.add(Event(group_prefix + f".{rank}.pack", tuple(deps),
                        {self.scalar_resource(rank): 2 * elements, rank + ".hbm": accesses},
                        slots=(rank + ".compute_stream",),
                        allocs=(self.allocation(buffer_id, rank, payload, "conversion_buffer", wire),),
                        metadata={**metadata, "category": "precision_conversion", "kernel": "communication_pack",
                                  "conversion": "communication_pack", "working_set_bytes": accesses,
                                  "source_precision": self.precision["activation"],
                                  "destination_precision": wire, "tensor_elements": elements, "wire_metadata_bytes": extra})))
            terminals = self.topo.collective(self.graph, group_prefix, kind, ranks,
                size, deps=tuple(network_deps), payload_matrix=matrix, metadata=metadata,
                algorithm=algorithm, apply_profiles=not outer)
            group_ends = []
            if convert:
                # Uniform all-to-all and ring reduction reuse their packed
                # per-rank wire buffer for the receive result until unpack ends.
                for index, rank in enumerate(ranks):
                    payload = sum(row[index] for row in matrix) if matrix is not None else size
                    elements, extra = wire_payload_parts(payload, wire, metadata)
                    accesses = self.size("activation", elements) + payload + extra
                    group_ends.append(self.add(Event(group_prefix + f".{rank}.unpack", tuple(terminals),
                        {self.scalar_resource(rank): 2 * elements, rank + ".hbm": accesses},
                        slots=(rank + ".compute_stream",), releases=(buffers[index],),
                        metadata={**metadata, "category": "precision_conversion", "kernel": "communication_unpack",
                                  "conversion": "communication_unpack", "working_set_bytes": accesses,
                                  "source_precision": wire,
                                  "destination_precision": self.precision["activation"],
                                  "tensor_elements": elements, "wire_metadata_bytes": extra})))
            else:
                group_ends.extend(terminals)
            if outer:
                group_ends = self.topo.complete_collective_profile(self.graph, group_prefix, kind, ranks, size,
                    deps=original_deps, terminals=tuple(group_ends), payload_matrix=matrix,
                    metadata=metadata, algorithm=algorithm, timing_scope='activation_collective')
            ends.extend(group_ends)
        return ends or list(deps)

    def phase_name(self, step: int, stage: int, phase: str, micro: int) -> str:
        return f"train.s{step}.p{stage}.{phase}.m{micro}"

    def layer_name(self, step: int, stage: int, phase: str, micro: int, layer: int) -> str:
        return self.phase_name(step, stage, phase, micro) + f".l{layer}"

    def phase_previous(self, step: int, stage: int, phase: str, micro: int) -> tuple[str, ...]:
        schedule = pipeline_schedule(stage, self.pp, self.microbatches)
        pos = schedule.index((phase, micro))
        if pos:
            last_phase, last_micro = schedule[pos - 1]
            return (self.phase_name(step, stage, last_phase, last_micro) + ".done",)
        return (f"train.step{step - 1}.done",) if step else ()

    def gather_name(self, step, stage, phase, micro, layer) -> str:
        if not self.reshard:
            phase, micro = "forward", 0
        return self.layer_name(step, stage, phase, micro, layer) + ".gather.done"

    def add_gather(self, step: int, stage: int, phase: str, micro: int, layer: int,
                   deps) -> str:
        name = self.layer_name(step, stage, phase, micro, layer)
        allocations = []
        for rank in self.stage_ranks[stage]:
            allocations.append(self.allocation(name + f".{rank}.gathered", rank,
                self.size("weight", self.params_per_layer), "gathered_weights", self.precision["weight"]))
            scales = self.scale_bytes(self.params_per_layer, self.precision["weight"])
            if scales:
                allocations.append(self.allocation(name + f".{rank}.gathered_scales", rank,
                    scales, "quantization_metadata"))
        start = self.barrier(name + ".gather.start", deps, allocs=allocations)
        # FSDP materializes weights in their configured storage precision. Its
        # scale tensors travel separately from numerical weights.
        ends = self.collective(name + ".gather", stage, "fsdp", "all_gather",
            self.size("weight", self.params_per_layer), (start,),
            {"phase": phase, "category": "fsdp_weights", "layer": layer})
        scales = self.scale_bytes(self.params_per_layer, self.precision["weight"])
        if scales:
            ends = self.collective(name + ".gather_scales", stage, "fsdp", "all_gather", scales, ends,
                {"phase": phase, "category": "quantization_metadata", "layer": layer})
        return self.barrier(name + ".gather.done", ends)

    def activation_size(self) -> float:
        hidden, intermediate = self.model["hidden_dim"], self.model["intermediate_dim"]
        if self.checkpoint:
            elements_per_token = hidden
        else:
            elements_per_token = 2 * hidden + intermediate / self.tp * (1 + self.model["moe_layer_fraction"] * self.model["top_k"])
        return self.size("activation", self.tokens * elements_per_token / self.ep)

    def kernel(self, name: str, rank: str, deps, forward: bool, *, allocs=(), releases=(),
               recompute=False, layer=0) -> str:
        # 2 P_active FLOPs per token in forward; backward is twice forward.
        # Attention adds a causal streaming attention term independent of P.
        work = 2 * self.model["active_params"] / self.layers * self.tokens
        work += 2 * self.wl["batch_size"] * self.wl["sequence_length"] ** 2 * self.model["hidden_dim"]
        flops = work / self.tp / self.ep * (1 if forward else 2)
        reads = self.size("weight", self.params_per_layer) + 2 * self.activation_size()
        if not forward:
            reads += 2 * self.size("gradient", self.params_per_layer)
        metadata = {"phase": "recompute" if recompute else ("forward" if forward else "backward"),
                    "kernel": "training_recompute" if recompute else ("training_forward" if forward else "training_backward"),
                    "rank": rank, "layer": layer, "flops": flops,
                    "hbm_bytes": reads, "working_set_bytes": reads,
                    "compute_precision": self.precision["compute"]}
        return self.add(Event(name, tuple(deps), {rank + ".compute": flops, rank + ".hbm": reads},
            slots=(rank + ".compute_stream",), allocs=tuple(allocs), releases=tuple(releases), metadata=metadata))

    def phase(self, step: int, stage: int, phase: str, micro: int) -> None:
        forward = phase == "forward"
        prefix = self.phase_name(step, stage, phase, micro)
        deps = list(self.phase_previous(step, stage, phase, micro))
        if forward and stage:
            deps.append(self.phase_name(step, stage - 1, phase, micro) + ".send.done")
        if not forward:
            deps.append(self.phase_name(step, stage, "forward", micro) + ".done")
            if stage < self.pp - 1:
                deps.append(self.phase_name(step, stage + 1, phase, micro) + ".send.done")
        start = self.barrier(prefix + ".start", deps)
        layers = self.stage_layers[stage][::1 if forward else -1]
        previous = start
        for index, layer in enumerate(layers):
            name = self.layer_name(step, stage, phase, micro, layer)
            layer_deps = [previous]
            if self.fsdp > 1:
                if self.reshard or (forward and micro == 0):
                    # One-layer lookahead: next all-gather may overlap the
                    # previous layer's compute but not its all-gather.
                    gather_deps = [previous]
                    if self.exe["prefetch"] and index:
                        gather_deps = [self.gather_name(step, stage, phase, micro, layers[index - 1])]
                        if index > 1:
                            gather_deps.append(self.layer_name(step, stage, phase, micro, layers[index - 2]) + ".done")
                    layer_deps.append(self.add_gather(step, stage, phase, micro, layer, gather_deps))
                else:
                    layer_deps.append(self.gather_name(step, stage, phase, micro, layer))
            elif self.precision["compute"] != self.precision["weight"]:
                # Conversion buffers are accounted below for both sharded and
                # unsharded execution; no all-gather exists in this case.
                pass
            # A declared MoE layer fraction is applied continuously to each
            # layer's communication instead of guessing which layers are MoE.
            routed_tokens = self.tokens / self.ep * self.model["top_k"] * self.model["moe_layer_fraction"]
            width = self.model['hidden_dim']
            dispatch_wire = moe_wire_metadata(self.precision, 'dispatch', width, top_k=self.model['top_k'], num_experts=self.model['num_experts'])
            combine_wire = moe_wire_metadata(self.precision, 'combine', width, top_k=self.model['top_k'], num_experts=self.model['num_experts'])
            moe_bytes = routed_tokens * moe_token_bytes(self.precision, 'dispatch', width)
            combine_bytes = routed_tokens * moe_token_bytes(self.precision, 'combine', width)
            if self.ep > 1 and moe_bytes:
                layer_deps = self.collective(name + ".dispatch", stage, "ep", "all_to_all_v",
                    moe_bytes, layer_deps,
                    {"phase": "recompute" if self.checkpoint and not forward else phase,
                     "semantic": "dispatch", "layer": layer, **dispatch_wire})
            kernels, recomputes = [], []
            for rank in self.stage_ranks[stage]:
                rank_name = name + "." + rank
                activation_id = self.layer_name(step, stage, "forward", micro, layer) + f".{rank}.activation"
                allocations, releases = [], []
                if forward:
                    allocations.append(self.allocation(activation_id, rank, self.activation_size(),
                        "activation_checkpoint" if self.checkpoint else "activations", self.precision["activation"],
                        microbatch=micro, layer=layer))
                else:
                    releases.append(activation_id)
                rank_deps = list(layer_deps)
                conversion_id = rank_name + ".compute_weights"
                if self.precision["compute"] != self.precision["weight"]:
                    convert_bytes = self.size("compute", self.params_per_layer)
                    converted = self.allocation(conversion_id, rank, convert_bytes, "conversion_buffer", self.precision["compute"])
                    conversion = self.add(Event(rank_name + ".convert", tuple(rank_deps),
                        {self.scalar_resource(rank): 2 * self.params_per_layer,
                         rank + ".hbm": convert_bytes + self.size("weight", self.params_per_layer)},
                        slots=(rank + ".compute_stream",), allocs=(converted,),
                        metadata={"phase": phase, "category": "precision_conversion", "kernel": "weight_conversion", "layer": layer}))
                    rank_deps = [conversion]
                    releases.append(conversion_id)
                if not forward and self.checkpoint:
                    recompute_id = rank_name + ".recompute_activations"
                    # Recompute restores a full transient layer workspace, not
                    # merely the small saved checkpoint input.
                    factor = (2 * self.model["hidden_dim"] + self.model["intermediate_dim"] / self.tp *
                              (1 + self.model["moe_layer_fraction"] * self.model["top_k"]))
                    amount = self.size("activation", self.tokens * factor / self.ep)
                    rank_deps = [self.kernel(rank_name + ".recompute", rank, rank_deps, True,
                        allocs=(self.allocation(recompute_id, rank, amount, "recompute_workspace", self.precision["activation"]),),
                        recompute=True, layer=layer)]
                    recomputes.extend(rank_deps)
                    # The recomputed forward includes its own collectives.
                    # This marker is added after all rank kernels are known.
                    rank_deps = [name + ".recompute.done"]
                    releases.append(recompute_id)
                if not forward and self.fsdp > 1:
                    grad_id = f"train.s{step}.p{stage}.l{layer}.{rank}.full_gradient"
                    if self.exe["sync_each_microbatch"]:
                        grad_id += f".m{micro}"
                    if self.exe["sync_each_microbatch"] or micro == 0:
                        allocations.append(self.allocation(grad_id, rank, self.size("gradient", self.params_per_layer),
                            "unsharded_gradients", self.precision["gradient"]))
                kernels.append(self.kernel(rank_name + ".compute", rank, rank_deps, forward,
                                           allocs=allocations, releases=releases, layer=layer))
            if recomputes:
                recompute_terminals = recomputes
                if self.tp > 1:
                    activation_bytes = self.size("communication", self.tokens * self.model["hidden_dim"] / self.ep)
                    for component in ("attention", "mlp"):
                        recompute_terminals = self.collective(name + ".recompute.tp." + component,
                            stage, "tp", "all_reduce", activation_bytes, recompute_terminals,
                            {"phase": "recompute", "category": "tensor_parallel", "layer": layer})
                if self.ep > 1 and moe_bytes:
                    recompute_terminals = self.collective(name + ".recompute.combine", stage, "ep", "all_to_all_v",
                        combine_bytes, recompute_terminals, {"phase": "recompute", "semantic": "combine", "layer": layer, **combine_wire})
                    recompute_terminals = self.collective(name + ".backward_dispatch", stage, "ep", "all_to_all_v",
                        moe_bytes, recompute_terminals, {"phase": "backward", "semantic": "dispatch", "layer": layer, **dispatch_wire})
                self.barrier(name + ".recompute.done", recompute_terminals)
            terminals = kernels
            if self.tp > 1:
                activation_bytes = self.size("communication", self.tokens * self.model["hidden_dim"] / self.ep)
                for component in ("attention", "mlp"):
                    terminals = self.collective(name + ".tp." + component, stage, "tp", "all_reduce",
                        activation_bytes, terminals, {"phase": phase, "category": "tensor_parallel", "layer": layer})
            if self.ep > 1 and moe_bytes:
                terminals = self.collective(name + ".combine", stage, "ep", "all_to_all_v",
                    combine_bytes, terminals, {"phase": phase, "semantic": "combine", "layer": layer, **combine_wire})
            layer_releases = []
            if not forward and (self.exe["sync_each_microbatch"] or micro == self.microbatches - 1):
                if self.ep > 1 and self.dense_per_layer:
                    terminals = self.collective(name + ".dense_grad_ep", stage, "ep", "all_reduce",
                        self.size("gradient", self.dense_per_layer), terminals,
                        {"phase": phase, "category": "dense_replica_gradient", "layer": layer})
                if self.fsdp > 1:
                    terminals = self.collective(name + ".grad_rs", stage, "fsdp", "reduce_scatter",
                        self.size("gradient", self.params_per_layer), terminals,
                        {"phase": phase, "category": "fsdp_gradients", "layer": layer})
                    for rank in self.stage_ranks[stage]:
                        grad_id = f"train.s{step}.p{stage}.l{layer}.{rank}.full_gradient"
                        if self.exe["sync_each_microbatch"]:
                            grad_id += f".m{micro}"
                        layer_releases.append(grad_id)
                if self.dp > self.fsdp:
                    terminals = self.collective(name + ".grad_replica", stage, "replica", "all_reduce",
                        self.size("gradient", self.shard_params), terminals,
                        {"phase": phase, "category": "data_parallel_gradients", "layer": layer})
            if self.fsdp > 1 and (self.reshard or (not forward and micro == self.microbatches - 1)):
                gather_prefix = name if self.reshard else self.layer_name(step, stage, "forward", 0, layer)
                for rank in self.stage_ranks[stage]:
                    layer_releases.append(gather_prefix + f".{rank}.gathered")
                    if self.precision["weight"] == "fp8":
                        layer_releases.append(gather_prefix + f".{rank}.gathered_scales")
            previous = self.barrier(name + ".done", terminals, releases=layer_releases)
        # End markers exclude asynchronous PP sends. A receiver waits for its
        # incoming send marker; a sender may continue computing after enqueue.
        self.barrier(prefix + ".done", (previous,))
        neighbor = stage + 1 if forward else stage - 1
        transfers = []
        if 0 <= neighbor < self.pp:
            elements = self.tokens * self.model["hidden_dim"] / self.ep
            payload = self.size("communication", elements)
            for rank in self.stage_ranks[stage]:
                d, _, e, t = self.reverse[rank]
                dst = self.ranks[d, neighbor, e, t]
                transfer_name = prefix + ".send." + rank
                current = [previous]
                convert = self.precision["activation"] != self.precision["communication"]
                source_id, destination_id = transfer_name + ".src", transfer_name + ".dst"
                accesses = self.size("activation", elements) + payload
                if convert:
                    current = [self.add(Event(transfer_name + ".pack", tuple(current),
                        {self.scalar_resource(rank): 2 * elements, rank + ".hbm": accesses},
                        slots=(rank + ".compute_stream",),
                        allocs=(self.allocation(source_id, rank, payload, "conversion_buffer", self.precision["communication"]),),
                        metadata={"phase": phase, "category": "precision_conversion", "kernel": "pipeline_pack",
                                  "conversion": "pipeline_pack", "working_set_bytes": accesses}))]
                event = self.topo.transfer(transfer_name, rank, dst, payload, deps=tuple(current),
                    metadata={"phase": phase, "category": "pipeline_activation", "microbatch": micro})
                buffers = [self.allocation(destination_id, dst, payload, "communication_buffer", self.precision["communication"])]
                if not convert:
                    buffers.append(self.allocation(source_id, rank, payload, "communication_buffer", self.precision["communication"]))
                event.allocs = tuple(event.allocs) + tuple(buffers)
                event.releases = tuple(event.releases) + ((source_id,) if convert else (source_id, destination_id))
                terminal = self.add(event)
                if convert:
                    terminal = self.add(Event(transfer_name + ".unpack", (terminal,),
                        {self.scalar_resource(dst): 2 * elements, dst + ".hbm": accesses},
                        slots=(dst + ".compute_stream",), releases=(destination_id,),
                        metadata={"phase": phase, "category": "precision_conversion", "kernel": "pipeline_unpack",
                                  "conversion": "pipeline_unpack", "working_set_bytes": accesses}))
                transfers.append(terminal)
        self.barrier(prefix + ".send.done", transfers or (previous,))

    def optimizer(self, step: int) -> None:
        deps = tuple(self.phase_name(step, s, "backward", self.microbatches - 1) + ".done" for s in range(self.pp))
        finishes = []
        for rank, coord in self.reverse.items():
            host = self.topo.host_for(int(rank[3:]))
            device = host if self.offload else rank
            for layer in self.stage_layers[coord[1]]:
                prefix = f"train.s{step}.optimizer.{rank}.l{layer}"
                current = list(deps)
                releases = []
                if self.offload:
                    incoming = self.allocation(prefix + ".cpu_gradient", host,
                        self.size("gradient", self.shard_params), "offload_gradient", self.precision["gradient"])
                    event = self.topo.transfer(prefix + ".D2H", rank, host, incoming.size_bytes,
                        deps=tuple(current), metadata={"phase": "optimizer", "category": "optimizer_offload",
                            "direction": "D2H", "tensor": "gradient", "layer": layer})
                    event.allocs = tuple(event.allocs) + (incoming,)
                    current = [self.add(event)]
                    releases.append(incoming.id)
                # Adam's operation count is an explicit analytic convention.
                # The stream slot prevents two local optimizers claiming full
                # CPU throughput simultaneously; DRAM still contends with DMA.
                memory = (self.size("gradient", self.shard_params) +
                          2 * self.size("master", self.shard_params) +
                          4 * self.size("optimizer", self.shard_params))
                current = [self.add(Event(prefix + ".update", tuple(current),
                    {self.scalar_resource(device): 14 * self.shard_params, self.pool(device): memory},
                    slots=(device + ".compute_stream",), releases=tuple(releases),
                    metadata={"phase": "optimizer", "category": "adam", "kernel": "adam", "flops": 14 * self.shard_params,
                              "layer": layer, "rank": device}))]
                transfer_dtype = self.precision["weight_transfer"] if self.offload else self.precision["weight"]
                output_size = self.shard_params * DTYPE_BYTES[transfer_dtype]
                output_id = prefix + ".updated_weights"
                output = self.allocation(output_id, device, output_size, "conversion_buffer", transfer_dtype)
                current = [self.add(Event(prefix + ".pack", tuple(current),
                    {self.scalar_resource(device): 2 * self.shard_params,
                     self.pool(device): self.size("master", self.shard_params) + output_size},
                    slots=(device + ".compute_stream",), allocs=(output,),
                    metadata={"phase": "optimizer", "category": "precision_conversion", "kernel": "weight_conversion", "layer": layer,
                              "source_precision": self.precision["master"], "destination_precision": transfer_dtype}))]
                if self.offload:
                    event = self.topo.transfer(prefix + ".H2D", host, rank, output_size, deps=tuple(current),
                        metadata={"phase": "optimizer", "category": "optimizer_offload", "direction": "H2D",
                                  "tensor": "weight", "layer": layer})
                    received = self.allocation(prefix + ".received_weights", rank, output_size,
                                               "conversion_buffer", transfer_dtype)
                    event.allocs = tuple(event.allocs) + (received,)
                    event.releases = tuple(event.releases) + (output_id,)
                    current = [self.add(event)]
                    output_id = received.id
                    scales = self.scale_bytes(self.shard_params, transfer_dtype)
                    if scales:
                        metadata_event = self.topo.transfer(prefix + ".H2D.scales", host, rank, scales,
                            deps=tuple(current), metadata={"phase": "optimizer", "category": "quantization_metadata",
                                                         "direction": "H2D", "tensor": "weight_scales", "layer": layer})
                        metadata_event.allocs = tuple(metadata_event.allocs) + (
                            self.allocation(metadata_event.id + ".src", host, scales, "quantization_metadata"),
                            self.allocation(metadata_event.id + ".dst", rank, scales, "quantization_metadata"))
                        metadata_event.releases = tuple(metadata_event.releases) + (metadata_event.id + ".src", metadata_event.id + ".dst")
                        current = [self.add(metadata_event)]
                # Refresh persistent storage; destination precision may differ
                # from wire precision. Includes a copy even if no conversion.
                refresh_work = {rank + ".hbm": output_size + self.size("weight", self.shard_params)}
                if transfer_dtype != self.precision["weight"]:
                    refresh_work[self.scalar_resource(rank)] = 2 * self.shard_params
                if self.features.get("extra_weight_copy_dtype"):
                    refresh_work[rank + ".hbm"] += self.shard_params * DTYPE_BYTES[self.features["extra_weight_copy_dtype"]]
                    refresh_work[self.scalar_resource(rank)] = refresh_work.get(self.scalar_resource(rank), 0) + 2 * self.shard_params
                refresh = self.add(Event(prefix + ".refresh", tuple(current), refresh_work,
                    slots=(rank + ".compute_stream",), releases=(output_id,),
                    metadata={"phase": "optimizer", "category": "weight_refresh", "kernel": "weight_refresh", "layer": layer}))
                finishes.append(refresh)
        end = self.barrier(f"train.step{step}.done", finishes,
                           metadata={"phase": "step_complete", "step": step})
        self.ends.append(end)

    def build(self) -> Graph:
        self.initial_state()
        for step in range(self.wl["training_steps"]):
            # Dependencies may reference later-added events. The engine checks
            # the complete DAG before execution.
            for stage in range(self.pp):
                for phase, micro in pipeline_schedule(stage, self.pp, self.microbatches):
                    self.phase(step, stage, phase, micro)
            self.optimizer(step)
        count = self.model["total_params"]
        logical = {"weights": self.size("weight", count), "gradients": self.size("gradient", count),
                   "master_weights": self.size("master", count), "optimizer_states": 2 * self.size("optimizer", count)}
        logical["total"] = sum(logical.values())
        self.graph.metadata.update({
            "scenario": "training", "logical_memory_bytes": logical,
            "step_end_events": self.ends, "training_steps": self.wl["training_steps"],
            "tokens_per_step": self.tokens * self.microbatches * self.dp,
            "tokens_per_replica_step": self.tokens * self.microbatches,
            "data_parallel_replicas": self.dp,
            "rank_coordinates": {rank: list(coord) for rank, coord in self.reverse.items()},
            "pipeline_schedule": {str(s): pipeline_schedule(s, self.pp, self.microbatches) for s in range(self.pp)},
            "approximations": [
                "Declared parameter-budget model: parameters and active work are uniform over layers and experts.",
                "EP dense weights are replicated; expert weights and token work are evenly distributed across EP ranks.",
                "MoE communication uses uniform routing and a continuous MoE-layer fraction; no expert imbalance is inferred.",
                "Activation ledger assumes streaming causal attention without a full attention-score matrix.",
                "Hidden boundaries are replicated over TP; saved activation elements/token/layer = 2H + intermediate/TP*(1 + MoE_fraction*top_k); checkpoint retains H.",
                "Activation/wire dtype differences add explicit packing/unpacking and retained communication buffers; FSDP tensors retain their own weight/gradient dtype.",
                "Forward FLOPs = 2*active_parameters*tokens plus causal attention; backward = 2*forward.",
                "Adam update uses 14 FLOPs/parameter; precision conversion uses 2 FLOPs/parameter; calibrate to measured kernels.",
                "NPU scalar updates/conversions use the FP32 profile and share a stream with matrix compute; if unavailable, use selected compute throughput.",
                "Optimizer starts after all backward phases; optimizer/backward overlap is not modeled.",
                "FP8 scale metadata uses continuous block counts; tensor alignment/padding requires profile calibration.",
                "Collective staging and precision-conversion buffers are additional to logical tensor capacity.",
            ],
        })
        return self.graph


def build_training(config: dict, topology: Topology) -> Graph:
    """Build a training graph from a validated, normalized configuration."""
    return _TrainingBuilder(config, topology).build()

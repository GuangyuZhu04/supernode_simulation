"""Independent numerical anchors and execution invariants for training."""
import pytest

from supernode_sim.config import Config, deep_merge
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from supernode_sim.training import build_training, pipeline_schedule


def configuration(patch=None):
    raw = {
        "model": {"name": "synthetic-test", "total_params": 1e6, "active_params": 1e6,
                  "layers": 2, "hidden_dim": 128, "intermediate_dim": 512},
        "hardware": {"npu_count": 1, "su_size": 1,
                     "npu": {"runtime_bytes": 0, "workspace_bytes": 0}},
        "workload": {"scenario": "training", "batch_size": 1,
                     "sequence_length": 32, "gradient_accumulation": 2},
        "execution": {"tp": 1, "pp": 1, "ep": 1, "dp": 1, "fsdp": 1,
                      "precision": {"weight": "fp16", "compute": "fp16", "gradient": "fp16",
                                    "master": "fp32", "optimizer": "fp32", "weight_transfer": "fp16"}},
    }
    return Config.model_validate(deep_merge(raw, patch or {})).model_dump()


def compile_config(config):
    return build_training(config, Topology(config["hardware"], config["execution"]["precision"]))


@pytest.mark.parametrize("dtype,weight,total", [("fp8", 10e12, 150e12), ("fp16", 20e12, 160e12)])
def test_10t_precision_anchors(dtype, weight, total):
    config = configuration({"model": {"total_params": 1e13, "active_params": 1e13},
                            "execution": {"precision": {"weight": dtype}}})
    ledger = compile_config(config).metadata["logical_memory_bytes"]
    assert ledger == {"weights": weight, "gradients": 20e12, "master_weights": 40e12,
                      "optimizer_states": 80e12, "total": total}


@pytest.mark.parametrize("stored,wire,expected", [("fp8", "fp8", 10e12), ("fp16", "fp16", 20e12),
                                                 ("fp8", "fp16", 20e12), ("fp16", "fp8", 10e12)])
def test_10t_offload_wire_precision_is_independent(stored, wire, expected):
    config = configuration({"model": {"total_params": 1e13, "active_params": 1e13},
        "hardware": {"npu_count": 2, "su_size": 2},
        "execution": {"dp": 2, "fsdp": 2, "features": {"optimizer_offload": True},
                      "precision": {"weight": stored, "weight_transfer": wire}}})
    graph = compile_config(config)
    payloads = {direction: sum(e.metadata.get("payload_bytes", 0) for e in graph.events
        if e.metadata.get("direction") == direction and e.metadata.get("category") == "optimizer_offload")
        for direction in ("D2H", "H2D")}
    assert payloads == {"D2H": 20e12, "H2D": expected}
    cpu_states = sum(a.size_bytes for a in graph.initial_allocations
                     if a.pool.endswith(".dram") and a.category in ("master_weights", "optimizer_states"))
    assert cpu_states == 120e12
    if stored != wire:
        assert any(e.metadata.get("category") == "weight_refresh" and
                   any("compute" in resource for resource in e.work) for e in graph.events)


def test_dense_ep_replication_is_not_divided_by_total_cards():
    config = configuration({"model": {"num_experts": 8, "top_k": 2,
        "moe_layer_fraction": .5, "expert_parameter_fraction": .8},
        "hardware": {"npu_count": 4, "su_size": 4}, "execution": {"ep": 2, "dp": 2, "fsdp": 2}})
    graph = compile_config(config)
    weights = sum(a.size_bytes for a in graph.initial_allocations if a.category == "weights")
    # Dense 20% is replicated on each EP rank, expert 80% is sharded.
    assert weights == pytest.approx(2 * 1e6 * (2 * .2 + .8))
    assert len([e for e in graph.events if e.metadata.get("semantic") == "dispatch"]) > 0


def test_fsdp_reshard_controls_forward_and_backward_gathers():
    base = configuration({"hardware": {"npu_count": 2, "su_size": 2},
                          "execution": {"dp": 2, "fsdp": 2}})
    reshard = compile_config(base)
    base["execution"]["reshard_after_forward"] = False
    retained = compile_config(base)
    starts = lambda g: [e for e in g.events if e.id.endswith(".gather.start")]
    assert len(starts(reshard)) == 2 * 2 * 2  # layers * microbatches * F/B
    assert len(starts(retained)) == 2
    for graph in (reshard, retained):
        result = simulate(graph)
        assert result["status"] == "feasible", result["diagnostics"]
        initial = {pool: sum(a.size_bytes for a in graph.initial_allocations if a.pool == pool)
                   for pool in graph.memory_caps}
        assert {p: result["memory_final_bytes"].get(p, 0) for p in initial} == pytest.approx(initial)


def test_gradient_accumulation_changes_sync_count_without_changing_update_payload():
    base = configuration({"hardware": {"npu_count": 2, "su_size": 2},
                          "execution": {"dp": 2, "fsdp": 2}})
    accumulated = compile_config(base)
    base["execution"]["sync_each_microbatch"] = True
    per_micro = compile_config(base)
    count = lambda g: sum(1 for e in g.events if e.metadata.get("category") == "fsdp_gradients"
                         and e.metadata.get("payload_bytes", 0))
    assert count(per_micro) == 2 * count(accumulated)
    assert simulate(per_micro)["status"] == "feasible"


def test_noninterleaved_1f1b_has_valid_pipeline_dependencies():
    assert pipeline_schedule(0, 3, 4) == [("forward", 0), ("forward", 1), ("forward", 2),
        ("backward", 0), ("forward", 3), ("backward", 1), ("backward", 2), ("backward", 3)]
    config = configuration({"model": {"layers": 3}, "hardware": {"npu_count": 3, "su_size": 3},
                            "workload": {"gradient_accumulation": 4}, "execution": {"pp": 3}})
    graph = compile_config(config)
    result = simulate(graph)
    assert result["status"] == "feasible", result["diagnostics"]
    assert result["completed_events"] == len(graph.events)
    trace = {row["id"]: row for row in result["trace"]}
    for event in graph.events:
        for dep in event.deps:
            assert trace[event.id]["start_s"] >= trace[dep]["end_s"] - 1e-12


def test_checkpointing_trades_saved_activation_for_recomputation():
    base = configuration({"model": {"layers": 4, "hidden_dim": 2048, "intermediate_dim": 8192},
                          "workload": {"sequence_length": 2048}})
    ordinary = compile_config(base)
    base["execution"]["features"]["activation_checkpointing"] = True
    checkpoint = compile_config(base)
    saved = lambda g: sum(a.size_bytes for e in g.events for a in e.allocs
                         if a.category in ("activations", "activation_checkpoint"))
    assert saved(checkpoint) < saved(ordinary)
    assert any(e.metadata.get("phase") == "recompute" for e in checkpoint.events)
    assert simulate(checkpoint)["status"] == "feasible"


def test_prefetch_is_bounded_to_one_layer_ahead():
    config = configuration({"model": {"layers": 4}, "hardware": {"npu_count": 2, "su_size": 2},
                            "execution": {"dp": 2, "fsdp": 2, "prefetch": True}})
    graph = compile_config(config)
    event = next(e for e in graph.events if e.id == "train.s0.p0.forward.m0.l2.gather.start")
    assert "train.s0.p0.forward.m0.l0.done" in event.deps
    assert simulate(graph)["status"] == "feasible"


def test_checkpoint_recompute_repeats_expert_and_tensor_parallel_communication():
    config = configuration({"model": {"num_experts": 8, "top_k": 2,
        "moe_layer_fraction": .5, "expert_parameter_fraction": .8},
        "hardware": {"npu_count": 4, "su_size": 4}, "execution": {"ep": 2, "tp": 2}})
    ordinary = compile_config(config)
    config["execution"]["features"]["activation_checkpointing"] = True
    checkpoint = compile_config(config)
    for category in ("dispatch", "combine", "tensor_parallel"):
        payload = lambda g: sum(e.metadata.get("payload_bytes", 0) for e in g.events
            if e.metadata.get("semantic") == category or e.metadata.get("category") == category)
        assert payload(checkpoint) == pytest.approx(1.5 * payload(ordinary))
    assert simulate(checkpoint)["status"] == "feasible"


def test_rank_placement_multiple_steps_are_reproducible_and_release_transients():
    config = configuration({"hardware": {"npu_count": 4, "su_size": 2},
        "workload": {"training_steps": 2},
        "execution": {"pp": 2, "dp": 2, "fsdp": 2, "rank_order": [3, 1, 2, 0],
                      "features": {"optimizer_offload": True, "activation_checkpointing": True},
                      "precision": {"weight": "fp8", "weight_transfer": "fp16"}}})
    graph = compile_config(config)
    result = simulate(graph)
    assert result["status"] == "feasible", result["diagnostics"]
    assert result == simulate(graph)
    assert graph.metadata["rank_coordinates"]["npu3"] == [0, 0, 0, 0]
    initial = {pool: sum(a.size_bytes for a in graph.initial_allocations if a.pool == pool)
               for pool in graph.memory_caps}
    assert {p: result["memory_final_bytes"].get(p, 0) for p in initial} == pytest.approx(initial)


def test_activation_wire_conversion_covers_tp_ep_pp_and_releases_buffers():
    config = configuration({"model": {"num_experts": 8, "top_k": 2,
        "moe_layer_fraction": .5, "expert_parameter_fraction": .8},
        "hardware": {"npu_count": 8, "su_size": 8}, "execution": {"ep": 2, "tp": 2, "pp": 2}})
    full_precision = compile_config(config)
    config["execution"]["precision"]["communication"] = "fp8"
    compressed = compile_config(config)
    for category in ("dispatch", "combine", "tensor_parallel", "pipeline_activation"):
        payload = lambda g: sum(e.metadata.get("payload_bytes", 0) for e in g.events
            if e.metadata.get("semantic") == category or e.metadata.get("category") == category)
        assert payload(compressed) == pytest.approx(payload(full_precision) / 2)
    conversion_kinds = {e.metadata.get("conversion") for e in compressed.events}
    assert {"communication_pack", "communication_unpack", "pipeline_pack", "pipeline_unpack"} <= conversion_kinds
    assert not any(e.metadata.get("conversion") for e in full_precision.events)
    result = simulate(compressed)
    assert result["status"] == "feasible", result["diagnostics"]
    initial = {pool: sum(a.size_bytes for a in compressed.initial_allocations if a.pool == pool)
               for pool in compressed.memory_caps}
    assert {p: result["memory_final_bytes"].get(p, 0) for p in initial} == pytest.approx(initial)
    trace = {row["id"]: row for row in result["trace"]}
    for event in compressed.events:
        if event.metadata.get("conversion"):
            assert event.slots
            assert any("compute" in resource for resource in event.work)
            for dep in event.deps:
                assert trace[event.id]["start_s"] >= trace[dep]["end_s"] - 1e-12


def test_hidden_activations_replicate_across_tp_boundaries():
    config = configuration({"model": {"num_experts": 8, "top_k": 2,
        "moe_layer_fraction": .5, "expert_parameter_fraction": .8},
        "hardware": {"npu_count": 8, "su_size": 8}, "execution": {"ep": 2, "tp": 2, "pp": 2}})
    graph = compile_config(config)
    first_pp = next(e for e in graph.events if e.metadata.get("category") == "pipeline_activation")
    assert first_pp.metadata["payload_bytes"] == 32 * 128 / 2 * 2  # tokens * H / EP * FP16
    first_dispatch = next(e for e in graph.events if e.metadata.get("semantic") == "dispatch")
    assert first_dispatch.metadata["payload_bytes"] == 32 * 128 / 2 * 2 * .5 * 2 / 2
    config["execution"]["features"]["activation_checkpointing"] = True
    checkpoint = compile_config(config)
    activation = next(a for e in checkpoint.events for a in e.allocs if a.category == "activation_checkpoint")
    assert activation.size_bytes == 32 * 128 / 2 * 2

"""Execution and memory-lifetime checks for analytical CSA/HCA training."""
from collections import Counter

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config
from tests.test_sparse_attention import sparse_model


def training_config(kind="CSA", checkpoint=False, tp=1, ep=1):
    model = sparse_model(kind)
    model["layers"] = 3
    model["architecture"]["attention_by_layer"] = [kind] * 3
    model["architecture"]["ffn_by_layer"] = ["Dense"] * 3
    if ep > 1:
        model["num_experts"] = ep
        model["top_k"] = 1
        model["architecture"]["ffn_by_layer"] = ["MoE"] * 3
        model["architecture"]["moe"] = {"num_experts": ep, "top_k": 1, "intermediate_dim": 32}
    return Config.model_validate({
        "model": model,
        "hardware": {"npu_count": tp*ep, "su_size": tp*ep, "npus_per_host": tp*ep,
                     "npu": {"hbm_bytes": 1e8, "runtime_bytes": 0, "workspace_bytes": 0}},
        "workload": {"scenario": "training", "batch_size": 2, "sequence_length": 32,
                     "gradient_accumulation": 2, "training_steps": 2},
        "execution": {"tp": tp, "ep": ep, "features": {"activation_checkpointing": checkpoint}},
    })


@pytest.mark.parametrize("kind", ["CSA", "HCA"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_sparse_training_executes_and_every_temporary_state_is_released(kind, checkpoint):
    config = training_config(kind, checkpoint)
    graph = compile_graph(config)
    result = run(config)
    assert result["status"] == "feasible", result.get("simulation", {}).get("diagnostics")
    releases = Counter(aid for event in graph.events for aid in event.releases)
    categories = {"compressed_attention_pool", "attention_selection_indices", "indexer_score_workspace"}
    relevant = [(event, allocation) for event in graph.events for allocation in event.allocs
                if allocation.category in categories or allocation.metadata.get("storage_role") in categories]
    assert relevant
    for event, allocation in relevant:
        assert releases[allocation.id] == 1
        if allocation.category == "attention_selection_indices" or allocation.metadata.get("storage_role") == "attention_selection_indices":
            assert allocation.dtype is None
            assert allocation.metadata["storage_dtype"] == "int32"
        if checkpoint and event.metadata.get("phase") == "forward":
            assert allocation.id in event.releases
    assert any("indexer auxiliary loss" in note for note in result["assumptions"])
    if kind == "HCA":
        assert not any(a.category == "attention_selection_indices" for _, a in relevant)


def test_checkpoint_rebuilds_sparse_state_and_reduces_memory_with_more_work():
    ordinary, checkpointed = training_config(), training_config(checkpoint=True)
    original_graph, checkpoint_graph = compile_graph(ordinary), compile_graph(checkpointed)
    original, checkpoint = run(ordinary), run(checkpointed)
    original_flops = sum(e.metadata.get("flops", 0) for e in original_graph.events)
    checkpoint_flops = sum(e.metadata.get("flops", 0) for e in checkpoint_graph.events)
    assert checkpoint_flops > original_flops
    assert checkpoint["simulation"]["memory_peaks_bytes"]["npu0.hbm"] < original["simulation"]["memory_peaks_bytes"]["npu0.hbm"]
    rebuilt = [(e, a) for e in checkpoint_graph.events for a in e.allocs
               if a.metadata.get("storage_role") == "compressed_attention_pool"]
    assert rebuilt
    for event, allocation in rebuilt:
        assert event.metadata["phase"] == "recompute"
        assert allocation.id not in event.releases
        freeing = [e for e in checkpoint_graph.events if allocation.id in e.releases]
        assert len(freeing) == 1 and freeing[0].id.endswith(".done")


def test_csa_score_collective_reappears_in_recompute_and_respects_ep_partition():
    config = training_config(checkpoint=True, tp=2, ep=2)
    graph = compile_graph(config)
    score_events = [e for e in graph.events if ".index_scores.tp" in e.id]
    assert score_events
    assert any(".recompute." in e.id for e in score_events)
    assert not any(".backward." in e.id and ".recompute." not in e.id for e in score_events)
    result = run(config)
    assert result["status"] == "feasible", result.get("simulation", {}).get("diagnostics")
    pools = [a for e in graph.events for a in e.allocs if a.category == "compressed_attention_pool"]
    # Two source requests / EP2; eight compressed rows, main8 + index4, fp16.
    assert pools and all(a.size_bytes == 1*8*(8+4)*2 for a in pools)

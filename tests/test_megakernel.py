"""Fusion preserves dependencies/work/lifetimes and reports actual coverage."""
import pytest

from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.megakernel import (apply_megakernel, megakernel_summary,
                                      merge_megakernel_summaries)
from supernode_sim.calibration import apply_kernel_profiles


def graph_for(prefix=""):
    ids = lambda value: prefix + value
    slot = ("npu0.compute_stream",)
    meta = lambda name: {"kernel": name, "component": name.upper(), "layer": 0, "phase": "prefill"}
    return Graph(resources={
        "npu0.compute": Resource("npu0.compute", 1000, "FLOP/s", "compute"),
        "npu0.hbm": Resource("npu0.hbm", 1000, "byte/s")}, memory_caps={"npu0.hbm": 1000},
        initial_allocations=[Allocation(ids("old"), "npu0.hbm", 10)], events=[
            Event(ids("seed")),
            Event(ids("a"), (ids("seed"),), {"npu0.compute": 100, "npu0.hbm": 100}, .01, slot,
                  (Allocation(ids("internal_a"), "npu0.hbm", 60, "activation"),
                   Allocation(ids("saved_a"), "npu0.hbm", 20, "activation")), (ids("old"),), meta("a")),
            Event(ids("bridge"), (ids("a"),)),
            Event(ids("b"), (ids("bridge"),), {"npu0.compute": 200, "npu0.hbm": 200}, .02, slot,
                  (Allocation(ids("internal_b"), "npu0.hbm", 40, "activation"),
                   Allocation(ids("saved_b"), "npu0.hbm", 20, "activation")),
                  (ids("internal_a"), ids("internal_b")), meta("b")),
            Event(ids("backward"), (ids("b"),), {"npu0.compute": 50}, 0, slot, (),
                  (ids("saved_a"), ids("saved_b")), {"kernel": "backward", "layer": 0, "phase": "backward"}),
        ])


def config_for(**overrides):
    group = {"name": "ab", "kernels": ["a", "b"], "phases": ["prefill"],
             "fused_startup_s": .001, "saved_hbm_bytes": 80, "workspace_bytes": 5,
             "source": {"status": "assumed", "description": "synthetic arithmetic fixture"}, **overrides}
    return {"execution": {"precision": {"compute": "fp16"},
            "features": {"megakernel": {"enabled": True, "groups": [group]}}},
            "hardware": {"npu": {"flops": {"fp16": 2000}}, "kernel_profiles": {}}}


def test_fusion_conserves_compute_reduces_only_bounded_hbm_and_retains_saved_state():
    original, graph = graph_for(), graph_for()
    summary = apply_megakernel(graph, config_for())
    assert summary["matched_groups"] == 1 and summary["coverage_fraction"] == 1
    assert [event.id for event in graph.events] == ["seed", "b", "backward"]
    fused = graph.events[1]
    assert fused.metadata["kernel"] == "megakernel:ab"
    assert fused.deps == ("seed",) and graph.events[-1].deps == ("b",)
    assert fused.work["npu0.compute"] == 300
    assert fused.work["npu0.hbm"] == 220
    assert fused.startup_s == .001
    assert set(fused.releases) == {"old", "internal_a", "internal_b", "b.megakernel_workspace"}
    assert {a.id for a in fused.allocs} >= {"saved_a", "saved_b"}
    before, after = simulate(original), simulate(graph)
    assert before["status"] == after["status"] == "feasible"
    assert after["memory_final_bytes"]["npu0.hbm"] == 0
    assert after["memory_peaks_bytes"]["npu0.hbm"] >= before["memory_peaks_bytes"]["npu0.hbm"]
    assert after["duration_s"] < before["duration_s"]


def test_no_implicit_overlap_speedup_without_explicit_profile():
    graph = graph_for()
    graph.events[1].work = {"npu0.compute": 1000, "npu0.hbm": 0}
    graph.events[3].work = {"npu0.compute": 0, "npu0.hbm": 1000}
    apply_megakernel(graph, config_for(saved_hbm_bytes=0, fused_startup_s=0))
    fused = graph.events[1]
    # Sum of serial compute-only + memory-only stages, not max of totals.
    assert fused.work["kernel_limit:b"] == 2


@pytest.mark.parametrize("mutation", ["fanout", "fanin", "collective", "phase", "rank", "metadata"])
def test_unsafe_boundaries_are_skipped_and_never_claimed_as_matches(mutation):
    graph = graph_for()
    if mutation == "fanout":
        graph.events.append(Event("observer", ("a",)))
    elif mutation == "fanin":
        graph.events[3].deps += ("seed",)
    elif mutation == "collective":
        graph.events[3].metadata["collective"] = "all_reduce"
    elif mutation == "phase":
        graph.events[3].metadata["phase"] = "decode"
    elif mutation == "rank":
        graph.events[3].work = {"npu1.compute": 200}
    else:
        graph.metadata["observed_event"] = "a"
    expected_ids = [e.id for e in graph.events]
    result = apply_megakernel(graph, config_for())
    assert result["matched_events"] == result["matched_groups"] == 0
    assert result["skipped_candidates"] == 1 and result["skip_reasons"]
    assert [e.id for e in graph.events] == expected_ids


def test_hbm_savings_cannot_remove_unrelated_weight_or_cache_traffic():
    with pytest.raises(ValueError, match="bounded internal traffic"):
        apply_megakernel(graph_for(), config_for(saved_hbm_bytes=201))


def test_explicit_fused_duration_can_improve_utilization_but_not_exceed_physical_peak():
    graph = graph_for()
    apply_megakernel(graph, config_for(saved_hbm_bytes=200, duration_s=.16))
    event = graph.events[1]
    assert event.work["npu0.compute.megakernel.fp16"] == 300
    assert graph.resources["npu0.compute"].capacity == 1000
    assert graph.resources["npu0.compute.megakernel.fp16"].capacity == 2000
    assert event.work["kernel_limit:b"] == .16
    with pytest.raises(ValueError, match="physical"):
        apply_megakernel(graph_for(), config_for(saved_hbm_bytes=200, duration_s=.14))


def test_member_profiles_are_suppressed_and_fused_profile_replaces_not_adds_timer():
    graph, config = graph_for(), config_for()
    config["hardware"]["kernel_profiles"] = {
        "a": [{"flops": 100, "memory_bytes": 100, "duration_s": .1}],
        "megakernel:ab": [{"flops": 300, "memory_bytes": 220, "duration_s": .23}],
    }
    summary = apply_megakernel(graph, config)
    assert summary["suppressed_member_profiles"] == ["a"]
    assert graph.events[1].work["kernel_limit:b"] == pytest.approx(.3)
    apply_kernel_profiles(graph, config)
    assert graph.events[1].work["kernel_limit:b"] == .23
    assert graph.metadata["kernel_calibration"]["matched_events"] == 1


def test_pd_prefixed_ids_and_coverage_merge_are_supported():
    graph = graph_for("decode_pool.worker1.")
    summary = apply_megakernel(graph, config_for())
    assert summary["matched_groups"] == 1
    assert graph.events[1].id == "decode_pool.worker1.b"
    aggregate = megakernel_summary()
    merge_megakernel_summaries(aggregate, summary)
    merge_megakernel_summaries(aggregate, summary)
    assert aggregate["matched_groups"] == 2 and aggregate["coverage_fraction"] == 1
    assert aggregate["groups"][0]["matched_events"] == 4


def test_real_structured_chain_is_transformed_not_a_nominal_flag():
    from tests.test_component_integration import shape_config
    from supernode_sim.analysis import compile_graph
    spec = shape_config("inference")
    graph, config = compile_graph(spec), spec.model_dump()
    config["execution"]["features"]["megakernel"] = {"enabled": True, "groups": [
        {"name": "norm_attention", "kernels": ["prenorm", "gqa"],
         "phases": ["prefill", "decode"], "fused_startup_s": 0, "saved_hbm_bytes": 16}]}
    result = apply_megakernel(graph, config)
    assert result["matched_groups"] == 2 and result["matched_events"] == 4
    assert simulate(graph)["status"] == "feasible"


@pytest.mark.parametrize("checkpoint", [False, True])
def test_actual_training_saved_activation_and_recompute_lifetimes_survive_fusion(checkpoint):
    from tests.test_component_integration import shape_config
    from supernode_sim.analysis import compile_graph
    spec = shape_config("training", {"execution": {"features": {"activation_checkpointing": checkpoint}}})
    graph, config = compile_graph(spec), spec.model_dump()
    before = simulate(graph)
    config["execution"]["features"]["megakernel"] = {"enabled": True, "groups": [
        {"name": "training_norm_attention", "kernels": ["prenorm", "gqa"],
         "phases": ["forward", "recompute"], "fused_startup_s": 0, "saved_hbm_bytes": 0}]}
    summary = apply_megakernel(graph, config)
    after = simulate(graph)
    assert summary["matched_groups"] == (4 if checkpoint else 2)
    assert before["status"] == after["status"] == "feasible"
    assert before["memory_final_bytes"] == after["memory_final_bytes"]
    releases = [aid for event in graph.events for aid in event.releases]
    assert all(releases.count(a.id) == 1 for e in graph.events for a in e.allocs)

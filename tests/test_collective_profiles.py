"""Collective timing fits preserve actual traffic and physical constraints."""
from copy import deepcopy

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.collective_calibration import summarize_collective_calibration
from supernode_sim.config import HardwareSpec
from supernode_sim.core import Event, Graph
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import window_breakdown
from supernode_sim.streaming import run_streaming
from supernode_sim.topology import Topology
from tests.test_structured import configuration
from tests.test_topology import hardware


def profile(**overrides):
    return dict(id="measured-dispatch", collective="all_to_all_v", semantic="dispatch",
                phase="decode", world_size=2, dtype="fp8", algorithm="direct",
                timing_scope="transport", total_payload_bytes=200,
                send_payload_bytes=[100, 100], recv_payload_bytes=[100, 100],
                duration_s=2, source="Synthetic timing test; not a hardware benchmark",
                hardware_name="profile-test", **overrides)


def build(profiles=(), *, latency=0, prefix="dispatch", params=None,
          metadata=None, deps=(), graph=None, topology=None):
    if topology is None:
        params = params or hardware(npu_count=2, su_size=2, name="profile-test")
        params["collective_profiles"] = list(profiles)
        params["scale_up"]["latency_s"] = latency
        topology = Topology(params, {"communication": "fp8"})
        graph = graph or Graph()
        topology.populate(graph)
    ends = topology.collective(
        graph, prefix, "all_to_all_v", ["npu0", "npu1"], 0, deps,
        payload_matrix=[[999, 100], [100, 999]],  # Local values are excluded.
        metadata=metadata or {"operation": "dispatch", "phase": "decode", "wire_dtype": "fp8"},
        algorithm="direct")
    return topology, graph, ends


def test_measured_call_time_runs_beside_transfer_and_includes_startup():
    _, original, _ = build(latency=.2)
    _, calibrated, ends = build([profile()], latency=.2)
    baseline, result = simulate(original), simulate(calibrated)
    assert baseline["duration_s"] == pytest.approx(1.2)
    assert result["duration_s"] == pytest.approx(2)  # Not 2 + 1.2.
    assert result["resource_work"] == baseline["resource_work"]
    assert result["memory_peaks_bytes"] == baseline["memory_peaks_bytes"]
    assert sum(e.metadata.get("payload_bytes", 0) for e in calibrated.events) == 200
    summary = summarize_collective_calibration(calibrated, result)
    assert summary["matched_collectives"] == summary["profile_limited_collectives"] == 1
    assert summary["physical_limited_collectives"] == 0
    breakdown = window_breakdown(calibrated, result, ends[0])
    assert breakdown["components_s"]["communication_other_s"] == pytest.approx(2)
    assert breakdown["components_s"]["compute_s"] == 0
    assert breakdown["components_s"]["synchronization_wait_s"] == 0


def test_physical_service_and_contention_are_not_bypassed_by_short_profile():
    p = profile()
    p["duration_s"] = .5
    topology, graph, _ = build([p])
    _, _, ends = build(prefix="second", topology=topology, graph=graph)
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(2)  # Two flows share each link.
    assert result["resource_work"]["npu0.SU.tx"] == 200
    summary = summarize_collective_calibration(graph, result)
    assert summary["matched_collectives"] == 2
    assert summary["physical_limited_collectives"] == 2


@pytest.mark.parametrize("field,value", [
    ("phase", "prefill"), ("semantic", "combine"), ("dtype", "bf16"),
    ("algorithm", "pairwise"), ("world_size", 4),
    ("hardware_name", "another-machine"), ("timing_scope", "activation_collective"),
    ("send_payload_bytes", [50, 150]), ("recv_payload_bytes", [150, 50]),
])
def test_profile_does_not_match_a_different_call(field, value):
    p = profile()
    p[field] = value
    _, graph, _ = build([p])
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(1)
    summary = summarize_collective_calibration(graph, result)
    assert summary["matched_collectives"] == 0
    reason = "hardware_mismatch" if field == "hardware_name" else "no_exact_match"
    assert summary["unmatched_reasons"] == {reason: 1}


def test_timer_starts_after_original_dependencies():
    params = hardware(npu_count=2, su_size=2, name="profile-test")
    params["collective_profiles"] = [profile()]
    topology, graph = Topology(params, {"communication": "fp8"}), Graph()
    topology.populate(graph)
    graph.events.append(Event("preceding", startup_s=3))
    build(topology=topology, graph=graph, deps=("preceding",))
    assert simulate(graph)["duration_s"] == pytest.approx(5)


def test_ambiguous_matching_profiles_fail():
    with pytest.raises(ValueError, match="Ambiguous collective calibration"):
        build([profile(), {**profile(), "id": "duplicate-point"}])


@pytest.mark.parametrize("patch", [
    {"source": ""}, {"duration_s": 0}, {"duration_s": float("inf")},
    {"send_payload_bytes": [100]}, {"recv_payload_bytes": [100, -100]},
    {"send_payload_bytes": [50, 50]}, {"total_payload_bytes": float("nan")},
    {"world_size": True}, {"collective": "all_gather", "algorithm": "direct"},
])
def test_profile_schema_rejects_invalid_calibration(patch):
    params = hardware(npu_count=2, su_size=2, name="profile-test")
    params["collective_profiles"] = [{**profile(), **patch}]
    with pytest.raises(ValueError):
        HardwareSpec.model_validate(params)


def test_activation_call_boundary_prevents_recharging_pack_and_unpack():
    p = {**profile(), "duration_s": 4, "timing_scope": "activation_collective"}
    params = hardware(npu_count=2, su_size=2, name="profile-test",
                      collective_profiles=[p])
    topology, graph = Topology(params, {"communication": "fp8"}), Graph()
    topology.populate(graph)
    graph.events.append(Event("pack", startup_s=1))
    wire_meta = {"phase": "decode", "operation": "dispatch", "wire_dtype": "fp8"}
    matrix = [[0, 100], [100, 0]]
    network = topology.collective(graph, "comm", "all_to_all_v",
                                  ["npu0", "npu1"], 0, ("pack",),
                                  payload_matrix=matrix, metadata=wire_meta,
                                  algorithm="direct", apply_profiles=False)
    graph.events.append(Event("unpack", deps=network, startup_s=1))
    topology.complete_collective_profile(
        graph, "comm", "all_to_all_v", ["npu0", "npu1"], 0,
        deps=(), terminals=("unpack",), payload_matrix=matrix,
        metadata=wire_meta, algorithm="direct", timing_scope="activation_collective")
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(4)  # max(1 + 1 + 1, 4).
    assert summarize_collective_calibration(graph, result)["matched_collectives"] == 1


def test_normal_and_streaming_cached_replay_preserve_collective_coverage():
    patch = {
        "workload": {"scenario": "inference", "output_length": 3},
        "model": {"layers": 6, "architecture": {
            "attention_by_layer": ["MLA", "KDA", "MLA", "KDA", "MLA", "KDA"],
            "ffn_by_layer": ["Dense", "MoE", "MoE", "MoE", "MoE", "MoE"],
            "attnres_block_size": 2}},
        "hardware": {"name": "profile-test", "npu_count": 2, "su_size": 2},
        "execution": {"ep": 2, "precision": {"communication": "fp8"}},
    }
    original = compile_graph(configuration(patch))
    signature = next(e.metadata["collective_calibration"]["signature"] for e in original.events
                     if e.metadata.get("collective_calibration", {}).get("signature", {}).get("semantic") == "dispatch"
                     and e.metadata["collective_calibration"]["signature"]["phase"] == "decode")
    patch["hardware"]["collective_profiles"] = [{
        **deepcopy(signature), "id": "decoded-moe", "duration_s": .001,
        "source": "Synthetic call wall-time for regression testing", "hardware_name": "profile-test",
    }]
    config = configuration(patch)
    ordinary = run(config)
    cache = {}
    streamed = run_streaming(config, cache=cache)
    replayed = run_streaming(config, cache=cache)
    assert ordinary["simulation"]["status"] == streamed["status"] == "feasible"
    assert streamed["metrics"]["duration_s"] == pytest.approx(ordinary["simulation"]["duration_s"])
    expected = ordinary["model_metadata"]["collective_calibration"]
    assert expected["matched_collectives"] > 0
    assert streamed["model_metadata"]["collective_calibration"] == expected
    assert replayed["model_metadata"]["collective_calibration"] == expected
    assert any(c["timing_cache_hit"] for c in replayed["chunks"])
    assert any(c["component_macro_hit"] for c in streamed["chunks"])
    assert ordinary["model_metadata"]["fidelity"]["hardware_performance"] == "partially_collective_calibrated"
    assert ordinary["model_metadata"]["fidelity"]["serving_performance"] == "not_validated"

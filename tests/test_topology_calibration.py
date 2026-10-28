"""Guard collective timing fits against topology, physical rank and peer changes."""

from copy import deepcopy

import pytest

from supernode_sim.collective_calibration import (
    collective_applicability_signature, match_collective_profile,
    summarize_collective_calibration,
)
from supernode_sim.config import HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from tests.test_topology import hardware


RANKS = [f"npu{index}" for index in range(4)]
MATRIX = [[0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100], [100, 0, 0, 0]]
PRECISION = {"compute": "bf16", "communication": "fp8"}


def params():
    return hardware(npu_count=4, su_size=2, name="same-machine-name")


def fingerprint(config, ranks=RANKS, matrix=MATRIX, precision=PRECISION):
    return collective_applicability_signature(config, ranks, "all_to_all_v", 0, matrix, precision=precision)


def make_profile(config, guarded=True):
    profile = {
        "id": "guarded-test", "collective": "all_to_all_v", "semantic": "dispatch", "phase": "decode",
        "world_size": 4, "dtype": "fp8", "algorithm": "direct", "timing_scope": "transport",
        "total_payload_bytes": 400, "send_payload_bytes": [100] * 4, "recv_payload_bytes": [100] * 4,
        "duration_s": 10, "source": "synthetic regression fixture", "hardware_name": config["name"],
    }
    if guarded:
        profile["applicability_signature"] = fingerprint(config)
    return profile


def build(config, ranks=RANKS, matrix=MATRIX):
    topology, graph = Topology(config, PRECISION), Graph()
    topology.populate(graph)
    topology.collective(graph, "dispatch", "all_to_all_v", ranks, 0, payload_matrix=matrix,
                        algorithm="direct", metadata={"phase": "decode", "operation": "dispatch", "wire_dtype": "fp8"})
    return graph


def test_guarded_profile_matches_without_changing_physical_work_or_algorithm_time():
    config = params()
    baseline = simulate(build(config))
    config["collective_profiles"] = [make_profile(config)]
    graph = build(config)
    result = simulate(graph)
    summary = summarize_collective_calibration(graph, result)
    assert summary["matched_collectives"] == summary["guarded_matched_collectives"] == 1
    assert summary["unguarded_matched_collectives"] == 0
    assert result["resource_work"] == baseline["resource_work"]
    assert result["duration_s"] == pytest.approx(max(10, baseline["duration_s"]))


@pytest.mark.parametrize("patch", [
    {"scale_out": {"bandwidth_Bps": 80}},
    {"scale_out": {"host_injection_Bps": 200}},
    {"scale_up": {"bandwidth_Bps": 80}},
    {"host_link": {"bandwidth_Bps": 80}},
    {"npu": {"hbm_Bps": 1e10}},
    {"npu": {"flops": {"bf16": 1e12}}},
    {"npu": {"efficiency": .4}},
    {"npus_per_host": 4},
    {"routing_algorithm": "ecmp_hash"},
    {"routing_seed": 123},
    {"kernel_profiles": {"activation.pack": [{"flops": 0, "memory_bytes": 100, "duration_s": 1}]}},
])
def test_same_name_different_effective_hardware_rejects_old_profile(patch):
    config = params()
    config["collective_profiles"] = [make_profile(config)]
    for key, value in patch.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    graph = build(config)
    summary = summarize_collective_calibration(graph)
    assert summary["matched_collectives"] == 0
    assert summary["unmatched_reasons"] == {"applicability_mismatch": 1}


def test_full_peer_matrix_and_physical_rank_order_are_guarded_beyond_vectors():
    config = params()
    config["collective_profiles"] = [make_profile(config)]
    transposed = [list(row) for row in zip(*MATRIX)]
    assert [sum(row) for row in transposed] == [sum(row) for row in MATRIX]
    for ranks, matrix in ((RANKS, transposed), (list(reversed(RANKS)), MATRIX)):
        summary = summarize_collective_calibration(build(config, ranks, matrix))
        assert summary["unmatched_reasons"] == {"applicability_mismatch": 1}


def test_diagonal_and_activation_compute_precision_are_included():
    config = params()
    matrix = deepcopy(MATRIX)
    matrix[0][0] = 100
    assert fingerprint(config, matrix=matrix) != fingerprint(config)
    assert fingerprint(config, precision={"compute": "fp16", "communication": "fp8"}) != fingerprint(config)
    assert fingerprint(config, precision={**PRECISION, "activation": "bf16"}) != fingerprint(config)


def test_large_integer_ecmp_seeds_keep_distinct_paths_and_guard_signatures():
    from tests.test_ecmp_routing import fabric

    first = fabric()
    first.update(name="large-seed-regression", routing_seed=2**53)
    second = deepcopy(first)
    second["routing_seed"] += 1
    assert fingerprint(first) != fingerprint(second)
    left, right = Topology(first, PRECISION), Topology(second, PRECISION)
    pairs = [(f"npu{source}", f"npu{destination}") for source in range(4) for destination in range(4, 8)]
    assert any(left.path(*pair) != right.path(*pair) for pair in pairs)


def test_io_groups_guard_bandwidth_membership_and_duplex_not_cosmetic_order():
    config = params()
    config["io_groups"] = [
        {"id": "root-a", "members": [0, 1], "domains": ["SO", "H2D"], "bandwidth_Bps": 100},
        {"id": "root-b", "members": [2, 3], "domains": ["SO"], "bandwidth_Bps": 100},
    ]
    expected = fingerprint(config)
    equivalent = deepcopy(config)
    equivalent["io_groups"].reverse()
    for group in equivalent["io_groups"]:
        group["id"] += "-renamed"
        group["members"].reverse()
        group["domains"].reverse()
        group.update(efficiency=1.0, duplex="full", latency_s=0.0)
    assert fingerprint(equivalent) == expected
    for key, value in (("bandwidth_Bps", 50), ("members", [0, 2]), ("duplex", "shared"), ("latency_s", 1e-6)):
        changed = deepcopy(config)
        changed["io_groups"][0][key] = value
        assert fingerprint(changed) != expected


def test_explicit_links_routes_are_guarded_and_insertion_order_is_irrelevant():
    config = params()
    config.update(topology="explicit", explicit_links=[
        {"id": "a", "src": "npu0", "dst": "switch", "bandwidth_Bps": 100},
        {"id": "b", "src": "switch", "dst": "npu1", "bandwidth_Bps": 100},
    ], routes={"npu0->npu1": ["a", "b"]})
    expected = fingerprint(config)
    changed = deepcopy(config)
    changed["explicit_links"][0]["bandwidth_Bps"] = 50
    assert fingerprint(changed) != expected
    changed = deepcopy(config)
    changed["explicit_links"].reverse()
    assert fingerprint(changed) == expected
    changed = deepcopy(config)
    changed["routes"]["npu0->npu1"] = ["b", "a"]
    assert fingerprint(changed) != expected


def test_nonruntime_metadata_profiles_and_numeric_json_representation_do_not_change_guard():
    config = params()
    expected = fingerprint(config)
    config.update(collective_profiles=[make_profile(config)], source="another citation", description="another note")
    config["npu"]["runtime_bytes"] = 123
    config["npu"]["workspace_bytes"] = 456
    config["scale_up"]["bandwidth_Bps"] = float(config["scale_up"]["bandwidth_Bps"])
    float_matrix = [[float(value) for value in row] for row in MATRIX]
    assert fingerprint(config, matrix=float_matrix) == expected


def test_topology_defaults_are_not_confused_with_config_model_defaults():
    raw = {"name": "defaults", "npu_count": 4, "su_size": 2}
    normalized = HardwareSpec.model_validate(raw).model_dump()
    assert fingerprint(raw) != fingerprint(normalized)  # raw SU=4e11, normalized SU=9e11.
    same = params()  # Explicit network parameters have equal effective semantics.
    assert fingerprint(same) == fingerprint(HardwareSpec.model_validate(same).model_dump())


def test_legacy_profiles_still_apply_with_visible_unguarded_coverage():
    config = params()
    config["collective_profiles"] = [make_profile(config, guarded=False)]
    graph = build(config)
    summary = summarize_collective_calibration(graph, simulate(graph))
    assert summary["matched_collectives"] == summary["unguarded_matched_collectives"] == 1
    assert summary["guarded_matched_collectives"] == 0
    record = next(event.metadata["collective_calibration"] for event in graph.events if "collective_calibration" in event.metadata)
    assert record["applicability_guard"] == "unguarded_legacy"


def test_no_profile_and_legacy_profile_calls_skip_full_fingerprint_work(monkeypatch):
    def expensive_hash_must_not_run(*args, **kwargs):
        raise AssertionError("unguarded calls should not hash the hardware and full payload matrix")

    monkeypatch.setattr("supernode_sim.collective_calibration.collective_applicability_signature", expensive_hash_must_not_run)
    config = params()
    assert summarize_collective_calibration(build(config))["matched_collectives"] == 0
    config["collective_profiles"] = [make_profile(config, guarded=False)]
    assert summarize_collective_calibration(build(config))["unguarded_matched_collectives"] == 1


def test_guard_cannot_be_bypassed_by_a_duplicate_legacy_profile():
    config = params()
    guarded = make_profile(config)
    legacy = {**make_profile(config, guarded=False), "id": "legacy"}
    config["collective_profiles"] = [legacy, guarded]
    signature = {key: guarded[key] for key in ("collective", "semantic", "phase", "world_size", "dtype", "algorithm", "timing_scope", "total_payload_bytes", "send_payload_bytes", "recv_payload_bytes")}
    assert match_collective_profile(config, signature) == (None, "applicability_unavailable")
    signature["applicability_signature"] = "v1:sha256:" + "0" * 64
    assert match_collective_profile(config, signature) == (None, "applicability_mismatch")
    signature["applicability_signature"] = guarded["applicability_signature"]
    assert match_collective_profile(config, signature)[0]["id"] == guarded["id"]


@pytest.mark.parametrize("signature", ["", "abc", "v1:sha256:abc", "v1:sha256:" + "G" * 64])
def test_profile_schema_rejects_malformed_guard(signature):
    config = params()
    config["collective_profiles"] = [{**make_profile(config), "applicability_signature": signature}]
    with pytest.raises(ValueError):
        HardwareSpec.model_validate(config)

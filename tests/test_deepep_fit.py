"""Fit constraints and held-out evidence for the reproducible DeepEP experiment."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "skills/supernode-modeling/scripts/fit_deepep_collectives.py"
SPEC = importlib.util.spec_from_file_location("fit_deepep_collectives", SCRIPT)
fit_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit_tool)


def test_affine_fit_recovers_interior_without_efficiency_fitting():
    fitted = fit_tool.fit_nonnegative_affine([(1e-6, 5e-6), (3e-6, 9e-6), (5e-6, 13e-6)])
    assert fitted["startup_s"] == pytest.approx(3e-6)
    assert fitted["slope"] == pytest.approx(2)
    assert fitted["training_sse_s2"] == pytest.approx(0, abs=1e-35)
    assert fitted["active_constraint"] == "interior"


def test_slope_floor_prevents_unphysical_bandwidth_speedup():
    fitted = fit_tool.fit_nonnegative_affine([(1, 4), (3, 5)])
    assert fitted["startup_s"] == pytest.approx(2.5)
    assert fitted["slope"] == 1
    assert fitted["active_constraint"] == "slope=1"


def test_negative_intercept_projects_onto_startup_zero_boundary():
    fitted = fit_tool.fit_nonnegative_affine([(1, 1), (2, 3)])
    assert fitted["startup_s"] == 0
    assert fitted["slope"] == pytest.approx(1.4)
    assert fitted["active_constraint"] == "startup=0"


def test_both_constraints_and_equal_floors_are_supported():
    fitted = fit_tool.fit_nonnegative_affine([(1, 0), (2, 0)])
    assert fitted["startup_s"] == 0
    assert fitted["slope"] == 1
    equal = fit_tool.fit_nonnegative_affine([(2, 5), (2, 7)])
    assert equal["startup_s"] == 4
    assert equal["slope"] == 1


@pytest.mark.parametrize("points", [[], [(1, 2)], [(1, True), (2, 3)], [(-1, 2), (2, 3)],
    [(1, float("nan")), (2, 3)], [(1, 2), (float("inf"), 3)], [(1, 2, 3), (2, 3)]])
def test_invalid_training_points_are_rejected(points):
    with pytest.raises(fit_tool.FitError):
        fit_tool.fit_nonnegative_affine(points)


def test_holdout_values_and_floors_never_change_fitted_parameters():
    floors = {world: {semantic: (128 * 8 * size / 50e9) * ((world - 8) / world)
                      for semantic, size in fit_tool.BYTES_PER_ROUTED_TOKEN.items()}
              for world in (16, 32, 64, 128)}
    expected = fit_tool.fit_reference_models(floors)
    observations = copy.deepcopy(fit_tool.PUBLIC_LATENCIES_US)
    for world in (64, 128):
        for semantic in ("dispatch", "combine"):
            observations[world][semantic] = 1e30
            floors[world][semantic] = 1e20
    assert fit_tool.fit_reference_models(floors, observations) == expected
    assert expected["dispatch"]["slope"] == 1
    assert expected["dispatch"]["startup_s"] == pytest.approx(41.6776e-6)
    assert expected["combine"]["startup_s"] == pytest.approx(39e-6)


def test_uniform_payload_retains_diagonal_but_profile_vectors_exclude_it():
    matrix = fit_tool.payload_matrix(16, "dispatch")
    logical = 128 * 8 * 7408
    assert len(matrix) == 16
    assert all(sum(row) == logical for row in matrix)
    profile = fit_tool.make_profile(16, "dispatch", {"startup_s": 1e-6, "slope": 1}, 2e-6)
    assert profile["send_payload_bytes"] == [logical * 15 / 16] * 16
    assert profile["recv_payload_bytes"] == profile["send_payload_bytes"]
    assert profile["total_payload_bytes"] == logical * 15
    assert profile["dtype"] == "fp8"
    assert profile["timing_scope"] == "activation_collective"


@pytest.mark.parametrize("worlds", [(16, 64), (64, 64), (), (32,), (256,)])
def test_holdout_split_cannot_overlap_training(worlds):
    with pytest.raises(fit_tool.FitError, match="holdout world sizes"):
        fit_tool.run_experiment(worlds)


def test_small_collective_applies_profile_as_max_with_physical_floor():
    graph = fit_tool.build_collective_graph(16, "dispatch")
    floor = fit_tool.resource_floor(graph)
    expected_floor = 128 * 8 * 7408 * (1 - 8 / 16) / 50e9
    assert floor["duration_s"] == pytest.approx(expected_floor)
    assert any("SO.tx" in resource for resource in floor["limiting_resources"])
    before, _ = fit_tool._simulation_summary(graph)
    assert before["duration_s"] == pytest.approx(expected_floor)
    profile = fit_tool.make_profile(16, "dispatch", {"startup_s": 40e-6, "slope": 1}, expected_floor)
    after_graph = fit_tool.build_collective_graph(16, "dispatch", [profile])
    after, summary = fit_tool._simulation_summary(after_graph)
    assert summary["matched_collectives"] == 1
    assert profile["id"] in summary["matched_profile_ids"]
    assert math.isclose(after["duration_s"], max(before["duration_s"], profile["duration_s"]), rel_tol=1e-10)
    assert after["duration_s"] < before["duration_s"] + profile["duration_s"]

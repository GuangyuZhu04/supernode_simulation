from copy import deepcopy

import pytest

from examples.bandwidth_overlap import overlap, single_transfer
from supernode_sim.analysis import compile_graph, graph_estimate, run
from supernode_sim.config import HardwareSpec
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from supernode_sim.serving import simulate_serving
from tests.test_structured import configuration
from tests.test_topology_calibration import fingerprint, params


@pytest.mark.parametrize("schedule,end", [
    ((), .010), (((.004, 5e9), (.008, 10e9)), .012),
    (((0, 20e9), (.005, 2e9)), .005), (((.020, 1),), .010),
])
def test_transfer_accounting_and_lower_bound(schedule, end):
    graph = single_transfer(schedule)
    original = deepcopy(graph)
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(end)
    assert graph_estimate(graph)["duration_lower_bound_s"] == pytest.approx(end)
    metrics = result["resources"]["link"]
    assert metrics["work"] == pytest.approx(100e6)
    assert metrics["utilization"] == pytest.approx(1)
    assert graph == original


@pytest.mark.parametrize("shared,comm_ms", [(True, 10), (False, 5)])
@pytest.mark.parametrize("compute_ms", [3, 12])
def test_shared_egress_overlap(shared, comm_ms, compute_ms):
    result = simulate(overlap(shared, compute_ms))
    ends = {row["id"]: row["end_s"] * 1000 for row in result["trace"]}
    assert ends["communication_done"] == pytest.approx(comm_ms)
    assert ends["task_done"] == pytest.approx(max(compute_ms, comm_ms))


def test_simultaneous_changes_during_startup_and_bottleneck_switch():
    graph = Graph(resources={
        "a": Resource("a", 1, "byte/s", capacity_schedule=((1, 2), (2, 10))),
        "b": Resource("b", 10, "byte/s", capacity_schedule=((2, 1),)),
    }, events=[Event("send", startup_s=1, work={"a": 5, "b": 5})])
    assert simulate(graph)["duration_s"] == pytest.approx(5)
    assert graph_estimate(graph)["dependency_path_lower_bound_s"] == pytest.approx(5)


def test_small_residual_before_a_large_bandwidth_drop():
    link = Resource("link", 1, "byte/s", capacity_schedule=((1 - 2**-36, 2**-36),))
    graph = Graph(resources={link.id: link}, events=[Event("send", work={"link": 1})])
    assert simulate(graph)["duration_s"] == pytest.approx(2 - 2**-36)


def test_weighted_contenders_share_changing_capacity():
    link = Resource("shared:bus", 8, "byte/s", capacity_schedule=((1, 4),))
    graph = Graph(resources={link.id: link}, metadata={"training_qos": True},
                  events=[Event(name, work={link.id: 9}, metadata={"qos_weight": weight})
                          for name, weight in (("high", 3), ("low", 1))])
    result = simulate(graph)
    ends = {row["id"]: row["end_s"] for row in result["trace"]}
    assert ends == pytest.approx({"high": 2, "low": 3.5})


def test_independent_path_timing_survives_another_links_slowdown():
    graph = overlap(False, 3)
    graph.resources["link:branch0"].capacity_schedule = ((.002, 5e9),)
    result = simulate(graph)
    ends = {row["id"]: row["end_s"] for row in result["trace"]}
    assert ends == pytest.approx({"branch0": .008, "branch1": .005, "compute": .003,
                                 "communication_done": .008, "task_done": .008})


def test_serving_batches_use_one_clock_and_cache_by_effective_schedule():
    config = configuration({"workload": {"scenario": "inference"},
                            "hardware": {"bandwidth_schedules": {"npu0.hbm": [(1, 1e8)]}}})
    requests = [{"id": str(i), "input_tokens": 8, "output_tokens": 2, "arrival_s": 100 + i}
                for i in range(3)]
    batches = simulate_serving(config, {"requests": requests})["batches"]
    assert batches[1]["duration_s"] > batches[0]["duration_s"]
    assert batches[2]["duration_s"] == batches[1]["duration_s"]
    assert [batch["cache_hit"] for batch in batches] == [False, False, True]


@pytest.mark.parametrize("points", [[(-1, 1)], [(1, 0)], [(0, float("inf"))],
                                    [(float("nan"), 1)], [(1, 2), (1, 3)], [(1, 2), (0, 3)]])
def test_invalid_schedules(points):
    with pytest.raises(ValueError):
        HardwareSpec(bandwidth_schedules={"link": points})
    assert simulate(single_transfer(points))["diagnostics"][0]["code"] == "invalid_capacity_schedule"


def test_calibration_signature_changes_only_with_an_active_schedule():
    original = params()
    assert fingerprint(original) == fingerprint({**original, "bandwidth_schedules": {}})
    active = {**original, "bandwidth_schedules": {"npu0.hbm": [(1, 10e9)]}}
    assert fingerprint(original) != fingerprint(active)
    assert fingerprint(active) == fingerprint(HardwareSpec.model_validate(active).model_dump())


def test_absolute_clock_in_streaming_and_full_execution():
    config = configuration({"workload": {"scenario": "inference", "output_length": 3}})
    first = run(config)
    config.hardware.bandwidth_schedules = {"npu0.hbm": [(first["metrics"]["ttft_s"], 1e8)]}
    full = run(config)
    streamed = run_streaming(config)
    for key in ("ttft_s", "tpot_s", "duration_s"):
        assert streamed["metrics"][key] == pytest.approx(full["metrics"][key])
    assert full["metrics"]["ttft_s"] == pytest.approx(first["metrics"]["ttft_s"])
    assert full["metrics"]["tpot_s"] > first["metrics"]["tpot_s"]
    assert streamed["simulation"]["resources"]["npu0.hbm"]["utilization"] == pytest.approx(
        full["simulation"]["resources"]["npu0.hbm"]["utilization"])
    config.hardware.bandwidth_schedules = {"typo": [(0, 1)]}
    with pytest.raises(ValueError, match="Unknown bandwidth schedule"):
        compile_graph(config)

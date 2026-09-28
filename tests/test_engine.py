import pytest

from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import max_min_rates, simulate


def graph_with(*events, memory=100):
    return Graph(resources={"bus": Resource("bus", 100, "byte/s"), "other": Resource("other", 100, "byte/s")},
                 memory_caps={"mem": memory}, events=list(events))


def test_shared_bandwidth_and_independent_overlap():
    graph = graph_with(Event("a", work={"bus": 100}), Event("b", work={"bus": 100}), Event("c", work={"other": 100}))
    result = simulate(graph)
    assert result["status"] == "feasible"
    assert result["duration_s"] == pytest.approx(2)
    trace = {entry["id"]: entry for entry in result["trace"]}
    assert trace["c"]["end_s"] == pytest.approx(1)
    assert trace["a"]["end_s"] == trace["b"]["end_s"] == pytest.approx(2)
    assert result["resources"]["bus"]["utilization"] == pytest.approx(1)


def test_normalized_fairness_reallocates_after_bottleneck():
    rates = max_min_rates({"a": {"bus": 100, "other": 100}, "b": {"bus": 100}}, {"bus": 100, "other": 10})
    assert rates == pytest.approx({"a": 1 / 11, "b": 10 / 11})
    result = simulate(graph_with(Event("a", work={"bus": 100}), Event("b", work={"bus": 300})))
    trace = {entry["id"]: entry for entry in result["trace"]}
    assert result["duration_s"] == pytest.approx(4)
    assert trace["a"]["end_s"] == pytest.approx(2)
    assert trace["b"]["end_s"] == pytest.approx(4)


def test_unequal_flow_lengths_get_equal_bandwidth():
    graph = Graph(resources={"link": Resource("link", 10, "byte/s")},
                  events=[Event("small", work={"link": 10}), Event("large", work={"link": 100})])
    result = simulate(graph)
    trace = {entry["id"]: entry for entry in result["trace"]}
    assert trace["small"]["end_s"] == pytest.approx(2)
    assert result["duration_s"] == pytest.approx(11)


def test_slots_startup_and_dependencies():
    graph = graph_with(Event("a", work={"bus": 100}, startup_s=1, slots=("compute",)),
                       Event("b", work={"other": 100}, slots=("compute",)),
                       Event("c", deps=("b",), startup_s=.5))
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(3.5)
    trace = {entry["id"]: entry for entry in result["trace"]}
    assert trace["b"]["start_s"] == pytest.approx(2)


def test_transfer_buffers_both_resident_and_released_on_completion():
    graph = graph_with(Event("transfer", work={"bus": 100}, allocs=(Allocation("dst", "mem", 40),), releases=("src",)),
                       Event("use", deps=("transfer",), work={"other": 100}, releases=("dst",)))
    graph.initial_allocations = [Allocation("src", "mem", 40, "weight")]
    result = simulate(graph)
    assert result["memory_peaks_bytes"]["mem"] == 80
    assert result["memory_final_bytes"]["mem"] == 0


def test_memory_admission_waits_for_release():
    graph = graph_with(Event("a", work={"bus": 100}, allocs=(Allocation("a.buf", "mem", 80),), releases=("a.buf",)),
                       Event("b", work={"other": 100}, allocs=(Allocation("b.buf", "mem", 80),), releases=("b.buf",)))
    result = simulate(graph)
    assert result["status"] == "feasible"
    assert result["duration_s"] == pytest.approx(2)
    assert result["memory_peaks_bytes"]["mem"] == 80


def test_initial_capacity_and_admission_deadlock_are_infeasible():
    graph = graph_with()
    graph.initial_allocations = [Allocation("weights", "mem", 101)]
    result = simulate(graph)
    assert result["status"] == "infeasible"
    assert result["diagnostics"][0]["code"] == "initial_memory_exceeded"
    graph = graph_with(Event("a", allocs=(Allocation("buf", "mem", 101),)))
    result = simulate(graph)
    assert result["status"] == "infeasible"
    assert result["diagnostics"][0]["code"] == "memory_admission_deadlock"


@pytest.mark.parametrize("events,code", [
    ([Event("a", deps=("b",)), Event("b", deps=("a",))], "dependency_cycle"),
    ([Event("a", deps=("absent",))], "missing_dependency"),
    ([Event("a"), Event("a")], "duplicate_event"),
    ([Event("a", work={"missing": 1})], "invalid_work"),
    ([Event("a", releases=("missing",))], "invalid_release"),
])
def test_invalid_graphs_report_errors(events, code):
    result = simulate(graph_with(*events))
    assert result["status"] == "error"
    assert result["diagnostics"][0]["code"] == code


def test_reproducibility_and_graph_immutability():
    graph = graph_with(Event("a", work={"bus": 100}), Event("b", work={"bus": 50}, startup_s=.5))
    first = simulate(graph)
    assert first == simulate(graph)
    assert graph.events[0].work == {"bus": 100}


def test_resource_sharing_no_artificial_roofline_sum():
    result = simulate(graph_with(Event("kernel", work={"bus": 100, "other": 200})))
    assert result["duration_s"] == pytest.approx(2)


def test_zero_work_chain_and_empty_graph():
    result = simulate(graph_with(Event("a"), Event("b", deps=("a",))))
    assert result["status"] == "feasible"
    assert result["duration_s"] == 0
    assert simulate(Graph())["status"] == "feasible"

import pytest
import weakref

from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.latency_breakdown import _WindowAttributor, build_latency_breakdown, window_breakdown
from supernode_sim.topology import Topology


def compute_graph(events):
    return Graph(resources={
        "a.compute": Resource("a.compute", 10, "FLOP/s", "compute"),
        "b.compute": Resource("b.compute", 10, "FLOP/s", "compute"),
    }, events=events)


def assert_additive(record):
    assert record is not None
    assert all(value >= 0 for value in record["components_s"].values())
    assert sum(record["components_s"].values()) == pytest.approx(record["total_s"])


def test_parallel_fanin_uses_latest_actual_completion_without_double_counting():
    graph = compute_graph([
        Event("fast", work={"a.compute": 10}),
        Event("slow", work={"b.compute": 30}),
        Event("barrier", deps=("fast", "slow"), startup_s=.25, metadata={"type": "barrier"}),
    ])
    record = window_breakdown(graph, simulate(graph), "barrier")
    assert_additive(record)
    assert record["total_s"] == pytest.approx(3.25)
    assert record["components_s"]["compute_s"] == pytest.approx(3)
    assert record["components_s"]["synchronization_wait_s"] == pytest.approx(.25)


def test_fluid_resource_contention_is_separate_from_solo_roofline():
    graph = compute_graph([
        Event("a", work={"a.compute": 10}),
        Event("b", work={"a.compute": 10}),
        Event("done", deps=("a", "b")),
    ])
    record = window_breakdown(graph, simulate(graph), "done")
    assert_additive(record)
    assert record["components_s"]["compute_s"] == pytest.approx(1)
    assert record["components_s"]["resource_contention_s"] == pytest.approx(1)


@pytest.mark.parametrize("blocked_by", ["slot", "memory"])
def test_dependency_ready_admission_wait_is_attributed_once(blocked_by):
    events = [Event("a", work={"a.compute": 10}), Event("b", work={"b.compute": 10})]
    graph = compute_graph(events)
    if blocked_by == "slot":
        for event in events:
            event.slots = ("exclusive",)
        component, timing = "scheduling_wait_s", "slot_wait_s"
    else:
        graph.memory_caps = {"mem": 10}
        for event in events:
            event.allocs = (Allocation(event.id + ".buffer", "mem", 10),)
            event.releases = (event.id + ".buffer",)
        component, timing = "memory_wait_s", "memory_wait_s"
    simulation = simulate(graph)
    rows = {row["id"]: row for row in simulation["trace"]}
    assert rows["b"]["dependency_ready_s"] == 0
    assert rows["b"][timing] == pytest.approx(1)
    record = window_breakdown(graph, simulation, "b")
    assert_additive(record)
    assert record["total_s"] == pytest.approx(2)
    assert record["components_s"][component] == pytest.approx(1)
    assert record["components_s"]["compute_s"] == pytest.approx(1)


def test_ready_wait_tracks_a_change_from_slot_to_memory_blocking():
    graph = compute_graph([
        Event("a.slot", work={"a.compute": 10}, slots=("slot",)),
        Event("b.memory", work={"b.compute": 20}, allocs=(Allocation("buffer", "mem", 10),), releases=("buffer",)),
        Event("c.wait", slots=("slot",), allocs=(Allocation("last", "mem", 10),), releases=("last",)),
    ])
    graph.memory_caps = {"mem": 10}
    simulation = simulate(graph)
    row = next(row for row in simulation["trace"] if row["id"] == "c.wait")
    assert row["slot_wait_s"] == pytest.approx(1)
    assert row["memory_wait_s"] == pytest.approx(1)
    record = window_breakdown(graph, simulation, "c.wait")
    assert_additive(record)
    assert record["components_s"]["scheduling_wait_s"] == pytest.approx(1)
    assert record["components_s"]["memory_wait_s"] == pytest.approx(1)


@pytest.mark.parametrize("src,dst,expected", [
    ("npu0", "npu1", "communication_node_s"),
    ("npu0", "npu2", "communication_scaleup_s"),
    ("npu0", "npu4", "communication_scaleout_s"),
    ("cpu0", "npu0", "communication_node_s"),
])
def test_host_scaleup_and_scaleout_follow_actual_topology(src, dst, expected):
    topology = Topology({"npu_count": 8, "npus_per_host": 2, "su_size": 4,
                         "npu": {"hbm_Bps": 1e12}, "cpu": {"dram_Bps": 1e12},
                         "scale_up": {"bandwidth_Bps": 10, "efficiency": 1, "latency_s": .1},
                         "scale_out": {"bandwidth_Bps": 10, "efficiency": 1, "latency_s": .2},
                         "host_link": {"bandwidth_Bps": 10, "efficiency": 1, "latency_s": .3}}, {})
    graph = Graph()
    topology.populate(graph)
    graph.events = [topology.transfer("transfer", src, dst, 10)]
    record = window_breakdown(graph, simulate(graph), "transfer")
    assert_additive(record)
    assert record["components_s"][expected] == pytest.approx(record["total_s"])


def test_mixed_path_uses_slowest_resource_and_domain_latencies_not_byte_ratios():
    topology = Topology({"npu_count": 2, "npus_per_host": 1, "topology": "explicit",
                         "npu": {"hbm_Bps": 1e12},
                         "explicit_links": [
                             {"id": "local", "src": "npu0", "dst": "switch", "domain": "SU", "bandwidth_Bps": 100, "latency_s": .1},
                             {"id": "remote", "src": "switch", "dst": "npu1", "domain": "SO", "bandwidth_Bps": 10, "latency_s": .2},
                         ]}, {})
    graph = Graph()
    topology.populate(graph)
    graph.events = [topology.transfer("transfer", "npu0", "npu1", 10)]
    record = window_breakdown(graph, simulate(graph), "transfer")
    assert_additive(record)
    assert record["components_s"]["communication_scaleup_s"] == pytest.approx(.1)
    assert record["components_s"]["communication_scaleout_s"] == pytest.approx(1.2)


def test_endpoint_memory_transfer_bottleneck_is_not_labeled_network_time():
    topology = Topology({"npu_count": 2, "npu": {"hbm_Bps": 10},
                         "scale_up": {"bandwidth_Bps": 1000, "efficiency": 1, "latency_s": .1}}, {})
    graph = Graph()
    topology.populate(graph)
    graph.events = [topology.transfer("transfer", "npu0", "npu1", 10)]
    record = window_breakdown(graph, simulate(graph), "transfer")
    assert_additive(record)
    assert record["components_s"]["communication_other_s"] == pytest.approx(1)
    assert record["components_s"]["communication_node_s"] == pytest.approx(.1)


def test_tpot_clips_prior_work_and_full_compact_traces_agree():
    graph = compute_graph([
        Event("prefill", work={"a.compute": 20}),
        Event("decode", deps=("prefill",), work={"a.compute": 30}),
    ])
    full, compact = simulate(graph), simulate(graph, retain_event_payload=False)
    record = window_breakdown(graph, full, "decode", start_s=2)
    assert record == window_breakdown(graph, compact, "decode", start_s=2)
    assert_additive(record)
    assert record["total_s"] == pytest.approx(3)
    assert record["components_s"]["compute_s"] == pytest.approx(3)


def test_lazy_trace_index_does_not_retain_rehydrated_payload_rows():
    graph = compute_graph([Event("done", work={"a.compute": 10})])
    simulation = simulate(graph)
    references = []

    class PayloadRow(dict):
        pass

    class LazyRows:
        def __iter__(self):
            for source in simulation["trace"]:
                row = PayloadRow(source)
                references.append(weakref.ref(row))
                yield row

    attribution = _WindowAttributor(graph, {**simulation, "trace": LazyRows()})
    assert references and all(reference() is None for reference in references)
    assert attribution.window("done") == window_breakdown(graph, simulation, "done")


def test_aggregate_selects_worst_mean_request_instead_of_component_maxima():
    graph = compute_graph([
        Event("a.first", work={"a.compute": 20}),
        Event("a.second", deps=("a.first",), work={"a.compute": 10}),
        Event("b.first", work={"b.compute": 10}),
        Event("b.second", deps=("b.first",), startup_s=3, metadata={"type": "barrier"}),
    ])
    graph.metadata["sequence_token_completion_events"] = {"0": {
        "0": ["a.first", "a.second"], "1": ["b.first", "b.second"], "2": ["a.first"],
    }}
    requests = [
        {"replica": "0", "sequence": "0", "ttft_s": 2, "tpot_s": 1},
        {"replica": "0", "sequence": "1", "ttft_s": 1, "tpot_s": 3},
        {"replica": "0", "sequence": "2", "ttft_s": 2, "tpot_s": None},
    ]
    result = build_latency_breakdown(graph, simulate(graph), requests)
    assert result["ttft"]["sequence"] == "0"
    assert result["tpot"]["sequence"] == "1"
    assert result["tpot"]["components_s"]["compute_s"] == 0
    assert result["tpot"]["components_s"]["synchronization_wait_s"] == 3
    assert requests[2]["tpot_breakdown"] is None
    for record in (result["ttft"], result["tpot"]):
        assert_additive(record)


def test_unavailable_and_single_token_metrics_remain_null():
    graph = compute_graph([Event("only", work={"a.compute": 10})])
    graph.metadata["sequence_token_completion_events"] = {"0": {"0": ["only"]}}
    request = {"replica": "0", "sequence": "0", "ttft_s": 1, "tpot_s": None}
    result = build_latency_breakdown(graph, simulate(graph), [request])
    assert result["ttft"]["total_s"] == 1
    assert result["tpot"] is None
    assert result["tpot_unavailable_reason"]
    result = build_latency_breakdown(graph, {"status": "infeasible"}, [request])
    assert result["ttft"] is result["tpot"] is None
    assert result["unavailable_reason"]

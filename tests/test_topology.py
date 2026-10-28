import pytest

from supernode_sim.config import HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology


def hardware(**overrides):
    return {"npu_count": 4, "su_size": 4, "npus_per_host": 2,
            "npu": {"hbm_bytes": 1e9, "hbm_Bps": 1e12},
            "cpu": {"dram_bytes": 1e9, "dram_Bps": 1e12},
            "scale_up": {"bandwidth_Bps": 100, "latency_s": 0, "efficiency": 1},
            "scale_out": {"bandwidth_Bps": 100, "latency_s": 0, "efficiency": 1},
            "host_link": {"bandwidth_Bps": 100, "latency_s": 0, "efficiency": 1},
            **overrides}


@pytest.mark.parametrize("kind,multiplier", [("all_gather", 1), ("reduce_scatter", 1), ("all_reduce", 2)])
def test_ring_against_closed_form(kind, multiplier):
    topology = Topology(hardware(), {"compute": "fp16"})
    graph = Graph()
    topology.populate(graph)
    terminals = topology.collective(graph, "comm", kind, [f"npu{i}" for i in range(4)], 400)
    result = simulate(graph)
    assert result["status"] == "feasible"
    assert result["duration_s"] == pytest.approx(multiplier * 3)
    assert len(terminals) == 4
    assert sum(event.metadata["payload_bytes"] for event in graph.events) == multiplier * 1200
    assert all(event.metadata["domains"] == ["SU"] for event in graph.events)
    assert max(result["memory_peaks_bytes"].values()) == 200


def test_a2av_matrix_payload_no_double_counting():
    topology = Topology(hardware(), {})
    graph = Graph()
    topology.populate(graph)
    matrix = [[0 if i == j else (i + 1) * (j + 1) for j in range(4)] for i in range(4)]
    topology.collective(graph, "dispatch", "all_to_all_v", [f"npu{i}" for i in range(4)], 0,
                        payload_matrix=matrix, metadata={"semantic": "dispatch"})
    assert sum(event.metadata["payload_bytes"] for event in graph.events) == sum(map(sum, matrix))
    assert all(event.metadata["semantic"] == "dispatch" for event in graph.events)
    assert all(event.metadata["collective"] == "all_to_all_v" for event in graph.events)
    assert simulate(graph)["status"] == "feasible"


def test_so_is_separate_and_host_nic_shared():
    topology = Topology(hardware(su_size=2), {})
    graph = Graph()
    topology.populate(graph)
    first = topology.transfer("a", "npu0", "npu2", 100)
    second = topology.transfer("b", "npu1", "npu3", 100)
    graph.events.extend([first, second])
    assert first.metadata["domains"] == ["SO"]
    assert first.work["cpu0.nic.tx"] == 100
    assert first.work["cpu1.nic.rx"] == 100
    assert first.metadata["physical_bytes"] == 400
    assert simulate(graph)["duration_s"] == pytest.approx(2)


@pytest.mark.parametrize("host_rate,expected", [(100, 2), (200, 1), (400, 1)])
def test_so_host_aggregate_preserves_per_device_limits(host_rate, expected):
    params = hardware(su_size=2)
    params["scale_out"]["host_injection_Bps"] = host_rate
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("a", "npu0", "npu2", 100),
                         topology.transfer("b", "npu1", "npu3", 100)])
    assert graph.resources["cpu0.nic.tx"].capacity == host_rate
    assert graph.resources["cpu1.nic.rx"].capacity == host_rate
    assert graph.resources["npu0.SO.tx"].capacity == 100
    assert graph.resources["link:su0.gateway>so.spine:SO"].capacity == min(host_rate, 200)
    assert simulate(graph)["duration_s"] == pytest.approx(expected)


def test_so_host_aggregate_efficiency_and_explicit_uplink_are_independent():
    params = hardware(su_size=2)
    params["scale_out"].update(host_injection_Bps=200, uplink_Bps=50, efficiency=0.5)
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("a", "npu0", "npu2", 100),
                         topology.transfer("b", "npu1", "npu3", 100)])
    assert graph.resources["cpu0.nic.tx"].capacity == 100
    assert graph.resources["npu0.SO.tx"].capacity == 50
    assert graph.resources["link:su0.gateway>so.spine:SO"].capacity == 25
    assert simulate(graph)["duration_s"] == pytest.approx(8)


def test_so_default_uplink_sums_reachable_host_nic_capacity():
    params = hardware(npu_count=8, su_size=4)
    params["scale_out"].update(host_injection_Bps=150, injection_Bps=80)
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    # Two hosts in each SU domain: min(150, 2 * 80) each.
    assert graph.resources["link:su0.gateway>so.spine:SO"].capacity == 300
    graph.events.extend(topology.transfer(f"flow{i}", f"npu{i}", f"npu{i + 4}", 100)
                        for i in range(4))
    assert simulate(graph)["duration_s"] == pytest.approx(400 / 300)


def test_so_partial_host_keeps_one_shared_nic_across_domains():
    params = hardware(su_size=2, npus_per_host=4)
    params["scale_out"].update(host_injection_Bps=150)
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("a", "npu0", "npu2", 100),
                         topology.transfer("b", "npu1", "npu3", 100)])
    assert graph.events[0].work["cpu0.nic.tx"] == 100
    assert graph.events[0].work["cpu0.nic.rx"] == 100
    assert simulate(graph)["duration_s"] == pytest.approx(200 / 150)


def test_so_host_aggregate_preserves_explicit_link_capacities():
    links = [{"id": f"flow{i}", "src": f"npu{i}", "dst": f"npu{i + 2}",
              "domain": "SO", "bandwidth_Bps": 100, "latency_s": 0}
             for i in range(2)]
    params = hardware(su_size=2, topology="explicit", explicit_links=links)
    params["scale_out"]["host_injection_Bps"] = 200
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend(topology.transfer(f"a{i}", f"npu{i}", f"npu{i + 2}", 100)
                        for i in range(2))
    assert graph.resources["link:flow0"].capacity == 100
    assert simulate(graph)["duration_s"] == pytest.approx(1)


def test_so_host_aggregate_schema_is_optional_and_scale_out_only():
    original = HardwareSpec.model_validate(hardware())
    assert original.scale_out.host_injection_Bps is None
    params = hardware()
    params["scale_out"]["host_injection_Bps"] = 200
    normalized = HardwareSpec.model_validate(params).model_dump()
    assert normalized["scale_out"]["host_injection_Bps"] == 200
    assert Topology(normalized, {}).resources["cpu0.nic.tx"].capacity == 200
    params["scale_up"]["host_injection_Bps"] = 200
    with pytest.raises(ValueError, match="Extra inputs"):
        HardwareSpec.model_validate(params)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_so_host_aggregate_rejects_invalid_rates(value):
    params = hardware()
    params["scale_out"]["host_injection_Bps"] = value
    with pytest.raises(ValueError):
        HardwareSpec.model_validate(params)


def test_host_transfers_compete_with_ddr_lookup():
    topology = Topology(hardware(), {})
    graph = Graph()
    topology.populate(graph)
    graph.events.append(topology.transfer("kv", "cpu0", "npu0", 100))
    graph.events.append(topology.transfer("ngram", "cpu0", "npu1", 100))
    result = simulate(graph)
    assert result["duration_s"] == pytest.approx(2)
    assert result["resource_work"]["cpu0.dram"] == 200
    assert graph.events[0].metadata["domains"] == ["H2D"]


@pytest.mark.parametrize("kind,expected_hops", [("mesh", 2), ("clos", 4), ("fat_tree", 4)])
def test_template_routes(kind, expected_hops):
    topology = Topology(hardware(topology=kind, leaf_size=2), {})
    path = topology.path("npu0", "npu3")
    assert len(path) == expected_hops
    assert all(topology.links[eid]["domain"] == "SU" for eid in path)
    assert topology.path("npu0", "npu3") == path


def test_explicit_route_and_shared_half_duplex_link():
    links = [{"id": eid, "src": src, "dst": dst, "domain": "SU", "bandwidth_Bps": 100,
              "shared_group": "half_duplex", "shared_Bps": 100}
             for eid, src, dst in (("forward", "npu0", "npu1"), ("reverse", "npu1", "npu0"))]
    topology = Topology(hardware(topology="explicit", explicit_links=links,
                                 routes={"npu0->npu1": ["forward"]}), {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend([topology.transfer("a", "npu0", "npu1", 100), topology.transfer("b", "npu1", "npu0", 100)])
    assert simulate(graph)["duration_s"] == pytest.approx(2)
    with pytest.raises(ValueError, match="No directed"):
        topology.path("npu0", "npu3")


def policy_links():
    return [{"id": eid, "src": src, "dst": dst, "domain": domain, "bandwidth_Bps": 100}
            for eid, src, dst, domain in (
                ("su_attach", "npu0", "su0.leaf", "SU"),
                ("su_exit", "su0.leaf", "npu1", "SU"),
                ("so_shortcut", "npu0", "npu1", "SO"),
                ("host_out", "npu0", "cpu0", "D2H"),
                ("host_in", "cpu0", "npu1", "H2D"),
                ("su_cross", "npu0", "npu2", "SU"),
                ("so_attach", "npu0", "so0.gateway", "SO"),
                ("so_exit", "so0.gateway", "npu2", "SO"),
            )]


def test_explicit_su_so_policy_prevents_so_and_host_shortcuts():
    topology = Topology(hardware(topology="explicit", su_size=2, routing_policy="su_so",
                                 explicit_links=policy_links()), {})
    assert topology.path("npu0", "npu1") == ("su_attach", "su_exit")
    assert topology.transfer("within", "npu0", "npu1", 100).metadata["domains"] == ["SU"]

    without_su = [link for link in policy_links() if link["id"] not in ("su_attach", "su_exit")]
    isolated = Topology(hardware(topology="explicit", su_size=2, routing_policy="su_so",
                                explicit_links=without_su), {})
    with pytest.raises(ValueError, match="No directed"):
        isolated.path("npu0", "npu1")


def test_explicit_su_so_policy_uses_so_for_cross_su_pairs():
    topology = Topology(hardware(topology="explicit", su_size=2, routing_policy="su_so",
                                 explicit_links=policy_links()), {})
    assert topology.path("npu0", "npu2") == ("so_attach", "so_exit")
    event = topology.transfer("cross", "npu0", "npu2", 100)
    assert event.metadata["domains"] == ["SO"]
    assert event.work["cpu0.nic.tx"] == 100
    assert event.work["cpu1.nic.rx"] == 100


@pytest.mark.parametrize("destination,route", [
    ("npu1", ["so_shortcut"]),
    ("npu1", ["host_out", "host_in"]),
    ("npu2", ["su_cross"]),
])
def test_explicit_su_so_policy_rejects_override_that_changes_domains(destination, route):
    topology = Topology(hardware(topology="explicit", su_size=2, routing_policy="su_so",
                                 explicit_links=policy_links(), routes={f"npu0->{destination}": route}), {})
    with pytest.raises(ValueError, match="violates su_so routing policy"):
        topology.path("npu0", destination)


def test_explicit_legacy_default_preserves_arbitrary_mixed_domain_path():
    links = [{"id": eid, "src": src, "dst": dst, "domain": domain, "bandwidth_Bps": 100}
             for eid, src, dst, domain in (
                 ("local", "npu0", "switch", "SU"),
                 ("external", "switch", "npu2", "SO"),
             )]
    topology = Topology(hardware(topology="explicit", su_size=2, explicit_links=links), {})
    assert topology.path("npu0", "npu2") == ("local", "external")
    assert topology.transfer("legacy", "npu0", "npu2", 100).metadata["domains"] == ["SO", "SU"]

    override = Topology(hardware(topology="explicit", su_size=2, explicit_links=policy_links(),
                                routes={"npu0->npu1": ["host_out", "host_in"]}), {})
    assert override.path("npu0", "npu1") == ("host_out", "host_in")


def test_remote_route_charges_bridge_and_endpoint():
    topology = Topology(hardware(remote={"enabled": True, "capacity_bytes": 1e12, "bandwidth_Bps": 1000, "latency_s": 0}), {})
    event = topology.transfer("fetch", "remote0", "npu0", 100)
    assert event.work["remote0.memory"] == 100
    assert event.work["cpu0.dram"] == 100
    assert event.work["cpu0.host.tx"] == 100
    assert event.metadata["domains"] == ["H2D", "remote"]


def test_full_mesh_constructed_lazily_for_large_capacity_search():
    topology = Topology(hardware(npu_count=128, su_size=128), {})
    assert not any(edge["domain"] == "SU" for edge in topology.links.values())
    graph = Graph()
    topology.populate(graph)
    event = topology.transfer("one", "npu0", "npu127", 100)
    assert sum(edge["domain"] == "SU" for edge in topology.links.values()) == 1
    assert set(event.work).issubset(graph.resources)


def test_singleton_collective_is_identity():
    topology = Topology(hardware(), {})
    graph = Graph()
    assert topology.collective(graph, "a", "all_gather", ["npu0"], 400, deps=("previous",)) == ("previous",)
    assert graph.events == []


def test_local_transfer_has_no_network_or_duplicate_memory_charge():
    topology = Topology(hardware(), {})
    event = topology.transfer("local", "npu0", "npu0", 100)
    assert event.work == {"npu0.hbm": 100}
    assert event.metadata["physical_bytes"] == 0
    assert event.metadata["domains"] == []


@pytest.mark.parametrize("kind", ["full_mesh", "mesh", "clos", "fat_tree"])
def test_normalized_optional_null_knobs(kind):
    data = hardware(topology=kind, mesh_width=None, leaf_size=None)
    for key in ("scale_up", "scale_out", "host_link"):
        data[key].update(injection_Bps=None, uplink_Bps=None)
    topology = Topology(data, {})
    assert topology.transfer("a", "npu0", "npu3", 100).metadata["payload_bytes"] == 100

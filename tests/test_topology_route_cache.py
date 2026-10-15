"""Routing cache preserves the legacy shortest-hop/edge-ID path semantics."""
import heapq
import importlib.util
from pathlib import Path
import sys

import pytest

from supernode_sim.cache import BoundedCache
from supernode_sim.topology import Topology


SCRIPT = (Path(__file__).resolve().parents[1] / "skills" / "supernode-modeling" /
          "subskills" / "supernode-topology" / "scripts" / "build_topology.py")
MODULE_SPEC = importlib.util.spec_from_file_location("route_cache_topology_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = BUILDER
MODULE_SPEC.loader.exec_module(BUILDER)


def legacy_path(topology, src, dst):
    """Independent pre-cache heap algorithm, used as the compatibility oracle."""
    allowed = None
    if ((topology.kind != "explicit" or topology.routing_policy == "su_so")
            and src.startswith("npu") and dst.startswith("npu")):
        allowed = {"SU"} if int(src[3:]) // topology.su_size == int(dst[3:]) // topology.su_size else {"SO"}
    pending = [(0, (), src)]
    seen = set()
    while pending:
        hops, route, node = heapq.heappop(pending)
        if node == dst:
            return route
        if node in seen:
            continue
        seen.add(node)
        for eid in sorted(topology.adjacency.get(node, [])):
            edge = topology.links[eid]
            if (allowed is None or edge["domain"] in allowed) and edge["dst"] not in seen:
                heapq.heappush(pending, (hops + 1, route + (eid,), edge["dst"]))
    raise ValueError("No directed route")


def generated_spec(su_kind, so_kind):
    rate = {"bandwidth_Bps": 100.0, "latency_s": 0.0, "efficiency": 1.0}
    return {"schema_version": "1.0", "npu_count": 8, "su_size": 2, "npus_per_host": 1,
            "scale_up": {"topology": su_kind, **rate},
            "scale_out": {"topology": so_kind, **rate, "domains": [
                {"id": "east", "su_domains": [0, 1]},
                {"id": "west", "su_domains": [2, 3]}]},
            "host_link": rate.copy(), "inter_domain_links": [
                {"id": "east-west", "src_domain": "east", "dst_domain": "west", **rate},
                {"id": "west-east", "src_domain": "west", "dst_domain": "east", **rate}]}


@pytest.mark.parametrize("su_kind", ["full_mesh", "ring", "mesh", "clos", "fat_tree"])
@pytest.mark.parametrize("so_kind", ["full_mesh", "ring", "mesh", "clos", "fat_tree"])
def test_all_pairs_generated_graph_matches_legacy_heap(su_kind, so_kind):
    topology = Topology(BUILDER.compile_topology(generated_spec(su_kind, so_kind))["hardware"], {})
    nodes = sorted({edge[endpoint] for edge in topology.links.values() for endpoint in ("src", "dst")})
    for src in nodes:
        for dst in nodes:
            assert topology.path(src, dst) == legacy_path(topology, src, dst)
    # All destinations reuse a source tree; local SU and cross-SU SO remain separate.
    assert len(topology._route_trees[("SU",)]) == 8
    assert len(topology._route_trees[("SO",)]) == 8


def edge(eid, src, dst, domain="SU"):
    return {"id": eid, "src": src, "dst": dst, "domain": domain, "bandwidth_Bps": 100}


def test_equal_hop_tie_breaking_and_explicit_override_priority():
    links = [edge("z-first", "npu0", "a"), edge("a-first", "npu0", "b"),
             edge("a-last", "a", "npu1"), edge("z-last", "b", "npu1"),
             edge("c-direct", "npu0", "npu2"), edge("b-extra", "b", "npu2")]
    topology = Topology({"npu_count": 3, "su_size": 3, "topology": "explicit", "explicit_links": links,
                         "routes": {"npu0->npu2": ["a-first", "b-extra"]}}, {})
    assert topology.path("npu0", "npu1") == ("a-first", "z-last")
    tree = topology._route_trees[None]["npu0"]
    # An explicit non-shortest override takes precedence even after the source tree exists.
    assert topology.path("npu0", "npu2") == ("a-first", "b-extra")
    assert topology._route_trees[None]["npu0"] is tree
    with pytest.raises(ValueError, match="No directed network route"):
        topology.path("npu0", "unreachable")
    assert topology._route_trees[None]["npu0"] is tree


def test_lazy_su_edges_refresh_unrestricted_tree_without_invalidating_so():
    topology = Topology({"npu_count": 4, "su_size": 2, "npus_per_host": 1,
                         "topology": "full_mesh"}, {})
    topology.path("npu0", "npu2")
    so_tree = topology._route_trees[("SO",)]["npu0"]
    topology.path("cpu0", "npu1")
    assert None in topology._route_trees
    assert topology.path("npu0", "npu1") == ("npu0>npu1:SU",)
    assert None not in topology._route_trees
    assert topology._route_trees[("SO",)]["npu0"] is so_tree
    # A new unrestricted query sees the lazily inserted SU shortcut.
    assert topology.path("cpu1", "npu0") == legacy_path(topology, "cpu1", "npu0")


def test_uncached_destination_uses_new_lazy_shortcut():
    topology = Topology({"npu_count": 4, "su_size": 4, "npus_per_host": 1,
                         "topology": "full_mesh"}, {})
    assert len(topology.path("cpu0", "npu3")) == 3
    topology.path("npu0", "npu2")
    assert topology.path("cpu0", "npu2") == ("cpu0>npu0:H2D", "npu0>npu2:SU")


@pytest.mark.parametrize("max_bytes", [1, 16 * 1024])
def test_bounded_caches_preserve_routes_after_eviction_or_oversized_tree(max_bytes):
    topology = Topology(BUILDER.compile_topology(generated_spec("ring", "clos"))["hardware"], {})
    topology._route_cache = BoundedCache(max_bytes=max_bytes, max_entries=3)
    for mode in (None, ("SU",), ("SO",)):
        topology._route_trees[mode] = BoundedCache(max_bytes=max_bytes, max_entries=1)
    nodes = ["npu0", "npu1", "npu2", "npu3", "cpu0", "cpu1"]
    for order in (nodes, list(reversed(nodes))):
        for src in order:
            for dst in order:
                assert topology.path(src, dst) == legacy_path(topology, src, dst)
                assert len(topology._route_cache) <= 3
                assert topology._route_cache.current_bytes <= max_bytes
                for trees in topology._route_trees.values():
                    assert len(trees) <= 1
                    assert trees.current_bytes <= max_bytes
    if max_bytes == 1:
        assert not topology._route_cache
        assert all(not trees for trees in topology._route_trees.values())
    else:
        assert len(topology._route_cache) == 3
        assert all(len(trees) == 1 for trees in topology._route_trees.values())
    # A discarded source tree also preserves the unreachable-destination error.
    with pytest.raises(ValueError, match="No directed network route"):
        topology.path("cpu0", "unreachable")


def test_lazy_full_mesh_keeps_historical_pair_route_after_shortcut():
    topology = Topology({"npu_count": 4, "su_size": 4, "npus_per_host": 1,
                         "topology": "full_mesh"}, {})
    original = topology.path("cpu0", "npu3")
    assert len(original) == 3
    assert isinstance(topology._route_trees[None], BoundedCache)
    topology.path("npu0", "npu3")
    assert len(legacy_path(topology, "cpu0", "npu3")) == 2
    # Pair eviction would change this existing lazy-routing behavior. Until
    # routing semantics change separately, only its source trees are bounded.
    assert isinstance(topology._route_cache, dict)
    assert topology.path("cpu0", "npu3") == original

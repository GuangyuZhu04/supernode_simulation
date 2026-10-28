"""Equal-cost routing preserves capacity and physical-path accounting."""
import copy

import pytest

from supernode_sim.cache import BoundedCache
from supernode_sim.config import HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.routing import destination_hops, ecmp_path
from supernode_sim.topology import Topology


def fabric():
    links = []

    def edge(name, src, dst, rate=100, domain='SO'):
        links.append(dict(id=name, src=src, dst=dst, domain=domain,
                          bandwidth_Bps=rate, latency_s=0, efficiency=1))

    for index in range(4):
        edge(f'attach{index}', f'npu{index}', 'left', 10000)
        edge(f'exit{index}', 'right', f'npu{index + 4}', 10000)
    for spine in range(2):
        edge(f'spine{spine}.in', 'left', f'spine{spine}')
        edge(f'spine{spine}.out', f'spine{spine}', 'right')
    return dict(npu_count=8, su_size=4, npus_per_host=1, topology='explicit',
                routing_policy='su_so', routing_algorithm='ecmp_hash', routing_seed=0,
                explicit_links=links, scale_out=dict(bandwidth_Bps=10000, efficiency=1))


def test_hash_paths_are_shortest_and_independent_of_query_or_edge_order():
    first, second = Topology(fabric(), {}), Topology(fabric(), {})
    expected = {(f'npu{i}', f'npu{j}'): first.path(f'npu{i}', f'npu{j}')
                for i in range(4) for j in range(4, 8)}
    for pair in reversed(list(expected)):
        assert second.path(*pair) == expected[pair]
    shuffled = fabric()
    shuffled['explicit_links'].reverse()
    third = Topology(shuffled, {})
    assert {pair: third.path(*pair) for pair in expected} == expected
    assert all(len(route) == 4 for route in expected.values())
    assert len({route[1] for route in expected.values()}) == 2
    # Simulate eviction/disabled caching. The same flow still takes the same path.
    first._route_cache = BoundedCache(max_bytes=0, max_entries=0)
    first._ecmp_distances.clear()
    assert {pair: first.path(*pair) for pair in expected} == expected


def test_explicit_route_override_wins_over_ecmp():
    data = fabric()
    data['routes'] = {'npu0->npu4': ['attach0', 'spine1.in', 'spine1.out', 'exit0']}
    topology = Topology(data, {})
    assert topology.path('npu0', 'npu4') == tuple(data['routes']['npu0->npu4'])


def test_ecmp_obeys_su_so_and_unreachable_is_not_repaired():
    data = fabric()
    data['explicit_links'].append(dict(id='forbidden_su', src='npu0', dst='npu4',
                                       domain='SU', bandwidth_Bps=10000, latency_s=0))
    topology = Topology(data, {})
    assert len(topology.path('npu0', 'npu4')) == 4
    with pytest.raises(ValueError, match='No directed'):
        topology.path('npu4', 'npu0')
    with pytest.raises(ValueError, match='No directed'):
        topology.path('npu0', 'npu1')


def test_ecmp_does_not_invent_capacity_and_preserves_traffic_accounting():
    records = []
    for algorithm in ('shortest_hop', 'ecmp_hash'):
        data = fabric()
        data['routing_algorithm'] = algorithm
        topology, graph = Topology(data, {}), Graph()
        topology.populate(graph)
        graph.events.extend(topology.transfer(f'f{i}-{j}', f'npu{i}', f'npu{j}', 100)
                            for i in range(4) for j in range(4, 8))
        result = simulate(graph)
        per_spine = [result['resource_work'].get(f'link:spine{i}.in', 0) for i in range(2)]
        assert result['duration_s'] == pytest.approx(max(per_spine) / 100)
        assert sum(event.metadata['payload_bytes'] for event in graph.events) == 1600
        assert sum(event.metadata['physical_bytes'] for event in graph.events) == 6400
        records.append(result['duration_s'])
    assert records[0] == 16
    assert 8 <= records[1] < records[0]  # collisions remain; no ideal split is assumed.


def test_seed_changes_pair_assignment_without_new_paths_or_global_random_state():
    routes = []
    for seed in (0, 1):
        data = fabric()
        data['routing_seed'] = seed
        topology = Topology(data, {})
        routes.append([topology.path(f'npu{i}', f'npu{j}') for i in range(4) for j in range(4, 8)])
    assert routes[0] != routes[1]


def test_reverse_bfs_handles_cycles_and_rejects_nonshortest_detour():
    edges = {name: dict(src=src, dst=dst, domain='SO') for name, src, dst in (
        ('a', 's', 'x'), ('b', 'x', 't'), ('c', 'x', 's'), ('d', 's', 'y'),
        ('e', 'y', 'z'), ('f', 'z', 't'))}
    adjacency = {'s': ['a', 'd'], 'x': ['b', 'c'], 'y': ['e'], 'z': ['f']}
    distances = destination_hops(edges, 't', {'SO'})
    assert ecmp_path(edges, adjacency, 's', 't', {'SO'}, 0, distances) == ('a', 'b')


def test_ecmp_schema_opt_in_and_strict_seed():
    assert HardwareSpec.model_validate(fabric()).routing_algorithm == 'ecmp_hash'
    for field, invalid in (('routing_algorithm', 'perfect_balance'), ('routing_seed', True), ('routing_seed', 1.2)):
        data = copy.deepcopy(fabric())
        data[field] = invalid
        with pytest.raises(ValueError):
            HardwareSpec.model_validate(data)

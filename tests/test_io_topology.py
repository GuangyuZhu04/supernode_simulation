"""Endpoint affinity and shared I/O bottlenecks without extra network bytes."""
import pytest

from supernode_sim.analysis import compile_graph
from supernode_sim.config import Config, HardwareSpec
from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.streaming import run_streaming
from supernode_sim.topology import Topology
from tests.test_moe_wire_precision import configuration


def hardware(groups=None, **overrides):
    return {'npu_count': 4, 'su_size': 2, 'npus_per_host': 2,
        'npu': {'hbm_bytes': 1e9, 'hbm_Bps': 1e9},
        'cpu': {'dram_bytes': 1e9, 'dram_Bps': 1e9},
        'scale_up': {'bandwidth_Bps': 1000, 'latency_s': 0, 'efficiency': 1},
        'scale_out': {'bandwidth_Bps': 1000, 'injection_Bps': 1000, 'host_injection_Bps': 2000,
                      'latency_s': 0, 'efficiency': 1},
        'host_link': {'bandwidth_Bps': 1000, 'injection_Bps': 2000, 'latency_s': 0, 'efficiency': 1},
        **({'io_groups': groups} if groups is not None else {}), **overrides}


def group(id='nic0', members=None, domains=None, **overrides):
    return {'id': id, 'members': [0, 1] if members is None else members,
            'domains': ['SO'] if domains is None else domains, 'bandwidth_Bps': 100, **overrides}


def run(params, flows):
    topology = Topology(params, {})
    graph = Graph()
    topology.populate(graph)
    graph.events.extend(topology.transfer(f'flow{i}', src, dst, size) for i, (src, dst, size) in enumerate(flows))
    result = simulate(graph)
    assert result['status'] == 'feasible'
    return topology, graph, result


def test_unconfigured_groups_preserve_resources_traffic_and_timing():
    flows = [('npu0', 'npu2', 100), ('cpu0', 'npu1', 100)]
    first, graph, before = run(hardware(), flows)
    second, another, after = run(hardware([]), flows)
    assert first.resources == second.resources
    assert [event.work for event in graph.events] == [event.work for event in another.events]
    assert before == after
    assert not any(key.startswith('io:') for key in before['resource_work'])


def test_two_gpus_share_nic_while_independent_nics_do_not():
    flows = [('npu0', 'npu2', 100), ('npu1', 'npu3', 100)]
    _, shared, limited = run(hardware([group()]), flows)
    _, independent, parallel = run(hardware([group('nic0', [0]), group('nic1', [1])]), flows)
    assert limited['duration_s'] == pytest.approx(2)
    assert parallel['duration_s'] == pytest.approx(1)
    assert limited['resource_work']['io:nic0.tx'] == 200
    for left, right in zip(shared.events, independent.events):
        assert left.metadata['physical_bytes'] == right.metadata['physical_bytes'] == 400
        assert left.metadata['physical_bytes_by_domain'] == right.metadata['physical_bytes_by_domain']
        assert left.metadata['payload_bytes'] == right.metadata['payload_bytes'] == 100


@pytest.mark.parametrize('duplex,expected', [('full', 1), ('shared', 2)])
def test_duplex_separates_or_combines_opposite_directions(duplex, expected):
    _, graph, result = run(hardware([group(members=[0], duplex=duplex)]),
                           [('npu0', 'npu2', 100), ('npu2', 'npu0', 100)])
    assert result['duration_s'] == pytest.approx(expected)
    assert sum(row['bytes'] for event in graph.events for row in event.metadata['io_group_charges']) == 200
    if duplex == 'full':
        assert result['resource_work']['io:nic0.tx'] == result['resource_work']['io:nic0.rx'] == 100
    else:
        assert result['resource_work']['io:nic0.shared'] == 200


@pytest.mark.parametrize('duplex,expected', [('full', 1), ('shared', 2)])
def test_both_endpoints_in_one_group_have_explicit_per_endpoint_charges(duplex, expected):
    _, graph, result = run(hardware([group(members=[0, 2], duplex=duplex)]), [('npu0', 'npu2', 100)])
    charges = graph.events[0].metadata['io_group_charges']
    assert [(row['endpoint'], row['direction'], row['bytes']) for row in charges] == [
        ('npu0', 'tx', 100), ('npu2', 'rx', 100)]
    assert result['duration_s'] == pytest.approx(expected)
    assert graph.events[0].metadata['physical_bytes'] == 400


def test_different_endpoint_groups_each_charge_one_payload():
    _, graph, result = run(hardware([group('src_nic', [0]), group('dst_nic', [2])]), [('npu0', 'npu2', 100)])
    assert result['resource_work']['io:src_nic.tx'] == result['resource_work']['io:dst_nic.rx'] == 100
    assert result['duration_s'] == pytest.approx(1)
    assert len(graph.events[0].metadata['io_group_charges']) == 2


def test_so_and_host_offload_share_pcie_when_explicitly_selected():
    all_io = group('pcie0', domains=['SO', 'H2D', 'D2H'], duplex='shared')
    flows = [('npu0', 'npu2', 100), ('cpu0', 'npu1', 100)]
    _, _, shared = run(hardware([all_io]), flows)
    _, _, separate = run(hardware([group('pcie0', domains=['SO'], duplex='shared')]), flows)
    assert shared['duration_s'] == pytest.approx(2)
    assert separate['duration_s'] == pytest.approx(1)
    assert shared['resource_work']['io:pcie0.shared'] == 200


def test_host_domains_charge_only_the_matching_npu_direction():
    _, graph, result = run(hardware([group('h2d', [0], ['H2D']), group('d2h', [0], ['D2H'])]),
                           [('cpu0', 'npu0', 100), ('npu0', 'cpu0', 100)])
    assert result['resource_work']['io:h2d.rx'] == 100
    assert result['resource_work']['io:d2h.tx'] == 100
    assert result['resource_work']['io:h2d.tx'] == result['resource_work']['io:d2h.rx'] == 0
    assert graph.events[0].metadata['io_group_charges'][0]['group'] == 'h2d'
    assert graph.events[1].metadata['io_group_charges'][0]['group'] == 'd2h'


def test_intra_su_and_local_transfers_do_not_implicitly_use_nic_or_pcie():
    topology = Topology(hardware([group(domains=['SO', 'H2D', 'D2H'], latency_s=2)]), {})
    for destination in ('npu0', 'npu1'):
        event = topology.transfer('local', 'npu0', destination, 100)
        assert not event.metadata['io_group_charges']
        assert not any(resource.startswith('io:') for resource in event.work)
        assert event.startup_s == 0


def test_explicit_route_domains_override_endpoint_proximity():
    links = [{'id': 'to_host', 'src': 'npu0', 'dst': 'cpu0', 'domain': 'D2H', 'bandwidth_Bps': 1000, 'latency_s': 0},
             {'id': 'from_host', 'src': 'cpu0', 'dst': 'npu1', 'domain': 'H2D', 'bandwidth_Bps': 1000, 'latency_s': 0}]
    params = hardware([group('pcie', [0, 1], ['H2D', 'D2H'], duplex='shared')],
                      topology='explicit', explicit_links=links)
    _, graph, result = run(params, [('npu0', 'npu1', 100)])
    charges = graph.events[0].metadata['io_group_charges']
    assert [row['domains'] for row in charges] == [['D2H'], ['H2D']]
    assert result['resource_work']['io:pcie.shared'] == 200
    assert graph.events[0].metadata['physical_bytes'] == 200


def test_overlapping_groups_are_distinct_capacity_constraints_and_keep_byte_ledger():
    params = hardware([group('nic', [0]), group('pcie', [0, 1], ['SO', 'H2D', 'D2H'],
                      bandwidth_Bps=200, efficiency=0.25, latency_s=0.2)])
    _, graph, result = run(params, [('npu0', 'npu2', 100)])
    assert result['resource_work']['io:nic.tx'] == result['resource_work']['io:pcie.tx'] == 100
    assert result['duration_s'] == pytest.approx(2.2)
    assert graph.events[0].metadata['physical_bytes'] == 400
    assert graph.events[0].metadata['startup_s_by_domain'] == {'SO': pytest.approx(0.2)}


@pytest.mark.parametrize('patch', [
    {'id': ''}, {'id': 'bad:resource'}, {'members': []}, {'members': [0, 0]}, {'members': [-1]},
    {'members': [4]}, {'members': [True]}, {'members': ['0']}, {'domains': []}, {'domains': ['SO', 'SO']},
    {'domains': ['SU']}, {'domains': ['remote']}, {'bandwidth_Bps': 0}, {'bandwidth_Bps': float('inf')},
    {'bandwidth_Bps': True}, {'efficiency': 0}, {'efficiency': 1.1}, {'latency_s': -1}, {'latency_s': float('nan')},
    {'duplex': 'half'},
])
def test_invalid_groups_fail_configuration_and_direct_topology(patch):
    params = hardware([{**group(), **patch}])
    with pytest.raises(ValueError):
        HardwareSpec.model_validate(params)
    with pytest.raises(ValueError):
        Topology(params, {})


def test_duplicate_ids_are_rejected_but_overlapping_members_are_valid():
    with pytest.raises(ValueError, match='IDs must be unique'):
        HardwareSpec.model_validate(hardware([group(), group()]))
    with pytest.raises(ValueError, match='IDs must be unique'):
        Topology(hardware([group(), group()]), {})
    normalized = HardwareSpec.model_validate(hardware([group('first'), group('second')]))
    assert len(normalized.io_groups) == 2


@pytest.mark.parametrize('scenario', ['training', 'inference'])
def test_physical_membership_survives_rank_order_and_streaming(scenario):
    data = configuration(scenario=scenario).model_dump()
    data['hardware']['su_size'] = 1
    data['hardware']['io_groups'] = [group('physical0', [0])]
    data['execution']['rank_order'] = [1, 0]
    config = Config.model_validate(data)
    graph = compile_graph(config)
    result = simulate(graph)
    assert result['status'] == 'feasible'
    transfers = [event for event in graph.events if event.metadata.get('type') == 'transfer']
    assert transfers
    for event in transfers:
        for charge in event.metadata['io_group_charges']:
            assert charge['endpoint'] == 'npu0'
    expected_tx = sum(event.metadata['payload_bytes'] for event in transfers if event.metadata['src'] == 'npu0' and 'SO' in event.metadata['domains'])
    expected_rx = sum(event.metadata['payload_bytes'] for event in transfers if event.metadata['dst'] == 'npu0' and 'SO' in event.metadata['domains'])
    assert result['resource_work']['io:physical0.tx'] == expected_tx
    assert result['resource_work']['io:physical0.rx'] == expected_rx
    if scenario == 'inference':
        streamed = run_streaming(config)
        assert streamed['simulation']['duration_s'] == pytest.approx(result['duration_s'])
        assert streamed['simulation']['resource_work']['io:physical0.tx'] == expected_tx
        assert streamed['simulation']['resource_work']['io:physical0.rx'] == expected_rx

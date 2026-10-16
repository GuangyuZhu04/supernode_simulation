"""Latency attribution survives the public API, topology placement and checkpoints."""
import pytest

from scripts.run_reference_experiment import JSONLRecords, save_records
from supernode_sim.analysis import compile_graph, execution_metrics, run
from supernode_sim.config import Config
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from tests.test_inference import configuration


@pytest.mark.parametrize('hosts,su_size,component', [
    (2, 2, 'communication_node_s'),
    (1, 2, 'communication_scaleup_s'),
    (1, 1, 'communication_scaleout_s'),
])
def test_tp_transfers_follow_physical_node_su_so_placement(hosts, su_size, component):
    config = configuration({'hardware': {'npu_count': 2, 'su_size': su_size, 'npus_per_host': hosts},
                            'execution': {'tp': 2}})
    result = run(config)
    assert result['status'] == 'feasible'
    metrics = result['metrics']
    assert metrics['tpop_s'] == metrics['tpot_s']
    for key in ('ttft', 'tpot'):
        record = metrics['latency_breakdown'][key]
        assert record['total_s'] == pytest.approx(metrics[key + '_s'])
        assert sum(record['components_s'].values()) == pytest.approx(record['total_s'])
        assert record['components_s']['compute_s'] > 0
        assert record['components_s'][component] > 0
    # Payload still uses the original physical SU/SO ledger; node attribution
    # is an additional latency view and never relabels historical byte totals.
    if su_size == 2:
        assert result['estimate']['communication']['physical_bytes_by_domain']['SU'] > 0


def test_inference_checkpoint_record_stream_preserves_attribution(tmp_path):
    config = Config.model_validate(configuration({'execution': {'features': {'kv_offload': 'host'}}}))
    graph = compile_graph(config)
    simulation = simulate(graph, retain_event_payload=False)
    expected = execution_metrics(config, graph, simulation)
    path = tmp_path / 'trace.jsonl.gz'
    count = save_records(path, simulation['trace'])
    simulation['trace'] = JSONLRecords(path, count)
    actual = execution_metrics(config, graph, simulation)
    assert actual == expected


def test_latency_slo_failure_retains_decomposition_and_single_token_stays_undefined():
    config = configuration({'targets': {'ttft_s': 1e-30},
                            'workload': {'output_lengths': [1, 1]}})
    result = run(config)
    assert result['status'] == 'slo_failed'
    metrics = result['metrics']
    assert metrics['latency_breakdown']['ttft']['total_s'] == pytest.approx(metrics['ttft_s'])
    assert metrics['tpot_s'] is metrics['tpop_s'] is None
    assert metrics['latency_breakdown']['tpot'] is None


def test_explicit_su_route_preserves_local_hops_and_interhost_detours():
    hardware = {'npu_count': 4, 'su_size': 4, 'npus_per_host': 2, 'topology': 'explicit',
                'explicit_links': [
                    {'id': 'local', 'src': 'npu0', 'dst': 'npu1', 'domain': 'SU',
                     'bandwidth_Bps': 10, 'latency_s': 1},
                    {'id': 'remote', 'src': 'npu1', 'dst': 'npu2', 'domain': 'SU',
                     'bandwidth_Bps': 10, 'latency_s': 2},
                    {'id': 'back', 'src': 'npu2', 'dst': 'npu0', 'domain': 'SU',
                     'bandwidth_Bps': 10, 'latency_s': 3}]}
    topology = Topology(hardware, {})
    mixed = topology.transfer('mixed', 'npu0', 'npu2', 10)
    assert mixed.metadata['resource_domains']['link:local'] == 'node'
    assert mixed.metadata['resource_domains']['link:remote'] == 'SU'
    assert mixed.metadata['startup_s_by_domain'] == {'node': 1, 'SU': 2}
    # Same-host endpoints do not make a detour through another host local.
    detour = topology.transfer('detour', 'npu1', 'npu0', 10)
    assert detour.metadata['startup_s_by_domain'] == {'SU': 5}

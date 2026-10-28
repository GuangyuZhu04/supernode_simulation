"""Compilation acceleration must retain payload and environment evidence."""
from copy import deepcopy

import pytest

from supernode_sim import analysis
from supernode_sim.cache import BoundedCache
from supernode_sim.collective_calibration import match_collective_model
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from tests.test_collective_models import PRECISION, RANKS, matrix, model, moe_config, params


@pytest.mark.parametrize('max_bytes,max_entries', [(0, 0), (1, 1), (8192, 1)])
def test_eviction_and_disabled_cache_preserve_complete_compiled_graph(monkeypatch, max_bytes, max_entries):
    config = moe_config(layers=3, scopes=('activation_collective', 'transport'))
    expected = analysis.compile_graph(config)

    class LimitedTopology(Topology):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._collective_signatures = BoundedCache(max_bytes=max_bytes, max_entries=max_entries)

    monkeypatch.setattr(analysis, 'Topology', LimitedTopology)
    actual = analysis.compile_graph(config)
    assert actual == expected
    assert simulate(actual) == simulate(expected)


def calibrated_topology():
    hardware = params()
    hardware['collective_models'] = [model(hardware)]
    return hardware, Topology(hardware, PRECISION)


def signature(topology, values=None, **metadata):
    return topology._collective_signature('all_to_all_v', RANKS, 0,
        matrix(200) if values is None else values,
        {'phase': 'decode', 'operation': 'dispatch', 'wire_dtype': 'fp8', **metadata},
        'direct', 'transport')


def test_cache_keeps_full_matrix_and_current_metadata_and_returns_fresh_vectors():
    hardware, topology = calibrated_topology()
    first = signature(topology)
    assert match_collective_model(hardware, first)[1] == 'matched'
    # No payload change: phase/dtype must still be matched independently.
    changed_phase = signature(topology, phase='prefill')
    assert match_collective_model(hardware, changed_phase)[1] == 'no_model_match'
    changed_dtype = signature(topology, wire_dtype='bf16')
    assert match_collective_model(hardware, changed_dtype)[1] == 'no_model_match'
    # Diagonal is invisible to transport totals but affects activation evidence.
    changed_matrix = matrix(200)
    changed_matrix[0][0] += 1
    changed = signature(topology, changed_matrix)
    assert changed['send_payload_bytes'] == first['send_payload_bytes']
    assert changed['family_signature'] != first['family_signature']
    assert match_collective_model(hardware, changed)[1] == 'family_mismatch'
    first['send_payload_bytes'][0] = -1
    assert signature(topology)['send_payload_bytes'] == [100, 100]


def test_calibration_owns_snapshot_and_raw_matcher_still_validates_every_input():
    hardware, topology = calibrated_topology()
    expected = signature(topology)
    hardware['scale_up']['bandwidth_Bps'] *= 2
    hardware['collective_models'][0]['samples'][0]['duration_s'] = -1
    assert signature(topology) == expected
    with pytest.raises(ValueError):
        match_collective_model(hardware, expected)
    with pytest.raises(ValueError):
        Topology(hardware, PRECISION)


def test_route_mutation_cannot_separate_actual_path_from_calibration_snapshot():
    hardware = params()
    hardware.update(topology='explicit', explicit_links=[
        {'id': 'direct', 'src': 'npu0', 'dst': 'npu1', 'bandwidth_Bps': 100},
        {'id': 'via0', 'src': 'npu0', 'dst': 'switch', 'bandwidth_Bps': 50},
        {'id': 'via1', 'src': 'switch', 'dst': 'npu1', 'bandwidth_Bps': 50},
    ], routes={'npu0->npu1': ['direct']})
    hardware['collective_models'] = [model(hardware)]
    topology = Topology(hardware, PRECISION)
    expected = signature(topology)
    # Mutate before the first route lookup so an existing pair cache cannot
    # hide an alias to the caller's routes dictionary or its nested list.
    hardware['routes']['npu0->npu1'][:] = ['via0', 'via1']
    event = topology.transfer('send', 'npu0', 'npu1', 100)
    assert event.metadata['path'] == ['direct']
    assert event.work['link:direct'] == 100
    assert signature(topology) == expected
    changed_topology = Topology(hardware, PRECISION)
    assert changed_topology.path('npu0', 'npu1') == ('via0', 'via1')
    assert match_collective_model(hardware, signature(changed_topology))[1] == 'family_mismatch'


def test_boolean_payload_cannot_reuse_numeric_cache_entry():
    _, topology = calibrated_topology()
    values = [[1, 100], [100, 1]]
    signature(topology, values)
    changed = deepcopy(values)
    changed[0][0] = True
    with pytest.raises(ValueError, match='payloads must be finite nonnegative numbers'):
        signature(topology, changed)

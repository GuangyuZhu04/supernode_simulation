import gc

import pytest

from supernode_sim.cli import main
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.memory_breakdown import build_memory_breakdown


def test_compact_event_capture_preserves_physics_and_memory_replay():
    graph = Graph(resources={'bus': Resource('bus', 10, 'byte/s')}, memory_caps={'npu0.hbm': 32},
                  events=[Event('a', work={'bus': 10}, allocs=(Allocation('weight', 'npu0.hbm', 8, 'weight', metadata={'component': 'Dense'}),)),
                          Event('b', deps=('a',), work={'bus': 20}, allocs=(Allocation('buf', 'npu0.hbm', 16, 'communication_buffer'),), releases=('buf', 'weight'))])
    full = simulate(graph)
    compact = simulate(graph, retain_event_payload=False)
    for key in ('status', 'duration_s', 'memory_peaks_bytes', 'memory_log', 'resources', 'resource_work'):
        assert full[key] == compact[key]
    assert [(r['id'], r['start_s'], r['end_s']) for r in full['trace']] == [(r['id'], r['start_s'], r['end_s']) for r in compact['trace']]
    assert all('work' not in row and 'allocations' not in row and 'releases' not in row for row in compact['trace'])
    assert build_memory_breakdown(graph, full)['pools'] == build_memory_breakdown(graph, compact, include_ephemeral_details=False)['pools']


def test_cli_restores_collector_on_parser_exit():
    was_enabled = gc.isenabled()
    try:
        gc.enable()
        with pytest.raises(SystemExit):
            main(['--help'])
        assert gc.isenabled()
        gc.disable()
        with pytest.raises(SystemExit):
            main(['--help'])
        assert not gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()

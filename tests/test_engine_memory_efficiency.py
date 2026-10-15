"""Behavioral coverage for compact graph storage and short-lived DAG indexes."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from supernode_sim.analysis import graph_estimate
from supernode_sim.core import Allocation, Event, Graph, Resource
from supernode_sim.engine import simulate


def branching_graph():
    return Graph(
        resources={'bus': Resource('bus', 10, 'byte/s')},
        memory_caps={'mem': 32},
        initial_allocations=[Allocation('resident', 'mem', 8, 'weight')],
        events=[
            Event('start', work={'bus': 10}),
            Event('left', deps=('start', 'start'), work={'bus': 20},
                  allocs=(Allocation('left.buf', 'mem', 16, metadata={'component': 'test'}),),
                  releases=('left.buf',)),
            Event('right', deps=('start',), startup_s=1,
                  allocs=(Allocation('right.buf', 'mem', 16),), releases=('right.buf',)),
            Event('end', deps=('left', 'right', 'left'), releases=('resident',)),
        ],
    )


def test_released_dependency_indexes_preserve_duplicate_edges_and_memory_waits():
    graph = branching_graph()
    original = deepcopy(graph)
    simulation = simulate(graph)
    assert simulation['status'] == 'feasible'
    assert simulation['completed_events'] == 4
    assert simulation['duration_s'] == pytest.approx(4)
    assert simulation['memory_peaks_bytes'] == {'mem': 24}
    assert simulation['memory_final_bytes'] == {'mem': 0}
    assert {row['id']: (row['start_s'], row['end_s']) for row in simulation['trace']} == {
        'start': (0, 1), 'left': (1, 3), 'right': (3, 4), 'end': (4, 4),
    }
    assert graph == original
    assert simulate(graph) == simulation


def test_estimate_recomputed_starts_preserve_memory_and_completion_tables():
    graph = branching_graph()
    original = deepcopy(graph)
    estimate = graph_estimate(graph)
    assert estimate['duration_lower_bound_s'] == 3
    assert estimate['optimistic_event_completion_s'] == {'start': 1, 'left': 3, 'right': 2, 'end': 3}
    # The optimistic schedule overlaps both buffers despite the memory cap.
    assert estimate['optimistic_schedule_memory_peak_bytes'] == {'mem': 40}
    assert estimate['resident_memory_bytes'] == {'mem': 8}
    assert estimate['resource_work'] == {'bus': 30}
    assert estimate['event_count'] == 4
    assert graph == original


def test_compact_graph_records_preserve_copy_replace_and_dict_serialization():
    graph = branching_graph()
    clone = deepcopy(graph)
    assert asdict(clone) == asdict(graph)
    clone.events[1].allocs[0].metadata['component'] = 'changed'
    clone.events[0].work['bus'] = 50
    assert graph.events[1].allocs[0].metadata['component'] == 'test'
    assert graph.events[0].work['bus'] == 10
    event = replace(graph.events[0], startup_s=2)
    assert event.startup_s == 2
    assert event.id == graph.events[0].id
    assert graph.events[0].startup_s == 0


def test_allocation_validation_covers_initial_and_event_allocations():
    graph = branching_graph()
    graph.events[-1].allocs = (Allocation('resident', 'mem', 1),)
    result = simulate(graph)
    assert result['status'] == 'error'
    assert result['diagnostics'][0]['code'] == 'duplicate_allocation'
    assert result['completed_events'] == 0


def test_partial_completion_count_survives_release_failure():
    graph = Graph(events=[Event('start'), Event('end', deps=('start',), releases=('missing',))])
    result = simulate(graph)
    assert result['status'] == 'error'
    assert result['diagnostics'][0]['code'] == 'invalid_release'
    assert result['completed_events'] == 1

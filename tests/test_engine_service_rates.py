"""Prepared active-event service preserves the original water-filling schedule."""
from collections import defaultdict
from copy import deepcopy
import random

import pytest

from supernode_sim import engine
from supernode_sim.core import Allocation, Event, Graph, Resource


def reference_rates(work, capacities):
    """Independent original progressive-filling implementation as an oracle."""
    fractions = dict.fromkeys(work, 0.0)
    solo = {eid: min((capacities[key] / amount for key, amount in demands.items() if amount > 0), default=0.0)
            for eid, demands in work.items()}
    incidence = defaultdict(list)
    for eid, demands in work.items():
        for key, amount in demands.items():
            if amount > 0:
                incidence[key].append((eid, amount * solo[eid]))
    pending = {eid for eid in work if solo[eid] > 0}
    while pending:
        candidates = []
        for resource, participants in incidence.items():
            slope = sum(demand for eid, demand in participants if eid in pending)
            if slope > 0:
                used = sum(demand * fractions[eid] for eid, demand in participants)
                candidates.append((max(0.0, capacities[resource] - used) / slope, resource))
        if not candidates:
            raise ValueError('Events reference no finite-capacity resource')
        step = min(value for value, _ in candidates)
        for eid in pending:
            fractions[eid] += step
        tight = {resource for value, resource in candidates if value <= step * (1 + 1e-10) + 1e-30}
        frozen = {eid for resource in tight for eid, _ in incidence[resource] if eid in pending}
        if not frozen:
            raise ValueError('Unable to resolve resource sharing')
        pending.difference_update(frozen)
    return {eid: fraction * solo[eid] for eid, fraction in fractions.items()}


@pytest.mark.parametrize('seed', range(20))
def test_random_sparse_service_matches_reference_exactly(seed):
    rng = random.Random(seed)
    for _ in range(20):
        capacities = {f'r{i}': 10 ** rng.uniform(-6, 8) for i in range(rng.randint(1, 12))}
        work = {f'e{i}': {key: 10 ** rng.uniform(-6, 8) if rng.random() > .1 else 0.0
                          for key in capacities if rng.random() < .4}
                for i in range(rng.randint(0, 24))}
        actual = engine.max_min_rates(work, capacities)
        assert actual == reference_rates(work, capacities)
        # The exact reference comparison also covers underutilized resources;
        # independently check that no resource capacity is exceeded.
        for resource, capacity in capacities.items():
            used = sum(demands.get(resource, 0) * actual[eid] for eid, demands in work.items())
            assert used <= capacity * (1 + 1e-9)


@pytest.mark.parametrize('work,capacities', [
    ({}, {'a': 1}),
    ({'empty': {}}, {'a': 1}),
    ({'zero': {'a': 0}}, {'a': 1}),
    ({'alone': {'a': 3, 'b': 17}}, {'a': 19, 'b': 23}),
    ({'slow': {'a': 1, 'b': 100}, 'fast': {'a': 100}, 'independent': {'c': 5}},
     {'a': 100, 'b': 1, 'c': 2}),
    ({'a': {'x': 1, 'y': 1}, 'b': {'y': 1 + 1e-11}, 'c': {'z': 1}},
     {'x': 1, 'y': 2, 'z': 1}),
])
def test_empty_lone_tied_and_multiround_services(work, capacities):
    assert engine.max_min_rates(work, capacities) == reference_rates(work, capacities)


def mixed_graph(seed):
    rng = random.Random(seed)
    capacities = {f'r{i}': rng.uniform(1, 100) for i in range(5)}
    events = []
    for index in range(60):
        eid = f'e{index:03}'
        dependencies = tuple(event.id for event in events[-6:] if rng.random() < .25)
        work = {key: rng.uniform(.001, 100) for key in capacities if rng.random() < .4}
        events.append(Event(eid, deps=dependencies, work=work,
                            startup_s=rng.choice([0, 0, .001, .25, 2]),
                            slots=(f'slot{index % 4}',) if rng.random() < .5 else (),
                            allocs=(Allocation(f'{eid}.buffer', 'mem', rng.randint(1, 12)),),
                            releases=(f'{eid}.buffer',)))
    return Graph(resources={key: Resource(key, value, 'byte/s') for key, value in capacities.items()},
                 memory_caps={'mem': 32}, events=events)


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('retain_payload', [True, False])
def test_full_schedule_matches_reference_with_contention_startup_and_admission(monkeypatch, seed, retain_payload):
    graph = mixed_graph(seed)
    original = deepcopy(graph)
    actual = engine.simulate(graph, retain_event_payload=retain_payload)
    work = {event.id: event.work for event in graph.events}
    monkeypatch.setattr(engine, '_max_min_rates', lambda services, capacities:
                        reference_rates({eid: work[eid] for eid in services}, capacities))
    expected = engine.simulate(graph, retain_event_payload=retain_payload)
    assert actual == expected
    assert actual['status'] == 'feasible'
    assert graph == original
    assert any(row['memory_wait_s'] > 0 for row in actual['trace'])
    assert any(row['slot_wait_s'] > 0 for row in actual['trace'])


def test_prepared_service_is_not_reused_across_graph_changes():
    graph = Graph(resources={'bus': Resource('bus', 10, 'byte/s')},
                  events=[Event('e', work={'bus': 100})])
    assert engine.simulate(graph)['duration_s'] == 10
    graph.resources['bus'] = Resource('bus', 20, 'byte/s')
    assert engine.simulate(graph)['duration_s'] == 5
    graph.events[0].work['bus'] = 200
    assert engine.simulate(graph)['duration_s'] == 10

"""Deterministic fluid, operator-granularity discrete-event simulation.

Each event advances from zero to one. Its ``work`` is charged to every
resource simultaneously; normalized max-min fairness shares each event's
fraction of its solo roofline rate among contending events. This is a multi-resource
roofline approximation, not a packet or instruction simulator.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from itertools import chain
from typing import Any

from .core import Graph


def max_min_rates(work: dict[str, dict[str, float]], capacities: dict[str, float]) -> dict[str, float]:
    """Progress/s with max-min fair fractions of solo roofline service.

    A flow of 10 bytes and a flow of 100 bytes on the same link initially
    receive equal byte/s, rather than finishing at the same instant.
    Sparse incidence avoids scanning resources unused by running events.
    """
    return _max_min_rates({eid: _prepare_service(demands, capacities)
                           for eid, demands in work.items()}, capacities)


def _prepare_service(work, capacities):
    """Invariant roofline service, retained only while an event is active."""
    solo = min((capacities[key] / amount for key, amount in work.items() if amount > 0), default=0.0)
    return solo, tuple((key, amount * solo) for key, amount in work.items() if amount > 0)


def _max_min_rates(services, capacities):
    """Water filling over prepared demands, with an exact first-round shortcut."""
    if len(services) == 1:
        eid, (solo, demands) = next(iter(services.items()))
        if solo <= 0:
            return {eid: 0.0}
        # Keep the normalized ratio rather than assuming its rounded value is
        # exactly one, so a lone event uses the same arithmetic as water filling.
        step = min((max(0.0, capacities[key]) / demand for key, demand in demands if demand > 0), default=None)
        if step is None:
            raise ValueError("Events reference no finite-capacity resource")
        return {eid: step * solo}
    incidence: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for eid, (_, demands) in services.items():
        for key, demand in demands:
            incidence[key].append((eid, demand))
    pending = {eid for eid, (solo, _) in services.items() if solo > 0}
    if not pending:
        return dict.fromkeys(services, 0.0)

    # Every fraction is initially zero. The first round needs neither the
    # used-capacity sums nor a fractions table. Most operator batches freeze
    # completely in this round; preserve the same sums, tie tolerance and
    # multiplication order as the general solver.
    candidates = []
    for resource, participants in incidence.items():
        slope = sum(demand for eid, demand in participants if eid in pending)
        if slope > 0:
            candidates.append((max(0.0, capacities[resource]) / slope, resource))
    if not candidates:
        raise ValueError("Events reference no finite-capacity resource")
    step = min(value for value, _ in candidates)
    tight = {resource for value, resource in candidates if value <= step * (1 + 1e-10) + 1e-30}
    frozen = {eid for resource in tight for eid, _ in incidence[resource] if eid in pending}
    if not frozen:
        raise ValueError("Unable to resolve resource sharing")
    if frozen == pending:
        return {eid: step * solo if solo > 0 else 0.0 for eid, (solo, _) in services.items()}
    fractions = {eid: step if solo > 0 else 0.0 for eid, (solo, _) in services.items()}
    pending.difference_update(frozen)
    while pending:
        candidates = []
        for resource, participants in incidence.items():
            capacity = capacities[resource]
            slope = sum(demand for eid, demand in participants if eid in pending)
            if slope <= 0:
                continue
            used = sum(demand * fractions[eid] for eid, demand in participants)
            candidates.append((max(0.0, capacity - used) / slope, resource))
        if not candidates:
            raise ValueError("Events reference no finite-capacity resource")
        step = min(value for value, _ in candidates)
        for eid in pending:
            fractions[eid] += step
        tight = {resource for value, resource in candidates if value <= step * (1 + 1e-10) + 1e-30}
        frozen = {eid for resource in tight for eid, _ in incidence[resource] if eid in pending}
        if not frozen:
            raise ValueError("Unable to resolve resource sharing")
        pending.difference_update(frozen)
    return {eid: fraction * services[eid][0] for eid, fraction in fractions.items()}


def simulate(graph: Graph, *, retain_event_payload: bool = True) -> dict[str, Any]:
    """Simulate one compiled graph without mutating it.

    Allocations occur atomically at event start and releases at event end.
    A ready event waits if memory or exclusive execution slots are occupied.
    Initial allocations remain resident unless explicitly released.  Missing
    dependencies, cycles and malformed graphs are errors; capacity exhaustion
    and a memory admission deadlock are infeasible executions.
    """
    capacities = {key: value.capacity for key, value in graph.resources.items()}
    events = {event.id: event for event in graph.events}
    used: dict[str, float] = defaultdict(float)
    peaks: dict[str, float] = {key: 0.0 for key in graph.memory_caps}
    allocations: dict[str, Any] = {}
    category_used: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    category_peaks: dict[str, dict[str, float]] = {}
    trace: list[dict[str, Any]] = []
    memory_log: list[dict[str, Any]] = []
    busy: dict[str, float] = defaultdict(float)
    work_done: dict[str, float] = defaultdict(float)
    completed_count = 0
    active: dict[str, dict[str, Any]] = {}
    held_slots: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    now = 0.0

    def result(status: str) -> dict[str, Any]:
        metrics = {}
        for key, resource in graph.resources.items():
            metrics[key] = {
                "capacity": resource.capacity, "unit": resource.unit,
                "kind": resource.kind, "work": work_done[key],
                "busy_s": busy[key],
                "utilization": work_done[key] / (resource.capacity * now) if now and resource.capacity > 0 else 0.0,
            }
        return {
            "status": status, "duration_s": now, "trace": trace,
            "trace_payload_fields_omitted": [] if retain_event_payload else ['work', 'allocations', 'releases'],
            "memory_log": memory_log,
            "memory_peaks_bytes": dict(peaks),
            "memory_categories_at_peak_bytes": category_peaks,
            "memory_final_bytes": dict(used), "resources": metrics,
            "resource_busy_s": dict(busy), "resource_work": dict(work_done),
            "diagnostics": diagnostics, "completed_events": completed_count,
            "total_events": len(graph.events),
        }

    def fail(status: str, code: str, message: str, **detail: Any) -> dict[str, Any]:
        diagnostics.append({"code": code, "message": message, **detail})
        return result(status)

    if len(events) != len(graph.events):
        return fail("error", "duplicate_event", "Event IDs must be unique")
    for key, capacity in capacities.items():
        if not math.isfinite(capacity) or capacity <= 0:
            return fail("error", "invalid_resource", "Resource capacity must be finite and positive", resource=key)
    for key, capacity in graph.memory_caps.items():
        if not math.isfinite(capacity) or capacity < 0:
            return fail("error", "invalid_memory", "Memory capacity must be finite and nonnegative", pool=key)
    allocation_ids: set[str] = set()
    for alloc in chain(graph.initial_allocations, (a for event in graph.events for a in event.allocs)):
        if alloc.id in allocation_ids:
            return fail("error", "duplicate_allocation", "Allocation IDs must be unique", allocation=alloc.id)
        allocation_ids.add(alloc.id)
        if alloc.pool not in graph.memory_caps or not math.isfinite(alloc.size_bytes) or alloc.size_bytes < 0:
            return fail("error", "invalid_allocation", "Allocation has unknown pool or invalid size", allocation=alloc.id)
    del allocation_ids
    for event in graph.events:
        missing = set(event.deps).difference(events)
        if missing:
            return fail("error", "missing_dependency", "Event dependency does not exist", event=event.id, missing=sorted(missing))
        if not math.isfinite(event.startup_s) or event.startup_s < 0:
            return fail("error", "invalid_startup", "Startup must be finite and nonnegative", event=event.id)
        for key, amount in event.work.items():
            if key not in capacities or not math.isfinite(amount) or amount < 0:
                return fail("error", "invalid_work", "Work must reference a resource and be nonnegative", event=event.id, resource=key)
    # Diagnose cycles independently of memory admission or numerical behavior.
    indegree = {eid: len(set(event.deps)) for eid, event in events.items()}
    successors: dict[str, list[str]] = defaultdict(list)
    for event in graph.events:
        for dep in set(event.deps):
            successors[dep].append(event.id)
    queue = [eid for eid, count in indegree.items() if count == 0]
    visited = 0
    while queue:
        eid = queue.pop()
        visited += 1
        for successor in successors.get(eid, ()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if visited != len(events):
        return fail("error", "dependency_cycle", "Dependency graph contains a cycle")
    del indegree, queue

    def allocate(alloc: Any) -> None:
        allocations[alloc.id] = alloc
        memory_log.append({"sequence": len(memory_log), "time_s": now, "action": "allocate", "allocation_id": alloc.id, "pool": alloc.pool, "bytes": alloc.size_bytes})
        used[alloc.pool] += alloc.size_bytes
        category_used[alloc.pool][alloc.category] += alloc.size_bytes
        if used[alloc.pool] >= peaks.get(alloc.pool, 0):
            peaks[alloc.pool] = used[alloc.pool]
            category_peaks[alloc.pool] = dict(category_used[alloc.pool])

    for alloc in graph.initial_allocations:
        allocate(alloc)
    for pool, value in used.items():
        if value > graph.memory_caps[pool] * (1 + 1e-12):
            return fail("infeasible", "initial_memory_exceeded", "Resident state exceeds memory capacity", pool=pool, required_bytes=value, capacity_bytes=graph.memory_caps[pool])

    waiting_dependencies = {eid: len(set(event.deps)) for eid, event in events.items()}
    ready = {eid for eid, count in waiting_dependencies.items() if count == 0}
    # Keep admission timing only for currently ready/running events. Completed
    # timings live in the normal trace, including its compact payload mode.
    ready_timing = {eid: {"dependency_ready_s": 0.0, "slot_wait_s": 0.0, "memory_wait_s": 0.0}
                    for eid in ready}
    while completed_count < len(events):
        # Complete events before admitting new ones, so same-time releases can
        # unblock allocations without introducing artificial overlap.
        finished = []
        time_tolerance = max(1e-15, abs(now) * 1e-14)
        for eid, state in active.items():
            startup_over = now >= state["ready"] - time_tolerance
            no_work = not state["service"][1]
            if startup_over and (state["remaining"] <= 1e-10 or no_work):
                finished.append(eid)
        for eid in sorted(finished):
            state = active.pop(eid)
            event = events[eid]
            for alloc_id in event.releases:
                alloc = allocations.pop(alloc_id, None)
                if alloc is None:
                    return fail("error", "invalid_release", "Releasing an allocation that is not live", event=eid, allocation=alloc_id)
                used[alloc.pool] = max(0.0, used[alloc.pool] - alloc.size_bytes)
                memory_log.append({"sequence": len(memory_log), "time_s": now, "action": "release", "allocation_id": alloc.id, "pool": alloc.pool, "bytes": alloc.size_bytes})
                category_used[alloc.pool][alloc.category] -= alloc.size_bytes
            held_slots.difference_update(event.slots)
            completed_count += 1
            # Dependency edges have no further use after their source finishes.
            for successor in successors.pop(eid, ()):
                waiting_dependencies[successor] -= 1
                if waiting_dependencies[successor] == 0:
                    ready.add(successor)
                    ready_timing[successor] = {"dependency_ready_s": now, "slot_wait_s": 0.0, "memory_wait_s": 0.0}
            row = {"id": eid, "start_s": state["start"], "end_s": now,
                          "duration_s": now - state["start"], "startup_s": event.startup_s,
                          "metadata": dict(event.metadata), "slots": list(event.slots)}
            row.update(state["admission_timing"])
            if retain_event_payload:
                row.update(work=dict(event.work), allocations=[asdict(a) for a in event.allocs], releases=list(event.releases))
            trace.append(row)
        admitted = False
        blocked = []
        wait_reasons = {}
        for eid in sorted(ready):
            event = events[eid]
            if held_slots.intersection(event.slots):
                wait_reasons[eid] = "slot_wait_s"
                continue
            requested: dict[str, float] = defaultdict(float)
            for alloc in event.allocs:
                requested[alloc.pool] += alloc.size_bytes
            shortage = {pool: used[pool] + amount - graph.memory_caps[pool]
                        for pool, amount in requested.items()
                        if used[pool] + amount > graph.memory_caps[pool] * (1 + 1e-12)}
            if shortage:
                blocked.append({"event": eid, "shortage_bytes": shortage})
                wait_reasons[eid] = "memory_wait_s"
                continue
            for alloc in event.allocs:
                allocate(alloc)
            ready.remove(eid)
            del waiting_dependencies[eid]
            held_slots.update(event.slots)
            active[eid] = {"start": now, "ready": now + event.startup_s, "remaining": 1.0,
                           "service": _prepare_service(event.work, capacities),
                           "admission_timing": ready_timing.pop(eid)}
            admitted = True
        if admitted or finished:
            continue
        if not active:
            if blocked:
                return fail("infeasible", "memory_admission_deadlock", "Ready events cannot fit and no running event can release memory", blocked=blocked)
            if completed_count == len(events):
                break
            return fail("error", "scheduler_deadlock", "No executable event remains")
        working = {eid: state["service"] for eid, state in active.items() if now >= state["ready"]}
        rates = _max_min_rates(working, capacities)
        intervals = [state["ready"] - now for state in active.values() if state["ready"] > now]
        intervals.extend(active[eid]["remaining"] / rate for eid, rate in rates.items() if rate > 0)
        if not intervals:
            return fail("error", "resource_deadlock", "Running events cannot make progress")
        delta = min(intervals)
        if not math.isfinite(delta) or delta < 0:
            return fail("error", "invalid_time_step", "Nonfinite or negative simulation time step")
        busy_resources: set[str] = set()
        for eid, rate in rates.items():
            progress = min(active[eid]["remaining"], delta * rate)
            active[eid]["remaining"] -= progress
            for key, amount in events[eid].work.items():
                work_done[key] += amount * progress
                if amount > 0 and progress > 0:
                    busy_resources.add(key)
        for key in busy_resources:
            busy[key] += delta
        # Slot blocking takes precedence when both admission constraints apply;
        # these disjoint elapsed intervals sum to start - dependency-ready.
        for eid, reason in wait_reasons.items():
            ready_timing[eid][reason] += delta
        now += delta
    trace.sort(key=lambda item: (item["start_s"], item["id"]))
    return result("feasible")

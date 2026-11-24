"""Shared tensor and execution ledger. All quantities use bytes, seconds and FLOPs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

DTYPE_BYTES = {"fp4": 0.5, "fp8": 1, "fp16": 2, "bf16": 2, "fp32": 4, "int8": 1}


@dataclass(slots=True)
class Allocation:
    id: str
    pool: str
    size_bytes: float
    category: str = "workspace"
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Resource:
    id: str
    capacity: float
    unit: str
    kind: str = "bandwidth"
    capacity_schedule: tuple[tuple[float, float], ...] = ()

    def validate_schedule(self):
        previous = -1.0
        for time, capacity in self.capacity_schedule:
            if not math.isfinite(time) or time < 0 or time <= previous:
                raise ValueError('Schedule times must be finite, nonnegative and strictly increasing')
            if not math.isfinite(capacity) or capacity <= 0:
                raise ValueError('Scheduled capacity must be finite and positive')
            previous = time

    def available_work(self, end):
        """Integral of effective capacity over [0, end]."""
        start, capacity, total = 0.0, self.capacity, 0.0
        for time, value in self.capacity_schedule:
            if time >= end:
                break
            total += (time - start) * capacity
            start, capacity = time, value
        return total + (end - start) * capacity

    def relative_to(self, start):
        """Express the global schedule on a streaming chunk's local clock."""
        schedule = shift_schedule(self.capacity_schedule, start)
        if schedule and schedule[0][0] == 0:
            return replace(self, capacity=schedule[0][1], capacity_schedule=schedule[1:])
        return replace(self, capacity_schedule=schedule)


def shift_schedule(points, start):
    """Retain the effective rate at start and express future changes in local seconds."""
    schedule = []
    for time, value in points:
        if time <= start:
            schedule = [(0.0, value)]
        else:
            schedule.append((time - start, value))
    return tuple(schedule)


def scheduled_solo_service(work, resources, start):
    """Solo duration and longest-serving bottleneck for simultaneous resource demand."""
    capacities, changes = {}, {}
    for key, amount in work.items():
        if amount <= 0:
            continue
        resource = resources[key]
        capacities[key] = resource.capacity
        for time, value in resource.capacity_schedule:
            if time <= start:
                capacities[key] = value
            else:
                changes.setdefault(time, {})[key] = value
    if not capacities:
        return 0.0, None
    remaining, elapsed, previous = 1.0, 0.0, start
    limiting = {}
    for time in [*sorted(changes), math.inf]:
        key = min(capacities, key=lambda k: (capacities[k] / work[k], k))
        full_duration = work[key] / capacities[key]
        needed = remaining * full_duration
        interval = min(needed, time - previous)
        limiting[key] = limiting.get(key, 0.0) + interval
        elapsed += interval
        if interval == needed:
            return elapsed, max(limiting, key=limiting.get)
        remaining -= interval / full_duration
        capacities.update(changes[time])
        previous = time


@dataclass(slots=True)
class Event:
    id: str
    deps: tuple[str, ...] = ()
    work: dict[str, float] = field(default_factory=dict)
    startup_s: float = 0.0
    slots: tuple[str, ...] = ()
    allocs: tuple[Allocation, ...] = ()
    releases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Graph:
    resources: dict[str, Resource] = field(default_factory=dict)
    memory_caps: dict[str, float] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    initial_allocations: list[Allocation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

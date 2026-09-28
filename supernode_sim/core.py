"""Shared tensor and execution ledger. All quantities use bytes, seconds and FLOPs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DTYPE_BYTES = {"fp4": 0.5, "fp8": 1, "fp16": 2, "bf16": 2, "fp32": 4, "int8": 1}


@dataclass
class Allocation:
    id: str
    pool: str
    size_bytes: float
    category: str = "workspace"
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    id: str
    capacity: float
    unit: str
    kind: str = "bandwidth"


@dataclass
class Event:
    id: str
    deps: tuple[str, ...] = ()
    work: dict[str, float] = field(default_factory=dict)
    startup_s: float = 0.0
    slots: tuple[str, ...] = ()
    allocs: tuple[Allocation, ...] = ()
    releases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    resources: dict[str, Resource] = field(default_factory=dict)
    memory_caps: dict[str, float] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    initial_allocations: list[Allocation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

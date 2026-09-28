"""Auditable tensor/state attribution and concurrent memory-peak composition.

Independent component peaks are useful for sizing a component in isolation, but
are never added to claim a device or cluster peak. Actual peak attribution is a
snapshot of the engine's ordered allocation/release log.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from collections import defaultdict
from typing import Any

from .core import Allocation, Graph


_STATE_ALIASES = {
    "weight": "weights", "parameters": "weights", "parameter": "weights",
    "gradient": "gradients", "grad": "gradients", "master": "master_weights",
    "master_weight": "master_weights", "adam_m": "adam_momentum",
    "exp_avg": "adam_momentum", "adam_v": "adam_variance",
    "exp_avg_sq": "adam_variance", "optimizer_momentum": "momentum",
    "optimizer_variance": "variance", "muon_buffer": "muon_momentum",
    "activations": "activation", "activation_checkpoint": "activation",
    "recompute_workspace": "activation", "kv": "kv_cache", "kv_payload": "kv_cache",
    "recurrent": "recurrent_state", "quantization": "quantization_metadata",
    "weight_scales": "quantization_metadata", "kv_scales": "quantization_metadata",
    "communication_buffer": "buffers", "conversion_buffer": "buffers",
    "workspace": "buffers", "offload_gradient": "buffers", "runtime_reserve": "runtime",
    "optimizer_states": "optimizer_combined", "unsharded_gradients": "gradients",
    "gathered_weights": "weight_materialization", "extra_weight_copy": "extra_weight_copy",
}
_SYSTEM_CATEGORIES = {"runtime", "runtime_reserve", "workspace", "communication_buffer",
                      "conversion_buffer", "buffers"}


def _pool_kind(pool: str) -> str:
    if pool.endswith(".hbm"):
        return "device_hbm"
    if pool.endswith(".dram"):
        return "host_dram"
    if pool.startswith("ssd"):
        return "ssd"
    if pool.startswith("remote"):
        return "remote_memory"
    return "other"


def _labels(allocation: Allocation) -> tuple[str, str, str]:
    metadata = allocation.metadata
    explicit = metadata.get("component")
    if explicit is not None and str(explicit).strip():
        component, attribution = str(explicit), "explicit"
    elif allocation.category in _SYSTEM_CATEGORIES:
        component, attribution = "System", "system"
    else:
        component, attribution = "Unassigned", "unassigned"
    state = str(metadata.get("state") or allocation.category or "unassigned")
    return component, _STATE_ALIASES.get(state, state), attribution


def _composition(values: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    return [{"component": component, "state": state, "bytes": amount}
            for (component, state), amount in sorted(values.items()) if amount > 0]


def _sum_by(values: dict[tuple[str, str], float], index: int) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for labels, amount in values.items():
        result[labels[index]] += amount
    return {key: amount for key, amount in sorted(result.items()) if amount > 0}


def _pool_label(pools: list[str]) -> str:
    pools = sorted(pools, key=lambda name: tuple(int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)))
    if len(pools) <= 4:
        return ", ".join(pools)
    return f"{pools[0]} … {pools[-1]} ({len(pools)} pools)"


def _equivalent_groups(pools: list[dict]) -> list[dict]:
    """Group only equal capacities and equal reported per-pool compositions."""
    grouped: dict[tuple, dict] = {}
    for pool in pools:
        signature = (pool["pool_kind"], pool["capacity_bytes"], pool["resident_bytes"],
                     pool["peak_bytes"], pool["peak_basis"], pool["optimistic_peak_bytes"],
                     tuple((r["component"], r["state"], r["bytes"]) for r in pool["resident_composition"]),
                     tuple((r["component"], r["state"], r["bytes"]) for r in pool["composition_at_peak"]),
                     tuple(pool["component_independent_peak_bytes"].items()),
                     tuple(pool["state_independent_peak_bytes"].items()))
        if signature not in grouped:
            grouped[signature] = {
                "pool_kind": pool["pool_kind"], "pools": [], "pool_count": 0,
                "capacity_bytes_per_pool": pool["capacity_bytes"],
                "resident_bytes_per_pool": pool["resident_bytes"],
                "peak_bytes_per_pool": pool["peak_bytes"], "peak_basis": pool["peak_basis"],
                "optimistic_peak_bytes_per_pool": pool["optimistic_peak_bytes"],
                "composition_at_peak_per_pool": pool["composition_at_peak"],
                "component_independent_peak_bytes_per_pool": pool["component_independent_peak_bytes"],
                "state_independent_peak_bytes_per_pool": pool["state_independent_peak_bytes"],
                "peak_times_s": [],
            }
        group = grouped[signature]
        group["pools"].append(pool["pool"])
        group["pool_count"] += 1
        if pool["peak_time_s"] is not None:
            group["peak_times_s"].append(pool["peak_time_s"])
    result = []
    for index, group in enumerate(grouped.values(), 1):
        times = group.pop("peak_times_s")
        group.update(group_id=f"G{index}", pool_label=_pool_label(group["pools"]),
                     peak_time_range_s=[min(times), max(times)] if times else None)
        result.append(group)
    return result


def build_memory_breakdown(graph: Graph, simulation: dict | None = None,
                           estimate: dict | None = None, *,
                           include_ephemeral_details: bool = True) -> dict[str, Any]:
    """Attribute graph allocations and replay an optional exact engine log.

    ``simulation.memory_log`` must include successful initial allocations, then
    successful event allocations/releases with monotonic ``sequence`` and
    ``time_s``. No log means resident-only attribution, never reconstructed
    chronology from ambiguous same-time event traces.

    With ``include_ephemeral_details=False``, every allocation still participates
    in exact replay, but temporary state=buffers allocations keep only their
    existing Allocation reference rather than a wide copied tensor-detail row.
    """
    tensors: dict[str, dict] = {}
    allocations: dict[str, Allocation | None] = {}
    resident: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    pool_names = set(graph.memory_caps)

    def add_tensor(allocation: Allocation, event_id: str | None) -> None:
        if allocation.id in allocations:
            raise ValueError(f"Duplicate allocation ID in memory ledger: {allocation.id}")
        if allocation.size_bytes < 0 or not math.isfinite(allocation.size_bytes):
            raise ValueError(f"Invalid memory allocation size: {allocation.id}")
        component, state, attribution = _labels(allocation)
        allocations[allocation.id] = allocation
        pool_names.add(allocation.pool)
        if event_id is None:
            resident[allocation.pool][component, state] += allocation.size_bytes
        if not include_ephemeral_details and event_id is not None and state == "buffers":
            return
        metadata = dict(allocation.metadata)
        row = {
            "allocation_id": allocation.id, "pool": allocation.pool, "pool_kind": _pool_kind(allocation.pool),
            "component": component, "layer": metadata.get("layer"),
            "tensor_name": str(metadata.get("tensor_name") or allocation.id),
            "tensor_members": list(metadata.get("tensor_members") or []),
            "state": state, "category": allocation.category, "dtype": allocation.dtype,
            "shape": list(allocation.shape), "bytes": allocation.size_bytes,
            "is_initial": event_id is None, "allocation_event": event_id,
            "release_events": [], "attribution": attribution,
            "observed": False, "allocated_at_s": None, "released_at_s": None,
            "metadata": metadata,
        }
        tensors[allocation.id] = row

    for allocation in graph.initial_allocations:
        add_tensor(allocation, None)
    for event in graph.events:
        for allocation in event.allocs:
            add_tensor(allocation, event.id)
    # Do not build an enormous parallel release index for collective buffers.
    # Model rows own their existing release lists directly.
    for event in graph.events:
        for aid in event.releases:
            tensor = tensors.get(aid)
            if tensor is not None:
                tensor["release_events"].append(event.id)

    has_log = simulation is not None and "memory_log" in simulation
    mode = ("simulated" if simulation.get("status") == "feasible" else "observed_partial") if has_log else "resident_only"
    basis = "observed_execution_peak" if has_log else "initial_resident_only"
    all_pools = sorted(pool_names)
    current: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    totals: dict[str, float] = defaultdict(float)
    peak_values = {pool: 0.0 for pool in all_pools}
    peak_times: dict[str, float | None] = {pool: None for pool in all_pools}
    peak_compositions: dict[str, dict] = {pool: {} for pool in all_pools}
    component_current: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    state_current: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    component_peaks: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    state_peaks: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_peaks: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    cluster_current: dict[tuple[str, str], float] = defaultdict(float)
    cluster_total, cluster_peak, cluster_peak_time = 0.0, 0.0, None
    cluster_composition: dict[tuple[str, str], float] = {}

    def change(pool: str, component: str, state: str, delta: float, time_s: float | None) -> None:
        nonlocal cluster_total, cluster_peak, cluster_peak_time, cluster_composition
        pair = component, state
        totals[pool] += delta
        current[pool][pair] += delta
        component_current[pool][component] += delta
        state_current[pool][state] += delta
        cluster_total += delta
        cluster_current[pair] += delta
        if totals[pool] > peak_values[pool]:
            peak_values[pool], peak_times[pool] = totals[pool], time_s
            peak_compositions[pool] = dict(current[pool])
        if cluster_total > cluster_peak:
            cluster_peak, cluster_peak_time = cluster_total, time_s
            cluster_composition = dict(cluster_current)
        component_peaks[pool][component] = max(component_peaks[pool][component], component_current[pool][component])
        state_peaks[pool][state] = max(state_peaks[pool][state], state_current[pool][state])
        pair_peaks[pool][pair] = max(pair_peaks[pool][pair], current[pool][pair])

    log = simulation.get("memory_log", []) if has_log else []
    if has_log:
        # Engine logs already arrive ordered. Reuse that list instead of copying
        # millions of references plus a sequence list and a sequence set.
        if isinstance(log, Sequence):
            ordered = log if all(log[i - 1]["sequence"] < log[i]["sequence"] for i in range(1, len(log))) else sorted(log, key=lambda row: row["sequence"])
        else:
            # A checkpoint can stream an already-ordered log without loading
            # millions of dictionaries; validate ordering during the replay.
            ordered = log
        active: set[str] = set()
        last_time = 0.0
        last_sequence = None
        for row in ordered:
            if last_sequence is not None and row["sequence"] <= last_sequence:
                raise ValueError("Memory log sequence numbers must be unique and increasing")
            last_sequence = row["sequence"]
            aid, action = row["allocation_id"], row["action"]
            if aid not in allocations:
                raise ValueError(f"Memory log references unknown allocation: {aid}")
            allocation = allocations[aid]
            if allocation is None:
                if action == "allocate":
                    raise ValueError(f"Memory allocation repeated in log: {aid}")
                raise ValueError(f"Memory release before allocation: {aid}")
            tensor = tensors.get(aid)
            component, state, _ = _labels(allocation)
            time_s = float(row["time_s"])
            if not math.isfinite(time_s) or time_s < last_time - 1e-12:
                raise ValueError("Memory log is not chronological")
            last_time = time_s
            if row["pool"] != allocation.pool or not math.isclose(row["bytes"], allocation.size_bytes, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"Memory log disagrees with allocation ledger: {aid}")
            if action == "allocate":
                if aid in active:
                    raise ValueError(f"Memory allocation repeated in log: {aid}")
                active.add(aid)
                if tensor is not None:
                    tensor.update(observed=True, allocated_at_s=time_s)
                delta = allocation.size_bytes
            elif action == "release":
                if aid not in active:
                    raise ValueError(f"Memory release before allocation: {aid}")
                active.remove(aid)
                if tensor is not None:
                    tensor["released_at_s"] = time_s
                delta = -allocation.size_bytes
                # The existing lookup slot also records that the ID completed;
                # no additional all-observed-ID set is needed for validation.
                allocations[aid] = None
            else:
                raise ValueError(f"Unknown memory log action: {action}")
            change(allocation.pool, component, state, delta, time_s)
        if simulation.get("status") == "feasible":
            missing = [a.id for a in graph.initial_allocations if not tensors[a.id]["observed"]]
            if missing:
                raise ValueError(f"Successful simulation log omitted initial allocations: {missing[:3]}")
    else:
        for pool, values in resident.items():
            for (component, state), amount in values.items():
                change(pool, component, state, amount, None)

    pools, component_rows, state_rows = [], [], []
    optimistic = (estimate or {}).get("optimistic_schedule_memory_peak_bytes", {})
    for pool in all_pools:
        observed = peak_compositions[pool]
        state_at_peak = _sum_by(observed, 1)
        component_at_peak = _sum_by(observed, 0)
        initial_bytes = sum(resident[pool].values())
        capacity = graph.memory_caps.get(pool)
        entry = {
            "pool": pool, "pool_kind": _pool_kind(pool), "capacity_bytes": capacity,
            "resident_bytes": initial_bytes, "resident_composition": _composition(resident[pool]),
            "peak_bytes": peak_values[pool], "peak_basis": basis, "peak_time_s": peak_times[pool],
            "composition_at_peak": _composition(observed), "state_bytes_at_peak": state_at_peak,
            "component_bytes_at_peak": component_at_peak,
            "component_independent_peak_bytes": dict(sorted(component_peaks[pool].items())),
            "state_independent_peak_bytes": dict(sorted(state_peaks[pool].items())),
            "optimistic_peak_bytes": optimistic.get(pool),
            "resident_exceeds_capacity": capacity is not None and initial_bytes > capacity,
            "observed_peak_exceeds_capacity": has_log and capacity is not None and peak_values[pool] > capacity,
        }
        pools.append(entry)
        for pair in sorted(set(resident[pool]) | set(pair_peaks[pool])):
            component, state = pair
            component_rows.append({"pool": pool, "pool_kind": _pool_kind(pool), "component": component, "state": state,
                "resident_bytes": resident[pool].get(pair, 0), "bytes_at_pool_peak": observed.get(pair, 0),
                "independent_component_state_peak_bytes": pair_peaks[pool].get(pair, 0),
                "independent_component_peak_bytes": component_peaks[pool].get(component, 0),
                "pool_peak_bytes": peak_values[pool], "pool_peak_time_s": peak_times[pool], "peak_basis": basis})
        resident_states = _sum_by(resident[pool], 1)
        for state in sorted(set(resident_states) | set(state_peaks[pool])):
            state_rows.append({"pool": pool, "pool_kind": _pool_kind(pool), "state": state,
                "resident_bytes": resident_states.get(state, 0), "bytes_at_pool_peak": state_at_peak.get(state, 0),
                "independent_state_peak_bytes": state_peaks[pool].get(state, 0),
                "pool_peak_bytes": peak_values[pool], "pool_peak_time_s": peak_times[pool], "peak_basis": basis})
    notes = [
        "All sizes are bytes; GB/TB are decimal and GiB/TiB use powers of 1024.",
        "Each pool's composition_at_peak is one concurrent snapshot; component/state independent peaks occur at potentially different times and must not be summed.",
        "Different pools may peak at different times; cluster peak is replayed concurrently, never the sum of pool peaks.",
        "Unknown model attribution remains Unassigned; generic infrastructure allocations remain System. Legacy combined optimizer state is not split into invented tensors.",
        "Equivalent groups have equal per-pool capacities and reported compositions; group count is not a concurrency multiplier for peak time.",
    ]
    if mode == "resident_only":
        notes.append("No ordered execution memory log is available: composition is initial residency only. Optimistic schedule peaks, when provided, are separate estimates with no claimed component attribution.")
    elif mode == "observed_partial":
        notes.append("Execution stopped before completion: reported peaks cover only admitted allocations before the stop, not full-workload capacity requirements.")
    omitted_details = len(allocations) - len(tensors)
    if omitted_details:
        notes.append("Ephemeral state=buffers tensor detail rows were not materialized to bound report memory; every allocation/release remains included in exact peak and component/state replay.")
    logical_states: dict[str, dict[str, float]] = {}
    for component, states in graph.metadata.get("logical_component_state_bytes", {}).items():
        logical_states[component] = {}
        for state, amount in states.items():
            canonical = _STATE_ALIASES.get(state, state)
            logical_states[component][canonical] = logical_states[component].get(canonical, 0) + amount
    return {
        "schema_version": "1.0", "mode": mode, "units": "byte",
        "component_inventory": graph.metadata.get("component_inventory"),
        "logical_component_state_bytes": logical_states,
        "tensor_detail_export": {"total_rows": len(allocations), "retained_rows": len(tensors),
                                 "omitted_rows": omitted_details,
                                 "include_ephemeral_details": include_ephemeral_details,
                                 "filter": "All model allocations and initial buffers; temporary state=buffers details omitted." if omitted_details else "Complete allocation detail."},
        "tensors": list(tensors.values()), "pools": pools,
        "equivalent_pool_groups": _equivalent_groups(pools),
        "component_state_rows": component_rows, "state_rows": state_rows,
        "cluster": {"resident_bytes": sum(sum(v.values()) for v in resident.values()),
                    "peak_bytes": cluster_peak, "peak_time_s": cluster_peak_time, "peak_basis": basis,
                    "composition_at_peak": _composition(cluster_composition)},
        "notes": notes,
    }

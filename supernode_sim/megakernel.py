"""Conservative, profile-driven fusion of exclusive local kernel chains.

SuperKernel describes binary kernel scheduling/fusion, not a universal speedup:
https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md
Memory traffic savings are separate explicit evidence; graph capture alone is
not assumed to remove intermediate global-memory accesses.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite
import re

from .core import Allocation, Event, Graph, Resource


POLICY = (
    "Only exclusive same-rank/phase/layer serial kernel chains are fused; no collective, "
    "cross-rank edge, fan-out or internal external dependency may be crossed. "
    "All original allocations move to fused start and releases to fused end (conservative_union); "
    "saved activations remain live outside the group, and peak memory may be overestimated. "
    "FLOPs and other resource work are conserved except explicit bounded HBM savings. "
    "Without a fused duration profile, a serial subkernel service floor prevents invented compute/memory overlap. "
    "Coverage measures matched graph events, not measured serving accuracy or guaranteed acceleration."
)


def megakernel_summary(enabled=False) -> dict:
    return {"enabled": enabled, "eligible_events": 0, "matched_events": 0,
            "matched_groups": 0, "skipped_candidates": 0, "coverage_fraction": 0.0,
            "saved_hbm_bytes": 0.0, "original_startup_s": 0.0, "fused_startup_s": 0.0,
            "skip_reasons": {}, "groups": [], "member_profiles_not_applied": [],
            "suppressed_member_profiles": [],
            "member_profile_policy": "Member profiles are not inherited; calibrate megakernel:<groupname> separately.",
            "lifecycle_mode": "conservative_union", "policy": POLICY}


def merge_megakernel_summaries(target: dict, source: dict) -> None:
    """Accumulate independently transformed streaming chunks or PD stage graphs."""
    target["enabled"] = target.get("enabled", False) or source.get("enabled", False)
    for key in ("eligible_events", "matched_events", "matched_groups", "skipped_candidates",
                "saved_hbm_bytes", "original_startup_s", "fused_startup_s"):
        target[key] = target.get(key, 0) + source.get(key, 0)
    reasons = Counter(target.get("skip_reasons", {}))
    reasons.update(source.get("skip_reasons", {}))
    target["skip_reasons"] = dict(reasons)
    target["member_profiles_not_applied"] = sorted(set(target.get("member_profiles_not_applied", [])) |
                                                   set(source.get("member_profiles_not_applied", [])))
    target["suppressed_member_profiles"] = list(target["member_profiles_not_applied"])
    groups = {row["name"]: row for row in target.get("groups", [])}
    for row in source.get("groups", []):
        out = groups.setdefault(row["name"], {"name": row["name"], "matched_groups": 0,
            "matched_events": 0, "skipped_candidates": 0, "saved_hbm_bytes": 0.0})
        for key in ("matched_groups", "matched_events", "skipped_candidates", "saved_hbm_bytes"):
            out[key] += row.get(key, 0)
    target["groups"] = list(groups.values())
    target["coverage_fraction"] = target["matched_events"] / target["eligible_events"] if target["eligible_events"] else 0.0


def _rank(event: Event) -> str | None:
    ranks = {key.split(".", 1)[0] for key in event.work if re.match(r"^npu\d+\.", key)}
    return next(iter(ranks)) if len(ranks) == 1 else None


def _selected(event, group):
    return (not group.get("phases") or event.metadata.get("phase") in group["phases"]) and (
        not group.get("layers") or event.metadata.get("layer") in group["layers"]) and (
        not group.get("ranks") or _rank(event) in group["ranks"])


def _metadata_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _metadata_strings(item)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            yield from _metadata_strings(item)


def _kernel_reason(event, rank, phase, layer):
    if _rank(event) != rank:
        return "different_or_multiple_ranks"
    if event.metadata.get("phase") != phase or event.metadata.get("layer") != layer:
        return "different_phase_or_layer"
    if event.metadata.get("collective") or event.metadata.get("operation") in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all_v"):
        return "communication_boundary"
    if event.metadata.get("calibration") or event.metadata.get("megakernel"):
        return "already_calibrated_or_fused"
    if any(not key.startswith(rank + ".") or not any(key.startswith(rank + suffix) for suffix in
            (".compute", ".hbm", ".selection")) for key in event.work):
        return "nonlocal_or_unsupported_resource"
    if any(slot != rank + ".compute_stream" for slot in event.slots):
        return "unsupported_execution_slot"
    if any(allocation.pool != rank + ".hbm" for allocation in event.allocs):
        return "nonlocal_allocation"
    return None


def _chain(start, group, by_id, children, consumed, metadata_refs):
    rank = _rank(start)
    phase, layer = start.metadata.get("phase"), start.metadata.get("layer")
    if rank is None or phase is None or layer is None:
        return None, "missing_rank_phase_or_layer"
    reason = _kernel_reason(start, rank, phase, layer)
    if reason:
        return None, reason
    members, bridges = [start], []
    visited = {start.id}
    current = start
    for kernel in group["kernels"][1:]:
        while True:
            next_ids = children[current.id]
            if len(next_ids) != 1:
                return None, "fanout_or_no_successor"
            next_event = by_id[next(iter(next_ids))]
            if next_event.id in visited:
                return None, "cyclic_chain"
            visited.add(next_event.id)
            if next_event.id in consumed:
                return None, "overlapping_group"
            if set(next_event.deps) != {current.id}:
                return None, "external_dependency_inside_group"
            if not next_event.metadata.get("kernel"):
                if (next_event.work or next_event.startup_s or next_event.slots or next_event.allocs or
                        next_event.releases or next_event.metadata):
                    return None, "nontransparent_barrier_or_communication"
                bridges.append(next_event)
                current = next_event
                continue
            if next_event.metadata.get("kernel") != kernel:
                return None, "kernel_order_mismatch"
            reason = _kernel_reason(next_event, rank, phase, layer)
            if reason:
                return None, reason
            if next_event.slots != start.slots:
                return None, "different_execution_slots"
            members.append(next_event)
            current = next_event
            break
    merged_ids = {event.id for event in members + bridges}
    if (merged_ids - {members[-1].id}) & metadata_refs:
        return None, "external_metadata_reference"
    compute_keys = {key for event in members for key in event.work if key.startswith(rank + ".compute")}
    if len(compute_keys) != 1:
        return None, "mixed_or_missing_compute_resource"
    return (members, bridges, rank), None


def _fuse(graph, group, members, bridges, rank, config):
    work = defaultdict(float)
    for event in members:
        for resource, amount in event.work.items():
            work[resource] += amount
    hbm = rank + ".hbm"
    original_hbm = work.get(hbm, 0.0)
    allocations = tuple(a for event in members for a in event.allocs)
    releases = tuple(dict.fromkeys(aid for event in members for aid in event.releases))
    released = set(releases)
    internal_bytes = sum(a.size_bytes for a in allocations if a.id in released and a.pool == hbm)
    savings = group.get("saved_hbm_bytes", 0.0)
    limit = min(original_hbm, 2 * internal_bytes)
    if savings > limit + 1e-9:
        raise ValueError(f"Megakernel {group['name']}: saved_hbm_bytes={savings} exceeds bounded internal traffic {limit}")
    work[hbm] = original_hbm - savings
    # Keep serial subkernel service absent an explicit fused timing profile.
    # Credit bounded savings proportionally across original HBM demands only.
    ratio = work[hbm] / original_hbm if original_hbm else 1.0
    serial_floor = sum(max((amount * (ratio if key == hbm else 1.0) / graph.resources[key].capacity
                            for key, amount in event.work.items()), default=0.0) for event in members)
    service = group.get("duration_s")
    if service is not None:
        compute = next(key for key in work if key.startswith(rank + ".compute"))
        dtype = "fp32" if compute.endswith(".fp32") else config.get("execution", {}).get("precision", {}).get("compute", "fp16")
        peak = config.get("hardware", {}).get("npu", {}).get("flops", {}).get(dtype, graph.resources[compute].capacity)
        calibrated = rank + ".compute.megakernel." + dtype
        graph.resources[calibrated] = Resource(calibrated, peak, "FLOP/s", "compute")
        work[calibrated] = work.pop(compute)
    physical_floor = max((amount / graph.resources[key].capacity for key, amount in work.items()), default=0.0)
    if service is None:
        service = serial_floor
    if service < physical_floor * (1 - 1e-9):
        raise ValueError(f"Megakernel {group['name']}: duration_s is below conserved physical compute/memory limits")
    final_id = members[-1].id  # Preserve all downstream users of group completion.
    limiter = "kernel_limit:" + final_id
    graph.resources[limiter] = Resource(limiter, 1.0, "second/second", "kernel_timing")
    work[limiter] = service
    workspace_bytes = group.get("workspace_bytes", 0.0)
    if workspace_bytes:
        workspace = Allocation(final_id + ".megakernel_workspace", hbm, workspace_bytes, "megakernel_workspace",
            metadata={"component": "Megakernel", "state": "megakernel_workspace", "layer": members[0].metadata["layer"]})
        allocations += (workspace,)
        releases += (workspace.id,)
    ledger = {"group": group["name"], "member_ids": [e.id for e in members],
        "member_kernels": [e.metadata["kernel"] for e in members], "bridge_ids": [e.id for e in bridges],
        "original_startup_s": sum(e.startup_s for e in members), "fused_startup_s": group["fused_startup_s"],
        "saved_hbm_bytes": savings, "internal_buffer_bytes": internal_bytes,
        "workspace_bytes": workspace_bytes, "service_time_s": service,
        "workspace_policy": "Capacity budget only; no unobserved workspace read/write traffic is invented.",
        "service_time_source": "explicit_profile" if group.get("duration_s") is not None else "serial_resource_floor",
        "source": group.get("source", {"status": "assumed"}), "lifecycle_mode": "conservative_union",
        "member_metadata": [{"id": e.id, **e.metadata} for e in members]}
    metadata = {"kernel": "megakernel:" + group["name"], "component": "Megakernel",
        "phase": members[0].metadata["phase"], "layer": members[0].metadata["layer"], "rank": rank,
        "flops": sum(amount for key, amount in work.items() if key.startswith(rank + ".compute")),
        "working_set_bytes": original_hbm + workspace_bytes, "megakernel": ledger}
    return Event(final_id, members[0].deps, dict(work), group["fused_startup_s"], members[0].slots,
                 allocations, releases, metadata)


def apply_megakernel(graph: Graph, config: dict) -> dict:
    """Transform graph in place before apply_kernel_profiles; return coverage.

    Calibration of ``megakernel:<name>`` can subsequently replace the service
    limiter by exact-work matching. Original member profiles are not inherited.
    No enabled groups / no safe matches produces no implicit speedup.
    """
    feature = config.get("execution", {}).get("features", {}).get("megakernel", {})
    summary = megakernel_summary(feature.get("enabled", False))
    if not summary["enabled"]:
        graph.metadata["megakernel"] = summary
        return summary
    groups = feature.get("groups", [])
    if not groups:
        raise ValueError("enabled megakernel requires explicit groups")
    by_id = {event.id: event for event in graph.events}
    if len(by_id) != len(graph.events):
        raise ValueError("Megakernel requires unique graph event IDs")
    children = defaultdict(set)
    for event in graph.events:
        for dep in event.deps:
            children[dep].add(event.id)
    metadata_refs = set(_metadata_strings(graph.metadata))
    eligible = {event.id for event in graph.events if any(event.metadata.get("kernel") in group["kernels"]
                    and _selected(event, group) for group in groups)}
    summary["eligible_events"] = len(eligible)
    consumed, replacements, reasons, member_profiles = set(), {}, Counter(), set()
    profiles = config.get("hardware", {}).get("kernel_profiles", {})
    for group in groups:
        if len(group.get("kernels", [])) < 2:
            raise ValueError("Megakernel groups require at least two ordered kernel names")
        for key in ("fused_startup_s", "saved_hbm_bytes", "workspace_bytes"):
            value = group.get(key, 0.0)
            if not isfinite(value) or value < 0:
                raise ValueError(f"megakernel {key} must be finite and nonnegative")
        if "fused_startup_s" not in group:
            raise ValueError("fused_startup_s must be explicitly supplied")
        if group.get("duration_s") is not None and (not isfinite(group["duration_s"]) or group["duration_s"] <= 0):
            raise ValueError("megakernel duration_s must be finite and positive")
        counts = {"name": group["name"], "matched_groups": 0, "matched_events": 0,
                  "skipped_candidates": 0, "saved_hbm_bytes": 0.0}
        for event in graph.events:
            if event.id in consumed or event.metadata.get("kernel") != group["kernels"][0] or not _selected(event, group):
                continue
            result, reason = _chain(event, group, by_id, children, consumed, metadata_refs)
            if reason:
                reasons[reason] += 1
                counts["skipped_candidates"] += 1
                continue
            members, bridges, rank = result
            fused = _fuse(graph, group, members, bridges, rank, config)
            consumed.update(item.id for item in members + bridges)
            replacements[event.id] = fused
            counts["matched_groups"] += 1
            counts["matched_events"] += len(members)
            counts["saved_hbm_bytes"] += group.get("saved_hbm_bytes", 0.0)
            summary["original_startup_s"] += sum(e.startup_s for e in members)
            summary["fused_startup_s"] += group["fused_startup_s"]
            member_profiles.update(e.metadata["kernel"] for e in members if e.metadata["kernel"] in profiles)
        summary["groups"].append(counts)
        for key in ("matched_groups", "matched_events", "skipped_candidates", "saved_hbm_bytes"):
            summary[key] += counts[key]
    graph.events = [replacements[event.id] if event.id in replacements else event
                    for event in graph.events if event.id not in consumed or event.id in replacements]
    summary["skip_reasons"] = dict(reasons)
    summary["member_profiles_not_applied"] = sorted(member_profiles)
    summary["suppressed_member_profiles"] = sorted(member_profiles)
    summary["coverage_fraction"] = summary["matched_events"] / len(eligible) if eligible else 0.0
    graph.metadata["megakernel"] = summary
    if POLICY not in graph.metadata.setdefault("assumptions", []):
        graph.metadata["assumptions"].append(POLICY)
    return summary

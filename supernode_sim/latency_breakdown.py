"""Additive latency attribution along the observed dependency critical path.

This is a wall-clock attribution, not the sum of work across parallel ranks.
Only a single latest-finishing dependency is followed at each fan-in. The
fluid engine's simultaneous resource demands use a roofline maximum, so a
transfer's baseline service belongs to its limiting resource's domain.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from .core import Event, Graph
from .operator_costs import OperatorLedger, empty_costs, operator_identity


COMPONENTS = {
    "compute_s": "Compute/operator service (including local memory access)",
    "communication_node_s": "Communication within a host/node",
    "communication_scaleup_s": "Communication across hosts in scale-up fabric",
    "communication_scaleout_s": "Communication over scale-out fabric",
    "communication_other_s": "Other transfer service (including endpoint memory)",
    "synchronization_wait_s": "Explicit synchronization/barrier delay",
    "scheduling_wait_s": "Ready-event execution-slot wait",
    "memory_wait_s": "Ready-event memory admission wait",
    "resource_contention_s": "Active service slowdown from shared resources",
    "other_s": "Other or unclassified elapsed time",
}


def latency_breakdown_metadata() -> dict[str, Any]:
    """Shared schema for full-graph and streaming inference reports."""
    return {
        "method": "actual_dependency_critical_path",
        "components": dict(COMPONENTS),
        "notes": [
            "A matched collective call timer is a concurrent completion lower bound, classified as other communication when it dominates. Its whole-call measurement may include pack/unpack; it is not split into unsupported fabric or compute measurements.",
            "Components are additive wall-clock seconds on one observed latest-finishing dependency path; parallel ranks/events are not summed. Equal completion times choose the lexically smallest event ID.",
            "TTFT uses [0, first-token completion]. TPOT is the mean of consecutive output-token windows; aggregate TTFT and TPOT each select the same worst request as the scalar metric, not the maximum of each component.",
            "Compute includes operator-local memory service and launch time. Transfer baseline uses the maximum work/capacity resource time, attributed to that bottleneck's path domain; equal resource times choose the lexically smallest resource ID. Endpoint memory bottlenecks are other communication, not fabricated network time.",
            "Path startup is allocated by link-domain latency. node means one CPU host and its attached NPUs; scale-up means inter-host SU links, and scale-out follows SO links regardless of host numbering.",
            "Dependency fan-in waiting is already represented by the selected predecessor and is not added again. Synchronization includes explicit barrier startup only; zero-duration barriers can therefore report zero synchronization time.",
            "Resource contention is active work duration minus solo roofline time. Slot wait takes precedence when slot and memory admission both block. A window clipping part of an event prorates its service/contending share or admission wait reasons; exact sub-event progress is not retained.",
            "Fixed-batch model: no arrival queue, tokenization, serving frontend, or unmodeled host scheduling delay. TPOP is an alias of TPOT (time per output token), excluding the first token.",
            "operator_breakdown refines the same window by phase/component/operator. Compute and memory overlap: baseline service is assigned to the limiting resource, not summed. Kernel launch, waits and contention are separate; opaque kernel profile time remains unattributed_service_s. Operator shares divide by the window; internal shares divide by that operator's attributed time.",
        ],
        "ttft": None,
        "tpot": None,
        "unavailable_reason": None,
    }


def _empty_components() -> dict[str, float]:
    return {key: 0.0 for key in COMPONENTS}


def combine_breakdowns(records: Iterable[dict[str, Any]], divisor: float | None = None) -> dict[str, Any] | None:
    """Sum windows, or average them when divisor is omitted; recompute shares."""
    totals = _empty_components()
    count, duration = 0, 0.0
    operators = OperatorLedger()
    complete_operators = True
    for record in records:
        count += 1
        duration += record["total_s"]
        for key in totals:
            totals[key] += record["components_s"].get(key, 0.0)
        if record.get('operator_breakdown') is not None:
            operators.merge(record['operator_breakdown'])
        else:
            complete_operators = False
    if not count:
        return None
    denominator = count if divisor is None else divisor
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Breakdown divisor must be finite and positive')
    result = {"total_s": duration / denominator,
              "components_s": {key: value / denominator for key, value in totals.items()}}
    if divisor is None:
        result['interval_count'] = count
    if complete_operators:
        result['operator_breakdown'] = operators.finish(result['total_s'], denominator)
    return result


def mean_breakdowns(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return combine_breakdowns(records)


def _domain_component(domain: str) -> str:
    return {
        "node": "communication_node_s", "host": "communication_node_s",
        "H2D": "communication_node_s", "D2H": "communication_node_s",
        "SU": "communication_scaleup_s", "SO": "communication_scaleout_s",
    }.get(domain, "communication_other_s")


class _WindowAttributor:
    def __init__(self, graph: Graph, simulation: dict[str, Any]):
        self.graph = graph
        self.events = {event.id: event for event in graph.events}
        trace = simulation.get("trace", ())
        if isinstance(trace, (list, tuple)):
            self.rows = {row["id"]: row for row in trace}
        else:
            # Checkpoint/JSONL traces may be repeatable lazy iterables. Keep
            # only timings, not rehydrated work/allocations/metadata payloads.
            fields = ("start_s", "end_s", "slot_wait_s", "memory_wait_s")
            self.rows = {row["id"]: {key: row[key] for key in fields if key in row} for row in trace}
        self.feasible = simulation.get("status") == "feasible"

    def _resource_domain(self, event: Event, resource: str) -> str:
        metadata = event.metadata
        domains = metadata.get("resource_domains", {})
        if resource in domains:
            return domains[resource]
        # Compatibility for hand-built or older graphs. Current topology
        # events carry explicit per-resource domains, including shared links.
        if ".SO." in resource or ".nic." in resource or resource.endswith(":SO"):
            return "SO"
        if ".host." in resource or resource.endswith((":H2D", ":D2H")):
            return "node"
        if ".SU." in resource or resource.endswith(":SU"):
            per_host = self.graph.metadata.get("topology", {}).get("npus_per_host")
            src, dst = metadata.get("src", ""), metadata.get("dst", "")
            if (per_host and src.startswith("npu") and dst.startswith("npu")
                    and src[3:].isdigit() and dst[3:].isdigit()
                    and int(src[3:]) // per_host == int(dst[3:]) // per_host
                    and "SO" not in metadata.get("domains", ())):
                return "node"
            return "SU"
        return "other"

    def _operator_component(self, event: Event) -> str:
        metadata = event.metadata
        if metadata.get("type") == "collective_timing":
            return "communication_other_s"
        if (metadata.get("type") == "barrier" or metadata.get("operation") == "barrier"
                or metadata.get("kind") == "barrier") and not any(event.work.values()):
            return "synchronization_wait_s"
        if any(self.graph.resources[key].kind in {"compute", "iops"}
               or key.endswith((".hbm", ".dram", ".memory"))
               for key, amount in event.work.items() if amount > 0):
            return "compute_s"
        if metadata.get("operation") or metadata.get("kernel"):
            return "compute_s"
        return "other_s"

    @staticmethod
    def _overlap(left: float, right: float, start: float, end: float) -> float:
        return max(0.0, min(right, end) - max(left, start))

    def _add_event(self, event: Event, row: dict[str, Any], dependency_ready: float,
                   start: float, end: float, parts: dict[str, float], costs: dict[str, float]) -> None:
        event_start, event_end = row["start_s"], row["end_s"]
        waiting = max(0.0, event_start - dependency_ready)
        clipped_wait = self._overlap(dependency_ready, event_start, start, end)
        if waiting and clipped_wait:
            slot = max(0.0, row.get("slot_wait_s", 0.0))
            memory = max(0.0, row.get("memory_wait_s", 0.0))
            # Older traces do not distinguish ready-but-not-running reasons.
            if "slot_wait_s" not in row and "memory_wait_s" not in row:
                slot = waiting
            scale = clipped_wait / max(waiting, slot + memory)
            parts["scheduling_wait_s"] += slot * scale
            parts["memory_wait_s"] += memory * scale
            parts["other_s"] += max(0.0, clipped_wait - (slot + memory) * scale)
            costs["scheduling_wait_s"] += slot * scale
            costs["memory_wait_s"] += memory * scale
            costs["other_s"] += max(0.0, clipped_wait - (slot + memory) * scale)
        startup_end = min(event_end, event_start + event.startup_s)
        clipped_startup = self._overlap(event_start, startup_end, start, end)
        transfer = event.metadata.get("type") == "transfer"
        if clipped_startup:
            if transfer:
                by_domain = event.metadata.get("startup_s_by_domain", {})
                domain_total = sum(max(0.0, value) for value in by_domain.values())
                denominator = max(event.startup_s, domain_total)
                if domain_total > 0 and abs(domain_total - event.startup_s) <= 8 * math.ulp(denominator):
                    denominator = domain_total
                for domain, value in by_domain.items():
                    parts[_domain_component(domain)] += clipped_startup * max(0.0, value) / denominator
                parts["communication_other_s"] += clipped_startup * max(0.0, 1.0 - domain_total / denominator)
            else:
                parts[self._operator_component(event)] += clipped_startup
            if transfer or event.metadata.get('type') == 'collective_timing':
                costs['communication_s'] += clipped_startup
            else:
                component = self._operator_component(event)
                costs['launch_s' if component == 'compute_s' else component] += clipped_startup
        active = max(0.0, event_end - startup_end)
        clipped_active = self._overlap(startup_end, event_end, start, end)
        if not clipped_active or not active:
            return
        demands = [(amount / self.graph.resources[key].capacity, key)
                   for key, amount in event.work.items() if amount > 0]
        if not demands:
            parts[self._operator_component(event)] += clipped_active
            costs['other_s'] += clipped_active
            return
        solo, bottleneck = min(demands, key=lambda item: (-item[0], item[1]))
        roundoff = 8 * (math.ulp(event_end) + math.ulp(startup_end) + math.ulp(solo))
        baseline = clipped_active if active - solo <= roundoff else clipped_active * min(1.0, solo / active)
        component = (_domain_component(self._resource_domain(event, bottleneck))
                     if transfer else self._operator_component(event))
        parts[component] += baseline
        parts["resource_contention_s"] += clipped_active - baseline
        resource = self.graph.resources[bottleneck]
        if resource.kind == 'kernel_timing':
            detailed = 'unattributed_service_s'
        elif bottleneck.endswith(('.hbm', '.dram', '.memory')):
            detailed = 'memory_access_s'
        elif resource.kind in {'compute', 'iops'}:
            detailed = 'compute_s'
        elif transfer:
            detailed = 'communication_s'
        else:
            detailed = 'other_s'
        costs[detailed] += baseline
        costs['resource_contention_s'] += clipped_active - baseline

    def window(self, end_event_id: str, start_s: float = 0.0) -> dict[str, Any] | None:
        if not self.feasible or end_event_id not in self.rows:
            return None
        end = self.rows[end_event_id]["end_s"]
        if not math.isfinite(start_s) or start_s < 0 or start_s > end:
            raise ValueError("Latency window start must be finite and between zero and completion")
        parts = _empty_components()
        operators = OperatorLedger()
        current = end_event_id
        while current is not None:
            event, row = self.events[current], self.rows[current]
            if row["end_s"] <= start_s:
                break
            predecessor = min(event.deps, key=lambda dep: (-self.rows[dep]["end_s"], dep)) if event.deps else None
            ready = self.rows[predecessor]["end_s"] if predecessor is not None else 0.0
            costs = empty_costs()
            self._add_event(event, row, ready, start_s, end, parts, costs)
            # Zero-duration completion markers do not represent operators.
            if any(costs.values()):
                operators.add(operator_identity(event.metadata), costs)
            current = predecessor
        total = end - start_s
        # Eliminate only floating-point accumulation residual; never rescale
        # categories or invent time from the sum of parallel branches.
        residual = total - math.fsum(parts.values())
        if abs(residual) > 1e-12 + total * 1e-9:
            raise ValueError(f"Latency attribution does not cover its window: residual={residual}s")
        if residual and parts:
            largest = max(parts, key=parts.get)
            parts[largest] = max(0.0, parts[largest] + residual)
        return {"total_s": total, "components_s": parts,
                "operator_breakdown": operators.finish(total)}


def window_breakdown(graph: Graph, simulation: dict[str, Any], end_event_id: str,
                     start_s: float = 0.0) -> dict[str, Any] | None:
    """Attribute one observed latency window; unavailable executions return None."""
    return _WindowAttributor(graph, simulation).window(end_event_id, start_s)


def build_training_latency_breakdown(graph: Graph, simulation: dict[str, Any]) -> dict[str, Any]:
    """Use completed step windows, including their actual pipeline/optimizer tail."""
    result = {'method': 'actual_dependency_critical_path', 'step': None,
              'first_step': None, 'steady_step': None, 'unavailable_reason': None}
    if simulation.get('status') != 'feasible':
        result['unavailable_reason'] = 'execution_incomplete'
        return result
    step_ids = graph.metadata.get('step_end_events', [])
    if not step_ids:
        result['unavailable_reason'] = 'no_training_step_completions'
        return result
    attribution = _WindowAttributor(graph, simulation)
    previous, records = 0.0, []
    for event_id in step_ids:
        record = attribution.window(event_id, previous)
        if record is None:
            result['unavailable_reason'] = 'missing_training_step_completion'
            return result
        records.append(record)
        previous = attribution.rows[event_id]['end_s']
    result.update(step=mean_breakdowns(records), first_step=records[0],
                  steady_step=mean_breakdowns(records[1:]))
    if result['steady_step'] is None:
        result['steady_step_unavailable_reason'] = 'At least two training steps are required.'
    return result


def build_latency_breakdown(graph: Graph, simulation: dict[str, Any] | None,
                            requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach request breakdowns and select worst-request TTFT and mean TPOT.

    Indices are built once. Window results are reused across sequences sharing
    a batch completion event; no cumulative vector is retained for every DAG
    event and decode windows stop traversal at their preceding token time.
    """
    result = latency_breakdown_metadata()
    if simulation is None or simulation.get("status") != "feasible":
        result["unavailable_reason"] = "Requires a feasible simulated inference trace."
        return result
    completion = graph.metadata.get("sequence_token_completion_events", {})
    if not completion or not requests:
        result["unavailable_reason"] = "No inference token-completion metadata; TTFT/TPOT do not apply to training."
        return result
    attribution = _WindowAttributor(graph, simulation)
    windows: dict[tuple[str, float], dict[str, Any] | None] = {}
    request_windows: dict[tuple[str, ...], tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}

    def cached_window(event_id: str, start: float) -> dict[str, Any] | None:
        key = event_id, start
        if key not in windows:
            windows[key] = attribution.window(event_id, start)
        return windows[key]

    for request in requests:
        replica, sequence = str(request["replica"]), str(request["sequence"])
        ids = tuple(completion.get(replica, {}).get(sequence, ()))
        if ids not in request_windows:
            first = cached_window(ids[0], 0.0) if ids else None
            intervals = [cached_window(after, attribution.rows[before]["end_s"])
                         for before, after in zip(ids, ids[1:])]
            mean = mean_breakdowns(intervals) if intervals and all(item is not None for item in intervals) else None
            request_windows[ids] = first, mean
        request["ttft_breakdown"], request["tpot_breakdown"] = request_windows[ids]
    for metric in ("ttft", "tpot"):
        valid = [request for request in requests if request.get(metric + "_breakdown") is not None]
        if valid:
            selected = max(valid, key=lambda request: request[metric + "_s"])
            result[metric] = {**selected[metric + "_breakdown"],
                              "replica": selected["replica"], "sequence": selected["sequence"]}
    if result["ttft"] is None:
        result["unavailable_reason"] = "No completed output-token event is available."
    elif result["tpot"] is None:
        result["tpot_unavailable_reason"] = "At least two output tokens per request are required for TPOT."
    return result

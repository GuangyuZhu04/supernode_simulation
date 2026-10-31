"""Exact collective wall-time floors with unchanged physical transfer work.

The measured call timer runs beside the analytical transfers. The completion
join waits for both, so startup is not charged twice and contention cannot be
bypassed. Profiles constrain completion, not packet scheduling or GPU occupancy.
"""
from __future__ import annotations

import hashlib
import json
import math

from .core import Event, Graph

POLICY = (
    "Exact collective/semantic/phase/world-size/dtype/algorithm/timing-scope and per-rank send/receive bytes; "
    "vectors exclude diagonal/local traffic and include all modeled rounds. Equal vectors do not "
    "prove equal routing matrices, token routing, runtime or topology. Guarded profiles additionally "
    "require a versioned exact digest of effective abstract hardware, ordered physical ranks and "
    "the complete payload matrix. Legacy profiles remain explicitly unguarded. A digest locks "
    "the declared abstraction; it is not real hardware or benchmark authentication. duration_s includes the "
    "measured call startup and excludes warmup and external queueing. Completion is the maximum of "
    "this timer and unchanged physical transfers, never their sum. Coverage is by collective call, "
    "including cached replay; it does not validate end-to-end serving."
    " Collective models additionally require complete software/transport/timer context and a versioned "
    "normalized peer-family digest (1e-12 proportions with nonzero support retained). Exact profiles take "
    "precedence; model samples use piecewise-linear interpolation only inside their measured payload range. "
    "Exact sample, interpolated, and out-of-range calls are counted separately; no payload extrapolation occurs."
)


def _counts():
    return dict(eligible_collectives=0, matched_collectives=0, unmatched_collectives=0,
                guarded_matched_collectives=0, unguarded_matched_collectives=0,
                exact_profile_collectives=0, exact_model_collectives=0,
                interpolated_collectives=0, out_of_range_collectives=0,
                profile_limited_collectives=0, physical_limited_collectives=0,
                unevaluated_collectives=0, coverage_fraction=0.0)


def collective_calibration_summary():
    return {**_counts(), "by_phase": {}, "unmatched_reasons": {},
            "matched_profile_ids": [], "matched_model_ids": [], "policy": POLICY}


def _coverage(counts):
    counts["unmatched_collectives"] = counts["eligible_collectives"] - counts["matched_collectives"]
    counts["coverage_fraction"] = (counts["matched_collectives"] / counts["eligible_collectives"]
                                  if counts["eligible_collectives"] else 0.0)


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def _link_parameters(raw, bandwidth, latency, efficiency):
    """Normalize the defaults used by Topology, which differ from config defaults."""
    return {
        "bandwidth_Bps": raw.get("bandwidth_Bps", bandwidth),
        "latency_s": raw.get("latency_s", latency),
        "efficiency": raw.get("efficiency", efficiency),
        "injection_Bps": raw.get("injection_Bps") or raw.get("bandwidth_Bps", bandwidth),
    }


def _effective_hardware(hardware):
    count = int(hardware.get("npu_count", 8))
    su_size = int(hardware.get("su_size", count))
    host_size = int(hardware.get("npus_per_host", 8))
    kind = hardware.get("topology", "full_mesh").lower().replace("-", "_")
    npu, cpu = hardware.get("npu", {}), hardware.get("cpu", {})
    su_raw, so_raw = hardware.get("scale_up", {}), hardware.get("scale_out", {})
    su = _link_parameters(su_raw, 4e11, 1e-6, .85)
    so = _link_parameters(so_raw, 5e10, 5e-6, .8)
    so["host_injection_Bps"] = so_raw.get("host_injection_Bps") or so["injection_Bps"]
    domain_members = [list(range(start, min(start + su_size, count))) for start in range(0, count, su_size)]
    uplinks = []
    for members in domain_members:
        hosts = {}
        for rank in members:
            host = rank // host_size
            hosts[host] = hosts.get(host, 0) + 1
        default = so["bandwidth_Bps"] * len(hosts)
        if so_raw.get("host_injection_Bps") is not None:
            per_device = min(so["bandwidth_Bps"], so["injection_Bps"])
            default = sum(min(so_raw["host_injection_Bps"], number * per_device) for number in hosts.values())
        uplinks.append(so_raw.get("uplink_Bps") or default)
    so["uplink_Bps_by_su"] = uplinks
    result = {
        "name": hardware.get("name"), "npu_count": count, "su_size": su_size,
        "npus_per_host": host_size, "topology": kind,
        "routing_policy": hardware.get("routing_policy", "shortest_hop"),
        "routing_algorithm": hardware.get("routing_algorithm", "shortest_hop"),
        "routing_seed": hardware.get("routing_seed", 0),
        "npu": {"hbm_bytes": npu.get("hbm_bytes", 80e9), "hbm_Bps": npu.get("hbm_Bps", 3e12),
                "efficiency": npu.get("efficiency", .6),
                "flops": npu.get("flops", {"fp8": 2e15, "fp16": 1e15, "bf16": 1e15, "fp32": 5e13})},
        "cpu": {"dram_bytes": cpu.get("dram_bytes", 2e12), "dram_Bps": cpu.get("dram_Bps", 4e11),
                "flops": cpu.get("flops", 1e12), "efficiency": cpu.get("efficiency", .7), "iops": cpu.get("iops", 1e8)},
        "scale_up": su, "scale_out": so,
        "host_link": _link_parameters(hardware.get("host_link", {}), 64e9, 2e-6, .85),
        "routes": hardware.get("routes") or {},
    }
    if kind == "mesh":
        result["mesh_width_by_su"] = [hardware.get("mesh_width") or math.ceil(math.sqrt(len(members))) for members in domain_members]
    if kind in {"clos", "fat_tree"}:
        leaves = [hardware.get("leaf_size") or max(1, min(8, len(members))) for members in domain_members]
        result["leaf_size_by_su"] = leaves
        su["uplink_Bps_by_su"] = [su_raw.get("uplink_Bps") or su["bandwidth_Bps"] * leaf for leaf in leaves]
    if kind == "explicit":
        # Shortest-hop and ECMP routing both use deterministic edge-ID ordering.
        result["explicit_links"] = [{
            "id": link["id"], "src": link["src"], "dst": link["dst"],
            "bandwidth_Bps": link["bandwidth_Bps"], "latency_s": link.get("latency_s", 0),
            "domain": link.get("domain", "SU"), "efficiency": link.get("efficiency", 1.0),
            "shared_group": link.get("shared_group") or None,
            "shared_Bps": (link.get("shared_Bps") if link.get("shared_Bps") is not None else link["bandwidth_Bps"]) if link.get("shared_group") else None,
        } for link in sorted(hardware.get("explicit_links", []), key=lambda link: link["id"])]
    io_groups = [{
        "members": sorted(group["members"]), "domains": sorted(group["domains"]),
        "bandwidth_Bps": group["bandwidth_Bps"], "efficiency": group.get("efficiency", 1.0),
        "duplex": group.get("duplex", "full"), "latency_s": group.get("latency_s", 0),
    } for group in hardware.get("io_groups", [])]
    # IO group names/order do not affect capacity or membership; duplicate groups remain present.
    result["io_groups"] = sorted(io_groups, key=lambda group: json.dumps(_canonical(group), sort_keys=True))
    result["kernel_profiles"] = {
        operation: [{**{key: entry.get(key) for key in ("flops", "memory_bytes", "duration_s", "startup_s", "dtype", "phase")},
                     **{key: entry[key] for key in ('hardware_name', 'calibration_context', 'metadata_match') if entry.get(key)}}
                    for entry in entries]
        for operation, entries in hardware.get("kernel_profiles", {}).items()
    }
    for name in ("remote", "ssd"):
        value = hardware.get(name, {})
        if value.get("enabled", False):
            result[name] = {"enabled": True, "capacity_bytes": value.get("capacity_bytes", 10e12),
                            "bandwidth_Bps": value.get("bandwidth_Bps", 1e11 if name == "remote" else 1e10),
                            "latency_s": value.get("latency_s", 5e-6), "iops": value.get("iops", 1e7 if name == "remote" else 1e6)}
    # New runtime hardware controls fail conservatively until their defaults are normalized.
    # Empty optional traces retain the existing profile identity. An active
    # decomposition can change local service/contending work, so guard it.
    if hardware.get('kernel_traces'):
        result['kernel_traces'] = hardware['kernel_traces']
    handled = set(result) | {"mesh_width", "leaf_size", "explicit_links", "remote", "ssd", "kernel_traces"}
    if hardware.get('calibration_context'):
        result['calibration_context'] = hardware['calibration_context']
    ignored = {"collective_profiles", "collective_models", "calibration_context", "source", "description", "notes", "metadata"}
    result["additional_hardware_controls"] = {key: value for key, value in hardware.items() if key not in handled | ignored}
    return result


def _canonical(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("collective applicability signature requires finite numeric inputs")
        # Keep exact integer seeds/rank IDs; converting them through float can
        # collapse distinct routing assignments above 2**53. Numeric 100 and
        # 100.0 still canonicalize equally without rounding any integer input.
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise ValueError("collective applicability signature requires JSON-compatible inputs")


def collective_applicability_signature(hardware, ranks, kind, size_bytes, payload_matrix=None, *, precision=None):
    """Hash effective abstract hardware and the exact physical peer assignment.

    This does not authenticate hardware/runtime evidence. Matrix entries include
    the diagonal, which can affect activation pack/unpack despite no wire work.
    Profiles themselves and explanatory provenance do not affect the digest.
    """
    precision = precision or {}
    descriptor = {
        "version": 1, "hardware": _effective_hardware(hardware),
        "ranks": list(ranks), "collective": kind, "size_bytes": size_bytes,
        "payload_matrix": payload_matrix,
        "precision": {"compute": precision.get("compute", "fp16"), "activation": precision.get("activation", "fp16")},
    }
    encoded = json.dumps(_canonical(descriptor), sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return "v1:sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_collective_model_hardware(hardware):
    """Validate raw Topology/API dictionaries as strictly as canonical configs."""
    from .config import CollectiveModel, validate_calibration_context
    raw = hardware.get('collective_models', [])
    if not isinstance(raw, list):
        raise ValueError('collective_models must be a list')
    validate_calibration_context(hardware.get('calibration_context', {}), require_complete=bool(raw))
    models = [CollectiveModel.model_validate(model).model_dump() for model in raw]
    if len({model['id'] for model in models}) != len(models):
        raise ValueError('collective model IDs must be unique')
    return models


def _family_metadata_signature(base_signature, metadata_match):
    if not metadata_match:
        return base_signature
    from .config import validate_json_metadata
    validate_json_metadata(metadata_match)
    encoded = json.dumps(_canonical({'family_base': base_signature, 'metadata_match': metadata_match}),
                         sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return 'collective-family-v1:sha256:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def metadata_subset_matches(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and metadata_subset_matches(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(metadata_subset_matches(a, b) for a, b in zip(expected, actual))
    return expected == actual and isinstance(expected, bool) == isinstance(actual, bool)


def collective_family_signature(hardware, ranks, kind, size_bytes, payload_matrix=None, *, precision=None, metadata_match=None):
    """Lock environment/topology/ranks and normalized complete peer assignment.

    Byte proportions are rounded to 1e-12 absolute resolution to tolerate
    floating-point rescaling. A separate nonzero mask preserves tiny routes;
    diagonal proportions are retained because activation pack work can use them.
    This permits interpolation along payload scale only, not EP, routing shape,
    software, transport, datatype, or timer-boundary extrapolation.
    """
    from .config import validate_calibration_context
    validate_calibration_context(hardware.get('calibration_context', {}), require_complete=True)
    n = len(ranks)
    if n < 2 or len(set(ranks)) != n or any(not isinstance(rank, str) or not rank for rank in ranks):
        raise ValueError('collective family requires distinct nonempty physical ranks')

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('collective family payloads must be finite nonnegative numbers')
        return value

    number(size_bytes)
    normalized, support = None, None
    if kind == 'all_to_all_v':
        if not isinstance(payload_matrix, list) or len(payload_matrix) != n or any(not isinstance(row, list) or len(row) != n for row in payload_matrix):
            raise ValueError('collective family requires an N x N payload matrix')
        for row in payload_matrix:
            for value in row:
                number(value)
        total = math.fsum(value for i, row in enumerate(payload_matrix) for j, value in enumerate(row) if i != j)
        if total <= 0 or not math.isfinite(total):
            raise ValueError('collective family requires positive finite off-diagonal payload')
        normalized = [[round(value / total, 12) for value in row] for row in payload_matrix]
        support = [[value > 0 for value in row] for row in payload_matrix]
        normalized_size = None
    elif kind in ('all_gather', 'reduce_scatter', 'all_reduce'):
        total = size_bytes * (n - 1) * (2 if kind == 'all_reduce' else 1)
        if total <= 0 or not math.isfinite(total):
            raise ValueError('collective family requires positive finite off-diagonal payload')
        normalized_size = round(size_bytes / total, 12)
    else:
        raise ValueError('unsupported collective family')
    precision = precision or {}
    descriptor = {
        'version': 'collective-family-v1', 'proportion_resolution': 1e-12,
        'hardware': _effective_hardware(hardware), 'ranks': list(ranks), 'collective': kind,
        'normalized_size_bytes': normalized_size, 'normalized_payload_matrix': normalized,
        'payload_support': support,
        'precision': {'compute': precision.get('compute', 'fp16'), 'activation': precision.get('activation', 'fp16')},
    }
    encoded = json.dumps(_canonical(descriptor), sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    base_signature = 'collective-family-v1:sha256:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    return _family_metadata_signature(base_signature, metadata_match or {})


def match_collective_model(hardware, signature):
    models = validate_collective_model_hardware(hardware)
    return _match_collective_model(hardware, signature, models)


def _match_collective_model(hardware, signature, models):
    """Match models already validated for an immutable Topology snapshot."""
    if not models:
        return None, 'no_models'
    keys = ('collective', 'semantic', 'phase', 'world_size', 'dtype', 'algorithm', 'timing_scope')
    candidates = [model for model in models if model['hardware_name'] == hardware.get('name')
                  and all(model[key] == signature.get(key) for key in keys)]
    if not candidates:
        return None, 'no_model_match'
    if not signature.get('family_signature'):
        return None, 'family_unavailable'
    candidates = [model for model in candidates if model['family_signature'] == _family_metadata_signature(
        signature['family_signature'], model['metadata_match'])]
    if not candidates:
        return None, 'family_mismatch'
    candidates = [model for model in candidates if metadata_subset_matches(model['metadata_match'], signature.get('operation_metadata', {}))]
    if not candidates:
        return None, 'metadata_mismatch'
    payload = signature['total_payload_bytes']
    covering = [model for model in candidates if model['samples'][0]['payload_bytes'] <= payload <= model['samples'][-1]['payload_bytes']]
    if not covering:
        return None, 'out_of_range'
    if len(covering) > 1:
        raise ValueError('Ambiguous collective models: ' + ', '.join(model['id'] for model in covering))
    model = covering[0]
    # Do not use a relative tolerance to admit extrapolation past either bound.
    exact = next((sample for sample in model['samples'] if sample['payload_bytes'] == payload), None)
    if exact:
        duration, bracket, weight, method = exact['duration_s'], [payload, payload], 0.0, 'exact_model'
    else:
        left, right = next((left, right) for left, right in zip(model['samples'], model['samples'][1:])
                           if left['payload_bytes'] < payload < right['payload_bytes'])
        weight = (payload - left['payload_bytes']) / (right['payload_bytes'] - left['payload_bytes'])
        duration = (1 - weight) * left['duration_s'] + weight * right['duration_s']
        bracket, method = [left['payload_bytes'], right['payload_bytes']], 'interpolated'
    return {'model_id': model['id'], 'duration_s': duration, 'source': model['source'],
            'method': method, 'sample_bracket_payload_bytes': bracket, 'interpolation_weight': weight}, 'matched'


def match_collective_profile(hardware, signature):
    profiles = hardware.get("collective_profiles", [])
    if not profiles:
        return None, "no_profiles"
    platform = [p for p in profiles if p["hardware_name"] == hardware.get("name")]
    if not platform:
        return None, "hardware_mismatch"
    keys = ("collective", "semantic", "phase", "world_size", "dtype", "algorithm", "timing_scope")
    fitting = [
        p for p in platform
        if all(p.get(key, "transport" if key == "timing_scope" else None) == signature[key] for key in keys)
        and _close(p["total_payload_bytes"], signature["total_payload_bytes"])
        and all(len(p[key]) == len(signature[key])
                and all(_close(a, b) for a, b in zip(p[key], signature[key]))
                for key in ("send_payload_bytes", "recv_payload_bytes"))
    ]
    guarded = [profile for profile in fitting if profile.get("applicability_signature") is not None]
    if guarded:
        if signature.get("applicability_signature") is None:
            return None, "applicability_unavailable"
        fitting = [profile for profile in guarded if profile["applicability_signature"] == signature["applicability_signature"]]
        if not fitting:
            return None, "applicability_mismatch"
    if len(fitting) > 1:
        raise ValueError("Ambiguous collective calibration: " + ", ".join(p["id"] for p in fitting))
    return (fitting[0], "matched") if fitting else (None, "no_exact_match")


def apply_collective_profile(graph, hardware, prefix, signature, deps, terminals, anchor, *, scope_selection=None,
                             _validated_models=None):
    """Tag one call; matched calls join a concurrent timing floor and transfers."""
    profile, reason = match_collective_profile(hardware, signature)
    model = None
    if profile is None and hardware.get('collective_models'):
        model, reason = (match_collective_model(hardware, signature) if _validated_models is None else
                         _match_collective_model(hardware, signature, _validated_models))
    record = {"status": reason, "phase": signature["phase"],
              "match": "exact_send_receive_vectors", "signature": signature,
              "applicability_guard": "not_matched", 'selected_scope': signature['timing_scope']}
    if scope_selection:
        record['scope_selection'] = scope_selection
    if hardware.get('calibration_context'):
        record['calibration_context'] = hardware['calibration_context']
    if profile is None and model is None:
        if anchor is not None:
            anchor.metadata["collective_calibration"] = record
        return terminals
    timer_id = prefix + ".calibration_timer"
    join_id = prefix + ".calibration_done"
    if profile is not None:
        duration = profile['duration_s']
        record.update(profile_id=profile["id"], source=profile["source"], method='exact_profile',
            applicability_guard="guarded" if profile.get("applicability_signature") is not None else "unguarded_legacy",
            match="exact_hardware_ranks_and_payload_matrix" if profile.get("applicability_signature") is not None else "exact_send_receive_vectors",
            measured_duration_s=duration)
    else:
        duration = model['duration_s']
        record.update({key: value for key, value in model.items() if key != 'duration_s'})
        record.update(applicability_guard='guarded', match='normalized_family_payload_curve',
                      service_duration_s=duration, extrapolated=False)
    source_boundary = hardware.get('calibration_context', {}).get('timer_boundary')
    record.update(timer_boundary="dependency_ready_to_collective_completion",
                  simulated_timer_boundary='dependency_ready_to_collective_completion',
                  source_timer_boundary=source_boundary,
                  source_includes_host_startup={'whole_call': True, 'device_only': False}.get(source_boundary),
                  warmup="excluded", completion_event=join_id)
    graph.events.append(Event(
        timer_id, deps=tuple(deps), startup_s=duration,
        metadata={"type": "collective_timing", "phase": signature["phase"],
                  "collective_calibration": record}))
    graph.events.append(Event(
        join_id, deps=tuple(dict.fromkeys((*terminals, timer_id))),
        metadata={"type": "collective_completion", "phase": signature["phase"],
                  "collective_timing_timer": timer_id}))
    return (join_id,)


def summarize_collective_calibration(graph: Graph, simulation=None):
    summary = collective_calibration_summary()
    rows = ({row["id"]: row for row in simulation.get("trace", ())}
            if simulation is not None else {})
    events = {event.id: event for event in graph.events}
    for event in graph.events:
        record = event.metadata.get("collective_calibration")
        if record is None:
            continue
        counts = summary["by_phase"].setdefault(str(record["phase"] or "unspecified"), _counts())
        for current in (summary, counts):
            current["eligible_collectives"] += 1
        if record["status"] != "matched":
            reason = record["status"]
            summary["unmatched_reasons"][reason] = summary["unmatched_reasons"].get(reason, 0) + 1
            if reason == 'out_of_range':
                for current in (summary, counts):
                    current['out_of_range_collectives'] += 1
            continue
        if 'profile_id' in record:
            summary["matched_profile_ids"].append(record["profile_id"])
        if 'model_id' in record:
            summary['matched_model_ids'].append(record['model_id'])
        outcome = "unevaluated_collectives"
        timer = rows.get(event.id)
        join = events.get(record["completion_event"])
        if timer is not None and join is not None and all(dep in rows for dep in join.deps):
            transport_end = max((rows[dep]["end_s"] for dep in join.deps if dep != event.id),
                                default=timer["start_s"])
            tolerance = max(1e-12, abs(timer["end_s"]) * 1e-9)
            outcome = ("physical_limited_collectives"
                       if transport_end > timer["end_s"] + tolerance
                       else "profile_limited_collectives")
        for current in (summary, counts):
            current["matched_collectives"] += 1
            current["guarded_matched_collectives" if record.get("applicability_guard") == "guarded" else "unguarded_matched_collectives"] += 1
            current[record.get('method', 'exact_profile') + '_collectives'] += 1
            current[outcome] += 1
    summary["matched_profile_ids"] = sorted(set(summary["matched_profile_ids"]))
    summary['matched_model_ids'] = sorted(set(summary['matched_model_ids']))
    _coverage(summary)
    for counts in summary["by_phase"].values():
        _coverage(counts)
    return summary


def merge_collective_calibration_summaries(target, source):
    keys = ("eligible_collectives", "matched_collectives", "profile_limited_collectives",
            "physical_limited_collectives", "unevaluated_collectives",
            "guarded_matched_collectives", "unguarded_matched_collectives",
            'exact_profile_collectives', 'exact_model_collectives', 'interpolated_collectives', 'out_of_range_collectives')
    for key in keys:
        target[key] += source[key]
    for phase, counts in source["by_phase"].items():
        combined = target["by_phase"].setdefault(phase, _counts())
        for key in keys:
            combined[key] += counts[key]
        _coverage(combined)
    for reason, count in source["unmatched_reasons"].items():
        target["unmatched_reasons"][reason] = target["unmatched_reasons"].get(reason, 0) + count
    target["matched_profile_ids"] = sorted(set(target["matched_profile_ids"]) | set(source["matched_profile_ids"]))
    target['matched_model_ids'] = sorted(set(target['matched_model_ids']) | set(source['matched_model_ids']))
    _coverage(target)

#!/usr/bin/env python3
"""Build an Atlas A3 HiAM hierarchy abstraction from explicit user link values.

Usage:
    python build_architecture_topology.py --spec a3-topology.example.json --output fragment.json

The output is a Config-compatible fragment (hardware/sources/assumptions), not a
complete model scenario. No network access or real-device inspection is used.
All bandwidths are bytes/second per direction, before the optional efficiency.
SIO is a direct link per NPU pair. HCCS-L1 is a per-NPU attachment to an abstract
host fabric; HCCS-L2 is one aggregate uplink per host to an abstract SU fabric.
Neither abstract fabric represents the vendor's exact internal switch wiring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARCHITECTURE_REVISION = "c38f378dbf2070399b4ade868abfdbfcaa7c8736"
ARCHITECTURE_URL = (
    "https://github.com/Ascend/mind-cluster/blob/" + ARCHITECTURE_REVISION
    + "/docs/zh/scheduling/04_usage/03_basic_scheduling/01_affinity_scheduling/"
      "03_ascend_ai_processor_based_affinity.md"
)
TIERS = ("sio", "hccs_l1", "hccs_l2")


def _number(value, name, *, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _source(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name}.source must declare provenance")
    if set(value) - {"status", "description", "uri", "revision"}:
        raise ValueError(f"Unknown {name}.source fields")
    if value.get("status") not in {"assumed", "user", "public", "measured"}:
        raise ValueError(f"{name}.source.status must be assumed, user, public or measured")
    if not isinstance(value.get("description"), str) or not value["description"].strip():
        raise ValueError(f"{name}.source.description must be nonempty")
    for key in ("uri", "revision"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise ValueError(f"{name}.source.{key} must be a string or null")
    return dict(value)


def _tier(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if set(value) - {"bandwidth_Bps", "latency_s", "efficiency", "source"}:
        raise ValueError(f"Unknown {name} fields")
    if "bandwidth_Bps" not in value or "latency_s" not in value:
        raise ValueError(f"{name}.bandwidth_Bps and {name}.latency_s must be supplied explicitly")
    bandwidth = _number(value["bandwidth_Bps"], f"{name}.bandwidth_Bps", positive=True)
    latency = _number(value["latency_s"], f"{name}.latency_s", positive=False)
    efficiency = _number(value.get("efficiency", 1), f"{name}.efficiency", positive=True)
    if efficiency > 1:
        raise ValueError(f"{name}.efficiency cannot exceed 1")
    return {"bandwidth_Bps": bandwidth, "latency_s": latency, "efficiency": efficiency}, _source(value.get("source"), name)


def build_topology_fragment(spec):
    """Return a hardware fragment; preserve numerical provenance tier by tier."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    if set(spec) - {"hosts", "sio", "hccs_l1", "hccs_l2", "host_link"}:
        raise ValueError("Unknown topology spec fields")
    hosts = spec.get("hosts")
    if isinstance(hosts, bool) or not isinstance(hosts, int) or hosts < 1:
        raise ValueError("hosts must be a positive integer")
    tiers, tier_sources = {}, {}
    for name in (*TIERS, *(("host_link",) if "host_link" in spec else ())):
        tiers[name], tier_sources[name] = _tier(spec.get(name), name)
    identity = {"hosts": hosts, **tiers}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    links = []

    def directed(src, dst, tier, domain):
        links.append({"id": f"{tier}:{src}>{dst}", "src": src, "dst": dst,
                      "domain": domain, **tiers[tier]})

    def pair(src, dst, tier):
        directed(src, dst, tier, "SU")
        directed(dst, src, tier, "SU")

    for host in range(hosts):
        first = host * 16
        fabric = f"a3.host{host}.hccs_l1"
        for module in range(8):
            left = first + module * 2
            pair(f"npu{left}", f"npu{left + 1}", "sio")
        for device in range(first, first + 16):
            pair(f"npu{device}", fabric, "hccs_l1")
            if "host_link" in tiers:
                directed(f"cpu{host}", f"npu{device}", "host_link", "H2D")
                directed(f"npu{device}", f"cpu{host}", "host_link", "D2H")
        # A single directed uplink per host shares capacity across that host's
        # cross-host flows. Do not create one full-capacity L2 link per NPU pair.
        if hosts > 1:
            pair(fabric, "a3.supod.hccs_l2", "hccs_l2")

    endpoint_cap = max(tiers[name]["bandwidth_Bps"] * tiers[name]["efficiency"]
                       for name in ("sio", "hccs_l1"))
    hardware = {
        "name": f"Atlas A3 HiAM hierarchy abstraction ({hosts} hosts; {digest})",
        "npu_count": 16 * hosts,
        "npus_per_host": 16,
        "su_size": 16 * hosts,
        "topology": "explicit",
        "routing_policy": "su_so",
        # Every path is explicit. Keep the shared per-NPU injection ceiling high
        # enough for the fastest local tier; tier edges impose their own bounds.
        "scale_up": {"bandwidth_Bps": endpoint_cap, "injection_Bps": endpoint_cap,
                     "efficiency": 1, "latency_s": 0},
        "explicit_links": links,
    }
    if "host_link" in tiers:
        hardware["host_link"] = dict(tiers["host_link"])
    sources = {
        "hardware.topology": {
            "status": "assumed",
            "description": "Engineering hierarchy abstraction, not vendor physical switch/port wiring or a measured performance claim.",
            "uri": ARCHITECTURE_URL, "revision": ARCHITECTURE_REVISION,
        },
        "hardware.topology.architecture": {
            "status": "public",
            "description": "Official A3 structure: two NPUs per SIO-connected HiAM, eight HiAMs per node, HCCS-L1 between modules, HCCS-L2 between nodes. Rates and latencies are not supplied by this source.",
            "uri": ARCHITECTURE_URL, "revision": ARCHITECTURE_REVISION,
        },
    }
    sources.update({f"hardware.topology.{name}": value for name, value in tier_sources.items()})
    assumptions = [
        "Atlas A3 HiAM hierarchy abstraction: 16 modeled NPU endpoints per host and adjacent even/odd pairs per HiAM; endpoint numbering is an explicit modeling assumption.",
        "SIO edges connect each modeled pair directly. HCCS-L1 is an abstract host fabric with one attachment per NPU; HCCS-L2 is an abstract SU fabric with one aggregate directed uplink per host. This is not the exact vendor switch/port topology.",
        "All tier bandwidths, latencies and efficiencies are explicitly supplied in the input spec; preserve the individual sources. Official architecture evidence does not establish those numerical values.",
        "All NPUs are in one logical scale-up domain; cross-supernode RoCE, adaptive routing, lane degradation and exact port sharing are not modeled.",
        "Bandwidths are per direction. Opposite directions have separate capacity; per-NPU SU injection is bounded by the fastest supplied endpoint tier. Concurrent SIO and L1 traffic share this declared ceiling.",
        "The fragment does not supply model, compute/HBM, CPU, workload or serving parameters; merge it deliberately with a compatible base scenario.",
    ]
    assumptions.append(
        "Explicit H2D/D2H edges are included using the supplied host_link values; existing simulator per-host host-link aggregation semantics still apply."
        if "host_link" in tiers else
        "No H2D/D2H edges are included. This fragment cannot execute host offload/host copy paths until explicit host_link input is supplied or compatible directed host edges are added."
    )
    return {"hardware": hardware, "sources": sources, "assumptions": assumptions}


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_constant(value):
    raise ValueError(f"Non-finite JSON constant is not accepted: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path, help="JSON with hosts and explicit per-tier numerical values/provenance")
    parser.add_argument("--output", required=True, type=Path, help="Config-compatible hardware fragment JSON")
    args = parser.parse_args(argv)
    try:
        same_path = args.spec.resolve() == args.output.resolve()
        same_file = (args.spec.exists() and args.output.exists()
                     and args.spec.samefile(args.output))
        if same_path or same_file:
            raise ValueError("--output must not overwrite the input --spec")
        with args.spec.open(encoding="utf-8") as handle:
            fragment = build_topology_fragment(json.load(
                handle, object_pairs_hook=_json_object, parse_constant=_json_constant))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(fragment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote architecture abstraction: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

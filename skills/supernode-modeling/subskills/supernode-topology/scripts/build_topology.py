"""Compile an explicit, versioned supernode topology specification.

Natural-language interpretation belongs to the accompanying skill. This script
only validates and compiles structured input; it never guesses hardware or SLOs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# A copied skill can target a checkout explicitly, discover a nearby checkout,
# or use an installed engine package. Never assume a fixed directory depth.
_configured_root = os.environ.get("SUPERNODE_SIM_ROOT")
_candidate_roots = ([Path(_configured_root).expanduser().resolve()] if _configured_root else [])
_candidate_roots += list(Path(__file__).resolve().parents)
_candidate_roots += [Path.cwd(), *Path.cwd().parents]
for _candidate in _candidate_roots:
    if (_candidate / "supernode_sim").is_dir() and (_candidate / "pyproject.toml").is_file():
        sys.path.insert(0, str(_candidate))
        break
from supernode_sim.config import Config, HardwareSpec, Source, load_config, read_data

TOPOLOGY_FIELDS = ("npu_count", "su_size", "npus_per_host", "topology", "routing_policy",
                   "scale_up", "scale_out", "host_link", "explicit_links", "routes")


def _topology_source(key: str) -> bool:
    return any(key == f"hardware.{field}" or key.startswith(f"hardware.{field}.")
               for field in (*TOPOLOGY_FIELDS, "name"))


class StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HostLink(StrictSpec):
    bandwidth_Bps: float = Field(gt=0)
    latency_s: float = Field(ge=0)
    efficiency: float = Field(gt=0, le=1)
    injection_Bps: float | None = Field(default=None, gt=0)


class Fabric(HostLink):
    topology: Literal["full_mesh", "ring", "mesh", "clos", "fat_tree"]
    uplink_Bps: float | None = Field(default=None, gt=0)
    mesh_width: int | None = Field(default=None, gt=0)
    leaf_size: int | None = Field(default=None, gt=0)
    spine_count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def relevant_knobs(self):
        if self.mesh_width is not None and self.topology != "mesh":
            raise ValueError("mesh_width requires topology mesh")
        if (self.leaf_size is not None or self.spine_count is not None) and self.topology not in {"clos", "fat_tree"}:
            raise ValueError("leaf_size and spine_count require clos or fat_tree")
        return self


class ScaleUp(Fabric):
    @model_validator(mode="after")
    def relevant_uplink(self):
        if self.uplink_Bps is not None and self.topology not in {"clos", "fat_tree"}:
            raise ValueError("scale_up.uplink_Bps requires clos or fat_tree")
        return self


class ScaleOutDomain(StrictSpec):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    su_domains: list[int] = Field(min_length=1)


class ScaleOut(Fabric):
    domains: list[ScaleOutDomain] = Field(min_length=1)


class InterDomainLink(StrictSpec):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    src_domain: str
    dst_domain: str
    bandwidth_Bps: float = Field(gt=0)
    latency_s: float = Field(ge=0)
    efficiency: float = Field(gt=0, le=1)
    shared_group: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    shared_Bps: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def shared_capacity(self):
        if (self.shared_group is None) != (self.shared_Bps is None):
            raise ValueError("shared_group and shared_Bps must be supplied together")
        return self


class TopologySpec(StrictSpec):
    schema_version: Literal["1.0"]
    name: str | None = None
    npu_count: int = Field(gt=0)
    su_size: int = Field(gt=0)
    npus_per_host: int = Field(gt=0)
    scale_up: ScaleUp
    scale_out: ScaleOut
    host_link: HostLink
    inter_domain_links: list[InterDomainLink] = Field(default_factory=list)
    sources: dict[str, Source] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if any(not _topology_source(key) for key in self.sources):
            raise ValueError("topology sources must use hardware paths owned by this topology specification")
        if self.su_size > self.npu_count or self.npu_count % self.su_size:
            raise ValueError("su_size must divide npu_count")
        if self.npus_per_host > self.su_size or self.su_size % self.npus_per_host:
            raise ValueError("npus_per_host must divide su_size; hosts spanning SU domains are not supported in v1")
        ids = [domain.id for domain in self.scale_out.domains]
        if len(ids) != len(set(ids)):
            raise ValueError("scale_out domain IDs must be unique")
        members = [index for domain in self.scale_out.domains for index in domain.su_domains]
        if sorted(members) != list(range(self.npu_count // self.su_size)):
            raise ValueError("scale_out domains must assign every SU domain index exactly once")
        link_ids = [link.id for link in self.inter_domain_links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("inter_domain_links IDs must be unique")
        capacities: dict[str, float] = {}
        for link in self.inter_domain_links:
            if link.src_domain not in ids or link.dst_domain not in ids:
                raise ValueError("inter_domain_links must reference declared domain IDs")
            if link.src_domain == link.dst_domain:
                raise ValueError("inter_domain_links must connect different domains")
            if link.shared_group:
                effective = link.shared_Bps * link.efficiency
                if link.shared_group in capacities and capacities[link.shared_group] != effective:
                    raise ValueError("inconsistent effective capacity for inter_domain shared_group")
                capacities[link.shared_group] = effective
        # A domain fabric is bidirectional and connected by construction. Checking
        # this small graph suffices for all-pair SO reachability without O(N²) routes.
        if len(ids) > 1:
            for reverse in (False, True):
                adjacent: dict[str, list[str]] = {key: [] for key in ids}
                for link in self.inter_domain_links:
                    src, dst = (link.dst_domain, link.src_domain) if reverse else (link.src_domain, link.dst_domain)
                    adjacent[src].append(dst)
                reached, stack = set(), [ids[0]]
                while stack:
                    node = stack.pop()
                    if node not in reached:
                        reached.add(node)
                        stack.extend(adjacent[node])
                if len(reached) != len(ids):
                    raise ValueError("inter_domain_links must provide strongly connected directed reachability between all scale_out domains")
        return self


def _link_parameters(params: HostLink) -> dict[str, float]:
    data = {key: getattr(params, key) for key in ("bandwidth_Bps", "latency_s", "efficiency")}
    data["injection_Bps"] = params.injection_Bps if params.injection_Bps is not None else params.bandwidth_Bps
    if isinstance(params, Fabric) and params.uplink_Bps is not None:
        data["uplink_Bps"] = params.uplink_Bps
    return data


def _fabric_size(count: int, params: Fabric) -> int:
    if params.topology == "full_mesh":
        return count * (count - 1)
    if params.topology == "ring":
        return 0 if count < 2 else 2 if count == 2 else 2 * count
    if params.topology == "mesh":
        width = params.mesh_width or math.ceil(math.sqrt(count))
        return 2 * (count - math.ceil(count / width) + max(0, count - width))
    leaf_size = params.leaf_size or min(8, count)
    return 2 * count + 2 * math.ceil(count / leaf_size) * (params.spine_count or 1)


class NetworkBuilder:
    def __init__(self):
        self.links: list[dict[str, Any]] = []

    def edge(self, src: str, dst: str, params: dict, domain: str, *, id: str | None = None,
             shared_group: str | None = None, shared_Bps: float | None = None) -> None:
        edge = {"id": id or f"{src}>{dst}:{domain}", "src": src, "dst": dst,
                "domain": domain, "bandwidth_Bps": params["bandwidth_Bps"],
                "latency_s": params["latency_s"], "efficiency": params.get("efficiency", 1.0)}
        if shared_group is not None:
            edge.update(shared_group=shared_group, shared_Bps=shared_Bps)
        self.links.append(edge)

    def pair(self, src: str, dst: str, params: dict, domain: str) -> None:
        self.edge(src, dst, params, domain)
        self.edge(dst, src, params, domain)

    def fabric(self, terminals: list[str], namespace: str, params: Fabric, domain: str) -> None:
        count, data = len(terminals), _link_parameters(params)
        if params.topology == "full_mesh":
            for i, src in enumerate(terminals):
                for dst in terminals[i + 1:]:
                    self.pair(src, dst, data, domain)
        elif params.topology == "ring":
            if count == 2:
                self.pair(*terminals, data, domain)
            elif count > 2:
                for i, src in enumerate(terminals):
                    self.pair(src, terminals[(i + 1) % count], data, domain)
        elif params.topology == "mesh":
            width = params.mesh_width or math.ceil(math.sqrt(count))
            for i, terminal in enumerate(terminals):
                if i % width:
                    self.pair(terminals[i - 1], terminal, data, domain)
                if i >= width:
                    self.pair(terminals[i - width], terminal, data, domain)
        else:
            leaf_size, spine_count = params.leaf_size or min(8, count), params.spine_count or 1
            for i, terminal in enumerate(terminals):
                self.pair(terminal, f"{namespace}.leaf{i // leaf_size}", data, domain)
            uplink = params.uplink_Bps or params.bandwidth_Bps * leaf_size
            for i in range(math.ceil(count / leaf_size)):
                leaf = f"{namespace}.leaf{i}"
                for j in range(spine_count):
                    spine = f"{namespace}.spine{j}"
                    uplink_data = {**data, "bandwidth_Bps": uplink}
                    self.edge(leaf, spine, uplink_data, domain, shared_group=f"{leaf}.uplink.tx", shared_Bps=uplink)
                    self.edge(spine, leaf, uplink_data, domain, shared_group=f"{leaf}.uplink.rx", shared_Bps=uplink)


def compile_topology(spec: dict | TopologySpec, base: dict | None = None, *,
                     max_links: int = 200_000, source_uri: str | None = None) -> dict:
    """Return a hardware fragment, or a checked full scenario when base is given.

    API callers should resolve a base's calibration first with load_config;
    the CLI does this automatically. Unresolved calibration is rejected.
    """
    spec = spec if isinstance(spec, TopologySpec) else TopologySpec.model_validate(spec)
    if isinstance(max_links, bool) or not isinstance(max_links, int) or max_links < 1:
        raise ValueError("max_links must be a positive integer")
    if base is not None and base.get("calibration"):
        raise ValueError("resolve base calibration with load_config before compilation")
    hardware = copy.deepcopy((base or {}).get("hardware", {}))
    npu_count, su_size = spec.npu_count, spec.su_size
    su_count, host_count = npu_count // su_size, npu_count // spec.npus_per_host
    estimated = su_count * _fabric_size(su_size, spec.scale_up) + 4 * npu_count
    estimated += sum(_fabric_size(len(domain.su_domains) + 1, spec.scale_out) for domain in spec.scale_out.domains)
    estimated += len(spec.inter_domain_links)
    estimated += sum(2 * host_count for name in ("remote", "ssd") if hardware.get(name, {}).get("enabled", False))
    if estimated > max_links:
        raise ValueError(f"topology requires {estimated} directed links, exceeding max_links={max_links}; use a sparse topology or an explicit larger limit")
    network = NetworkBuilder()
    su, so, host = map(_link_parameters, (spec.scale_up, spec.scale_out, spec.host_link))
    for index in range(su_count):
        members = [f"npu{i}" for i in range(index * su_size, (index + 1) * su_size)]
        network.fabric(members, f"su{index}", spec.scale_up, "SU")
        gateway = f"su{index}.gateway"
        uplink = spec.scale_out.uplink_Bps or spec.scale_out.bandwidth_Bps * (su_size // spec.npus_per_host)
        for member in members:
            network.edge(member, gateway, so, "SO", shared_group=f"{gateway}.attachment.tx", shared_Bps=uplink)
            network.edge(gateway, member, so, "SO", shared_group=f"{gateway}.attachment.rx", shared_Bps=uplink)
    for domain in sorted(spec.scale_out.domains, key=lambda value: value.id):
        terminals = [f"su{i}.gateway" for i in sorted(domain.su_domains)] + [f"so.{domain.id}.gateway"]
        network.fabric(terminals, f"so.{domain.id}", spec.scale_out, "SO")
    for link in sorted(spec.inter_domain_links, key=lambda value: value.id):
        network.edge(f"so.{link.src_domain}.gateway", f"so.{link.dst_domain}.gateway",
                     link.model_dump(), "SO", id=f"inter:{link.id}",
                     shared_group=f"inter:{link.shared_group}" if link.shared_group else None, shared_Bps=link.shared_Bps)
    for index in range(npu_count):
        cpu, npu = f"cpu{index // spec.npus_per_host}", f"npu{index}"
        network.edge(cpu, npu, host, "H2D")
        network.edge(npu, cpu, host, "D2H")
    for name in ("remote", "ssd"):
        memory = hardware.get(name, {})
        if memory.get("enabled", False):
            # Defaults are normalized by HardwareSpec; this validates rather than
            # guesses remote/SSD properties when API callers supply partial bases.
            memory = HardwareSpec.model_validate(hardware).model_dump()[name]
            for index in range(host_count):
                network.pair(f"cpu{index}", f"{name}0", memory, name)
    if len(network.links) != estimated:
        raise RuntimeError("internal topology link estimate mismatch")
    for obsolete in ("mesh_width", "leaf_size", "explicit_links", "routes", "routing_policy"):
        hardware.pop(obsolete, None)
    hardware.update(npu_count=npu_count, su_size=su_size, npus_per_host=spec.npus_per_host,
                    topology="explicit", routing_policy="su_so", scale_up=su, scale_out=so,
                    host_link=host, explicit_links=network.links, routes={})
    if spec.name is not None:
        hardware["name"] = spec.name
    HardwareSpec.model_validate(hardware)
    sources = {key: copy.deepcopy(value) for key, value in (base or {}).get("sources", {}).items()
               if not _topology_source(key) or (spec.name is None and (key == "hardware.name" or key.startswith("hardware.name.")))}
    for key in (*TOPOLOGY_FIELDS, *(("name",) if spec.name is not None else ())):
        sources[f"hardware.{key}"] = {"status": "user", "description": "Compiled from the supplied structured topology specification."}
        if source_uri is not None:
            sources[f"hardware.{key}"]["uri"] = source_uri
    assumptions = list((base or {}).get("assumptions", [])) + list(spec.assumptions)
    assumptions.extend([
        "Topology bandwidths are bytes/s per direction; generated reverse links have independent capacity unless an explicit shared_group couples them.",
        "Uniform SU domains use contiguous NPU indices; each SO fabric includes its SU gateways followed by one domain gateway terminal.",
        "NPU routes use SU only within one SU domain and SO only across SU domains; deterministic shortest-hop routing does not model ECMP or packet switching.",
    ])
    for key, params in (("scale_up", spec.scale_up), ("scale_out", spec.scale_out), ("host_link", spec.host_link)):
        if params.injection_Bps is None:
            field = f"hardware.{key}.injection_Bps"
            sources[field] = {"status": "assumed", "description": f"Omitted {key}.injection_Bps defaults to its supplied bandwidth_Bps."}
            assumptions.append(sources[field]["description"])
        if isinstance(params, Fabric):
            if params.topology == "mesh" and params.mesh_width is None:
                assumptions.append(f"{key}.mesh_width defaults to ceil(sqrt(terminal_count)) separately for each fabric.")
            if params.topology in {"clos", "fat_tree"}:
                assumptions.append(f"{key}.{params.topology} is a two-tier leaf/spine fluid abstraction; each leaf's spine links share one aggregate capacity per direction.")
                if params.leaf_size is None:
                    assumptions.append(f"{key}.leaf_size defaults to min(8, terminal_count) separately for each fabric.")
                if params.spine_count is None:
                    assumptions.append(f"{key}.spine_count defaults to 1.")
                if params.uplink_Bps is None:
                    assumptions.append(f"{key} leaf uplink capacity defaults to bandwidth_Bps * leaf_size per direction.")
            if params.topology == "ring":
                assumptions.append(f"{key}.ring is a bidirectional closed cycle; two terminals use one duplex pair.")
    if spec.scale_out.uplink_Bps is None:
        assumptions.append("scale_out SU gateway attachment aggregate defaults to bandwidth_Bps * hosts_in_SU per direction.")
    sources.update({key: value.model_dump(exclude_none=True) for key, value in spec.sources.items()})
    output = copy.deepcopy(base) if base is not None else {}
    output.update(schema_version="1.0", hardware=hardware, sources=sources,
                  assumptions=list(dict.fromkeys(assumptions)))
    if base is not None:
        output["calibration"] = None
        Config.model_validate(output)
    return output


def _write(data: dict, output: Path | None) -> None:
    is_json = output is None or output.suffix.lower() == ".json"
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n" if is_json else yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    if output is None:
        sys.stdout.write(serialized)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, nargs="?", help="structured topology YAML or JSON")
    parser.add_argument("--base", type=Path, help="existing complete scenario; resolve calibration then replace topology")
    parser.add_argument("-o", "--output", type=Path, help="output .json or YAML path; omitted prints JSON")
    parser.add_argument("--max-links", type=int, default=200_000, help="maximum directed links, default 200000")
    parser.add_argument("--schema", action="store_true", help="write topology input JSON Schema without compiling")
    args = parser.parse_args(argv)
    try:
        if args.schema:
            if args.spec is not None or args.base is not None:
                raise ValueError("--schema cannot be combined with a spec or --base")
            _write(TopologySpec.model_json_schema(), args.output)
            return 0
        if args.spec is None:
            raise ValueError("a topology spec is required unless --schema is used")
        base = None
        if args.base is not None:
            base = load_config(args.base).model_dump(mode="json")
            base["calibration"] = None
        output = compile_topology(read_data(args.spec), base, max_links=args.max_links,
                                  source_uri=str(args.spec.resolve()))
        _write(output, args.output)
        print(f"Compiled {len(output['hardware']['explicit_links'])} directed links; output {'scenario' if base is not None else 'hardware fragment'}.", file=sys.stderr)
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Topology error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Directed interconnect templates, deterministic routes and collectives.

Bandwidths are per direction. SU and SO have separate injection resources;
SO additionally shares a host NIC and SU uplink. Full-mesh edges are created
on demand, so very large synthetic capacity studies do not build O(N²) links.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

from .cache import BoundedCache
from .config import IOGroupSpec, validate_io_groups
from .core import Allocation, Event, Graph, Resource
from .routing import destination_hops, ecmp_path


class Topology:
    def __init__(self, hardware: dict[str, Any], precision: dict[str, Any]):
        from .collective_calibration import validate_collective_model_hardware
        models = validate_collective_model_hardware(hardware)
        # Calibration must describe the same fixed hardware as the resources
        # built below. Own its nested inputs so callers cannot mutate a cached
        # fingerprint or validated model through their configuration dictionary.
        guarded_profiles = any(profile.get('applicability_signature') is not None
                               for profile in hardware.get('collective_profiles', ()))
        self.hardware = deepcopy(hardware) if models or guarded_profiles else hardware
        if models:
            self.hardware['collective_models'] = models
        hardware = self.hardware
        self.precision = dict(precision)
        self._collective_models = models
        self._guarded_collective_profiles = guarded_profiles
        self._collective_signatures = BoundedCache(max_bytes=8 * 1024 * 1024, max_entries=1024)
        self.npu_count = int(hardware.get("npu_count", 8))
        self.su_size = int(hardware.get("su_size", self.npu_count))
        self.npus_per_host = int(hardware.get("npus_per_host", 8))
        self.kind = hardware.get("topology", "full_mesh").lower().replace("-", "_")
        self.routing_policy = hardware.get("routing_policy", "shortest_hop")
        self.routing_algorithm = hardware.get('routing_algorithm', 'shortest_hop')
        self.routing_seed = hardware.get('routing_seed', 0)
        if self.routing_algorithm not in ('shortest_hop', 'ecmp_hash'):
            raise ValueError('routing_algorithm must be shortest_hop or ecmp_hash')
        if type(self.routing_seed) is not int:
            raise ValueError('routing_seed must be an integer')
        self.resources: dict[str, Resource] = {}
        self.memory_caps: dict[str, float] = {}
        self.links: dict[str, dict[str, Any]] = {}
        self.adjacency: dict[str, list[str]] = defaultdict(list)
        self.graphs: list[Graph] = []
        self.routes = hardware.get("routes") or {}
        raw_io_groups = hardware.get('io_groups', [])
        if not isinstance(raw_io_groups, list):
            raise ValueError('io_groups must be a list')
        validated_io_groups = [IOGroupSpec.model_validate(group) for group in raw_io_groups]
        validate_io_groups(validated_io_groups, self.npu_count)
        self.io_groups = [group.model_dump() for group in validated_io_groups]
        self._io_groups_by_member: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in self.io_groups:
            for member in group['members']:
                self._io_groups_by_member[f'npu{member}'].append(group)
        # Lazy full-mesh edges make historical pair routes observable: after a
        # shortcut is added, an existing pair must still use its earlier route.
        # Preserve that behavior; static topologies can safely evict and reroute.
        self._route_cache = ({} if self.kind == "full_mesh" else
                             BoundedCache(max_bytes=16 * 1024 * 1024, max_entries=65_536))
        # Reuse shortest-path trees without retaining all O(N²) source/destination
        # combinations. Each of the unrestricted, SU and SO modes has a budget.
        self._route_trees: dict[tuple[str, ...] | None, BoundedCache] = {}
        self._ecmp_distances: dict[tuple[str, ...] | None, BoundedCache] = {}
        self._route_adjacency: dict[tuple[str, ...] | None, dict[str, tuple[str, ...]]] = {}
        self.su = {"bandwidth_Bps": 4e11, "latency_s": 1e-6, "efficiency": .85, **hardware.get("scale_up", {})}
        self.so = {"bandwidth_Bps": 5e10, "latency_s": 5e-6, "efficiency": .8, **hardware.get("scale_out", {})}
        self.host = {"bandwidth_Bps": 64e9, "latency_s": 2e-6, "efficiency": .85, **hardware.get("host_link", {})}
        self._build_resources()
        self._build_network()

    def _resource(self, key: str, capacity: float, unit: str = "byte/s", kind: str = "bandwidth") -> None:
        schedule = self.hardware.get('bandwidth_schedules', {}).get(key, ())
        if schedule and unit != 'byte/s':
            raise ValueError(f'{key} is not a bandwidth resource')
        resource = Resource(key, float(capacity), unit, kind, tuple(map(tuple, schedule)))
        self.resources[key] = resource
        for graph in self.graphs:
            graph.resources[key] = resource

    def validate_scheduled_resources(self):
        unknown = set(self.hardware.get('bandwidth_schedules', ())).difference(self.resources)
        if unknown:
            raise ValueError(f'Unknown bandwidth schedule resources: {sorted(unknown)}')

    def _memory(self, key: str, capacity: float, bandwidth: float) -> None:
        self.memory_caps[key] = float(capacity)
        self._resource(key, bandwidth)

    def host_for(self, npu_index: int) -> str:
        return f"cpu{npu_index // self.npus_per_host}"

    @staticmethod
    def memory_for(endpoint: str) -> str:
        if endpoint.startswith("npu"):
            return f"{endpoint}.hbm"
        if endpoint.startswith("cpu"):
            return f"{endpoint}.dram"
        return f"{endpoint}.memory"

    def _build_resources(self) -> None:
        npu = self.hardware.get("npu", {})
        cpu = self.hardware.get("cpu", {})
        flops = npu.get("flops", {"fp8": 2e15, "fp16": 1e15, "bf16": 1e15, "fp32": 5e13})
        effective_flops = flops.get(self.precision.get("compute", "fp16"), 1e15) if isinstance(flops, dict) else flops
        for index in range(self.npu_count):
            node = f"npu{index}"
            self._memory(f"{node}.hbm", npu.get("hbm_bytes", 80e9), npu.get("hbm_Bps", 3e12))
            self._resource(f"{node}.compute", effective_flops * npu.get("efficiency", .6), "FLOP/s", "compute")
            for domain, params in (("SU", self.su), ("SO", self.so), ("host", self.host)):
                for direction in ("tx", "rx"):
                    self._resource(f"{node}.{domain}.{direction}", (params.get("injection_Bps") or params["bandwidth_Bps"]) * params["efficiency"])
        for index in range(math.ceil(self.npu_count / self.npus_per_host)):
            node = f"cpu{index}"
            self._memory(f"{node}.dram", cpu.get("dram_bytes", 2e12), cpu.get("dram_Bps", 4e11))
            self._resource(f"{node}.compute", cpu.get("flops", 1e12) * cpu.get("efficiency", .7), "FLOP/s", "compute")
            self._resource(f"{node}.iops", cpu.get("iops", 1e8), "operation/s", "iops")
            for direction in ("tx", "rx"):
                # Host aggregate is independent of each accelerator's injection
                # limit. An omitted override preserves the historical shared NIC.
                host_injection = (self.so.get("host_injection_Bps") or
                                  self.so.get("injection_Bps") or self.so["bandwidth_Bps"])
                self._resource(f"{node}.nic.{direction}", host_injection * self.so["efficiency"])
                self._resource(f"{node}.host.{direction}", (self.host.get("injection_Bps") or self.host["bandwidth_Bps"]) * self.host["efficiency"])
        for name in ("remote", "ssd"):
            params = self.hardware.get(name, {})
            if params.get("enabled", False):
                self._memory(f"{name}0.memory", params.get("capacity_bytes", 10e12), params.get("bandwidth_Bps", 1e11 if name == "remote" else 1e10))
                self._resource(f"{name}0.iops", params.get("iops", 1e7 if name == "remote" else 1e6), "operation/s", "iops")
        for group in self.io_groups:
            directions = ('tx', 'rx') if group['duplex'] == 'full' else ('shared',)
            for direction in directions:
                self._resource(f"io:{group['id']}.{direction}", group['bandwidth_Bps'] * group['efficiency'])

    def _edge(self, eid: str, src: str, dst: str, params: dict[str, Any], domain: str, shared_group: str | None = None, shared_Bps: float | None = None) -> None:
        if eid in self.links:
            raise ValueError(f"Duplicate directed link ID: {eid}")
        bandwidth = float(params["bandwidth_Bps"]) * params.get("efficiency", 1.0)
        resource = f"link:{eid}"
        self._resource(resource, bandwidth)
        shared = None
        if shared_group:
            shared = f"shared:{shared_group}"
            capacity = float(shared_Bps if shared_Bps is not None else params["bandwidth_Bps"]) * params.get("efficiency", 1.0)
            if shared in self.resources and self.resources[shared].capacity != capacity:
                raise ValueError(f"Inconsistent capacities for shared link group {shared_group}")
            self._resource(shared, capacity)
        self.links[eid] = {"id": eid, "src": src, "dst": dst, "resource": resource,
                           "bandwidth_Bps": bandwidth, "latency_s": float(params.get("latency_s", 0)),
                           "domain": domain, "shared_resource": shared}
        self.adjacency[src].append(eid)
        # Lazy full-mesh SU edges must not invalidate cached SO routing trees.
        # Pair routes retain their existing cache semantics.
        for mode in (None, (domain,)):
            self._route_trees.pop(mode, None)
            self._ecmp_distances.pop(mode, None)
            self._route_adjacency.pop(mode, None)

    def _pair(self, src: str, dst: str, params: dict[str, Any], domain: str) -> None:
        self._edge(f"{src}>{dst}:{domain}", src, dst, params, domain)
        self._edge(f"{dst}>{src}:{domain}", dst, src, params, domain)

    def _build_network(self) -> None:
        if self.kind == "explicit":
            for link in self.hardware.get("explicit_links", []):
                self._edge(link["id"], link["src"], link["dst"], link,
                           link.get("domain", "SU"), link.get("shared_group"), link.get("shared_Bps"))
            return
        if self.kind not in {"full_mesh", "mesh", "clos", "fat_tree"}:
            raise ValueError(f"Unknown topology: {self.kind}")
        for domain in range(math.ceil(self.npu_count / self.su_size)):
            members = list(range(domain * self.su_size, min((domain + 1) * self.su_size, self.npu_count)))
            gateway = f"su{domain}.gateway"
            if self.kind == "mesh":
                width = int(self.hardware.get("mesh_width") or math.ceil(math.sqrt(len(members))))
                for offset, index in enumerate(members):
                    if offset % width and offset - 1 >= 0:
                        self._pair(f"npu{members[offset - 1]}", f"npu{index}", self.su, "SU")
                    if offset >= width:
                        self._pair(f"npu{members[offset - width]}", f"npu{index}", self.su, "SU")
            elif self.kind in {"clos", "fat_tree"}:
                leaf_size = int(self.hardware.get("leaf_size") or max(1, min(8, self.su_size)))
                leaf_nodes = set()
                for offset, index in enumerate(members):
                    leaf = f"su{domain}.leaf{offset // leaf_size}"
                    leaf_nodes.add(leaf)
                    self._pair(f"npu{index}", leaf, self.su, "SU")
                uplink = {**self.su, "bandwidth_Bps": self.su.get("uplink_Bps") or self.su["bandwidth_Bps"] * leaf_size}
                for leaf in sorted(leaf_nodes):
                    self._pair(leaf, f"su{domain}.spine", uplink, "SU")
            # A gateway's edge resources represent SO NIC attachment, separate
            # from the SU fabric. Shared host NIC work is added per transfer.
            for index in members:
                self._pair(f"npu{index}", gateway, self.so, "SO")
            host_members: dict[str, int] = defaultdict(int)
            for index in members:
                host_members[self.host_for(index)] += 1
            default_uplink = self.so["bandwidth_Bps"] * len(host_members)
            if self.so.get("host_injection_Bps") is not None:
                # Aggregate only the NIC capacity reachable from this domain;
                # partial hosts retain their shared NIC resource across domains.
                per_device = min(self.so["bandwidth_Bps"],
                                 self.so.get("injection_Bps") or self.so["bandwidth_Bps"])
                default_uplink = sum(min(self.so["host_injection_Bps"], count * per_device)
                                     for count in host_members.values())
            uplink = {**self.so, "bandwidth_Bps": self.so.get("uplink_Bps") or default_uplink}
            self._pair(gateway, "so.spine", uplink, "SO")
        for index in range(self.npu_count):
            node, host = f"npu{index}", self.host_for(index)
            self._edge(f"{host}>{node}:H2D", host, node, self.host, "H2D")
            self._edge(f"{node}>{host}:D2H", node, host, self.host, "D2H")
        for name in ("remote", "ssd"):
            params = self.hardware.get(name, {})
            if params.get("enabled", False):
                params = {"bandwidth_Bps": 1e11 if name == "remote" else 1e10, "latency_s": 5e-6, **params}
                for index in range(math.ceil(self.npu_count / self.npus_per_host)):
                    self._pair(f"cpu{index}", f"{name}0", params, "remote")

    def populate(self, graph: Graph, *, training_hbm_fraction: float | None = None) -> None:
        graph.resources.update(self.resources)
        graph.memory_caps.update(self.memory_caps)
        if training_hbm_fraction is not None:
            physical = self.memory_caps['npu0.hbm']
            usable = physical * training_hbm_fraction
            for rank in range(self.npu_count):
                graph.memory_caps[f'npu{rank}.hbm'] = usable
            graph.metadata['training_hbm'] = {
                'usable_fraction': training_hbm_fraction, 'physical_bytes_per_npu': physical,
                'usable_bytes_per_npu': usable, 'reserved_bytes_per_npu': physical - usable,
            }
        self.graphs.append(graph)
        graph.metadata.setdefault("topology", {"kind": self.kind, "npu_count": self.npu_count,
                                               "su_size": self.su_size, "npus_per_host": self.npus_per_host,
                                               "routing_policy": self.routing_policy,
                                               "routing_algorithm": self.routing_algorithm,
                                               "routing_seed": self.routing_seed,
                                               "routing": ("explicit overrides, otherwise stable per-pair equal-hop ECMP; no packet spraying"
                                                           if self.routing_algorithm == 'ecmp_hash' else
                                                           "explicit or deterministic shortest-hop"),
                                               "bandwidth_direction": "full_duplex",
                                               "clos_model": ("explicit directed fabric; shared capacity from supplied links"
                                                              if self.kind == "explicit" else
                                                              "one deterministic spine; uplinks share capacity")})
        if self.io_groups:
            graph.metadata['topology']['io_groups'] = self.io_groups
            graph.metadata['topology']['io_group_policy'] = (
                'Physical endpoint membership plus actual route domains; SO charges each NPU endpoint, '
                'D2H only source/tx and H2D only destination/rx. Each endpoint charges once per group; '
                'full duplex separates directions, shared duplex sums both. These capacity constraints '
                'do not add physical link bytes or imply GPU-NIC switching/rail selection.')

    def path(self, src: str, dst: str) -> tuple[str, ...]:
        if src == dst:
            return ()
        key = (src, dst)
        if key in self._route_cache:
            return self._route_cache[key]
        allowed = None
        if (self.kind != "explicit" or self.routing_policy == "su_so") and src.startswith("npu") and dst.startswith("npu"):
            src_domain, dst_domain = int(src[3:]) // self.su_size, int(dst[3:]) // self.su_size
            allowed = {"SU"} if src_domain == dst_domain else {"SO"}
        explicit = self.routes.get(f"{src}->{dst}")
        if explicit is not None:
            current = src
            for eid in explicit:
                if eid not in self.links or self.links[eid]["src"] != current:
                    raise ValueError(f"Invalid explicit route {src}->{dst}: {eid}")
                if self.routing_policy == "su_so" and allowed is not None and self.links[eid]["domain"] not in allowed:
                    raise ValueError(f"Explicit route {src}->{dst} violates su_so routing policy: {eid}")
                current = self.links[eid]["dst"]
            if current != dst:
                raise ValueError(f"Explicit route {src}->{dst} does not reach destination")
            self._route_cache[key] = tuple(explicit)
            return tuple(explicit)
        if self.kind != "explicit" and src.startswith("npu") and dst.startswith("npu"):
            if allowed == {"SU"} and self.kind == "full_mesh":
                eid = f"{src}>{dst}:SU"
                if eid not in self.links:
                    self._edge(eid, src, dst, self.su, "SU")
                self._route_cache[key] = (eid,)
                return (eid,)
        mode = None if allowed is None else tuple(sorted(allowed))
        if self.routing_algorithm == 'ecmp_hash':
            cache = self._ecmp_distances.get(mode)
            if cache is None:
                cache = BoundedCache(max_bytes=16 * 1024 * 1024, max_entries=128)
                self._ecmp_distances[mode] = cache
            distances = cache.get(dst)
            if distances is None:
                distances = destination_hops(self.links, dst, allowed)
                cache[dst] = distances
            route = ecmp_path(self.links, self.adjacency, src, dst, allowed,
                              self.routing_seed, distances)
            self._route_cache[key] = route
            return route
        trees = self._route_trees.get(mode)
        if trees is None:
            trees = BoundedCache(max_bytes=16 * 1024 * 1024, max_entries=128)
            self._route_trees[mode] = trees
        previous = trees.get(src)
        if previous is None:
            adjacent = self._route_adjacency.get(mode)
            if adjacent is None:
                adjacent = {
                    node: tuple(eid for eid in sorted(edges)
                                if allowed is None or self.links[eid]["domain"] in allowed)
                    for node, edges in self.adjacency.items()
                }
                self._route_adjacency[mode] = adjacent
            previous = {src: None}
            pending = deque([src])
            # FIFO traversal of sorted edges visits equal-hop routes in lexical
            # edge-ID order, matching the old (hops, full_route) heap ordering.
            while pending:
                node = pending.popleft()
                for eid in adjacent.get(node, ()):
                    target = self.links[eid]["dst"]
                    if target not in previous:
                        previous[target] = (node, eid)
                        pending.append(target)
            trees[src] = previous
        # Oversized trees are deliberately not retained. The locally computed
        # tree still serves this query, so budgets never change reachability.
        if dst not in previous:
            raise ValueError(f"No directed network route from {src} to {dst}")
        reverse_route: list[str] = []
        node = dst
        while node != src:
            node, eid = previous[node]
            reverse_route.append(eid)
        route = tuple(reversed(reverse_route))
        self._route_cache[key] = route
        return route

    def transfer(self, id: str, src: str, dst: str, size_bytes: float, deps: tuple[str, ...] = (), metadata: dict[str, Any] | None = None) -> Event:
        if not math.isfinite(size_bytes) or size_bytes < 0:
            raise ValueError("Transfer size must be finite and nonnegative")
        route = self.path(src, dst) if size_bytes else ()
        work: dict[str, float] = defaultdict(float)
        for endpoint in dict.fromkeys((src, dst)):
            resource = self.memory_for(endpoint)
            if resource not in self.resources:
                raise ValueError(f"Unknown memory endpoint: {endpoint}")
            work[resource] += size_bytes
        domains = sorted({self.links[eid]["domain"] for eid in route})
        latency = 0.0
        physical: dict[str, float] = defaultdict(float)
        # Latency attribution uses physical paths and endpoint placement, not
        # transport-byte proportions. Keep the existing SU/SO byte ledger intact.
        def endpoint_host(endpoint: str) -> str | None:
            if endpoint.startswith('npu') and endpoint[3:].isdigit():
                return self.host_for(int(endpoint[3:]))
            if endpoint.startswith('cpu') and endpoint[3:].isdigit():
                return endpoint
            return None

        src_host, dst_host = endpoint_host(src), endpoint_host(dst)
        route_hosts = {host for eid in route for endpoint in (self.links[eid]['src'], self.links[eid]['dst'])
                       if (host := endpoint_host(endpoint)) is not None}
        within_host = (src_host is not None and src_host == dst_host and 'SO' not in domains
                       and route_hosts <= {src_host})
        resource_domains: dict[str, str] = {key: 'other' for key in work}
        startup_by_domain: dict[str, float] = defaultdict(float)
        su_scopes = set()
        for eid in route:
            edge = self.links[eid]
            domain = edge['domain']
            edge_src_host, edge_dst_host = endpoint_host(edge['src']), endpoint_host(edge['dst'])
            edge_within_host = (edge_src_host is not None and edge_src_host == edge_dst_host)
            attribution = ('node' if domain in ('H2D', 'D2H', 'host') or (domain == 'SU' and (within_host or edge_within_host))
                           else domain if domain in ('SU', 'SO') else 'other')
            if domain == 'SU':
                su_scopes.add(attribution)
            work[edge["resource"]] += size_bytes
            resource_domains[edge['resource']] = attribution
            physical[edge["domain"]] += size_bytes
            if edge["shared_resource"]:
                work[edge["shared_resource"]] += size_bytes
                previous = resource_domains.get(edge['shared_resource'], attribution)
                resource_domains[edge['shared_resource']] = attribution if previous == attribution else 'other'
            latency += edge["latency_s"]
            startup_by_domain[attribution] += edge['latency_s']
        for node, direction in ((src, "tx"), (dst, "rx")):
            for domain in ("SU", "SO"):
                resource = f"{node}.{domain}.{direction}"
                if domain in domains and resource in self.resources:
                    work[resource] += size_bytes
                    resource_domains[resource] = 'node' if domain == 'SU' and su_scopes == {'node'} else domain
            if "SO" in domains:
                host = self.host_for(int(node[3:])) if node.startswith("npu") else node
                resource = f"{host}.nic.{direction}"
                if resource in self.resources:
                    work[resource] += size_bytes
                    resource_domains[resource] = 'SO'
            if "H2D" in domains or "D2H" in domains:
                resource = f"{node}.host.{direction}"
                if resource in self.resources:
                    work[resource] += size_bytes
                    resource_domains[resource] = 'node'
        # Transfers to remote memory traverse a CPU memory-controller/host
        # bridge. Charge that shared host fabric even though CPU is not an
        # application endpoint.
        bridges = {self.links[eid]["src"] for eid in route} | {self.links[eid]["dst"] for eid in route}
        for bridge in bridges.difference({src, dst}):
            if f"{bridge}.dram" in self.resources:
                work[f"{bridge}.dram"] += size_bytes
                resource_domains[f'{bridge}.dram'] = 'other'
                direction = "tx" if dst.startswith("npu") else "rx"
                work[f"{bridge}.host.{direction}"] += size_bytes
                resource_domains[f'{bridge}.host.{direction}'] = 'node'
        io_charges = []
        for endpoint, direction, eligible_domains in ((src, 'tx', {'SO', 'D2H'}),
                                                       (dst, 'rx', {'SO', 'H2D'})):
            used_domains = set(domains) & eligible_domains
            if not used_domains:
                continue
            for group in self._io_groups_by_member.get(endpoint, ()):
                matched_domains = used_domains & set(group['domains'])
                if not matched_domains:
                    continue
                # A selected group is an endpoint capacity constraint, not a
                # new route edge. Two endpoints in one shared-duplex group
                # intentionally consume two payloads on that single resource.
                resource_direction = direction if group['duplex'] == 'full' else 'shared'
                resource = f"io:{group['id']}.{resource_direction}"
                work[resource] += size_bytes
                attribution = ('SO' if matched_domains == {'SO'} else
                               'node' if 'SO' not in matched_domains else 'other')
                previous = resource_domains.get(resource, attribution)
                resource_domains[resource] = attribution if previous == attribution else 'other'
                latency += group['latency_s']
                startup_by_domain[attribution] += group['latency_s']
                io_charges.append({'group': group['id'], 'endpoint': endpoint, 'direction': direction,
                    'resource': resource, 'domains': sorted(matched_domains), 'bytes': size_bytes,
                    'latency_s': group['latency_s']})
        event_metadata = {**(metadata or {}), "type": "transfer", "src": src, "dst": dst,
                          "payload_bytes": size_bytes, "path": list(route), "domains": domains,
                          "resource_domains": resource_domains, "startup_s_by_domain": dict(startup_by_domain),
                          "physical_bytes": sum(physical.values()), "physical_bytes_by_domain": dict(physical)}
        if self.io_groups:
            event_metadata['io_group_charges'] = io_charges
        return Event(id, deps=tuple(deps), work=dict(work), startup_s=latency, metadata=event_metadata)

    def collective(self, graph: Graph, prefix: str, kind: str, ranks: list[str], size_bytes: float,
                   deps: tuple[str, ...] = (), payload_matrix: list[list[float]] | None = None,
                   metadata: dict[str, Any] | None = None, algorithm: str = 'pairwise',
                   apply_profiles: bool = True, timing_scope: str = 'transport') -> tuple[str, ...]:
        """Append ring or pairwise rounds, returning terminal dependency IDs.

        AG/RS/AR sizes are full tensor bytes per rank; all-to-all-v supplies
        the rank-to-rank matrix. Diagonal entries need no network transfer.
        A send and receive staging buffer live for each transfer's duration.
        """
        aliases = {"all-gather": "all_gather", "reduce-scatter": "reduce_scatter", "all-reduce": "all_reduce", "all-to-all-v": "all_to_all_v"}
        kind = aliases.get(kind, kind)
        n = len(ranks)
        if algorithm not in ('pairwise', 'direct'):
            raise ValueError('all-to-all algorithm must be pairwise or direct')
        if n == 0 or len(set(ranks)) != n:
            raise ValueError("Collective requires distinct, nonempty ranks")
        if kind not in {"all_gather", "reduce_scatter", "all_reduce", "all_to_all_v"}:
            raise ValueError(f"Unknown collective: {kind}")
        if size_bytes < 0 or not math.isfinite(size_bytes):
            raise ValueError("Collective size must be finite and nonnegative")
        if kind == "all_to_all_v":
            if payload_matrix is None or len(payload_matrix) != n or any(len(row) != n for row in payload_matrix):
                raise ValueError("all_to_all_v requires an N x N payload matrix")
            if any(not math.isfinite(value) or value < 0 for row in payload_matrix for value in row):
                raise ValueError("Payload matrix entries must be finite and nonnegative")
        if n == 1:
            return tuple(deps)
        previous = tuple(deps)
        direct_ends = []
        phases = ("reduce_scatter", "all_gather") if kind == "all_reduce" else (kind,)
        for phase in phases:
            for step in range(n - 1):
                current = []
                for index, src in enumerate(ranks):
                    destination = (index + (step + 1 if kind == "all_to_all_v" else 1)) % n
                    dst = ranks[destination]
                    amount = payload_matrix[index][destination] if kind == "all_to_all_v" else size_bytes / n
                    if amount == 0:
                        continue
                    eid = f"{prefix}.{phase}.s{step}.r{index}"
                    event = self.transfer(eid, src, dst, amount, previous,
                                          {**(metadata or {}), "collective": kind, "collective_phase": phase,
                                           "algorithm": algorithm if kind == "all_to_all_v" else "ring",
                                           "communicator": list(ranks), "round": step})
                    allocations = tuple(Allocation(f"{eid}.{role}", self.memory_for(node), amount,
                                                   "communication_buffer", (metadata or {}).get("buffer_dtype", self.precision.get("communication", "fp16")))
                                        for role, node in (("send", src), ("recv", dst)))
                    event.allocs = allocations
                    event.releases = tuple(alloc.id for alloc in allocations)
                    if kind != 'all_to_all_v' or algorithm != 'direct':
                        event.slots = (f"{src}.collective.tx", f"{dst}.collective.rx")
                    graph.events.append(event)
                    current.append(eid)
                if current:
                    if kind == 'all_to_all_v' and algorithm == 'direct':
                        direct_ends.extend(current)
                    else:
                        previous = tuple(current)
        terminals = tuple(direct_ends) if direct_ends else previous
        if not apply_profiles:
            return terminals
        return self.complete_collective_profile(
            graph, prefix, kind, ranks, size_bytes, tuple(deps), terminals,
            payload_matrix, metadata, algorithm, timing_scope)

    def _collective_signature(self, kind, ranks, size_bytes, payload_matrix, metadata, algorithm, timing_scope):
        aliases = {"all-gather": "all_gather", "reduce-scatter": "reduce_scatter",
                   "all-reduce": "all_reduce", "all-to-all-v": "all_to_all_v"}
        kind = aliases.get(kind, kind)
        n = len(ranks)
        if timing_scope not in ('transport', 'activation_collective'):
            raise ValueError('Unknown collective profile timing_scope')
        if n <= 1:
            return None
        # Layers and decode steps repeatedly use the same physical peers and
        # byte matrix. Cache only payload-dependent evidence; phase, operation,
        # dtype, timing boundary and metadata are still matched on every call.
        # Include scalar types so a boolean cannot hit an earlier numeric key
        # and evade the public signature functions' input checks.
        cache_key = None
        cached = None
        if self._guarded_collective_profiles or self._collective_models:
            matrix_key = (tuple(tuple((type(value), value) for value in row) for row in payload_matrix)
                          if payload_matrix is not None else None)
            cache_key = (kind, tuple(ranks), type(size_bytes), size_bytes, matrix_key)
            cached = self._collective_signatures.get(cache_key)
        if cached is None:
            payload = self._collective_payload_signature(kind, ranks, size_bytes, payload_matrix)
            if payload is None:
                return None
            if cache_key is not None:
                self._collective_signatures[cache_key] = payload
        else:
            payload = cached
        meta = metadata or {}
        signature = {
            'collective': kind, 'semantic': meta.get('operation', meta.get('semantic', kind)),
            'phase': meta.get('phase'), 'world_size': n,
            'dtype': meta.get('wire_dtype', meta.get('buffer_dtype', self.precision.get('communication', 'fp16'))),
            'algorithm': algorithm if kind == 'all_to_all_v' else 'ring',
            'timing_scope': timing_scope,
            'total_payload_bytes': payload['total_payload_bytes'],
            'send_payload_bytes': list(payload['send_payload_bytes']),
            'recv_payload_bytes': list(payload['recv_payload_bytes']),
            'applicability_signature': payload['applicability_signature'],
        }
        if self._collective_models:
            signature['family_signature'] = payload['family_signature']
            signature['operation_metadata'] = dict(meta)
        return signature

    def _collective_payload_signature(self, kind, ranks, size_bytes, payload_matrix):
        from .collective_calibration import collective_applicability_signature, collective_family_signature
        n = len(ranks)
        if kind == 'all_to_all_v':
            if payload_matrix is None or len(payload_matrix) != n or any(len(row) != n for row in payload_matrix):
                raise ValueError('all_to_all_v calibration requires an N x N payload matrix')
            sends = [math.fsum(value for j, value in enumerate(row) if i != j)
                     for i, row in enumerate(payload_matrix)]
            receives = [math.fsum(payload_matrix[i][j] for i in range(n) if i != j)
                        for j in range(n)]
        else:
            amount = size_bytes * (n - 1) / n * (2 if kind == 'all_reduce' else 1)
            sends, receives = [amount] * n, [amount] * n
        total = math.fsum(sends)
        if total == 0:
            return None
        signature = {
            'total_payload_bytes': total, 'send_payload_bytes': tuple(sends),
            'recv_payload_bytes': tuple(receives),
            'applicability_signature': collective_applicability_signature(
                self.hardware, ranks, kind, size_bytes, payload_matrix, precision=self.precision)
                if self._guarded_collective_profiles else None,
        }
        if self._collective_models:
            signature['family_signature'] = collective_family_signature(
                self.hardware, ranks, kind, size_bytes, payload_matrix, precision=self.precision)
        return signature

    def select_collective_timing_scope(self, kind, ranks, size_bytes, payload_matrix=None, metadata=None, algorithm='pairwise'):
        """Prefer an applicable whole activation call, otherwise fit transport.

        One selected boundary is recorded per call. This avoids duplicate timers
        and coverage counts while letting HCCL transport curves reach MoE graphs.
        """
        from .collective_calibration import match_collective_profile, _match_collective_model
        entries = [*self.hardware.get('collective_profiles', ()), *self.hardware.get('collective_models', ())]
        if not entries:
            return {'selected_scope': 'activation_collective', 'activation_collective_status': 'no_calibration',
                    'transport_status': 'no_calibration'}
        signature = self._collective_signature(kind, ranks, size_bytes, payload_matrix, metadata, algorithm, 'activation_collective')
        if signature is None:
            return {'selected_scope': 'activation_collective', 'activation_collective_status': 'no_remote_payload',
                    'transport_status': 'no_remote_payload'}
        statuses = {}
        for scope in ('activation_collective', 'transport'):
            scoped = {**signature, 'timing_scope': scope}
            profile, reason = match_collective_profile(self.hardware, scoped)
            model = None
            if profile is None and self.hardware.get('collective_models'):
                model, reason = _match_collective_model(self.hardware, scoped, self._collective_models)
            statuses[scope + '_status'] = reason
            if profile is not None or model is not None:
                if scope == 'activation_collective':
                    statuses['transport_status'] = 'not_checked_outer_preferred'
                return {'selected_scope': scope, **statuses}
        configured = {entry.get('timing_scope', 'transport') for entry in entries
                      if entry.get('collective') == signature['collective'] and entry.get('semantic') == signature['semantic']}
        selected = ('activation_collective' if 'activation_collective' in configured or 'transport' not in configured else 'transport')
        return {'selected_scope': selected, **statuses}

    def complete_collective_profile(self, graph: Graph, prefix: str, kind: str, ranks: list[str],
                                    size_bytes: float, deps: tuple[str, ...] = (),
                                    terminals: tuple[str, ...] = (),
                                    payload_matrix: list[list[float]] | None = None,
                                    metadata: dict[str, Any] | None = None,
                                    algorithm: str = 'pairwise',
                                    timing_scope: str = 'transport') -> tuple[str, ...]:
        """Apply the selected timing boundary while retaining all physical work."""
        from .collective_calibration import apply_collective_profile
        signature = self._collective_signature(kind, ranks, size_bytes, payload_matrix, metadata, algorithm, timing_scope)
        if signature is None:
            return tuple(terminals)
        terminal_ids = set(terminals)
        anchor = next((event for event in reversed(graph.events) if event.id in terminal_ids), None)
        return apply_collective_profile(graph, self.hardware, prefix, signature,
            tuple(deps), tuple(terminals), anchor,
            scope_selection=(metadata or {}).get('collective_scope_selection'),
            _validated_models=self._collective_models)

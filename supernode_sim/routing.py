"""Deterministic per-pair ECMP over an explicitly supplied directed fabric.

Every eligible edge has unit routing cost. This chooses one shortest path per
source/destination pair, not packet spraying, adaptive routing, or perfect load
balance. Actual bandwidth, latency and shared resources are charged downstream.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json


def destination_hops(links: dict, destination: str, allowed: set[str] | None) -> dict[str, int]:
    """Reverse BFS; cycles cannot create a forwarding loop in decreasing hops."""
    incoming = defaultdict(list)
    for edge in links.values():
        if allowed is None or edge['domain'] in allowed:
            incoming[edge['dst']].append(edge['src'])
    distance = {destination: 0}
    pending = deque([destination])
    while pending:
        node = pending.popleft()
        for previous in incoming.get(node, ()):
            if previous not in distance:
                distance[previous] = distance[node] + 1
                pending.append(previous)
    return distance


def ecmp_path(links: dict, adjacency: dict, source: str, destination: str,
              allowed: set[str] | None, seed: int, distance: dict[str, int]) -> tuple[str, ...]:
    """Select equal-hop next hops using stable SHA256, independent of query order."""
    if source not in distance:
        raise ValueError(f'No directed network route from {source} to {destination}')
    node, route = source, []
    while node != destination:
        choices = sorted(eid for eid in adjacency.get(node, ())
                         if (allowed is None or links[eid]['domain'] in allowed)
                         and distance.get(links[eid]['dst']) == distance[node] - 1)
        if not choices:
            raise ValueError(f'No directed network route from {source} to {destination}')
        identity = json.dumps([source, destination, node, seed], ensure_ascii=False, separators=(',', ':'))
        index = int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:8], 'big') % len(choices)
        selected = choices[index]
        route.append(selected)
        node = links[selected]['dst']
    return tuple(route)

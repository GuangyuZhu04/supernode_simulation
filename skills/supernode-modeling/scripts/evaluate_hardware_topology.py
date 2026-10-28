#!/usr/bin/env python3
"""Reproduce architectural capacity/path checks, not hardware benchmark accuracy.

Run from the repository: python skills/supernode-modeling/scripts/evaluate_hardware_topology.py
All numerical link values in these probes are explicit engineering assumptions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supernode_sim.core import Graph
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology
from build_architecture_topology import build_topology_fragment


def run_flows(hardware, flows):
    graph, topology = Graph(), Topology(hardware, {})
    topology.populate(graph)
    graph.events.extend(topology.transfer(f'flow{i}', src, dst, size)
                        for i, (src, dst, size) in enumerate(flows))
    result = simulate(graph)
    if result['status'] != 'feasible':
        raise ValueError(f"Network probe failed: {result['diagnostics']}")
    loads = {key: amount / graph.resources[key].capacity for key, amount in result['resource_work'].items() if amount}
    largest = max(loads.values(), default=0)
    return {
        'duration_s': result['duration_s'],
        'busiest_resources': sorted(key for key, duration in loads.items() if math.isclose(duration, largest, rel_tol=1e-9)),
        'payload_bytes': sum(event.metadata['payload_bytes'] for event in graph.events),
        'physical_bytes': sum(event.metadata['physical_bytes'] for event in graph.events),
        'flows': [{'src': event.metadata['src'], 'dst': event.metadata['dst'],
                   'payload_bytes': event.metadata['payload_bytes'], 'path': event.metadata['path'],
                   'io_group_charges': event.metadata.get('io_group_charges', [])} for event in graph.events],
        'hardware': hardware,
    }


def base_hardware():
    return {
        'name': 'assumed two-host network probe', 'npu_count': 4, 'su_size': 2, 'npus_per_host': 2,
        'scale_out': {'bandwidth_Bps': 50e9, 'injection_Bps': 50e9, 'host_injection_Bps': 100e9,
                      'latency_s': 0, 'efficiency': 1},
        'host_link': {'bandwidth_Bps': 64e9, 'injection_Bps': 64e9, 'latency_s': 0, 'efficiency': 1},
    }


def ecmp_fabric():
    links = []
    def edge(id, src, dst, rate):
        links.append(dict(id=id, src=src, dst=dst, bandwidth_Bps=rate, latency_s=0, efficiency=1, domain='SO'))
    for index in range(4):
        edge(f'attach{index}', f'npu{index}', 'left', 1e12)
        edge(f'exit{index}', 'right', f'npu{index+4}', 1e12)
    for spine in range(2):
        edge(f'spine{spine}.in', 'left', f'spine{spine}', 50e9)
        edge(f'spine{spine}.out', f'spine{spine}', 'right', 50e9)
    return {'name': 'assumed two-spine directed probe', 'npu_count': 8, 'su_size': 4, 'npus_per_host': 4,
            'topology': 'explicit', 'routing_policy': 'su_so', 'routing_seed': 0,
            'scale_out': {'bandwidth_Bps': 1e12, 'injection_Bps': 1e12, 'host_injection_Bps': 4e12, 'efficiency': 1},
            'explicit_links': links}


def experiment():
    cases = {}
    flows = [('npu0', 'npu2', 1e9), ('npu1', 'npu3', 1e9)]
    for name, members in [('independent_nics', [[0], [1]]), ('shared_nic', [[0, 1]])]:
        hardware = base_hardware()
        hardware['io_groups'] = [dict(id=f'nic{i}', members=group, domains=['SO'], bandwidth_Bps=50e9)
                                 for i, group in enumerate(members)]
        cases[name] = run_flows(hardware, flows)
    flows = [('npu0', 'npu2', 1e9), ('npu1', 'cpu0', 1e9)]
    hardware = base_hardware()
    cases['offload_without_pcie_group'] = run_flows(hardware, flows)
    hardware = copy.deepcopy(hardware)
    hardware['io_groups'] = [dict(id='shared_pcie', members=[0, 1], domains=['SO', 'D2H', 'H2D'],
                                  bandwidth_Bps=32e9, duplex='full')]
    cases['offload_with_pcie_group'] = run_flows(hardware, flows)
    for algorithm in ('shortest_hop', 'ecmp_hash'):
        hardware = ecmp_fabric()
        hardware['routing_algorithm'] = algorithm
        cases[algorithm] = run_flows(hardware, [(f'npu{i}', f'npu{j}', 1e9) for i in range(4) for j in range(4, 8)])
    spec = json.loads((Path(__file__).resolve().parents[1] / 'assets/a3-topology.example.json').read_text(encoding='utf-8'))
    fragment = build_topology_fragment(spec)
    for name, endpoints in [('a3_same_hiam', [(2*i, 2*i+1) for i in range(8)]),
                            ('a3_cross_hiam', [(i, i+8) for i in range(8)]),
                            ('a3_cross_host', [(i, i+16) for i in range(8)])]:
        cases[name] = run_flows(fragment['hardware'], [(f'npu{i}', f'npu{j}', 1e9) for i, j in endpoints])
    expected = {'independent_nics': .02, 'shared_nic': .04, 'offload_without_pcie_group': .02,
                'offload_with_pcie_group': .0625, 'shortest_hop': .32,
                'a3_same_hiam': .0050004, 'a3_cross_hiam': .0100016, 'a3_cross_host': .0500046}
    for name, value in expected.items():
        if not math.isclose(cases[name]['duration_s'], value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Capacity/path invariant failed for {name}: {cases[name]['duration_s']} != {value}")
    return {'schema_version': 1, 'evidence_kind': 'analytical_capacity_checks', 'measured_accuracy': None,
            'cases': cases, 'a3_fragment': fragment,
            'notes': ['All numerical bandwidths and latencies in these examples are assumptions, not new measured vendor results.',
                      'The A3 hierarchy follows the cited official structural description; fabrics are lumped abstractions.',
                      'ECMP hashes one path per source/destination pair; collisions remain and no extra capacity is created.',
                      'These probes verify the effect of topology on simulated execution. They do not establish end-to-end inference accuracy.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/hardware-topology-fit'))
    args = parser.parse_args(argv)
    result = experiment()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'a3-hardware-fragment.json').write_text(json.dumps(result['a3_fragment'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 硬件架构与拓扑仿真检查', '', '这些是显式假设下的容量/路径检查，不是新增实机 benchmark 或整网拟合精度。', '',
             '| 场景 | 耗时 ms | 应用 payload GB | 逐跳物理 GB |', '| --- | ---: | ---: | ---: |']
    for name, case in result['cases'].items():
        lines.append(f"| {name} | {case['duration_s']*1000:.4f} | {case['payload_bytes']/1e9:.1f} | {case['physical_bytes']/1e9:.1f} |")
    lines += ['', '同样的传输量，会因共享 NIC、PCIe 争用、路由及节点层级产生不同耗时。JSON 包含每条流的路径、共享资源收费、硬件输入及瓶颈。', '',
              '复现：`python skills/supernode-modeling/scripts/evaluate_hardware_topology.py --output outputs/hardware-topology-fit`', '']
    (args.output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({name: round(case['duration_s']*1000, 6) for name, case in result['cases'].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Compact operator cost ledgers for additive critical-path time windows."""
from __future__ import annotations

import math


COST_LABELS = {
    'compute_s': '计算',
    'communication_s': '通信',
    'memory_access_s': '访存',
    'launch_s': '算子启动',
    'synchronization_wait_s': '显式同步等待',
    'scheduling_wait_s': '调度槽位等待',
    'memory_wait_s': '内存准入等待',
    'resource_contention_s': '资源争用',
    'unattributed_service_s': '校准服务（未细分）',
    'other_s': '其他',
}


def empty_costs():
    return dict.fromkeys(COST_LABELS, 0.0)


def operator_identity(metadata):
    """Use emitted identities; never invent subkernels or communication owners."""
    operator = next((metadata[key] for key in
                     ('kernel', 'semantic', 'operation', 'collective', 'category', 'type', 'kind')
                     if metadata.get(key)), 'Unassigned')
    return (str(metadata.get('phase') or 'unspecified'),
            str(metadata.get('component') or 'Unassigned'), str(operator))


class OperatorLedger:
    def __init__(self):
        self.rows = {}

    def add(self, identity, costs, scale=1.0):
        row = self.rows.setdefault(identity, empty_costs())
        for key in row:
            row[key] += costs.get(key, 0.0) * scale

    def merge(self, breakdown, scale=1.0):
        for row in breakdown['operators']:
            self.add(tuple(row[key] for key in ('phase', 'component', 'operator')),
                     row['components_s'], scale)

    def finish(self, total_s, divisor=1.0):
        rows = [(identity, {key: value / divisor for key, value in costs.items()})
                for identity, costs in self.rows.items()]
        residual = total_s - math.fsum(value for _, costs in rows for value in costs.values())
        if abs(residual) > 1e-12 + total_s * 1e-9:
            raise ValueError(f'Operator attribution does not cover its window: residual={residual}s')
        if residual and rows:
            _, costs = max(rows, key=lambda item: max(item[1].values()))
            key = max(costs, key=costs.get)
            costs[key] = max(0.0, costs[key] + residual)
        operators = []
        components = empty_costs()
        for identity, costs in rows:
            duration = math.fsum(costs.values())
            for key, value in costs.items():
                components[key] += value
            operators.append(dict(zip(('phase', 'component', 'operator'), identity)) | {
                'total_s': duration,
                'share_pct': 100 * duration / total_s if total_s > 0 else None,
                'components_s': costs,
                'component_shares_pct': {key: 100 * value / duration if duration > 0 else None
                                         for key, value in costs.items()},
            })
        operators.sort(key=lambda row: (-row['total_s'], row['phase'], row['component'], row['operator']))
        return {
            'method': 'critical_path_limiting_resource',
            'total_s': total_s,
            'components_s': components,
            'component_shares_pct': {key: 100 * value / total_s if total_s > 0 else None
                                     for key, value in components.items()},
            'operators': operators,
        }

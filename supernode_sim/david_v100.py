"""Opt-in DavidV100 service bounds, never a measured kernel timing profile.

The source is the November 2026 V1.0 MicroBenchmark report, as curated by
ascend-david-v100-operator-optimization. No clock, cache hit rate, padding,
launch cost or mixed MTE/SIMT bandwidth is inferred.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from .config import Spec


CUBE_OPS_PER_CYCLE = {
    'fp32': 512, 'hf32': 4068, 'fp16': 8192, 'bf16': 8192,
    'int8': 16384, 'fp8': 16384, 'hif8': 16384,
}
SIMT_BANDWIDTH_BPS = {
    'hbm': {'read': 1506e9, 'write': 1546e9, 'mixed': 1289e9},
    'l2': {'read': 5305e9, 'write': 5235e9, 'mixed': 5445e9},
}
REFERENCE = 'DavidV100 MicroBenchmark V1.0 (2026-11), CANN 9.1.T560, 32 AI Core'


class CubeWork(Spec):
    mode: Literal['fp32', 'hf32', 'fp16', 'bf16', 'int8', 'fp8', 'hif8']
    useful_ops: float = Field(ge=0, strict=True)
    executed_ops: float = Field(gt=0, strict=True)
    active_aic: StrictInt = Field(ge=1, le=32)

    @model_validator(mode='after')
    def executed_work(self):
        if self.executed_ops < self.useful_ops:
            raise ValueError('executed_ops must include all useful_ops and explicit padding/tail work')
        return self


class MemoryWork(Spec):
    level: Literal['hbm', 'l2']
    access_path: Literal['simt', 'mte', 'mixed']
    read_bytes: float = Field(ge=0, strict=True)
    write_bytes: float = Field(ge=0, strict=True)
    effective_Bps: float | None = Field(default=None, gt=0, strict=True)
    source: str | None = None
    evidence_kind: Literal['measured_reference', 'assumed'] = 'assumed'

    @model_validator(mode='after')
    def bandwidth_scope(self):
        total = self.read_bytes + self.write_bytes
        if not math.isfinite(total) or total <= 0:
            raise ValueError('Each memory level needs a finite positive total of actual transferred bytes')
        if self.effective_Bps is None:
            if self.access_path != 'simt':
                raise ValueError('MTE or mixed paths require effective_Bps and a source; peaks cannot be added')
            if self.source is not None or self.evidence_kind != 'assumed':
                raise ValueError('source/evidence_kind overrides require an explicit effective_Bps')
        elif not self.source or not self.source.strip():
            raise ValueError('An explicit effective_Bps requires a nonempty source with matching measurement conditions')
        return self


class DavidV100Workload(Spec):
    device_sku: Literal['DavidV100']
    cann_version: Literal['9.1.T560']
    enabled_ai_cores: StrictInt = Field(ge=32, le=32)
    frequency_hz: float | None = Field(default=None, gt=0, strict=True)
    cube: CubeWork | None = None
    memory: list[MemoryWork] = Field(default_factory=list)
    dependency_floor_s: float = Field(default=0.0, ge=0, strict=True)
    dependency_source: str | None = None

    @model_validator(mode='after')
    def comparable_scope(self):
        if self.cube is not None and self.frequency_hz is None:
            raise ValueError('Cube cycle conversion requires the corresponding measured frequency_hz')
        if len({item.level for item in self.memory}) != len(self.memory):
            raise ValueError('Aggregate all paths at each memory level into one read/write budget')
        if self.dependency_floor_s and (not self.dependency_source or not self.dependency_source.strip()):
            raise ValueError('dependency_floor_s requires a dependency_source describing its timing boundary')
        if self.cube is None and not self.memory and not self.dependency_floor_s:
            raise ValueError('Provide Cube work, actual memory transfers, or a dependency floor')
        return self


def build_demands(document: dict) -> dict:
    """Return trace-ready resource demands and the assumptions behind each.

    Different memory engines use the SAME resource name for each level.
    A caller-supplied mixed bandwidth must already include their contention.
    The regular graph HBM resource remains active alongside this constraint.
    """
    workload = DavidV100Workload.model_validate(document)
    demands = []
    accounting = {'useful_ops': 0.0, 'executed_ops': 0.0, 'padding_ops': 0.0,
                  'hbm_bytes': 0.0, 'l2_bytes': 0.0, 'memory': []}
    limitations = [
        'Analytical resource lower bounds, not a measured kernel time or validated prediction error.',
        'The reference uses 32 AI Core (32 AIC, 64 AIV); do not scale to physical 36 cores or other SKUs.',
        'Memory reference rates are whole-chip steady-state observations, not guaranteed at small granularity or low concurrency.',
        'No clock, cache hit rate, launch, bank penalty, overlap fraction or UB/DCache partition is inferred.',
        'Bind to an explicit kernel trace with shape, dtype/mode, hardware and runtime guards; no automatic global hardware update.',
        'Original FLOPs/HBM and any stronger configured resource constraints remain active; a service bound can only add a limit.',
        'David memory service resources cover only annotated operators; offload/communication still contend on the original HBM resource. Configure global hbm_Bps separately.',
        'Shared service seconds approximate a fixed resource budget; they do not reconstruct instruction scheduling or concurrent occupancy.',
    ]
    if workload.cube:
        cube = workload.cube
        rate = CUBE_OPS_PER_CYCLE[cube.mode] * workload.frequency_hz * cube.active_aic
        service = cube.executed_ops / rate
        if not math.isfinite(rate) or not math.isfinite(service) or service <= 0:
            raise ValueError('Cube rate/service conversion must remain finite and positive')
        demands.append(dict(resource='david_cube', service_s=service, kind='compute',
                            source=REFERENCE + ' §2.2.1–§2.2.3; dtype-specific steady-state MMAD throughput',
                            evidence_kind='measured_reference'))
        accounting.update(useful_ops=cube.useful_ops, executed_ops=cube.executed_ops,
                          padding_ops=cube.executed_ops - cube.useful_ops,
                          cube_mode=cube.mode, active_aic=cube.active_aic,
                          cube_ops_per_cycle_per_aic=CUBE_OPS_PER_CYCLE[cube.mode],
                          cube_effective_ops_per_s=rate)
        limitations.append('MMAD counts FMA as 2 ops; 92 cycles is not a complete 64³ instruction time. Active-AIC scaling assumes balanced supply.')
        if cube.mode == 'hf32':
            limitations.append('HF32 changes FP32 numerical behavior; this declaration does not authorize changing the precision contract.')
    for transfer in workload.memory:
        direction = ('mixed' if transfer.read_bytes and transfer.write_bytes else
                     'read' if transfer.read_bytes else 'write')
        automatic = transfer.effective_Bps is None
        rate = (SIMT_BANDWIDTH_BPS[transfer.level][direction] if automatic
                else transfer.effective_Bps)
        total = transfer.read_bytes + transfer.write_bytes
        service = total / rate
        if not math.isfinite(service) or service <= 0:
            raise ValueError('Memory service conversion must remain finite and positive')
        source = (REFERENCE + ' §3.4.2/§3.4.4.2; SIMT ' + transfer.level.upper() + ' ' + direction
                  if automatic else transfer.source)
        demands.append(dict(resource='david_' + transfer.level, service_s=service, kind='memory',
                            source=source, evidence_kind='measured_reference' if automatic else transfer.evidence_kind))
        accounting[transfer.level + '_bytes'] = total
        accounting['memory'].append(dict(level=transfer.level, access_path=transfer.access_path,
            read_bytes=transfer.read_bytes, write_bytes=transfer.write_bytes,
            direction=direction, effective_Bps=rate, reference_default=automatic))
        if automatic and direction == 'mixed':
            limitations.append(transfer.level.upper() + ': mixed rate is a reference scenario; the report does not establish a curve for arbitrary read/write ratios.')
        if automatic and transfer.level == 'l2':
            limitations.append('SIMT L2 reference constructs reuse with about 0.99 hit rate; L2 traffic must not be subtracted from HBM merely from capacity.')
    return {
        'schema_version': '1.0', 'model_kind': 'analytical_lower_bound',
        'reference_context': {'device_sku': workload.device_sku, 'cann_version': workload.cann_version,
                              'enabled_ai_cores': 32, 'aic_count': 32, 'aiv_count': 64,
                              'frequency_hz': workload.frequency_hz, 'source': REFERENCE},
        'resource_demands': demands, 'dependency_floor_s': workload.dependency_floor_s,
        'dependency_source': workload.dependency_source,
        'resource_lower_bound_s': max([workload.dependency_floor_s] + [row['service_s'] for row in demands]),
        'accounting': accounting, 'workload': workload.model_dump(exclude_none=True),
        'limitations': limitations,
    }


def operator_document(result: dict, *, kernel: str, trace_id: str) -> dict:
    """A single explicitly budgeted operator for build_operator_trace.py.

    The existing binder enforces work conservation and pins the target shape.
    This does not turn a component budget into an actual implementation trace.
    """
    if not kernel.strip() or not trace_id.strip():
        raise ValueError('kernel and trace_id must be nonempty')
    return {
        'id': trace_id,
        'source': {'status': 'assumed', 'description':
                   'Analytical DavidV100 resource bounds using report references and caller work budgets; not measured kernel timing.'},
        'operations': [{
            'kernel': kernel, 'flops': result['accounting']['useful_ops'],
            'memory_bytes': result['accounting']['hbm_bytes'],
            'resource_demands': result['resource_demands'],
            'dependency_floor_s': result['dependency_floor_s'],
            'implementation': {'david_v100_service_model': result['workload']},
        }],
    }

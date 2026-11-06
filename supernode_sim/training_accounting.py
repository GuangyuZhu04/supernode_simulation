"""Training FLOPs and peak-normalized utilization with explicit numerators.

The model numerator is the compiled forward/backward proxy, not a universal
paper MFU definition. Recompute, optimizer and auxiliary execution remain
separate; calibration timers are never floating-point work.
"""
from __future__ import annotations

from collections import defaultdict
import re

from .config import Config
from .core import Graph


POLICY = (
    'model_mfu is the current compiler model forward/backward FLOPs proxy, '
    'excluding recompute, optimizer, conversions and communication packing; '
    'it is not automatically equivalent to a paper or external simulator MFU. '
    'Each dtype is divided by its declared bare NPU peak, then summed and '
    'divided by configured NPU count and full execution duration. '
    'executed_compute_utilization includes every NPU compute resource FLOP. '
    'These are peak-normalized work ratios, not measured device busy fractions. '
    'CPU work uses its own peak and device count. Phase totals partition '
    'execution; model/recompute fields are subsets, not extra work to add. '
    'Task schedules include only trainable actor/critic F+B in model_mfu; '
    'rollout, frozen reference/reward/value scoring and objectives are auxiliary.'
)


def _model_work(metadata: dict) -> tuple[float, float]:
    """Return F+B and recompute work, including preserved fusion members."""
    fusion = metadata.get('megakernel')
    if fusion:
        pieces = [_model_work(member) for member in fusion.get('member_metadata', [])]
        return tuple(sum(piece[i] for piece in pieces) for i in (0, 1))
    if metadata.get('training_update') is False:
        return 0.0, 0.0
    kernel = metadata.get('original_kernel', metadata.get('kernel'))
    component = metadata.get('component')
    is_model = (kernel in ('training_forward', 'training_backward', 'training_recompute')
                or bool(component and kernel == str(component).lower()))
    if not is_model:
        return 0.0, 0.0
    flops = metadata.get('flops', 0.0)
    phase = metadata.get('phase')
    return (flops if phase in ('forward', 'backward') else 0.0,
            flops if phase == 'recompute' else 0.0)


def _optimizer_work(metadata: dict) -> float:
    fusion = metadata.get('megakernel')
    if fusion:
        return sum(_optimizer_work(member) for member in fusion.get('member_metadata', []))
    return (metadata.get('flops', 0.0)
            if metadata.get('phase') == 'optimizer'
            and metadata.get('original_kernel', metadata.get('kernel')) in ('adam', 'muon') else 0.0)


def training_accounting(config: Config, graph: Graph, simulation: dict) -> dict:
    if config.workload.scenario != 'training':
        raise ValueError('Training accounting requires a training workload')
    duration = simulation.get('duration_s', 0.0)
    if simulation.get('status') != 'feasible' or duration <= 0:
        return {'status': 'unavailable', 'model_mfu': None,
                'executed_compute_utilization': None,
                'reason': 'Complete feasible execution with positive duration is required.',
                'policy': POLICY}

    npu_phases, cpu_phases = defaultdict(float), defaultdict(float)
    npu_roles = dict.fromkeys(('model_forward_backward', 'model_recompute', 'optimizer', 'auxiliary'), 0.0)
    unknown_work = defaultdict(float)
    by_dtype = {}
    npu_ranks, cpu_ranks = set(), set()
    unknown_resources, unknown_phases = set(), set()
    known_phases = {'forward', 'backward', 'recompute', 'optimizer', 'weight_refresh', 'prefill', 'decode', 'loss', 'reward'}
    task_roles = defaultdict(float)
    cpu_devices = {key.split('.')[0] for key, resource in graph.resources.items()
                   if re.fullmatch(r'cpu\d+\.compute', key)
                   and resource.kind == 'compute' and resource.unit == 'FLOP/s'}
    for event in graph.events:
        computations = [(key, amount) for key, amount in event.work.items()
                        if graph.resources[key].kind == 'compute'
                        and graph.resources[key].unit == 'FLOP/s' and amount > 0]
        if not computations:
            continue
        phase = str(event.metadata.get('phase') or 'unspecified')
        if phase not in known_phases:
            unknown_phases.add(phase)
        model_flops, recompute_flops = _model_work(event.metadata)
        optimizer_flops = _optimizer_work(event.metadata)
        # Supported compilers have exactly one compute resource per event.
        # Avoid attributing a metadata total twice if an imported graph does not.
        if len(computations) != 1:
            model_flops = recompute_flops = optimizer_flops = 0.0
            unknown_resources.update(key for key, _ in computations)
        for key, amount in computations:
            if re.fullmatch(r'cpu\d+\.compute', key):
                cpu_ranks.add(key.split('.')[0])
                cpu_phases[phase] += amount
                continue
            match = re.fullmatch(r'(npu\d+)\.compute(?:\.(fp32)|\.(?:calibrated|megakernel)\.(fp8|fp16|bf16|fp32))?', key)
            if not match:
                unknown_resources.add(key)
                unknown_work[key] += amount
                continue
            rank, scalar_dtype, timed_dtype = match.groups()
            dtype = timed_dtype or scalar_dtype or config.execution.precision.compute
            npu_ranks.add(rank)
            npu_phases[phase] += amount
            if event.metadata.get('training_role'):
                task_roles[event.metadata['training_role']] += amount
            row = by_dtype.setdefault(dtype, {
                'executed_flops': 0.0, 'model_forward_backward_flops': 0.0,
                'model_recompute_flops': 0.0, 'flops_by_phase': defaultdict(float),
                'peak_flops_per_device_s': config.hardware.npu.flops.get(dtype),
            })
            row['executed_flops'] += amount
            row['flops_by_phase'][phase] += amount
            # A malformed metadata counter must not exceed the resource work.
            if model_flops + recompute_flops + optimizer_flops > amount * (1 + 1e-9) + 1e-6:
                raise ValueError(f'Model FLOPs metadata exceeds executed work: {event.id}')
            row['model_forward_backward_flops'] += model_flops
            row['model_recompute_flops'] += recompute_flops
            npu_roles['model_forward_backward'] += model_flops
            npu_roles['model_recompute'] += recompute_flops
            npu_roles['optimizer'] += optimizer_flops
            npu_roles['auxiliary'] += max(0.0, amount - model_flops - recompute_flops - optimizer_flops)
            if row['peak_flops_per_device_s'] is None:
                unknown_resources.add(key)

    ideal_model_s = ideal_executed_s = 0.0
    for row in by_dtype.values():
        row['flops_by_phase'] = dict(sorted(row['flops_by_phase'].items()))
        peak = row['peak_flops_per_device_s']
        if peak is not None:
            row['model_peak_device_seconds'] = row['model_forward_backward_flops'] / peak
            row['executed_peak_device_seconds'] = row['executed_flops'] / peak
            ideal_model_s += row['model_peak_device_seconds']
            ideal_executed_s += row['executed_peak_device_seconds']
    denominator = config.hardware.npu_count * duration
    cpu_flops = sum(cpu_phases.values())
    cpu_denominator = len(cpu_devices) * config.hardware.cpu.flops * duration
    return {
        'status': 'complete' if not unknown_resources else 'incomplete_resource_identity',
        'duration_s': duration, 'npu_count': config.hardware.npu_count,
        'active_npu_ranks': sorted(npu_ranks),
        'model_mfu': ideal_model_s / denominator if not unknown_resources else None,
        'executed_compute_utilization': ideal_executed_s / denominator if not unknown_resources else None,
        'model_forward_backward_flops': sum(row['model_forward_backward_flops'] for row in by_dtype.values()),
        'model_recompute_flops': sum(row['model_recompute_flops'] for row in by_dtype.values()),
        'executed_npu_flops': sum(row['executed_flops'] for row in by_dtype.values()),
        'npu_flops_by_phase': dict(sorted(npu_phases.items())),
        'npu_flops_by_role': npu_roles,
        'npu_flops_by_training_role': dict(task_roles),
        'npu_by_dtype': dict(sorted(by_dtype.items())),
        'unknown_compute_resources': sorted(unknown_resources),
        'unknown_compute_flops': dict(sorted(unknown_work.items())),
        'unclassified_phase_names': sorted(unknown_phases),
        'cpu': {'device_count': len(cpu_devices), 'active_ranks': sorted(cpu_ranks),
                'executed_flops': cpu_flops, 'optimizer_phase_flops': cpu_phases.get('optimizer', 0.0),
                'flops_by_phase': dict(sorted(cpu_phases.items())),
                'peak_flops_per_device_s': config.hardware.cpu.flops,
                'executed_compute_utilization': cpu_flops / cpu_denominator if cpu_denominator else None},
        'policy': POLICY,
    }

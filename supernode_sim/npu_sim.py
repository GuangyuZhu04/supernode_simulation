"""Import one NPUSim CA kernel summary without replaying instruction lanes.

NPUSim's kernel clock window already includes intra-kernel pipeline overlap.
Only that window becomes a service-time constraint. Pipeline utilization and
on-chip bandwidth remain diagnostics, never FLOPs or HBM traffic estimates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, StrictInt, StrictStr, model_validator

from .analysis import compile_graph
from .calibration import _same_declaration
from .config import (Config, KernelProfile, Source, Spec,
                     validate_calibration_context, validate_json_metadata)


REFERENCE_REVISION = '8dc64db200f497c55db31d3f970e4cc054786a25'
REFERENCE_URL = ('https://gitcode.com/cann/npu-simulator/blob/'
                 + REFERENCE_REVISION + '/npusim/prof_demo/reports/summary.py')
SCOPE_FIELDS = (
    'component', 'layer', 'model_shape_signature', 'tokens', 'workload_shape',
    'training_task', 'training_role', 'training_stage', 'training_update',
    'tensor_parallel_degree', 'expert_parallel_degree', 'sequence_parallel_degree',
    'trace_id', 'original_kernel', 'operation_index', 'matmul', 'implementation',
)


def _number(value, field: str, *, positive: bool = False) -> float:
    # SQLite-backed NPUSim versions can serialize numeric values as strings.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f'{field} must be a finite number')
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{field} must be a finite number') from exc
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f'{field} must be finite and {"positive" if positive else "nonnegative"}')
    return number


def analyze_summary(document: dict) -> dict:
    """Normalize the official per-launch summary, preserving raw diagnostics."""
    if not isinstance(document, dict) or not isinstance(document.get('kernel_info'), dict):
        raise ValueError('Expected one NPUSim summary.json with kernel_info, not a trace or run directory')
    validate_json_metadata(document)
    info = document['kernel_info']
    clocks = _number(info.get('kernel_total_clocks'), 'kernel_total_clocks', positive=True)
    frequency = _number(info.get('frequency_ghz'), 'frequency_ghz', positive=True)
    active = _number(info.get('ai_core_active'), 'ai_core_active', positive=True)
    if not active.is_integer():
        raise ValueError('ai_core_active must be an integer')
    duration = clocks / frequency / 1e9
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Derived kernel duration must be finite and positive')
    diagnostics = {key: deepcopy(document[key]) for key in (
        'pipe_utilization', 'top_level_diagnosis', 'pipeline_overlap', 'bandwidth',
        'scalar_instructions', 'aiv_vector_instructions', 'cache', 'simd_vf_metrics',
    ) if key in document}
    pipes = diagnostics.get('pipe_utilization', {})
    if not isinstance(pipes, dict) or not isinstance(pipes.get('pipeline_util_summary', {}), dict):
        raise ValueError('pipe_utilization.pipeline_util_summary must be an object')
    for name, stats in pipes.get('pipeline_util_summary', {}).items():
        if not isinstance(stats, dict):
            raise ValueError(f'Pipeline {name} must contain utilization statistics')
        for key in ('mean', 'min', 'max'):
            if key in stats and _number(stats[key], f'{name}.{key}') > 1:
                raise ValueError('NPUSim pipeline utilization must be a fraction in [0, 1]')
        if 'per_core' in stats:
            if not isinstance(stats['per_core'], list) or any(
                    _number(value, f'{name}.per_core') > 1 for value in stats['per_core']):
                raise ValueError('NPUSim per_core must be an array of fractions in [0, 1]')
    return {
        'schema_version': '1.0', 'evidence_kind': 'unspecified',
        'duration_s': duration, 'duration_us': duration * 1e6,
        'kernel_total_clocks': clocks, 'frequency_ghz': frequency,
        'ai_core_active': int(active), 'diagnostics': diagnostics,
        'timing_basis': 'kernel_info.kernel_total_clocks / (frequency_ghz * 1e9)',
        'timing_scope': 'device_kernel_window',
        'reference': REFERENCE_URL,
        'limitations': [
            'Summary alone does not establish CA/PV mode, correctness, capture coverage or operator identity.',
            'Core and pipeline durations overlap; they are not summed or multiplied by active cores.',
            'On-chip bandwidth and utilization are diagnostics, not HBM bytes or resource capacity.',
            'In the referenced version kernel_info is kernel-wide; pipeline/scalar diagnostics may cover selected report cores.',
        ],
    }


def unprofiled_graph(config: Config):
    """Retain operator expansion; remove only timing profiles for inspection."""
    data = config.model_dump()
    data['hardware']['kernel_profiles'] = {}
    return compile_graph(data)


def describe_target(event, config: Config) -> dict | None:
    """Only an explicitly implemented suboperator can bind a device kernel."""
    metadata = event.metadata
    if (not metadata.get('trace_parent_event') or not metadata.get('implementation')
            or not metadata.get('model_shape_signature') or not metadata.get('workload_shape')
            or metadata.get('megakernel') or metadata.get('collective')):
        return None
    compute_keys = [key for key in event.work if re.fullmatch(r'npu\d+\.compute(?:\.fp32)?', key)]
    if len(compute_keys) != 1 or not metadata.get('kernel') or not metadata.get('phase'):
        return None
    key = compute_keys[0]
    node = key.split('.')[0]
    if set(event.work) - {key, node + '.hbm'} or event.slots != (node + '.compute_stream',):
        return None
    return {
        'event_id': event.id, 'kernel': metadata['kernel'], 'phase': metadata['phase'],
        'dtype': 'fp32' if key.endswith('.fp32') else config.execution.precision.compute,
        'flops': event.work[key], 'memory_bytes': event.work.get(node + '.hbm', 0),
        'original_startup_s': event.startup_s,
        'metadata_match': {key: deepcopy(metadata[key]) for key in SCOPE_FIELDS if key in metadata},
    }


def list_targets(config: Config) -> list[dict]:
    return [target for event in unprofiled_graph(config).events
            if (target := describe_target(event, config)) is not None]


def _validate_implementation(implementation: dict) -> None:
    """These fields must come from compilation and the actual invocation."""
    for key in ('op_type', 'kernel_name', 'tiling_key'):
        if not isinstance(implementation.get(key), str) or not implementation[key].strip():
            raise ValueError(f'implementation.{key} must be a nonempty string')
    if not isinstance(implementation.get('binary_sha256'), str) or not re.fullmatch(
            r'[0-9a-f]{64}', implementation['binary_sha256']):
        raise ValueError('implementation.binary_sha256 must be the lowercase SHA-256 of the compiled kernel')
    if type(implementation.get('block_dim')) is not int or implementation['block_dim'] < 1:
        raise ValueError('implementation.block_dim must be a positive integer')
    for key in ('inputs', 'outputs'):
        tensors = implementation.get(key)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f'implementation.{key} must describe tensor shapes, dtypes and formats')
        for tensor in tensors:
            if not isinstance(tensor, dict) or not isinstance(tensor.get('shape'), list):
                raise ValueError(f'implementation.{key} tensors require a shape array')
            if any(type(dim) is not int or dim < 1 for dim in tensor['shape']):
                raise ValueError('Tensor dimensions must be positive integers; use [] for a scalar')
            if any(not isinstance(tensor.get(field), str) or not tensor[field].strip()
                   for field in ('dtype', 'format')):
                raise ValueError('Tensor dtype and format must be nonempty strings')


class NpuSimManifest(Spec):
    """Caller-owned link between a single launch and an engine suboperator."""
    schema_version: Literal['1.0'] = '1.0'
    mode: Literal['ca']
    correctness: Literal['passed']
    capture_scope: Literal['all_active_cores']
    expected_active_cores: StrictInt = Field(gt=0)
    hardware_name: StrictStr = Field(min_length=1)
    simulator_version: StrictStr = Field(min_length=1)
    simulator_revision: StrictStr = Field(min_length=1)
    command: StrictStr = Field(min_length=1)
    summary_sha256: StrictStr = Field(pattern=r'^[0-9a-f]{64}$')
    calibration_context: dict[str, StrictStr]
    target: dict
    evidence_kind: Literal['simulator', 'synthetic'] = 'simulator'
    startup_s: float | None = Field(default=None, ge=0,
        description='Optional separately justified startup, outside the device kernel window.')
    source: Source

    @model_validator(mode='after')
    def identity(self):
        validate_calibration_context(self.calibration_context, require_complete=True)
        if self.calibration_context['timer_boundary'] != 'device_only':
            raise ValueError('NPUSim summary requires device_only timer_boundary')
        if self.source.status == 'measured':
            raise ValueError('NPUSim simulator/synthetic evidence is not hardware measured evidence')
        for key in ('hardware_name', 'simulator_version', 'simulator_revision', 'command'):
            if not getattr(self, key).strip():
                raise ValueError(f'{key} must be nonempty')
        if re.search(r'(^|\s)--pv(?:\s|$|=)', self.command):
            raise ValueError('PV functional simulation cannot supply performance timing')
        validate_json_metadata(self.target)
        return self


def build_calibration(config: Config, summary_path: str | Path, manifest: dict) -> dict:
    """Return an additive overlay plus diagnostics; do not mutate the config."""
    spec = NpuSimManifest.model_validate(manifest)
    payload = Path(summary_path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec.summary_sha256:
        raise ValueError('summary_sha256 does not match the selected report')
    report = analyze_summary(json.loads(payload.decode('utf-8-sig')))
    if report['ai_core_active'] != spec.expected_active_cores:
        raise ValueError('Active core count differs from the capture manifest')
    if spec.hardware_name != config.hardware.name or not _same_declaration(
            spec.calibration_context, config.hardware.calibration_context):
        raise ValueError('Manifest hardware/environment does not match the target configuration')
    event_id = spec.target.get('event_id')
    graph = unprofiled_graph(config)
    event = next((item for item in graph.events if item.id == event_id), None)
    target = describe_target(event, config) if event is not None else None
    if target is None:
        raise ValueError('Target must be an expanded operator with explicit implementation and workload shape; use --list')
    if not _same_declaration(spec.target, target):
        raise ValueError('Manifest target differs from the compiled operator work/shape/implementation')
    _validate_implementation(target['metadata_match']['implementation'])
    source = spec.source.model_dump(exclude_none=True)
    provenance = {
        'tool': 'npusim', 'mode': spec.mode, 'correctness': spec.correctness,
        'simulator_version': spec.simulator_version, 'simulator_revision': spec.simulator_revision,
        'command': spec.command, 'summary_sha256': digest,
        'capture_scope': spec.capture_scope, 'expected_active_cores': spec.expected_active_cores,
        'kernel_total_clocks': report['kernel_total_clocks'], 'frequency_ghz': report['frequency_ghz'],
        'timing_basis': report['timing_basis'], 'timing_scope': report['timing_scope'],
        'source': source, 'reference': REFERENCE_URL,
    }
    profile = KernelProfile.model_validate({
        **{key: target[key] for key in ('flops', 'memory_bytes', 'dtype', 'phase', 'metadata_match')},
        'hardware_name': config.hardware.name, 'calibration_context': spec.calibration_context,
        'duration_s': report['duration_s'], 'startup_s': spec.startup_s,
        'evidence_kind': spec.evidence_kind, 'provenance': provenance,
        'source': spec.source.description + (f' ({spec.source.uri})' if spec.source.uri else ''),
    }).model_dump(exclude_none=True)
    # Validate with existing profiles still present: conflicting imports and
    # impossible service times fail here rather than only in the next run.
    candidate = config.model_dump()
    candidate['hardware']['kernel_profiles'].setdefault(target['kernel'], []).append(profile)
    validated = compile_graph(candidate)
    applied = next(item for item in validated.events if item.id == event_id).metadata.get('calibration', {})
    if applied.get('provenance', {}).get('summary_sha256') != digest:
        raise ValueError('Imported NPUSim profile did not bind to the selected operator')
    report.update(evidence_kind=spec.evidence_kind, target=target, provenance=provenance,
                  startup_s=applied['startup_s'],
                  startup_policy='preserved_event_startup' if spec.startup_s is None else 'explicit_replacement',
                  matched_events=sum(item.metadata.get('calibration', {}).get('provenance', {}).get(
                      'summary_sha256') == digest for item in validated.events))
    return {
        'calibration': {'hardware': {'kernel_profiles': {target['kernel']: [profile]}}, 'source': source},
        'report': report,
    }

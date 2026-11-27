"""Training strong/weak scaling studies with a frozen profile and independent holdouts."""
from __future__ import annotations

import json
from math import fsum, isclose
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .analysis import run
from .config import Config, Spec, load_config, read_data
from .report import write_training_summary
from .search import set_field


# Other settings (model, precision, kernels, measured rates and runtime) are inherited
# from the same resolved config. Topology/placement changes are visible in each row.
LAYOUT_FIELDS = {
    'hardware.npu_count', 'hardware.su_size', 'hardware.npus_per_host', 'hardware.topology',
    'hardware.mesh_width', 'hardware.leaf_size', 'hardware.explicit_links', 'hardware.routes',
    'hardware.io_groups', 'execution.tp', 'execution.pp', 'execution.ep', 'execution.dp',
    'execution.fsdp', 'execution.dp_domain_size', 'execution.rank_order', 'execution.cp', 'execution.expert_tp',
    'execution.attention_dp', 'execution.expert_dp', 'execution.expert_rank_order',
    'execution.fsdp_domain', 'execution.fsdp_domain.dense_degree', 'execution.fsdp_domain.expert_degree',
    'execution.fsdp_domain.dense_groups', 'execution.fsdp_domain.expert_groups',
    'workload.batch_size', 'workload.gradient_accumulation',
}
SCORED = ('step_time_s', 'steady_step_time_s', 'peak_hbm_bytes', 'exposed_comm_s_per_rank_per_step')


class Observation(Spec):
    kind: Literal['measured', 'synthetic']
    source: str = Field(min_length=1)
    step_time_s: float | None = Field(default=None, gt=0)
    steady_step_time_s: float | None = Field(default=None, gt=0)
    peak_hbm_bytes: float | None = Field(default=None, ge=0)
    exposed_comm_s_per_rank_per_step: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def has_values(self):
        if not self.source.strip() or not any(getattr(self, key) is not None for key in SCORED):
            raise ValueError('An observation needs a source and at least one metric')
        return self


class ScalingCase(Spec):
    name: str = Field(min_length=1)
    changes: dict[str, Any] = Field(default_factory=dict)
    observation: Observation | None = None

    @model_validator(mode='after')
    def check_changes(self):
        frozen = self.changes.keys() - LAYOUT_FIELDS
        if frozen:
            raise ValueError(f'Frozen training settings cannot change within a scaling study: {sorted(frozen)}')
        return self


class ScalingStudy(Spec):
    config: str
    scaling: Literal['strong', 'weak']
    baseline: ScalingCase
    holdouts: list[ScalingCase] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_names(self):
        names = [case.name for case in [self.baseline, *self.holdouts]]
        if len(names) != len(set(names)):
            raise ValueError('Scaling case names must be unique')
        return self


def _native_case(config, case):
    data = config.model_dump()
    for path, value in case.changes.items():
        set_field(data, path, value)
    row = {'name': case.name, 'changes': case.changes, 'prediction': None,
           'observation': case.observation.model_dump(exclude_none=True) if case.observation else None}
    try:
        spec = Config.model_validate(data)
    except ValueError as error:
        row.update(simulation_status='invalid_config', diagnostics=[{'code': 'invalid_config', 'message': str(error)}])
        return row, {'schema_version': 1, 'status': 'unavailable', 'reason': str(error)}
    result = run(spec, 'simulate', full_detail=False)
    summary = result['training_summary']
    metadata = result['model_metadata']
    calibration = {}
    for key, fields in (
        ('kernel_calibration', ('eligible_events', 'matched_events', 'guarded_matched_events')),
        ('collective_calibration', ('eligible_collectives', 'matched_collectives', 'guarded_matched_collectives')),
    ):
        calibration[key] = {field: metadata[key][field] for field in fields}
    row.update({
        'config_sha256': result['config_sha256'],
        'devices': spec.hardware.npu_count, 'su_size': spec.hardware.su_size,
        'parallelism': {axis: getattr(spec.execution, axis) for axis in ('dp', 'tp', 'pp', 'ep', 'fsdp', 'dp_domain_size')},
        'simulation_status': result['status'], 'fidelity': metadata['fidelity'], 'coverage': calibration,
    })
    if 'parallel_groups' in metadata:
        row['parallelism'].update({k: metadata['parallel_groups'][k] for k in ('cp', 'expert_tp', 'attention_dp', 'expert_dp')})
    if 'fsdp_domains' in metadata:
        row['parallelism']['fsdp_domains'] = metadata['fsdp_domains']
    if summary['status'] != 'complete':
        row['diagnostics'] = result['simulation'].get('diagnostics', []) or [summary['reason']]
        return row, summary
    metrics = result['metrics']
    steps = spec.workload.training_steps
    rank_steps = spec.hardware.npu_count * steps
    exposure = fsum(state['exposed_transfer'] for state in summary['rank_activity_s'].values())
    breakdown = metrics['training_latency_breakdown'].get('step')
    path_comm = None if not breakdown else fsum(
        time for component, time in breakdown['components_s'].items() if component.startswith('communication_'))
    fabric = metrics['training_communication']['by_class']
    wire = summary.get('moe_wire', {}).get('off_rank_sent_bytes_by_phase', {})
    accounting = metrics['training_accounting']
    row['prediction'] = {
        'tokens_per_step': metrics['training_tokens'] / steps, 'step_time_s': metrics['step_time_s'],
        'steady_step_time_s': metrics['steady_step_time_s'],
        'throughput_tokens_s': metrics['throughput_tokens_s'], 'peak_hbm_bytes': summary['peak_hbm']['peak_bytes'],
        'exposed_comm_s_per_rank_per_step': exposure / rank_steps, 'critical_path_comm_service_s_per_step': path_comm,
        'sent_payload_bytes_per_rank_per_step_by_class': {axis: item['payload_bytes'] / rank_steps for axis, item in fabric.items()},
        'ep_sent_bytes_per_rank_per_step_by_phase': {phase: {k: v / rank_steps for k, v in ops.items()} for phase, ops in wire.items()},
        'model_mfu': accounting['model_mfu'],
        'expert_padding_flops_per_step': accounting['npu_flops_by_role'].get('expert_padding', 0.0) / steps,
    }
    return row, summary


def _error(predicted, measured):
    delta = predicted - measured
    return {'delta': delta, 'relative_pct': 100 * delta / measured if measured else None}


def _measured(row):
    obs = row['observation'] or {}
    return obs if obs.get('kind') == 'measured' else {}


def _compare(row, base, mode):
    pred, ref = row['prediction'], base['prediction']
    observed, baseline_observed = _measured(row), _measured(base)
    row['native_errors'] = {key: _error(pred[key], observed[key]) for key in SCORED
                            if pred is not None and pred[key] is not None and key in observed}
    evidence = (row['observation'] or {}).get('kind')
    row['validation_status'] = ('prediction_only' if evidence is None else
                                'synthetic_reference' if evidence == 'synthetic' else
                                'compared_to_measurement' if row['native_errors'] else 'prediction_unavailable')
    if row is base:
        row['role'] = 'baseline'
        return
    row['role'] = 'holdout'
    if pred is None or ref is None:
        row['comparability'] = 'incomplete_simulation'
        return
    if row['config_sha256'] == base['config_sha256']:
        row['comparability'] = 'same_configuration'
        return
    device_ratio = row['devices'] / base['devices']
    expected_tokens = ref['tokens_per_step'] * (device_ratio if mode == 'weak' else 1)
    if min(pred['tokens_per_step'], expected_tokens) <= 0 or not isclose(
            pred['tokens_per_step'], expected_tokens, rel_tol=1e-12, abs_tol=0):
        row['comparability'] = 'workload_mismatch'
        return
    row['comparability'] = 'matched'
    speedup = pred['throughput_tokens_s'] / ref['throughput_tokens_s']
    row['scaling'] = {'device_ratio': device_ratio, 'throughput_speedup': speedup, 'efficiency': speedup / device_ratio}
    if 'step_time_s' in baseline_observed:
        adjusted = baseline_observed['step_time_s'] * pred['step_time_s'] / ref['step_time_s']
        row['baseline_adjusted_step_time_s'] = adjusted
        if 'step_time_s' in observed:
            row['baseline_adjusted_step_error'] = _error(adjusted, observed['step_time_s'])
            measured_speedup = (pred['tokens_per_step'] / ref['tokens_per_step']
                                * baseline_observed['step_time_s'] / observed['step_time_s'])
            row['scaling']['measured_efficiency'] = measured_speedup / device_ratio


def run_training_scaling(manifest, *, output=None):
    path = Path(manifest)
    study = ScalingStudy.model_validate(read_data(path))
    config = load_config(path.parent / study.config)
    if config.workload.scenario != 'training':
        raise ValueError('training-scaling requires a training config')
    rows = []
    for index, case in enumerate([study.baseline, *study.holdouts]):
        row, summary = _native_case(config, case)
        if output is not None:
            directory = Path(output) / f'case_{index:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            write_training_summary(summary, directory)
            row['summary_file'] = f'{directory.name}/training_summary.json'
            if summary['status'] == 'complete':
                row['chart_file'] = f'{directory.name}/training_breakdown.svg'
        rows.append(row)
    for row in rows:
        _compare(row, rows[0], study.scaling)
    return {
        'schema_version': 1, 'scaling': study.scaling, 'config': study.config,
        'sources': {key: src.model_dump(exclude_none=True) for key, src in config.sources.items()},
        'calibration_context': config.hardware.calibration_context, 'assumptions': config.assumptions,
        'measured_holdouts_compared': len({r['config_sha256'] for r in rows[1:]
                                          if r['native_errors'] and r['comparability'] == 'matched'}),
        'cases': rows,
    }


def write_scaling_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'training_scaling.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return path

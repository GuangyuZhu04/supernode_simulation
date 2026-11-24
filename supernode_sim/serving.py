"""Arrival-driven FCFS batching around the existing fixed-batch execution engine.

Each instance owns an isolated copy of the configured hardware. A dispatched
batch holds its instance until the complete model execution finishes; new
requests cannot join it. This is neither continuous batching nor ServingCast.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from pydantic import Field, StrictInt, StrictStr, model_validator

from .analysis import run
from .cache import BoundedCache
from .config import Config, Spec
from .core import shift_schedule


class ServingRequest(Spec):
    id: StrictStr = Field(min_length=1)
    arrival_s: float = Field(ge=0, strict=True)
    input_tokens: StrictInt = Field(gt=0)
    output_tokens: StrictInt = Field(gt=0)

    @model_validator(mode='after')
    def named(self):
        if not self.id.strip():
            raise ValueError('serving request id must be nonempty')
        return self


class ServingSLO(Spec):
    ttft_s: float | None = Field(default=None, ge=0, strict=True)
    tpot_s: float | None = Field(default=None, ge=0, strict=True)


class ServingSpec(Spec):
    requests: list[ServingRequest] = Field(min_length=1)
    instance_count: StrictInt = Field(default=1, gt=0)
    max_batch_size: StrictInt = Field(default=1, gt=0)
    max_batch_wait_s: float = Field(default=0.0, ge=0, strict=True)
    scheduler_overhead_s: float = Field(default=0.0, ge=0, strict=True)
    slo: ServingSLO | None = None

    @model_validator(mode='after')
    def unique_requests(self):
        ids = [request.id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError('serving request ids must be unique')
        return self


LIMITATIONS = [
    'FCFS admission with non-preemptive fixed batches; no continuous batching, chunked prefill, token-level scheduling or preemption.',
    'Instances replicate the complete configured hardware and do not share GPUs/NPUs, host memory, NICs or network capacity.',
    'Weights are preloaded. Each batch starts with fresh request KV state; there is no prefix reuse, cross-batch cache residency or cold-start model loading.',
    'The model-start-to-token times come from supernode_sim analytical/event execution. HTTP, tokenization, client transport and response forwarding are excluded.',
    'Scheduler overhead is one declared wall-time cost per batch. It does not contend for a modeled CPU or network resource.',
    'Original config.targets are recorded but not used for serving admission or SLOs; serving.slo evaluates client TTFT and per-request mean TPOT.',
    'Infeasible batches are not resized or retried. Their elapsed execution and scheduler cost occupy the instance, then its batch state is reset.',
    'Request quantiles describe this finite synthetic arrival trace and model; they are not measured production tail latency or a full ServingCast reproduction.',
]


def _execute_batch(config: Config) -> dict:
    """Production execution path; kept separate for hand-computed scheduler tests."""
    return run(config, mode='simulate', full_detail=False)


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite nonnegative number')
    return float(value)


def _time_sum(*values: float) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('serving timestamps overflowed finite seconds') from exc
    return _number(result, 'serving timestamp')


def _validated_execution(result: dict, config: Config) -> dict:
    """Reject malformed/incomplete success records instead of inventing tokens."""
    if not isinstance(result, dict) or not isinstance(result.get('simulation'), dict):
        raise ValueError('Batch executor must return a simulation object')
    simulation = result['simulation']
    status = simulation.get('status')
    if status not in ('feasible', 'infeasible'):
        raise ValueError(f'Batch executor returned an engine error or unknown status: {status!r}')
    duration = _number(simulation.get('duration_s'), 'batch duration_s')
    token_times = {}
    if status == 'feasible':
        rows = result.get('metrics', {}).get('requests')
        if not isinstance(rows, list) or len(rows) != config.workload.batch_size:
            raise ValueError('Successful batch must contain exactly one token record per request')
        for row in rows:
            if not isinstance(row, dict) or type(row.get('replica')) not in (str, int) or str(row['replica']) != '0':
                raise ValueError('Serving token records must belong to DP replica 0')
            sequence = row.get('sequence')
            if type(sequence) not in (str, int) or not str(sequence).isdigit():
                raise ValueError('Serving token records require a nonnegative sequence index')
            index = int(sequence)
            if index >= config.workload.batch_size or index in token_times:
                raise ValueError('Batch token records contain an out-of-range or duplicate sequence index')
            times = row.get('token_completion_s')
            if not isinstance(times, list) or len(times) != config.workload.output_lengths[index]:
                raise ValueError('Batch token count must equal the requested output length')
            times = [_number(value, 'token_completion_s') for value in times]
            if any(right < left for left, right in zip(times, times[1:])):
                raise ValueError('Batch token completion times must be nondecreasing')
            if times[-1] > duration + max(1e-12, duration * 1e-9):
                raise ValueError('Batch token completion exceeds its execution duration')
            token_times[index] = times
        if set(token_times) != set(range(config.workload.batch_size)):
            raise ValueError('Batch token records do not cover every request')
    return {'engine_status': status, 'duration_s': duration,
            'token_times': token_times, 'diagnostics': deepcopy(simulation.get('diagnostics', [])),
            'memory_peaks_bytes': deepcopy(simulation.get('memory_peaks_bytes', {})),
            'config_sha256': result.get('config_sha256'),
            'model_metadata': deepcopy(result.get('model_metadata', {})),
            'sources': deepcopy(result.get('sources', {}))}


def _distribution(values: list[float], cohort: str) -> dict:
    values = sorted(values)
    count = len(values)
    summary = {'sample_count': count, 'cohort': cohort,
               'mean_s': math.fsum(value / count for value in values) if count else None,
               'min_s': values[0] if count else None, 'max_s': values[-1] if count else None}
    for percentile in (50, 90, 95, 99):
        position = (count - 1) * percentile / 100 if count else 0
        lower, upper = math.floor(position), math.ceil(position)
        summary[f'p{percentile}_s'] = (values[lower] + (values[upper] - values[lower]) * (position - lower)
                                         if count else None)
    return summary


def _slo_checks(row: dict, slo: ServingSLO | None) -> tuple[bool | None, list[dict]]:
    if slo is None or (slo.ttft_s is None and slo.tpot_s is None):
        return None, []
    checks = []
    for metric, target, value in (('client_ttft_s', slo.ttft_s, row['client_ttft_s']),
                                  ('tpot_s', slo.tpot_s, row['tpot_s'])):
        if target is None:
            continue
        not_applicable = metric == 'tpot_s' and row['status'] == 'completed' and row['requested_output_tokens'] == 1
        passed = None if not_applicable else value is not None and value <= target
        checks.append({'metric': metric, 'target_s': target, 'actual_s': value,
                       'status': 'not_applicable' if not_applicable else 'passed' if passed else 'failed',
                       'passed': passed})
    return row['status'] == 'completed' and all(check['passed'] is not False for check in checks), checks


def simulate_serving(config: Config | dict, serving: ServingSpec | dict) -> dict[str, Any]:
    base_spec = config if isinstance(config, Config) else Config.model_validate(config)
    spec = serving if isinstance(serving, ServingSpec) else ServingSpec.model_validate(serving)
    if base_spec.workload.scenario != 'inference':
        raise ValueError('Serving requires an inference configuration')
    if base_spec.execution.dp != 1:
        raise ValueError('Serving requires DP=1; instance_count explicitly replicates isolated hardware')
    features = base_spec.execution.features
    if features.speculative_decode.enabled or features.prefill_decode_disaggregation.enabled:
        raise ValueError('Serving does not implement speculative decode or prefill/decode disaggregated pool scheduling')
    base = base_spec.model_dump()
    base['calibration'] = None  # The parsed model already contains the resolved overrides.
    clock_origin = min(request.arrival_s for request in spec.requests)
    arrival = lambda request: request.arrival_s - clock_origin
    absolute = lambda value: _time_sum(clock_origin, value)
    queue = sorted(enumerate(spec.requests), key=lambda pair: (pair[1].arrival_s, pair[0]))
    available = [0.0] * spec.instance_count
    cache = BoundedCache()
    cache_hits = 0
    requests, batches, executions = [], [], []
    cursor = 0
    while cursor < len(queue):
        oldest = queue[cursor][1]
        # All instances already idle at this arrival are equally available;
        # deterministic ties use the lowest instance ID, not its idle age.
        instance = min(range(spec.instance_count), key=lambda i: (max(available[i], arrival(oldest)), i))
        ready = max(available[instance], arrival(oldest))
        deadline = max(ready, _time_sum(arrival(oldest), spec.max_batch_wait_s))
        last_full = cursor + spec.max_batch_size - 1
        dispatch = (max(ready, arrival(queue[last_full][1]))
                    if last_full < len(queue) and arrival(queue[last_full][1]) <= deadline else deadline)
        stop = cursor
        while stop < len(queue) and stop - cursor < spec.max_batch_size and arrival(queue[stop][1]) <= dispatch:
            stop += 1
        batch = queue[cursor:stop]
        cursor = stop
        input_lengths = tuple(request.input_tokens for _, request in batch)
        output_lengths = tuple(request.output_tokens for _, request in batch)
        key = (input_lengths, output_lengths)
        model_start = _time_sum(dispatch, spec.scheduler_overhead_s)
        schedules = {name: shift_schedule(points, model_start)
                     for name, points in base_spec.hardware.bandwidth_schedules.items()}
        if schedules:
            key += (tuple(sorted(schedules.items())),)
        execution = cache.get(key)
        cache_hit = execution is not None
        if execution is None:
            data = deepcopy(base)
            data['hardware']['bandwidth_schedules'] = schedules
            data['targets'] = {}
            data['workload'].update(batch_size=len(batch), input_length=input_lengths[0], output_length=output_lengths[0],
                                    input_lengths=list(input_lengths), output_lengths=list(output_lengths))
            batch_spec = Config.model_validate(data)
            execution = _validated_execution(_execute_batch(batch_spec), batch_spec)
            execution['execution_id'] = len(executions)
            execution['config_sha256'] = execution['config_sha256'] or hashlib.sha256(
                json.dumps(batch_spec.model_dump(), sort_keys=True, allow_nan=False).encode()).hexdigest()
            executions.append({name: deepcopy(value) for name, value in execution.items() if name != 'token_times'} |
                              {'input_lengths': list(input_lengths), 'output_lengths': list(output_lengths)})
            cache[key] = execution
        else:
            cache_hits += 1
        end = _time_sum(model_start, execution['duration_s'])
        available[instance] = end
        success = execution['engine_status'] == 'feasible'
        batch_id = len(batches)
        batches.append({'id': batch_id, 'instance_id': instance,
            'request_ids': [request.id for _, request in batch], 'admission_s': absolute(dispatch),
            'model_start_s': absolute(model_start), 'end_s': absolute(end), 'duration_s': execution['duration_s'],
            'admission_relative_s': dispatch, 'model_start_relative_s': model_start, 'end_relative_s': end,
            'batch_wait_s': dispatch - ready, 'scheduler_overhead_s': spec.scheduler_overhead_s,
            'status': 'completed' if success else 'failed', 'engine_status': execution['engine_status'],
            'execution_id': execution['execution_id'], 'cache_hit': cache_hit,
            'config_sha256': execution['config_sha256'], 'diagnostics': deepcopy(execution['diagnostics']),
            'memory_peaks_bytes': deepcopy(execution['memory_peaks_bytes'])})
        for sequence, (original_index, request) in enumerate(batch):
            local_times = execution['token_times'][sequence] if success else []
            relative_times = [_time_sum(model_start, value) for value in local_times]
            absolute_times = [absolute(value) for value in relative_times]
            first, last = (relative_times[0], relative_times[-1]) if success else (None, None)
            intervals = [right - left for left, right in zip(local_times, local_times[1:])]
            tpot = math.fsum(intervals) / len(intervals) if intervals else None
            wait = dispatch - arrival(request)
            row = {'id': request.id, 'input_order': original_index,
                'status': 'completed' if success else 'failed', 'instance_id': instance, 'batch_id': batch_id,
                'arrival_s': request.arrival_s, 'admission_s': absolute(dispatch), 'model_start_s': absolute(model_start),
                'first_token_s': absolute(first) if success else None,
                'last_token_s': absolute(last) if success else None, 'finish_s': absolute(last if success else end),
                'arrival_relative_s': arrival(request), 'admission_relative_s': dispatch,
                'model_start_relative_s': model_start, 'first_token_relative_s': first,
                'last_token_relative_s': last, 'finish_relative_s': last if success else end,
                'token_times_relative_s': relative_times,
                'token_times_s': absolute_times, 'input_tokens': request.input_tokens,
                'requested_output_tokens': request.output_tokens, 'output_tokens': len(absolute_times),
                'admission_wait_s': wait,
                'model_ttft_s': local_times[0] if success else None,
                'server_ttft_s': _time_sum(spec.scheduler_overhead_s, local_times[0]) if success else None,
                'client_ttft_s': _time_sum(wait, spec.scheduler_overhead_s, local_times[0]) if success else None,
                'tpot_s': tpot,
                'e2e_s': _time_sum(wait, spec.scheduler_overhead_s, local_times[-1]) if success else None,
                'failure_elapsed_s': _time_sum(wait, spec.scheduler_overhead_s, execution['duration_s']) if not success else None}
            row['slo_passed'], row['slo_checks'] = _slo_checks(row, spec.slo)
            requests.append(row)
    requests.sort(key=lambda row: row['input_order'])
    successful = [row for row in requests if row['status'] == 'completed']
    end = max(available)
    relative_window = {'start': 0.0, 'end': end, 'duration': end}
    window = {'start': clock_origin, 'end': absolute(end), 'duration': end}
    rate = lambda count: count / window['duration'] if window['duration'] > 0 else None
    specified = spec.slo is not None and (spec.slo.ttft_s is not None or spec.slo.tpot_s is not None)
    passed = [row for row in successful if row['slo_passed'] is True]
    distributions = {}
    for field in ('admission_wait_s', 'model_ttft_s', 'server_ttft_s', 'client_ttft_s', 'tpot_s', 'e2e_s'):
        cohort = requests if field == 'admission_wait_s' else successful
        distributions[field] = _distribution([row[field] for row in cohort if row[field] is not None],
            'all_admitted_requests' if field == 'admission_wait_s' else 'successful_requests_with_defined_metric')
    output_tokens = sum(row['output_tokens'] for row in successful)
    input_tokens = sum(row['input_tokens'] for row in successful)
    result = {
        'schema_version': '1.0', 'mode': 'serving',
        'status': 'completed' if len(successful) == len(requests) else 'completed_with_failures',
        'config': base, 'serving': spec.model_dump(),
        'hardware': {'instance_count': spec.instance_count, 'npus_per_instance': base_spec.hardware.npu_count,
                     'total_npus': spec.instance_count * base_spec.hardware.npu_count,
                     'isolation': 'Independent complete copies of base hardware, including host and network resources.'},
        'requests': requests, 'batches': batches, 'executions': executions,
        'clock_origin_s': clock_origin, 'time_window_s': window,
        'time_window_relative_s': relative_window,
        'metrics': {'request_count': len(requests), 'completed_requests': len(successful),
                    'failed_requests': len(requests) - len(successful), 'input_tokens': input_tokens,
                    'output_tokens': output_tokens, 'distributions': distributions},
        'throughput': {'completed_requests_per_s': rate(len(successful)),
                       'output_tokens_per_s': rate(output_tokens), 'input_tokens_per_s': rate(input_tokens)},
        'goodput': {'specified': specified, 'passed_requests': len(passed) if specified else None,
                    'joint_requests_per_s': rate(len(passed)) if specified else None,
                    'passed_output_tokens_per_s': rate(sum(row['output_tokens'] for row in passed)) if specified else None,
                    'tpot_not_applicable_requests': sum(row['requested_output_tokens'] == 1 for row in successful),
                    'scope': 'Successful requests meeting every applicable client TTFT / mean TPOT threshold.'},
        'provenance': {'scheduler': 'fcfs_nonpreemptive_fixed_batch', 'batch_executor': 'supernode_sim.analysis.run',
            'execution_cache': {'scope': 'Within this invocation only; exact ordered input/output length arrays.',
                                'hits': cache_hits, 'misses': len(executions), 'entries': len(cache),
                                'accounted_bytes': cache.current_bytes, 'max_bytes': cache.max_bytes,
                                'max_entries': cache.max_entries},
            'quantile_method': 'Linear interpolation of sorted per-request samples at (n - 1) * p; no population tail-confidence claim.',
            'clock_policy': 'Scheduling and metrics use seconds relative to clock_origin_s. Absolute timestamp fields are compatibility displays and may round away small intervals; use *_relative_s for timing analysis and trace export. time_window_s.duration is the relative elapsed time, not a subtraction of rounded absolute endpoints.',
            'throughput_window': 'Earliest arrival through final batch completion, including queue, deliberate batch wait, overhead and failed attempts.',
            'server_ttft_boundary': 'Admission through first token; model_ttft starts after scheduler overhead.',
            'failed_batch_policy': 'No successful tokens from incomplete execution; no retry or automatic batch shrinking.'},
        'limitations': list(LIMITATIONS),
    }
    return result

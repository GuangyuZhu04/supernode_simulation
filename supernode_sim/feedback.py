"""Exhaustive, bounded inference feedback with a fixed model and hardware."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections import Counter
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from .analysis import evaluate_targets, run
from .config import Config, Spec
from .parallel import ordered_map
from .streaming import run_streaming


PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class FeedbackSpec(Spec):
    """Finite search axes; omitted workload axes use the baseline as a bound."""

    ttft_s: float | None = Field(default=None, gt=0)
    tpot_s: float | None = Field(default=None, gt=0)
    prefill_lengths: list[PositiveInt] | None = Field(default=None, min_length=1)
    decode_batch_sizes: list[PositiveInt] | None = Field(default=None, min_length=1)
    tp_sizes: list[PositiveInt] | None = Field(default=None, min_length=1)
    ep_sizes: list[PositiveInt] | None = Field(default=None, min_length=1)
    sp_enabled: list[StrictBool] | None = Field(default=None, min_length=1)
    output_length: Annotated[int, Field(strict=True, ge=2)] | None = None
    max_candidates: PositiveInt | None = None
    engine: Literal['auto', 'simulate', 'streaming'] = 'auto'

    @model_validator(mode='before')
    @classmethod
    def tpop_alias(cls, value):
        if isinstance(value, dict) and 'tpop_s' in value:
            value = dict(value)
            alias = value.pop('tpop_s')
            if 'tpot_s' in value and value['tpot_s'] != alias:
                raise ValueError('tpop_s and tpot_s must agree when both are supplied')
            value['tpot_s'] = alias
        return value

    @model_validator(mode='after')
    def canonical_axes(self):
        for name in ('prefill_lengths', 'decode_batch_sizes', 'tp_sizes', 'ep_sizes', 'sp_enabled'):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, sorted(set(values)))
        return self


def _bounded_grid(limit: int) -> list[int]:
    """Powers of two plus the exact endpoint; never imply all integer lengths."""
    return sorted({limit, *(1 << exponent for exponent in range(limit.bit_length()))})


def _divisors(value: int) -> list[int]:
    small = [n for n in range(1, math.isqrt(value) + 1) if value % n == 0]
    return sorted({*small, *(value // n for n in small)})


def _layouts(config: Config, search: FeedbackSpec) -> list[dict]:
    copies = 2 if config.execution.features.prefill_decode_disaggregation.enabled else 1
    slots = config.hardware.npu_count // (copies * config.execution.dp * config.execution.pp)
    if search.tp_sizes is None and search.ep_sizes is None:
        pairs = [(slots // ep, ep) for ep in _divisors(math.gcd(slots, config.model.num_experts))]
    elif search.ep_sizes is None:
        pairs = [(tp, slots // tp if slots % tp == 0 else 1) for tp in search.tp_sizes]
    elif search.tp_sizes is None:
        pairs = [(slots // ep if slots % ep == 0 else 1, ep) for ep in search.ep_sizes]
    else:
        pairs = itertools.product(search.tp_sizes, search.ep_sizes)
    layouts = []
    # Explicit invalid axes remain visible even when the other axis is derived.
    for tp, ep in sorted(pairs):
        sp_values = search.sp_enabled
        if sp_values is None:
            sp_values = [False, True] if (tp > 1 and config.model.architecture is not None
                                          and config.execution.pp == 1) else [False]
        for enabled in sp_values:
            layouts.append({'tp': tp, 'ep': ep, 'sequence_parallel': enabled,
                            'sp': tp if enabled else 1})
    return layouts


def _candidate_config(base: dict, parameters: dict, output_length: int, targets: dict) -> Config:
    data = copy.deepcopy(base)
    data['execution'].update({key: parameters[key] for key in ('tp', 'ep', 'sequence_parallel')})
    data['workload'].update(input_length=parameters['prefill_length'],
                            batch_size=parameters['decode_batch_size'], output_length=output_length,
                            input_lengths=[], output_lengths=[])
    data['targets'].update(targets)
    return Config.model_validate(data)


def _decode_metrics(metrics: dict) -> dict:
    requests = metrics.get('requests', [])
    completions = [row['token_completion_s'] for row in requests]
    if not completions or any(len(times) < 2 for times in completions):
        raise ValueError('Decode throughput requires at least two completion times for every request')
    window = max(times[-1] for times in completions) - min(times[0] for times in completions)
    if not math.isfinite(window) or window <= 0:
        raise ValueError('Decode completion window must be finite and positive')
    tokens = sum(len(times) - 1 for times in completions)
    return {key: value for key, value in metrics.items() if key != 'requests'} | {
        'decode_tokens': tokens, 'decode_window_s': window,
        'decode_throughput_tokens_s': tokens / window,
    }


def _rank(row: dict) -> tuple:
    p, m = row['parameters'], row['metrics']
    # Stable, documented tie-breaking, independent of input axis ordering.
    return (-m['decode_throughput_tokens_s'], -p['prefill_length'], -p['decode_batch_size'],
            m['ttft_s'], m['tpot_s'], p['tp'], p['ep'], p['sequence_parallel'])


def _frontier(rows: list[dict]) -> list[int]:
    """Non-dominated triples: length, batch, measured decode throughput."""
    def point(row):
        return (row['parameters']['prefill_length'], row['parameters']['decode_batch_size'],
                row['metrics']['decode_throughput_tokens_s'])

    points = [(row['id'], point(row)) for row in rows]
    return [identifier for identifier, values in points if not any(
        all(a >= b for a, b in zip(other, values)) and any(a > b for a, b in zip(other, values))
        for other_id, other in points if other_id != identifier)]


def _feedback_candidate(arguments) -> dict:
    base, parameters, output_length, targets, engine, identifier = arguments
    row = {'id': identifier, 'parameters': parameters, 'engine': engine, 'metrics': {},
           'slo': {'specified': True, 'satisfied': False, 'checks': []},
           'memory_peaks_bytes': {}, 'diagnostics': []}
    try:
        candidate = _candidate_config(base, parameters, output_length, targets)
    except ValueError as exc:
        row.update(status='invalid_config', diagnostics=[str(exc)])
        return row
    try:
        simulated = (run_streaming(candidate, phase='all') if engine == 'streaming'
                     else run(candidate, 'simulate', full_detail=False))
        simulation = simulated.get('simulation', {})
        physical_status = simulation.get('status', simulated['status'])
        row['memory_peaks_bytes'] = simulation.get('memory_peaks_bytes', {})
        row['diagnostics'] = simulation.get('diagnostics', [])
        if physical_status in ('feasible', 'slo_failed'):
            row['metrics'] = _decode_metrics(simulated['metrics'])
            row['slo'] = evaluate_targets(candidate, row['metrics'], True)
            row['status'] = 'feasible' if row['slo']['satisfied'] else 'slo_failed'
        else:
            row['status'] = 'infeasible' if physical_status == 'infeasible' else 'error'
    except (ValueError, KeyError) as exc:
        row.update(status='error', diagnostics=[str(exc)])
    return row


def feedback(config: Config, search: FeedbackSpec | dict | None = None, *, workers: int = 1) -> dict:
    """Search each declared joint workload, retaining every failure and SLO check.

    Prefill and decode use the same fixed batch and parallel layout. Only the
    searched execution/workload axes and the two latency targets are changed.
    """
    search = search if isinstance(search, FeedbackSpec) else FeedbackSpec.model_validate({} if search is None else search)
    if config.workload.scenario != 'inference':
        raise ValueError('feedback requires an inference configuration')
    targets = {name: getattr(search, name) if getattr(search, name) is not None
               else getattr(config.targets, name) for name in ('ttft_s', 'tpot_s')}
    if any(value is None for value in targets.values()):
        raise ValueError('feedback requires both ttft_s and tpot_s (TPOP), in search options or config.targets')
    output_length = search.output_length
    if output_length is None:
        lengths = set(config.workload.output_lengths)
        if len(lengths) != 1:
            raise ValueError('Ragged output_lengths require explicit feedback output_length')
        output_length = next(iter(lengths))
    if output_length < 2:
        raise ValueError('feedback output_length must be at least 2 to measure TPOT')
    features = config.execution.features
    requires_graph = (features.speculative_decode.enabled or features.megakernel.enabled
                      or features.prefill_decode_disaggregation.enabled)
    supports_streaming = (config.model.architecture is not None
                          and config.execution.dp == config.execution.pp == 1 and not requires_graph)
    if search.engine == 'streaming' and not supports_streaming:
        raise ValueError('streaming feedback requires structured inference and DP=PP=1; '
                         'speculative decode, megakernel and PD require ordinary simulate')
    engine = ('streaming' if supports_streaming else 'simulate') if search.engine == 'auto' else search.engine
    prefill_lengths = search.prefill_lengths or _bounded_grid(max(config.workload.input_lengths))
    batch_sizes = search.decode_batch_sizes or _bounded_grid(config.workload.batch_size)
    layouts = _layouts(config, search)
    count = len(layouts) * len(prefill_lengths) * len(batch_sizes)
    limit = search.max_candidates or config.targets.max_candidates
    if count > limit:
        raise ValueError(f'{count} feedback combinations exceed max_candidates={limit}; narrow the axes or raise the explicit limit')
    base = config.model_dump()
    base['calibration'] = None  # load_config has already resolved hardware overrides.
    defaults = []
    if search.prefill_lengths is None:
        defaults.append('prefill_lengths: powers of two plus max(original input_lengths), bounded by that maximum')
    if search.decode_batch_sizes is None:
        defaults.append('decode_batch_sizes: powers of two plus original batch_size, bounded by that batch size')
    arguments = ((base, {**layout, 'prefill_length': length, 'decode_batch_size': batch},
                  output_length, targets, engine, identifier)
                 for identifier, (layout, length, batch) in enumerate(
                     itertools.product(layouts, prefill_lengths, batch_sizes)))
    rows = list(ordered_map(_feedback_candidate, arguments, workers=workers, count=count))
    feasible = sorted((row for row in rows if row['status'] == 'feasible'), key=_rank)
    best = None
    if feasible:
        best = copy.deepcopy(feasible[0])
        best['config'] = _candidate_config(base, best['parameters'], output_length, targets).model_dump()
    boundaries = []
    for layout in layouts:
        matches = [row for row in feasible if all(row['parameters'][key] == value for key, value in layout.items())]
        if matches:
            boundaries.append({**layout,
                'max_prefill_length': max(row['parameters']['prefill_length'] for row in matches),
                'max_decode_batch_size': max(row['parameters']['decode_batch_size'] for row in matches),
                'candidate_ids': sorted(row['id'] for row in matches)})
    counts = dict.fromkeys(('feasible', 'invalid_config', 'slo_failed', 'infeasible', 'error'), 0)
    counts.update(Counter(row['status'] for row in rows))
    counts['total'] = len(rows)
    return {
        'schema_version': '1.0', 'engine_version': '0.1.0', 'mode': 'feedback',
        'status': 'error' if counts['error'] else 'feasible' if best else 'no_feasible_candidate',
        'search_complete': not counts['error'],
        'objective': 'decode_throughput_tokens_s', 'targets': targets, 'config': base,
        'config_sha256': hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
        'search_space': {'layouts': layouts, 'prefill_lengths': prefill_lengths,
                         'decode_batch_sizes': batch_sizes, 'output_length': output_length,
                         'max_candidates': limit, 'defaults': defaults},
        'counts': counts, 'best': best, 'candidates': rows,
        'pareto_frontier': _frontier(feasible), 'feasible_boundaries': boundaries,
        'assumptions': [
            'Optimality is limited to the complete declared finite candidate grid and fixed simulator scheduler; no global optimum is claimed.',
            'Model, hardware/topology, PP, DP, FSDP, rank_order, precision and features are held fixed.',
            'Prefill length is uniform per request; prefill and decode share the same fixed batch per DP replica and the same parallel layout. No chunked prefill or online batching is modeled.',
            ('PD uses two equal TP/EP pools, counts both pools in the physical NPU budget, and retains/revalidates the explicit pool mappings. Decode throughput includes the handoff window.'
             if features.prefill_decode_disaggregation.enabled else 'Prefill and decode use the same physical pool.'),
            'SP shares TP ranks: degree is TP when enabled and 1 otherwise; SP does not multiply the device count.',
            'TPOP is accepted as an input alias of TPOT. TTFT is the maximum request TTFT; TPOT is the maximum per-request mean token interval.',
            'Decode throughput = total output tokens after each request first token / (latest last-token completion - earliest first-token completion), aggregated over fixed DP replicas.',
            'For PP>1 or DP>1, this decode window may overlap other requests remaining prefill; it is not an isolated decode benchmark.',
            'Both latency targets and any existing throughput target must pass; full prefill/decode memory admission must also succeed.',
            'Throughput ties prefer longer prefill, then larger batch, then lower TTFT/TPOT, then TP/EP/SP in ascending order.',
            'Per-layout maximum length and maximum batch can belong to different candidates; only an actual feasible candidate is a jointly validated recommendation.',
            'The frontier maximizes prefill length, decode batch and measured decode throughput simultaneously.',
            'Two output tokens measure one decode interval, not steady-state throughput; output length is fixed throughout a search.',
            'Execution latency excludes request queueing and cold weight loading; results are model predictions, not production P99 guarantees.',
        ],
    }

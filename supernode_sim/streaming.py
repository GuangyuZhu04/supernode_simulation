"""Bounded-memory execution of the existing structured inference graph.

The structured compiler has a full barrier between layers and output tokens.
For DP=PP=1 those barriers allow exact sequential simulation without retaining
the complete graph. This module changes graph retention, not operator kernels,
collective algorithms, placement, resource sharing, or memory admission.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
import copy
import hashlib
import json
import re
from typing import Callable

from .analysis import execution_metrics, model_fidelity
from .calibration import apply_kernel_profiles, calibration_summary, merge_calibration_summaries
from .cache import BoundedCache
from .config import Config
from .features import build_feature_plan
from .core import Allocation, Event, Graph
from .engine import simulate
from .latency_breakdown import latency_breakdown_metadata, window_breakdown
from .structured import Builder
from .topology import Topology


# These allocations are never released by structured fixed-batch inference.
# Combining them by pool/category/dtype conserves all admission constraints.
_PERMANENT = {
    'weight', 'quantization_metadata', 'extra_weight_copy', 'runtime', 'workspace',
    'kv_cache', 'kv_scales', 'page_table', 'recurrent_state', 'convolution_state',
    'lookup_table', 'lookup_cache',
    'window_elements', 'compressed_pool_elements', 'indexer_pool_elements',
    'compression_state_elements', 'indexer_compression_state_elements',
}
_STEP = re.compile(r'^infer\.d\d+\.(?:s(\d+)\.|token(\d+)$)')
_PREFIX = re.compile(r'^infer\.d\d+\.s\d+\.l\d+')


def _sum_latency(records, divisor=1):
    """Combine consecutive, non-overlapping critical-path windows."""
    components = defaultdict(float)
    total = 0.0
    for record in records:
        total += record['total_s']
        for name, value in record['components_s'].items():
            components[name] += value
    return {'total_s': total / divisor,
            'components_s': {name: value / divisor for name, value in components.items()}}


class _Stop(Exception):
    pass


class _EventSink(list):
    """A module-level type avoids a local class/closure reference cycle."""

    def __init__(self, accept):
        super().__init__()
        self.accept = accept

    def append(self, event):
        self.accept(event)


def _communication():
    return {'transport_payload_bytes': 0.0, 'physical_bytes_by_domain': defaultdict(float),
            'payload_by_semantic_bytes': defaultdict(float), 'payload_by_collective_bytes': defaultdict(float)}


def _add_traffic(target, event):
    meta = event.metadata
    payload = meta.get('payload_bytes', 0.0)
    target['transport_payload_bytes'] += payload
    if payload:
        semantic = str(meta.get('semantic', meta.get('operation', meta.get('category', meta.get('kind', 'transfer')))))
        target['payload_by_semantic_bytes'][semantic] += payload
        target['payload_by_collective_bytes'][str(meta.get('collective', 'point_to_point'))] += payload
    for domain, value in meta.get('physical_bytes_by_domain', {}).items():
        target['physical_bytes_by_domain'][domain] += value


def _merge_traffic(target, source):
    target['transport_payload_bytes'] += source['transport_payload_bytes']
    for key in ('physical_bytes_by_domain', 'payload_by_semantic_bytes', 'payload_by_collective_bytes'):
        for name, value in source[key].items():
            target[key][name] += value


def run_streaming(config: Config | dict, *, phase: str = 'all',
                  progress: Callable[[dict], None] | None = None,
                  cache: MutableMapping | None = None) -> dict:
    """Simulate structured DP1/PP1 inference using one layer of graph memory.

    ``phase='decode'`` seeds the same persistent KV/KDA state that a completed
    prefill would leave, but skips prefill execution and its transient memory.
    It therefore remains useful when unchunked prefill is infeasible. It is a
    standalone warm-cache decode experiment, and its TTFT is deliberately null.
    ``output_length=2`` gives one decode interval; longer outputs give their mean.

    The return value is a compact numerical artifact, not a write_report input.
    Per-event traces and component tensor lifetimes are intentionally omitted.
    The default timing cache is limited to 64 MiB of accounted Python objects
    and 256 entries. Pass a BoundedCache to share a budget across experiments;
    a caller-provided plain dict retains its existing unlimited-cache behavior.
    """
    if phase not in ('all', 'prefill', 'decode'):
        raise ValueError("phase must be 'all', 'prefill', or 'decode'")
    spec = config if isinstance(config, Config) else Config.model_validate(config)
    if spec.execution.features.speculative_decode.enabled:
        raise ValueError('Speculative decoding currently requires ordinary simulate; streaming round state is not implemented')
    if spec.workload.scenario != 'inference' or spec.model.architecture is None:
        raise ValueError('Streaming requires structured inference architecture')
    if spec.execution.dp != 1 or spec.execution.pp != 1:
        raise ValueError('Streaming currently requires DP=PP=1')
    if phase == 'decode' and max(spec.workload.output_lengths) < 2:
        raise ValueError('Standalone decode requires output_length >= 2')
    data = spec.model_dump()
    builder = Builder(data, Topology(data['hardware'], data['execution']['precision']))
    if phase == 'decode':
        original_collective, original_kernel = builder.collective, builder.kernel

        def skip_prefill_collective(prefix, kind, ranks, size, deps, *args, **kwargs):
            if '.s0.' in prefix:
                return list(deps)
            return original_collective(prefix, kind, ranks, size, deps, *args, **kwargs)

        def skip_prefill_kernel(name, *args, **kwargs):
            if '.s0.' in name:
                return name
            return original_kernel(name, *args, **kwargs)

        # Persistent attention state is allocated separately in the compiler's
        # per-layer state barriers. Avoid building/routing execution events for
        # the intentionally skipped prefill; the sink still records that state.
        builder.collective, builder.kernel = skip_prefill_collective, skip_prefill_kernel
    graph = builder.g
    permanent = defaultdict(float)
    named = {}
    initialized = False
    completed_barriers = set()
    marker_trace = []
    token_latency = {}
    chunks = []
    timing_cache = BoundedCache() if cache is None else cache
    initial_cache_enabled = not data['execution']['features']['engram']['enabled']
    graph.metadata['initial_resident_cache'] = {'enabled': initial_cache_enabled, 'hit': False}
    if initial_cache_enabled:
        initial_spec = {
            'model': data['model'], 'execution': data['execution'],
            'npu_count': data['hardware']['npu_count'],
            'npus_per_host': data['hardware']['npus_per_host'],
            'runtime_bytes': data['hardware']['npu']['runtime_bytes'],
            'workspace_bytes': data['hardware']['npu']['workspace_bytes'],
        }
        initial_key = ('structured_initial_resident_v1', hashlib.sha256(
            json.dumps(initial_spec, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest())
        original_initial = builder.initial

        def cached_initial():
            resident = timing_cache.get(initial_key)
            if resident is None:
                original_initial()
                folded = defaultdict(float)
                for allocation in graph.initial_allocations:
                    folded[allocation.pool, allocation.category, allocation.dtype] += allocation.size_bytes
                resident = {
                    'allocations': tuple((pool, size, category, dtype)
                        for (pool, category, dtype), size in folded.items()),
                    'metadata': {key: copy.deepcopy(graph.metadata[key]) for key in
                        ('logical_component_state_bytes', 'logical_memory_bytes')},
                }
                timing_cache[initial_key] = resident
            else:
                graph.initial_allocations.extend(Allocation(f'stream.initial.{i}', pool, size, category, dtype)
                    for i, (pool, size, category, dtype) in enumerate(resident['allocations']))
                graph.metadata.update(copy.deepcopy(resident['metadata']))
                graph.metadata['initial_resident_cache']['hit'] = True

        builder.initial = cached_initial
    component_macros = {}
    kernel_calibration = calibration_summary()
    phases = {}
    peaks = defaultdict(float)
    categories_at_peak = {}
    all_work = defaultdict(float)
    all_busy = defaultdict(float)
    diagnostics = []
    duration = 0.0
    total_events = completed_events = generated_events = 0
    status = 'feasible'
    final_memory = {}

    def remember(allocation):
        if allocation.category in _PERMANENT:
            permanent[allocation.pool, allocation.category, allocation.dtype] += allocation.size_bytes
        else:
            named[allocation.id] = allocation

    def seed_initial():
        nonlocal initialized
        if not initialized:
            for allocation in graph.initial_allocations:
                remember(allocation)
            graph.initial_allocations.clear()
            initialized = True

    def initial_allocations():
        return [Allocation(f'stream.resident.{i}', pool, size, category, dtype)
                for i, ((pool, category, dtype), size) in enumerate(permanent.items())] + list(named.values())

    def accept_event(event):
        nonlocal generated_events
        generated_events += 1
        seed_initial()
        match = _STEP.match(event.id)
        if not match:
            raise ValueError(f'Unexpected structured inference event: {event.id}')
        step = int(match.group(1) or match.group(2))
        if phase == 'decode' and step == 0:
            # Only persistent state survives a complete prefill; residuals
            # are released by token0 and temporary buffers by their users.
            for allocation in event.allocs:
                if allocation.category in _PERMANENT:
                    remember(allocation)
            if event.metadata.get('phase') == 'token_complete':
                completed_barriers.add(event.id)
            return
        list.append(sink, event)
        if (event.id.endswith('.layer_done') or event.metadata.get('phase') == 'token_complete'
                or event.metadata.get('_stream_boundary')):
            flush(step, event)

    sink = _EventSink(accept_event)
    graph.events = sink

    def flush(step, terminal):
        nonlocal duration, status, total_events, completed_events, final_memory
        label = 'prefill' if step == 0 else 'decode'
        phase_result = phases.setdefault(label, {
            'status': 'feasible', 'duration_s': 0.0, 'event_count': 0,
            'communication': _communication(), 'resource_work': defaultdict(float),
            'memory_peaks_bytes': defaultdict(float), 'steps': [],
        })
        local_ids = {event.id for event in sink}
        for event in sink:
            external = set(event.deps).difference(local_ids)
            if external.difference(completed_barriers):
                raise ValueError(f'Chunk has a dependency outside its completed barrier: {event.id}')
            event.deps = tuple(dep for dep in event.deps if dep in local_ids)
        chunk = Graph(resources=graph.resources, memory_caps=graph.memory_caps,
                      events=list(sink), initial_allocations=initial_allocations(),
                      metadata={'topology': graph.metadata.get('topology', {})})
        replay = terminal.metadata.get('_stream_replay')
        if replay:
            chunk_calibration = replay['kernel_calibration']
        else:
            apply_kernel_profiles(chunk, data)
            chunk_calibration = chunk.metadata['kernel_calibration']
        merge_calibration_summaries(kernel_calibration, chunk_calibration)
        used_resources = {key for event in chunk.events for key in event.work}
        chunk.resources = {key: chunk.resources[key] for key in used_resources}
        initial_bytes = defaultdict(float)
        initial_categories = defaultdict(lambda: defaultdict(float))
        for allocation in chunk.initial_allocations:
            initial_bytes[allocation.pool] += allocation.size_bytes
            initial_categories[allocation.pool][allocation.category] += allocation.size_bytes
        # Canonicalize only the shared layer prefix. Resource IDs, amounts,
        # precise topology, dependency structure, slot ordering and dtypes stay
        # in the key. Token releases can refer to earlier layers and are cheap,
        # so they deliberately bypass caching.
        cache_key = None
        if not replay and (terminal.id.endswith('.layer_done') or terminal.metadata.get('_stream_component')):
            digest = hashlib.sha256()
            # Attribution changes are part of cache identity even when event
            # timings happen to match. The version prevents old compact cache
            # entries without a latency ledger from being replayed.
            digest.update(b'stream_timing_latency_v1\n')
            digest.update(json.dumps(chunk.metadata, sort_keys=True).encode('utf-8'))
            for event in chunk.events:
                normalized = lambda value: _PREFIX.sub('infer.layer', value)
                attribution = {key: event.metadata[key] for key in (
                    'type', 'kernel', 'phase', 'operation', 'category', 'kind',
                    'resource_domains', 'startup_s_by_domain', 'domains', 'src', 'dst', 'path')
                    if key in event.metadata}
                signature = (normalized(event.id), tuple(normalized(d) for d in event.deps),
                    tuple((k, v, graph.resources[k].capacity, graph.resources[k].kind,
                           graph.resources[k].unit) for k, v in sorted(event.work.items())),
                    event.startup_s, event.slots,
                    tuple((normalized(a.id), a.pool, a.size_bytes, a.category, a.dtype) for a in event.allocs),
                    tuple(normalized(aid) for aid in event.releases),
                    json.dumps(attribution, sort_keys=True))
                digest.update(repr(signature).encode('utf-8'))
                digest.update(b'\n')
            cache_key = digest.hexdigest()
        cached = replay['template'] if replay else (timing_cache.get(cache_key) if cache_key else None)
        cache_hit = cached is not None
        if cached is None and cache_key:
            # Construct the unconstrained schedule once. Reuse is permitted
            # only when that schedule fits the real current resident memory;
            # otherwise the normal admission scheduler determines feasibility.
            unconstrained = Graph(resources=chunk.resources,
                memory_caps={pool: 1e300 for pool in chunk.memory_caps},
                events=chunk.events, initial_allocations=chunk.initial_allocations,
                metadata=chunk.metadata)
            template = simulate(unconstrained, retain_event_payload=False)
            if template['status'] == 'feasible':
                cached = {key: template[key] for key in (
                    'status', 'duration_s', 'diagnostics', 'completed_events', 'total_events')}
                cached['resource_work'] = {key: value for key, value in template['resource_work'].items() if value}
                cached['resource_busy_s'] = {key: value for key, value in template['resource_busy_s'].items() if value}
                cached['_latency_breakdown'] = window_breakdown(chunk, template, terminal.id)
                cached['_peak_delta'] = {pool: value - initial_bytes[pool] for pool, value in template['memory_peaks_bytes'].items()}
                cached['_final_delta'] = {pool: value - initial_bytes[pool] for pool, value in template['memory_final_bytes'].items()}
                cached['_category_delta'] = {pool: {category: value - initial_categories[pool].get(category, 0)
                    for category, value in values.items()} for pool, values in template['memory_categories_at_peak_bytes'].items()}
                timing_cache[cache_key] = cached
        fits = cached is not None and all(
            initial_bytes[pool] + cached['_peak_delta'].get(pool, 0) <= capacity * (1 + 1e-12)
            for pool, capacity in chunk.memory_caps.items())
        if fits:
            result = {key: value for key, value in cached.items() if not key.startswith('_')}
            result['_latency_breakdown'] = cached['_latency_breakdown']
            result['memory_peaks_bytes'] = {pool: initial_bytes[pool] + cached['_peak_delta'].get(pool, 0) for pool in chunk.memory_caps}
            result['memory_final_bytes'] = {pool: initial_bytes[pool] + cached['_final_delta'].get(pool, 0) for pool in chunk.memory_caps}
            result['memory_categories_at_peak_bytes'] = {pool: {
                category: initial_categories[pool].get(category, 0) + cached['_category_delta'].get(pool, {}).get(category, 0)
                for category in set(initial_categories[pool]) | set(cached['_category_delta'].get(pool, {}))}
                for pool in chunk.memory_caps}
        else:
            cache_hit = False
            result = simulate(chunk, retain_event_payload=False)
            if result['status'] == 'feasible':
                result['_latency_breakdown'] = window_breakdown(chunk, result, terminal.id)
        total_events += result['total_events']
        completed_events += result['completed_events']
        before = duration
        duration += result['duration_s']
        for pool, size in result['memory_peaks_bytes'].items():
            phase_result['memory_peaks_bytes'][pool] = max(phase_result['memory_peaks_bytes'][pool], size)
            if size >= peaks[pool]:
                peaks[pool] = size
                categories_at_peak[pool] = result['memory_categories_at_peak_bytes'].get(pool, {})
        for resource, amount in result['resource_work'].items():
            all_work[resource] += amount
            phase_result['resource_work'][resource] += amount
        for resource, seconds in result['resource_busy_s'].items():
            all_busy[resource] += seconds
        phase_result['duration_s'] += result['duration_s']
        phase_result['event_count'] += result['total_events']
        summary = {'id': terminal.id, 'phase': label, 'token_index': step,
                   'start_s': before, 'end_s': duration, 'duration_s': result['duration_s'],
                   'status': result['status'], 'event_count': result['total_events'],
                   'timing_cache_hit': cache_hit,
                   'component_macro_hit': bool(replay),
                   'component': terminal.metadata.get('_stream_component'),
                   'peak_hbm_bytes': max((v for k, v in result['memory_peaks_bytes'].items() if k.endswith('.hbm')), default=0)}
        chunks.append(summary)
        if progress:
            progress(summary)
        if result['status'] != 'feasible':
            status = phase_result['status'] = result['status']
            diagnostics.extend(result['diagnostics'])
            sink.clear()
            raise _Stop
        record = result['_latency_breakdown']
        token_latency[step] = _sum_latency([token_latency[step], record]) if step in token_latency else record
        traffic = replay['communication'] if replay else _communication()
        if not replay:
            for event in sink:
                _add_traffic(traffic, event)
        _merge_traffic(phase_result['communication'], traffic)
        # Every event completed and allocation IDs are unique; survivors can be
        # found structurally without retaining a potentially huge event trace.
        released = {aid for event in sink for aid in event.releases}
        survivors = [allocation for event in sink for allocation in event.allocs if allocation.id not in released]
        macro_key = terminal.metadata.get('_stream_macro_key')
        if macro_key is not None and fits and not survivors and not (released & named.keys()):
            component_macros[macro_key] = {'template': cached, 'communication': traffic,
                                           'kernel_calibration': chunk_calibration}
        for aid in released:
            named.pop(aid, None)
        for event in sink:
            for allocation in event.allocs:
                if allocation.id not in released:
                    named[allocation.id] = allocation
        for aid, allocation in list(named.items()):
            if allocation.category in _PERMANENT:
                del named[aid]
                remember(allocation)
        final_memory = result['memory_final_bytes']
        completed_barriers.add(terminal.id)
        if terminal.metadata.get('phase') == 'token_complete':
            marker_trace.append({'id': terminal.id, 'end_s': duration})
            phase_result['steps'].append({'token_index': step, 'completion_s': duration})
            # Macro keys include the token index; these templates can never be
            # reused by a later token. Release their references after each step.
            component_macros.clear()
            completed_barriers.intersection_update((terminal.id,))
        sink.clear()
        if phase == 'prefill' and step == 0 and terminal.metadata.get('phase') == 'token_complete':
            raise _Stop

    def state_hook(prefix, layer, ready, step):
        return [builder.barrier(prefix + '.state_done', ready, _stream_boundary=True)]

    def macro_key(layer, component, step):
        tensors = builder.local_tensors(layer, component)
        if component == 'Output' and not tensors:
            tensors = [t for t in builder.by_layer[layer] if t.component == 'Output']
        shapes = tuple((t.shape, t.role, t.tp_shard, t.ep_shard, t.active_fraction,
                        bool(t.alias_of), t.tp_max_shards, builder.weight_dtype(t)) for t in tensors)
        block = builder.a.get('attnres_block_size')
        engram = builder.f['engram']
        engram_lookup = component == 'PreNorm' and engram['enabled'] and layer in engram['layers']
        # Layer-selected Engram lookups change PreNorm work despite identical
        # parameter shapes. Measured limiters additionally have event-specific
        # IDs, so preserve those IDs by avoiding cross-layer replay for profiles.
        return (step, component, shapes, layer // block + 1 if component == 'AttnRes' and block else None,
                engram_lookup, layer if data['hardware'].get('kernel_profiles') else None)

    def before_component(prefix, layer, component, ready, step):
        if phase == 'decode' and step == 0:
            return None
        key = macro_key(layer, component, step)
        macro = component_macros.get(key)
        if macro is None:
            return None
        resident = defaultdict(float)
        for (pool, category, dtype), size in permanent.items():
            resident[pool] += size
        for allocation in named.values():
            resident[allocation.pool] += allocation.size_bytes
        if any(resident[pool] + macro['template']['_peak_delta'].get(pool, 0) > capacity * (1 + 1e-12)
               for pool, capacity in graph.memory_caps.items()):
            return None
        terminal = Event(prefix + '.component_done.' + component, tuple(ready), metadata={
            '_stream_boundary': True, '_stream_component': component,
            '_stream_macro_key': key, '_stream_replay': macro})
        sink.append(terminal)
        return [terminal.id]

    def after_component(prefix, layer, component, ready, step):
        return [builder.barrier(prefix + '.component_done.' + component, ready,
            _stream_boundary=True, _stream_component=component,
            _stream_macro_key=macro_key(layer, component, step))]

    builder._inference_state_hook = state_hook
    builder._inference_component_before = before_component
    builder._inference_component_after = after_component

    try:
        try:
            builder.inference()
        except _Stop:
            pass
        if sink:
            raise ValueError('Structured compiler ended without flushing its final barrier')
    finally:
        # Hooks and Sink methods close over this run's builder/graph/cache.
        # The CLI intentionally disables cyclic GC while simulating large DAGs,
        # so break the cycles even on infeasibility or a progress callback error.
        for name in ('initial', 'collective', 'kernel', '_inference_state_hook',
                     '_inference_component_before', '_inference_component_after'):
            builder.__dict__.pop(name, None)
        graph.events = []
        sink.clear()
        sink.accept = None
        sink = None

    metrics = {'duration_s': duration, 'ttft_s': None, 'tpot_s': None}
    latency = latency_breakdown_metadata()
    token_times = [row['end_s'] for row in marker_trace]
    if status == 'feasible' and token_times:
        if phase == 'all':
            metrics = execution_metrics(spec, graph, {'status': status, 'duration_s': duration, 'trace': marker_trace})
            # DP1/PP1 fixed-batch requests share token barriers. Select the
            # same request as the latency metric, then average that request's
            # intervals; categorywise maxima would not describe any request.
            for request in metrics['requests']:
                request['ttft_breakdown'] = _sum_latency([token_latency[0]])
                interval_count = len(request['itl_s'])
                request['tpot_breakdown'] = ({**_sum_latency(
                    (token_latency[index] for index in range(1, interval_count + 1)), interval_count),
                    'interval_count': interval_count} if interval_count else None)
            selected = max(metrics['requests'], key=lambda row: row['ttft_s'])
            latency['ttft'] = {**selected['ttft_breakdown'],
                               'replica': selected['replica'], 'sequence': selected['sequence']}
            with_decode = [request for request in metrics['requests'] if request['tpot_s'] is not None]
            if with_decode:
                selected = max(with_decode, key=lambda row: row['tpot_s'])
                latency['tpot'] = {**selected['tpot_breakdown'],
                                   'replica': selected['replica'], 'sequence': selected['sequence']}
        elif phase == 'prefill':
            metrics.update(ttft_s=token_times[0])
            latency['ttft'] = {**_sum_latency([token_latency[0]]), 'replica': '0', 'sequence': '0'}
        else:
            intervals = [b - a for a, b in zip([0.0] + token_times, token_times)]
            sequence_means = [(index, sum(intervals[:length - 1]) / (length - 1))
                              for index, length in enumerate(spec.workload.output_lengths) if length > 1]
            selected_index, selected_mean = max(sequence_means, key=lambda item: item[1])
            metrics.update(tpot_s=selected_mean, decode_intervals_s=intervals)
            interval_count = spec.workload.output_lengths[selected_index] - 1
            latency['tpot'] = {**_sum_latency(
                (token_latency[index] for index in range(1, interval_count + 1)), interval_count),
                'interval_count': interval_count, 'replica': '0', 'sequence': str(selected_index)}
    else:
        latency['unavailable_reason'] = ('execution_incomplete' if status != 'feasible'
                                       else 'no_token_completions')
    metrics['tpop_s'] = metrics['tpot_s']
    if status == 'feasible':
        if phase == 'decode':
            latency['ttft_unavailable_reason'] = 'Standalone decode excludes prefill and TTFT.'
        elif latency['tpot'] is None:
            latency['tpot_unavailable_reason'] = ('Prefill-only execution excludes decode intervals.'
                                                 if phase == 'prefill' else
                                                 'At least two output tokens per request are required for TPOT.')
    metrics['latency_breakdown'] = latency
    for phase_result in phases.values():
        phase_result['peak_hbm_bytes'] = max((v for k, v in phase_result['memory_peaks_bytes'].items() if k.endswith('.hbm')), default=0)
        phase_result['communication']['classification_note'] = 'Semantic and collective payloads are two views; physical bytes count link hops.'
    graph.metadata['kernel_calibration'] = kernel_calibration
    graph.metadata['fidelity'] = model_fidelity(spec, kernel_calibration)
    graph.metadata['feature_plan'] = build_feature_plan(spec, kernel_calibration)
    return {
        'status': status, 'execution_mode': 'structured_layer_streaming', 'phase': phase,
        'metrics': metrics, 'phase_metrics': phases, 'chunks': chunks,
        'simulation': {'status': status, 'duration_s': duration, 'total_events': total_events,
            'completed_events': completed_events, 'generated_events': generated_events,
            'memory_peaks_bytes': dict(peaks), 'memory_categories_at_peak_bytes': categories_at_peak,
            'memory_final_bytes': final_memory, 'resource_work': dict(all_work),
            'resource_busy_s': dict(all_busy), 'diagnostics': diagnostics},
        'model_metadata': graph.metadata,
        'assumptions': [*graph.metadata.get('assumptions', []), *spec.assumptions,
            'Layer streaming preserves existing barriers, kernels, collectives, and memory admission for DP=PP=1.',
            'Permanent allocations are combined by pool/category/dtype; per-tensor lifetime detail is not retained.',
            *(['Decode is seeded with persistent KV/KDA state; prefill execution and TTFT are excluded.'] if phase == 'decode' else [])],
    }

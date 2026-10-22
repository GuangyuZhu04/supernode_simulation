"""Profile-driven draft/verify/commit rounds on the target device resources.

This is a scheduling and resource model, not DSpark's token acceptance algorithm.
Acceptance counts come from an explicit trace or a declared fixed assumption.
The target compiler still accounts for every verified query, including rejected
ones. Optional commit/replay costs remain visibly unmodeled when not supplied.
"""
from __future__ import annotations

from math import ceil

from .components import attention_state
from .core import DTYPE_BYTES, Event, Resource
from .structured import Builder


def speculative_plan(config: dict) -> list[dict]:
    """Keep the requested output count and truncate the final round at EOS.

    accepted_lengths counts committed output tokens, including the replacement
    or bonus token. A trace entry describes a synchronized batch round, not one
    request. The initial prefill already emits the first output token.
    """
    workload = config['workload']
    feature = config['execution']['features']['speculative_decode']
    prompt = workload['input_lengths'][0]
    output = workload['output_lengths'][0]
    plan = [{'phase': 'prefill', 'round_index': -1, 'query_tokens': prompt,
             'context_before': 0, 'committed_tokens': 1, 'draft_tokens': 0}]
    emitted, index = 1, 0
    trace = feature['accepted_lengths']
    while emitted < output:
        if trace:
            if index >= len(trace):
                raise ValueError('accepted_lengths trace ended before the requested output_length; provide every round')
            accepted = trace[index]
        else:
            accepted = feature['accepted_length']
        queries = min(feature['draft_tokens'] + 1, output - emitted)
        committed = min(accepted, queries)
        plan.append({'phase': 'verify', 'round_index': index, 'query_tokens': queries,
                     'context_before': prompt + emitted - 1, 'committed_tokens': committed,
                     'draft_tokens': queries - 1, 'requested_acceptance': accepted,
                     'rejected_queries': queries - committed,
                     'eos_truncated': queries < feature['draft_tokens'] + 1 or committed < accepted})
        emitted += committed
        index += 1
    return plan


def _profile_event(builder: Builder, name: str, rank: str, profile: dict, deps,
                   operation: str) -> str:
    """Use the same HBM and exclusive compute stream as the target kernels."""
    dtype = profile.get('compute_dtype') or builder.p['compute']
    peak = builder.h['npu']['flops'][dtype]
    flops, memory = profile['flops'], profile['memory_bytes']
    measured = profile.get('duration_s')
    work = {}
    if measured is not None:
        floor = max(flops / peak, memory / builder.g.resources[rank + '.hbm'].capacity)
        if measured < floor * (1 - 1e-9):
            raise ValueError(f'{operation} duration_s is below physical compute/memory limits; startup is separate')
        limiter = 'speculative_service:' + name
        builder.g.resources[limiter] = Resource(limiter, 1.0, 'second/second', 'kernel_timing')
        work[limiter] = measured
    if flops:
        # A measured service may exceed the coarse default compute efficiency.
        # The same compute_stream prevents overlap with any target NPU kernel.
        key = rank + '.compute.speculative.' + dtype + ('.profile' if measured is not None else '.analytic')
        capacity = peak if measured is not None else peak * builder.h['npu']['efficiency']
        builder.g.resources[key] = Resource(key, capacity, 'FLOP/s', 'compute')
        work[key] = flops
    if memory:
        work[rank + '.hbm'] = memory
    return builder.add(Event(name, tuple(deps), work, startup_s=profile.get('startup_s', 0),
        slots=(rank + '.compute_stream',), metadata={
            'component': 'SpeculativeDecode', 'phase': 'draft' if operation == 'draft_forward' else operation,
            'operation': operation, 'service_time_s': measured, 'profile_source': profile.get('source'),
            'cost_scope': 'one participating rank, its complete local batch', 'compute_dtype': dtype}))


def _copy_event(builder: Builder, name: str, rank: str, size: float, deps, operation: str) -> str:
    return builder.add(Event(name, tuple(deps), {rank + '.hbm': 2 * size},
        slots=(rank + '.compute_stream',), metadata={
            'component': 'SpeculativeDecode', 'phase': operation, 'operation': operation,
            'state_bytes': size, 'copy_bytes': 2 * size}))


def _round_storage(builder: Builder, rank: str, item: dict) -> tuple[float, float]:
    """Snapshot mutable state and reserve verifier candidates in HBM.

    Dense/MLA candidates reserve all query KV rows, even if committed capacity
    has already been admitted. This conservative double buffering is explicit.
    Sparse windows need a snapshot because verification can overwrite the ring;
    only uncommitted compressed pool growth needs an additional pool buffer.
    """
    if item['draft_tokens'] == 0:
        # A final single-token target step has no speculative candidates and
        # cannot reject a proposal, so it needs no rollback buffer.
        return 0.0, 0.0
    lane = builder.coords[rank][2]
    batch = len(range(lane, builder.w['batch_size'], builder.ep))
    before, queries, commit = item['context_before'], item['query_tokens'], item['committed_tokens']
    snapshot = builder.f['speculative_decode']['snapshot_bytes_per_rank']
    candidates = 0.0
    for layer in range(builder.m['layers']):
        state = attention_state(builder.m, layer, batch, before, tp=builder.tp)
        if state['kind'] == 'recurrent':
            snapshot += (state['recurrent_elements'] * DTYPE_BYTES[builder.p['recurrent']]
                         + state['convolution_elements'] * DTYPE_BYTES[builder.p['activation']])
        elif state['kind'] == 'compressed_sparse':
            from .sparse_attention import sparse_attention_state
            snapshot += state['recurrent_elements'] * DTYPE_BYTES['fp32'] + builder.kv_bytes(state['window_elements'])
            committed = sparse_attention_state(builder.m, layer, [before + commit] * batch, tp=builder.tp)
            verified = sparse_attention_state(builder.m, layer, [before + queries] * batch, tp=builder.tp)
            for key in ('compressed_pool_elements', 'indexer_pool_elements'):
                candidates += builder.kv_bytes(max(0, verified[key] - committed[key]))
        else:
            pages = batch * ceil(queries / builder.f['kv_page_tokens'])
            elements = pages * builder.f['kv_page_tokens'] * state['elements_per_token']
            candidates += builder.kv_bytes(elements) + pages * builder.f['kv_page_entry_bytes']
    return snapshot, candidates


def configure_speculative(builder: Builder):
    """Install the same round schedule on full-graph and streaming builders."""
    config = builder.c
    feature = config['execution']['features']['speculative_decode']
    if not feature['enabled']:
        raise ValueError('build_speculative requires speculative_decode.enabled')
    graph = builder.g
    plan = speculative_plan(config)
    mutable_state = any(kind in ('KDA', 'CSA', 'HCA') for kind in builder.a['attention_by_layer'])
    if mutable_state and feature['replay_cost'] is None and any(item.get('rejected_queries') for item in plan):
        raise ValueError('Rejected verification with KDA/CSA/HCA requires replay_cost to restore committed state')
    builder._inference_plan = plan
    ranks = list(builder.coords)
    draft = feature['draft']
    for rank in ranks:
        builder.g.initial_allocations.append(builder.allocation(
            f'speculative.{rank}.draft_resident', rank, draft['resident_bytes_per_rank'],
            'DraftModel', 'draft_resident', dtype=None,
            assumption='Explicit per-rank budget includes draft weights, KV and runtime for this workload.'))
    live = {}

    def before(prefix, item, ready):
        if item['phase'] == 'prefill':
            if draft['prefilled']:
                return ready
            allocations = [builder.allocation(prefix + f'.{rank}.draft_prefill_workspace', rank,
                draft['workspace_bytes_per_rank'], 'DraftModel', 'draft_workspace')
                for rank in ranks if draft['workspace_bytes_per_rank']]
            allocated = builder.barrier(prefix + '.draft_prefill.allocate', ready, allocations)
            ends = [_profile_event(builder, prefix + f'.{rank}.draft_prefill', rank,
                draft['prefill_cost'], [allocated], 'draft_prefill') for rank in ranks]
            return [builder.barrier(prefix + '.draft_prefill.done', ends, releases=[a.id for a in allocations])]
        allocations, snapshots = [], {}
        for rank in ranks:
            snapshot, candidates = _round_storage(builder, rank, item)
            snapshots[rank] = snapshot
            for suffix, size, category in (
                ('snapshot', snapshot, 'speculative_snapshot'),
                ('candidates', candidates, 'speculative_candidates'),
                ('draft_workspace', draft['workspace_bytes_per_rank'] if item['draft_tokens'] else 0, 'draft_workspace'),
            ):
                if size:
                    allocations.append(builder.allocation(prefix + f'.{rank}.{suffix}', rank,
                        size, 'SpeculativeDecode', category))
        live[prefix] = {'allocation_ids': [a.id for a in allocations], 'snapshots': snapshots}
        current = [builder.barrier(prefix + '.speculative.allocate', ready, allocations,
            phase='speculative_allocate', round_index=item['round_index'])]
        snapshot_ends = [_copy_event(builder, prefix + f'.{rank}.snapshot', rank, size, current, 'snapshot')
                         for rank, size in snapshots.items() if size]
        if snapshot_ends:
            current = snapshot_ends
        if item['draft_tokens']:
            steps = item['draft_tokens'] if draft['cost_unit'] == 'per_step' else 1
            for index in range(steps):
                current = [_profile_event(builder, prefix + f'.draft{index}.{rank}', rank,
                    draft['cost'], current, 'draft_forward') for rank in ranks]
        return [builder.barrier(prefix + '.verify_ready', current,
            phase='verify_ready', query_tokens=item['query_tokens'], draft_tokens=item['draft_tokens'])]

    def after(prefix, item, ready):
        if item['phase'] == 'prefill':
            return ready
        state = live.pop(prefix)
        current = list(ready)
        if feature['commit_cost'] is not None:
            current = [_profile_event(builder, prefix + f'.{rank}.accept', rank,
                feature['commit_cost'], current, 'accept_commit') for rank in ranks]
        decision = builder.barrier(prefix + '.acceptance', current, phase='acceptance',
            committed_tokens_per_sequence=item['committed_tokens'],
            rejected_queries_per_sequence=item['rejected_queries'],
            acceptance_source=feature['acceptance_source'],
            extra_cost_status='profile_driven' if feature['commit_cost'] is not None else 'not_modeled')
        current = [decision]
        if item['rejected_queries']:
            restores = [_copy_event(builder, prefix + f'.{rank}.restore', rank, size, current, 'rollback')
                        for rank, size in state['snapshots'].items() if size]
            if restores:
                current = restores
            if feature['replay_cost'] is not None:
                current = [_profile_event(builder, prefix + f'.{rank}.replay', rank,
                    feature['replay_cost'], current, 'state_replay') for rank in ranks]
        return [builder.barrier(prefix + '.commit', current, releases=state['allocation_ids'],
            phase='commit', committed_tokens_per_sequence=item['committed_tokens'],
            state_replay_cost_status='profile_driven' if feature['replay_cost'] is not None else 'not_modeled')]

    builder._inference_round_before = before
    builder._inference_round_after = after
    rounds = plan[1:]
    graph.metadata['speculative_decode'] = {
        'enabled': True, 'method': feature['method'], 'implementation': 'profile_driven_round_schedule',
        'round_count': len(rounds), 'rounds': rounds,
        'draft_tokens_per_sequence': sum(row['draft_tokens'] for row in rounds),
        'verified_tokens_per_sequence': sum(row['query_tokens'] for row in rounds),
        'committed_tokens_per_sequence': 1 + sum(row['committed_tokens'] for row in rounds),
        'acceptance_source': feature['acceptance_source'],
        'acceptance_mode': 'trace' if feature['accepted_lengths'] else 'fixed_assumption',
        'draft_prefill': 'externally_prefilled' if draft['prefilled'] else 'profile_driven',
        'unused_acceptance_trace_entries': max(0, len(feature['accepted_lengths']) - len(rounds)),
        'commit_cost': 'profile_driven' if feature['commit_cost'] is not None else 'not_modeled',
        'replay_cost': 'profile_driven' if feature['replay_cost'] is not None else 'not_modeled',
        'timing_scope': 'model execution; committed tokens share the round completion timestamp',
    }
    graph.metadata['assumptions'].extend([
        'Speculative acceptance counts are explicit inputs; DSpark acceptance prediction and token correctness are not simulated.',
        'A trace entry is the committed output count including bonus/replacement for every request in a synchronized equal-length batch.',
        'Draft cost is per participating rank for its whole local batch; draft resident budget includes its weights, maximum KV/state and runtime.',
        'Uneven EP request placement uses actual local request counts for target state; scalar draft/commit/replay budgets are charged on every rank and must cover the maximum local batch.',
        'Draft, verifier and commit/rollback use the same NPU compute slots and shared HBM. No speculative CPU/network overhead is inferred.',
        'Verifier candidate KV is conservatively double-buffered for dense/MLA; sparse rolling windows and recurrent/compressor states are snapshotted.',
        'A supplied per-round draft profile is not scaled down for a shortened final round; a per-step profile charges only executed draft positions.',
        'Draft prefill is serialized before target prefill when profiled; externally_prefilled explicitly excludes draft prompt preparation from TTFT.',
        'Unspecified commit costs are not modeled. KDA/CSA/HCA rejection requires a state replay profile; pure KV rejection can truncate candidates without recurrent replay.',
        'Snapshot/rollback copies and a replay profile model resource costs, not the internal ReplaySSM or DSpark algorithm.',
        'Committed output timestamps represent a model-side burst. Zero intra-round intervals are not independently observed client token latencies.',
    ])
    return plan


def build_speculative(config: dict, topology):
    """Build one fixed homogeneous batch; feature validation lives in Config."""
    builder = Builder(config, topology)
    configure_speculative(builder)
    try:
        graph = builder.inference()
    finally:
        for name in ('_inference_round_before', '_inference_round_after', '_inference_plan'):
            builder.__dict__.pop(name, None)
    builder.annotate_collective_allocations()
    return graph

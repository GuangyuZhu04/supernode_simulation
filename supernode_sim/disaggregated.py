"""Fixed-batch prefill/decode disaggregation with explicit device-pool handoff.

The two pools have independent model residency and identical TP/EP layouts.
Prefill emits the first token; state and token IDs then travel over the actual
topology before decode can start. This does not model an online request queue.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

from .core import Graph
from .structured import Builder


def _namespace(graph: Graph, prefix: str, pool: str) -> None:
    """Keep physical resource IDs, but make all event/allocation IDs distinct."""
    event_ids = {event.id: prefix + event.id for event in graph.events}
    allocations = [*graph.initial_allocations, *(a for event in graph.events for a in event.allocs)]
    allocation_ids = {allocation.id: prefix + allocation.id for allocation in allocations}
    for allocation in allocations:
        allocation.id = allocation_ids[allocation.id]
        allocation.metadata['execution_pool'] = pool
    for event in graph.events:
        event.id = event_ids[event.id]
        event.deps = tuple(event_ids.get(dep, dep) for dep in event.deps)
        event.releases = tuple(allocation_ids.get(aid, aid) for aid in event.releases)
        event.metadata['execution_pool'] = pool
    for ids in graph.metadata.get('token_completion_events', {}).values():
        ids[:] = [event_ids[eid] for eid in ids]
    for sequences in graph.metadata.get('sequence_token_completion_events', {}).values():
        for ids in sequences.values():
            ids[:] = [event_ids[eid] for eid in ids]
    if 'seeded_inference_state_allocations' in graph.metadata:
        graph.metadata['seeded_inference_state_allocations'] = [
            allocation_ids[aid] for aid in graph.metadata['seeded_inference_state_allocations']]


def _state_key(allocation):
    return (allocation.metadata.get('layer'), allocation.category, allocation.dtype)


def build_disaggregated(config: dict, topology) -> Graph:
    """Compile both pools and the state transfers, without revalidating subplans.

    Config validates global device count and rank sets. Each internal Builder
    uses the complete physical topology and its own subset rank_order; it must
    not be passed back through the single-pool configuration validator.
    """
    feature = config['execution']['features']['prefill_decode_disaggregation']
    if not feature['enabled']:
        raise ValueError('build_disaggregated requires prefill_decode_disaggregation.enabled')
    pconfig, dconfig = deepcopy(config), deepcopy(config)
    pconfig['execution']['rank_order'] = feature['prefill_ranks']
    dconfig['execution']['rank_order'] = feature['decode_ranks']
    prefill, decode = Builder(pconfig, topology), Builder(dconfig, topology)
    pgraph = prefill.inference(phase='prefill')
    prefill.annotate_collective_allocations()
    needs_decode = max(config['workload']['output_lengths']) > 1
    handoff_id = 'pd.handoff.done' if needs_decode else None
    if needs_decode:
        dgraph = decode.inference(phase='decode', initial_dependencies=(handoff_id,))
        decode.annotate_collective_allocations()
    else:
        # Deployment still reserves the decode model, but it receives no cache
        # or first-token IDs when all requests stop after prefill.
        decode.initial()
        dgraph = decode.g
    _namespace(pgraph, 'pd.prefill.', 'prefill')
    _namespace(dgraph, 'pd.decode.', 'decode')
    pterminal = pgraph.metadata['token_completion_events']['0'][0]
    dtokens = dgraph.metadata.get('token_completion_events', {}).get('0', [])
    first_decode = dtokens[0] if dtokens else None
    seed_ids = set(dgraph.metadata.get('seeded_inference_state_allocations', []))
    seeded = [a for a in dgraph.initial_allocations if a.id in seed_ids]
    dgraph.initial_allocations = [a for a in dgraph.initial_allocations if a.id not in seed_ids]

    handoff_events = []
    handoff_rows = []
    decode_token_buffers = []
    unused_source_state = []
    source_states = defaultdict(dict)
    destination_states = defaultdict(dict)
    # The decode seeds name exactly the persistent categories that must cross
    # pools. Activation/AttnRes workspaces do not survive the first-token event.
    persistent_keys = {_state_key(a) for a in seeded}
    for event in pgraph.events:
        for allocation in event.allocs:
            key = _state_key(allocation)
            if key in persistent_keys:
                rank = allocation.pool.rsplit('.', 1)[0]
                if key in source_states[rank]:
                    raise ValueError('Ambiguous prefill persistent state for PD handoff')
                source_states[rank][key] = allocation
    for allocation in seeded:
        rank = allocation.pool.rsplit('.', 1)[0]
        destination_states[rank][_state_key(allocation)] = allocation

    for index, (source_index, destination_index) in enumerate(zip(feature['prefill_ranks'], feature['decode_ranks'])):
        if not needs_decode:
            break
        source, destination = f'npu{source_index}', f'npu{destination_index}'
        ep_rank = prefill.coords[source][2]
        sequences = [i for i, length in enumerate(config['workload']['output_lengths'])
                     if i % prefill.ep == ep_rank and length > 1]
        targets = destination_states[destination]
        sources = source_states[source]
        if sources.keys() != targets.keys():
            raise ValueError('Prefill and decode persistent layouts differ; PD requires matching TP/EP state layouts')
        if not sequences:
            # The seed cache accounting remains initialized, preventing later
            # zero-growth loops from allocating unused state for finished rows.
            unused_source_state.extend(a.id for a in sources.values())
            continue
        state_rows = []
        for key, target in targets.items():
            origin = sources[key]
            if not math.isclose(origin.size_bytes, target.size_bytes, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('Prefill/decode state size mismatch; resharding is not implemented')
            state_rows.append({'layer': key[0], 'state': key[1], 'dtype': key[2],
                               'bytes': target.size_bytes})
        state_bytes = sum(a.size_bytes for a in targets.values())
        token_bytes = len(sequences) * feature['token_id_bytes']
        src_token = prefill.allocation(f'pd.handoff.{index}.source_token_ids', source, token_bytes,
            'System', 'pd_token_ids', wire_integer_bytes=feature['token_id_bytes'], execution_pool='prefill')
        dst_token = decode.allocation(f'pd.handoff.{index}.decode_token_ids', destination, token_bytes,
            'System', 'pd_token_ids', wire_integer_bytes=feature['token_id_bytes'], execution_pool='decode')
        decode_token_buffers.append(dst_token.id)
        event = topology.transfer(f'pd.handoff.{index}', source, destination, state_bytes + token_bytes,
            (pterminal,), {'component': 'PDHandoff', 'phase': 'handoff', 'operation': 'pd_state_handoff',
                          'semantic': 'pd_state_handoff', 'state_payload_bytes': state_bytes,
                          'token_id_payload_bytes': token_bytes, 'decode_sequences': sequences,
                          'state_breakdown': state_rows})
        event.allocs = (*targets.values(), src_token, dst_token)
        event.releases = (*[a.id for a in sources.values()], src_token.id)
        handoff_events.append(event)
        handoff_rows.append({'source': source, 'destination': destination,
            'state_payload_bytes': state_bytes, 'token_id_payload_bytes': token_bytes,
            'payload_bytes': state_bytes + token_bytes,
            'path': event.metadata['path'], 'domains': event.metadata['domains'],
            'physical_bytes_by_domain': event.metadata['physical_bytes_by_domain'],
            'states': state_rows})
    if needs_decode:
        from .core import Event
        if unused_source_state:
            handoff_events.append(Event('pd.prefill.release_finished_state', (pterminal,),
                releases=tuple(unused_source_state), metadata={'phase': 'handoff', 'operation': 'release_finished_state'}))
        handoff_events.append(Event(handoff_id, tuple(event.id for event in handoff_events) or (pterminal,),
            metadata={'phase': 'handoff_complete', 'operation': 'pd_handoff_barrier'}))
        first = next(event for event in dgraph.events if event.id == first_decode)
        first.releases = (*first.releases, *decode_token_buffers)

    graph = Graph(resources={**pgraph.resources, **dgraph.resources},
        memory_caps=dict(pgraph.memory_caps), events=[*pgraph.events, *handoff_events, *dgraph.events],
        initial_allocations=[*pgraph.initial_allocations, *dgraph.initial_allocations],
        metadata=deepcopy(pgraph.metadata))
    graph.metadata['token_completion_events'] = {'0': [pterminal, *dtokens]}
    pseq = pgraph.metadata['sequence_token_completion_events']['0']
    dseq = dgraph.metadata.get('sequence_token_completion_events', {}).get('0', {})
    graph.metadata['sequence_token_completion_events'] = {
        '0': {sequence: [*ids, *dseq.get(sequence, [])] for sequence, ids in pseq.items()}}
    if 'sequence_parallel' in graph.metadata:
        graph.metadata['sequence_parallel'].setdefault('steps', []).extend(
            deepcopy(dgraph.metadata.get('sequence_parallel', {}).get('steps', [])))
    graph.metadata['prefill_decode_disaggregation'] = {
        'enabled': True, 'model_copies': 2, 'prefill_ranks': feature['prefill_ranks'],
        'decode_ranks': feature['decode_ranks'], 'layout': {'tp': prefill.tp, 'ep': prefill.ep, 'pp': 1, 'dp': 1},
        'physical_npu_count': config['hardware']['npu_count'],
        'prefill_completion_event': pterminal, 'handoff_completion_event': handoff_id,
        'first_decode_completion_event': first_decode,
        'state_payload_bytes': sum(row['state_payload_bytes'] for row in handoff_rows),
        'token_id_payload_bytes': sum(row['token_id_payload_bytes'] for row in handoff_rows),
        'handoff_payload_bytes': sum(row['payload_bytes'] for row in handoff_rows),
        'handoffs': handoff_rows,
        'timing_scope': 'TTFT ends at prefill-pool first-token completion; the first decode interval includes handoff.',
        'state_lifetime': 'Decode cache allocated at transfer admission; source cache released only after its transfer completes.',
    }
    graph.metadata['assumptions'].extend([
        'Prefill/decode disaggregation deploys two resident model copies on disjoint physical NPU pools with identical TP/EP layouts.',
        'Both model copies are already loaded. TTFT is prefill execution to first token, not client latency; request queues, RPC framing and first-token client transport are not modeled.',
        'State handoff uses configured topology routes, endpoint HBM, shared links and injection/NIC resources; decode waits for every participating rank.',
        'Persistent cache payload conservatively includes allocated padding, scales, page tables, recurrent/convolution state and sparse compressor/window/index pools.',
        'Page-table portability/remapping and allocator software costs are not modeled; matching physical state layouts are required.',
        'Within an active EP group, packed fixed-batch state includes completed one-token requests; first-token IDs are transferred only for requests requiring decode.',
        'A TP rank receives its own first-token ID copy; integer wire width is explicit in token_id_bytes. Source token buffers are released after transfer and decode buffers after the first decode step.',
    ])
    return graph

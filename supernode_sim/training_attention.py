"""Analytical FlashAttention tile dependencies for structured training."""
from collections import Counter, defaultdict

from .core import DTYPE_BYTES, Event


def causal_pairs(length, window=None):
    """Valid Q/K pairs for one unpadded document, including the diagonal."""
    width = length if window is None else min(length, window)
    return width * (width + 1) // 2 + (length - width) * width


def _triangle(x):
    return x * (x + 1) // 2 if x > 0 else 0


def _pairs_through(q0, q1, k0, k1):
    # Sum the clipped linear ramp of valid key counts over these query rows.
    return _triangle(q1 - k0) - _triangle(q0 - k0) - _triangle(q1 - k1) + _triangle(q0 - k1)


def causal_tiles(lengths, blocks, window=None):
    """Aggregate live tiles across documents, counting each query load/store once."""
    tiles = defaultdict(lambda: [0, 0, 0, 0])
    for length, count in Counter(lengths).items():
        n = min(blocks, length)
        edges = [i * length // n for i in range(n + 1)]
        for q in range(n):
            q0, q1 = edges[q:q + 2]
            first = True
            for k in range(q + 1):
                k0, k1 = edges[k:k + 2]
                pairs = _pairs_through(q0, q1, k0, k1)
                if window is not None:
                    pairs -= _pairs_through(q0 - window, q1 - window, k0, k1)
                if not pairs:
                    continue
                keys = k1 - (k0 if window is None else max(k0, q0 - window + 1))
                tile = tiles[q, k]
                tile[0] += count * pairs
                tile[1] += count * keys
                tile[2] += count * (q1 - q0) if first else 0
                tile[3] += count * (q1 - q0) if q == k else 0
                first = False
    return sorted(tiles.items())


def flash_attention(builder, event, rank, lengths, tensors, parameter_bytes):
    """Model projections, causal tiles and ping-pong reads on shared HBM.

    Backward keeps the compiler's 2x-forward FLOP convention. Expanded K/V
    are read once per visited query tile. The two input buffers are on chip.
    """
    m, p, meta = builder.m, builder.p, event.metadata
    component = meta['component']
    backward = meta['phase'] == 'backward'
    q_heads = m['num_heads'] / builder.tp
    q_dim = v_dim = m['head_dim']
    heads = {'MHA': m['num_heads'], 'MQA': 1, 'GQA': m['kv_heads'], 'MLA': m['num_heads']}
    kv_heads = max(meta.get('active_kv_heads', heads[component]) / builder.tp, 1)
    if component == 'MLA':
        mla = builder.a['mla']
        q_dim, v_dim = mla['qk_nope_head_dim'] + mla['qk_rope_head_dim'], mla['v_head_dim']
    width = DTYPE_BYTES[p['activation']]
    q_bytes, o_bytes = q_heads * q_dim * width, q_heads * v_dim * width
    kv_bytes = kv_heads * (q_dim + v_dim) * width
    lse_bytes = 4 * q_heads
    tokens = meta['tokens']
    qkv = tokens * (q_bytes + kv_bytes)
    output = tokens * o_bytes
    hidden = tokens * m['hidden_dim'] * width
    tiles = causal_tiles(lengths, builder.x['flash_attention_blocks'], meta.get('attention_window'))
    divisor = meta['workload_shape']['training_ep_divisor']
    flops_per_pair = 2 * q_heads * (q_dim + v_dim) * (2 if backward else 1) / divisor
    attention_flops = sum(tile[0] for _, tile in tiles) * flops_per_pair
    projection_flops = max(0.0, meta['flops'] - attention_flops)
    elements = sum(builder.kernel_elements(t) for t in tensors)
    out_fraction = sum(builder.kernel_elements(t) for t in tensors
                       if '.output.' in t.name) / elements
    qkv_io = parameter_bytes * (1 - out_fraction) + hidden + qkv
    out_io = parameter_bytes * out_fraction + hidden + output
    if component == 'MLA':
        qkv_io += sum(2 * row['elements'] * width for row in builder.activation_rows(meta['layer'], tokens)
                      if row['component'] == 'MLA' and row['name'] not in ('qkv_and_output', 'softmax_lse'))
    if backward:
        qkv_io += hidden
        out_io += output
    proj = ('flash_qkv_projection', projection_flops * (1 - out_fraction), qkv_io)
    out = ('flash_output_projection', projection_flops * out_fraction, out_io)
    first, last = (out, proj) if backward else (proj, out)

    def kernel(eid, kind, deps, flops, memory, slots, *, allocs=(), releases=(), **shape):
        work = {k: v for k, v in ((rank + '.compute', flops), (rank + '.hbm', memory)) if v}
        return builder.add(Event(eid, tuple(dict.fromkeys(deps)), work, slots=slots,
            allocs=allocs, releases=releases, metadata={**meta, 'kernel': kind,
                'training_attention_part': True, 'flops': flops, 'hbm_bytes': memory,
                'flash_attention_blocks': builder.x['flash_attention_blocks'], **shape}))

    start = kernel(event.id + '.input', first[0], event.deps, first[1], first[2], event.slots, allocs=event.allocs)
    previous_read = previous_compute = two_back = start
    stores, kv_updates, projections = [], {}, []
    stream_output = builder.x['overlap_attention_tp'] and not backward
    for (q, k), (pairs, keys, first_queries, last_queries) in tiles:
        prefix = f'{event.id}.q{q}.k{k}'
        shape = {'query_block': q, 'kv_block': k, 'causal_pairs': pairs / divisor}
        reads = keys * kv_bytes + first_queries * q_bytes
        writes = last_queries * (o_bytes + lse_bytes)
        read_deps = [previous_read, two_back]
        if backward:
            reads += first_queries * (2 * o_bytes + lse_bytes)
            if k in kv_updates:
                reads += keys * kv_bytes
                read_deps.append(kv_updates[k])
            writes = last_queries * q_bytes + keys * kv_bytes
        read = kernel(prefix + '.load', 'flash_tile_load', read_deps, 0, reads / divisor,
                      (rank + '.attention_read',), **shape)
        tile = kernel(prefix + '.compute', 'flash_tile_compute', [read, previous_compute],
                      pairs * flops_per_pair, 0, event.slots, **shape)
        if writes:
            store = kernel(prefix + '.store', 'flash_tile_store', [tile], 0, writes / divisor,
                           (rank + '.attention_write',), **shape)
            stores.append(store)
            if backward:
                kv_updates[k] = store
            elif stream_output:
                rows = last_queries / divisor
                share = rows / tokens
                eid = f'{event.id}.q{q}.output'
                output_buffer = builder.allocation(eid + '.partial', rank, rows * m['hidden_dim'] * width,
                    component, 'tp_partial_output', meta['layer'], p['activation'])
                projected = kernel(eid, last[0], [store], last[1] * share, last[2] * share, event.slots,
                                   allocs=(output_buffer,), query_block=q, output_rows=rows)
                projections.append((projected, rows, output_buffer.id))
        previous_read, two_back, previous_compute = read, previous_compute, tile
    if stream_output:
        builder.attention_tp_outputs[event.id] = projections
        return builder.barrier(event.id, [eid for eid, _, _ in projections], releases=event.releases)
    return kernel(event.id, last[0], [previous_compute, *stores], last[1], last[2], event.slots,
                  releases=event.releases)

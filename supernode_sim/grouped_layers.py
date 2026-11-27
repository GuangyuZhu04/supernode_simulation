"""Layer execution on distinct attention and routed-expert process groups."""
from collections import defaultdict
from math import ceil

from .components import ParameterTensor, attention_state, attention_training_state, attnres_state
from .core import DTYPE_BYTES, Event
from .moe_wire import moe_payload_matrix, moe_wire_metadata
from .rank_layout import shard_pairs, shard_rows


def budget_layers(model):
    """Place only the shared/routed budgets supplied by the profile."""
    layers = model['layers']
    routed = model['total_params'] * model['expert_parameter_fraction']
    dense = model['total_params'] - routed
    count = max(1, round(layers * model['moe_layer_fraction'])) if routed else 0
    kind = 'MLA' if model['attention_kind'] == 'latent' else model['attention_kind'].upper()
    arch = {'attention_by_layer': [kind] * layers, 'ffn_by_layer': ['Dense'] * (layers-count) + ['MoE'] * count,
            'moe': {}, 'mla': {'qk_nope_head_dim': model['head_dim'], 'qk_rope_head_dim': 0, 'v_head_dim': model['head_dim']}}
    tensors = []
    for layer in range(layers):
        tensors.append(ParameterTensor(f'layers.{layer}.shared_budget', kind, layer, (), role='unknown',
            budget_elements=dense/layers, active_fraction=min(model['active_params'], dense)/dense if dense else 0))
        if layer >= layers-count:
            tensors.append(ParameterTensor(f'layers.{layer}.routed_budget', 'MoE', layer, (), role='unknown',
                ep_shard=True, budget_elements=routed/count, active_fraction=max(0, model['active_params']-dense)/routed))
    return arch, tensors


class LayerBatch:
    """Requests retain their attention replica; context positions use cyclic ownership."""
    def __init__(self, mesh, records):
        self.mesh = mesh
        self.requests = [[] for _ in range(mesh.attention_dp)]
        for sequence, length, queries in records:
            self.requests[sequence % mesh.attention_dp].append((length, queries))

    def rows(self, slot, *, owner=False):
        replica, shard = divmod(slot, self.mesh.cp)
        if owner:
            return self.requests[replica] if shard == 0 else []
        return [(length, local) for length, queries in self.requests[replica]
                if (local := shard_rows(length, shard, self.mesh.cp) - shard_rows(length-queries, shard, self.mesh.cp))]

    def context(self, slot, *, owner=False):
        replica, shard = divmod(slot, self.mesh.cp)
        if owner:
            return [length for length, _ in self.requests[replica]] if shard == 0 else []
        return [rows for length, _ in self.requests[replica] if (rows := shard_rows(length, shard, self.mesh.cp))]

    def pairs(self, slot, window):
        replica, shard = divmod(slot, self.mesh.cp)
        return sum(shard_pairs(s, q, shard, self.mesh.cp, window) for s, q in self.requests[replica])

    def tokens(self, slot):
        return sum(q for _, q in self.rows(slot))

    def last_tokens(self, slot):
        replica, shard = divmod(slot, self.mesh.cp)
        return sum((s-1) % self.mesh.cp == shard for s, _ in self.requests[replica])


class GroupedLayer:
    def __init__(self, builder, batch, replica, stage, layer, name, phase):
        self.b, self.batch, self.layer, self.name, self.phase = builder, batch, layer, name, phase
        self.mesh = builder.mesh
        self.groups = builder.layer_groups(replica, stage)
        self.tp_groups = self.groups['attention_tp']
        self.tokens = [batch.tokens(i) for i in range(len(self.tp_groups))]
        self.retained = []
        self.output_retained = []
        self.cp_intermediates = []
        self.kind = builder.a['attention_by_layer'][layer]

    def transfer(self, name, src, dst, size, deps, component, axis, dtype):
        # Use the native collective transport so staging, topology and admission
        # use the same physical endpoints as all other parallel collectives.
        return self.b.topo.collective(self.b.g, name, 'all_to_all_v', [src, dst], 0, tuple(deps),
            payload_matrix=[[0, size], [0, 0]], metadata={'component': component, 'layer': self.layer,
                'phase': self.phase, 'parallel_axis': axis, 'operation': axis, 'buffer_dtype': dtype})

    def reduce(self, name, component, groups, rows, deps, axis, kind='all_reduce', width=None):
        b = self.b
        width = width or b.m['hidden_dim']
        terminals = []
        for i, (group, n) in enumerate(zip(groups, rows)):
            if n and len(group) > 1:
                terminals.extend(b.collective(f'{self.name}.{name}{i}', kind, group,
                    n * width * DTYPE_BYTES[b.p['communication']], deps, component, self.layer, activation=True,
                    phase=self.phase, wire_metadata={'parallel_axis': axis}))
        return terminals or deps

    def kernel(self, component, rank, tokens, deps, records, *, part=None, pairs=None, total=None):
        if not tokens:
            return []
        b, layer = self.b, self.layer
        name = f'{self.name}.{rank}.{component}' + (f'.{part}' if part else '')
        allocations, releases = [], []
        if part not in ('context', 'latent_up') and self.phase != 'backward':
            rows = b.activation_rows(layer, tokens, training=b.training)
            for i, row in enumerate(rows):
                if row['component'] != component:
                    continue
                routed = row['name'] == 'routed_gate_up'
                if part in ('shared', 'routed') and routed != (part == 'routed'):
                    continue
                dtype = row.get('dtype', b.p['activation'])
                amount = row['elements'] * DTYPE_BYTES[dtype]
                a = b.allocation(name+f'.a{i}', rank, amount, component, 'activation', layer, dtype, formula=row['formula'])
                allocations.append(a)
                save = (self.phase == 'recompute' or self.phase == 'forward' and
                        (not b.f['activation_checkpointing'] or component == b.component_order(layer)[0] and i == 0))
                if component == 'Output' and self.phase == 'forward' and getattr(b, '_training_loss', None):
                    self.output_retained.append(a.id)
                elif save and not b.forward_only and row.get('lifetime') != 'operator_working_set':
                    self.retained.append(a.id)
                elif part == 'qkv':
                    self.cp_intermediates.append(a.id)
                else:
                    releases.append(a.id)
            if component in ('KDA', 'CSA', 'HCA') and self.phase in ('forward', 'recompute'):
                sizes = [('recurrent_chunk_elements', 'saved_recurrent_states', b.p['recurrent'])] if component == 'KDA' else [
                    ('compressed_pool_elements', 'compressed_attention_pool', b.p['kv']),
                    ('selection_index_elements', 'attention_selection_indices', 'fp32')]
                states = [attention_training_state(b.m, layer, 1, length, tp=b.tp) for length, _ in records]
                for field, category, dtype in sizes:
                    amount = sum(state[field] for state in states)*DTYPE_BYTES[dtype]
                    if amount:
                        a = b.allocation(name+'.'+category, rank, amount, component, category, layer, dtype)
                        allocations.append(a)
                        (self.retained if not b.forward_only and (self.phase == 'recompute' or not b.f['activation_checkpointing']) else releases).append(a.id)
        lengths, queries = [s for s, _ in records], [q for _, q in records]
        return [b.kernel(name, rank, layer, component, tokens, deps, phase=self.phase,
            backward=self.phase == 'backward', lengths=lengths, query_lengths=queries, total_tokens=total,
            allocations=allocations, releases=releases, part=part, causal_work=pairs, local_work=True)]

    def regroup(self, origins, deps, returning):
        """Copy hidden rows only to TP ranks missing them on the receiving layout."""
        b = self.b
        dtype = b.p['activation']
        width = (b.a['moe'].get('latent_dim') or b.m['hidden_dim']) * DTYPE_BYTES[dtype]
        ends = []
        for (source, dest), n in origins.items():
            before, after = self.tp_groups[source], self.groups['expert_tp'][dest]
            if returning:
                before, after = after, before
            # The selected representative is in the intersection by construction.
            owner = next(r for r in before if r in after)
            for rank in after:
                if rank not in before:
                    prefix = f'{self.name}.moe.{"return" if returning else "enter"}.{source}.{dest}.{rank}'
                    ends.extend(self.transfer(prefix, owner, rank, n*width, deps, 'MoE', 'layout_exchange', dtype))
        return ends or deps

    def moe(self, deps):
        b, mesh = self.b, self.mesh
        origins = defaultdict(int)
        source_rows = [0] * len(self.groups['expert_tp'])
        replica_requests = [{} for _ in range(mesh.expert_dp)]
        expert_source = {rank: i for i, group in enumerate(self.groups['expert_tp']) for rank in group}
        for slot, ranks in enumerate(self.tp_groups):
            attention_replica, shard = divmod(slot, mesh.cp)
            offset = 0
            for sequence, (length, queries) in enumerate(self.batch.requests[attention_replica]):
                n = shard_rows(length, shard, mesh.cp) - shard_rows(length-queries, shard, mesh.cp)
                for lane in range(min(n, mesh.tp)):
                    source = expert_source[ranks[(offset+lane) % mesh.tp]]
                    count = shard_rows(n, lane, mesh.tp)
                    origins[slot, source] += count
                    source_rows[source] += count
                    record = replica_requests[source // mesh.ep].setdefault((attention_replica, sequence), [length, 0])
                    record[1] += count
                offset += n
        replica_records = [list(requests.values()) for requests in replica_requests]
        shared = [t for t in b.local_tensors(self.layer, 'MoE') if not t.ep_shard]
        replica_tokens = [sum(source_rows[i:i + mesh.ep]) for i in range(0, len(source_rows), mesh.ep)]
        has_up = any('.moe.latent_up.' in t.name for t in shared)
        def local(part, predecessors):
            ends = [end for i, group in enumerate(self.tp_groups) for rank in group
                    for end in self.kernel('MoE', rank, self.tokens[i], predecessors, self.batch.rows(i), part=part)]
            return ends or predecessors
        if has_up and self.phase == 'backward':
            deps = local('latent_up', deps)
        if shared and self.phase != 'backward':
            deps = local('shared', deps)
            if any(t.tp_shard for t in shared):
                deps = self.reduce('shared_tp', 'MoE', self.tp_groups, self.tokens, deps, 'tp')
        deps = self.regroup(origins, deps, False)
        width = b.a['moe'].get('latent_dim') or (b.routing or {}).get('routed_width') or b.m['hidden_dim']
        def exchange(operation, predecessors):
            ends = []
            for i, members in enumerate(self.groups['ep']):
                replica = i // mesh.expert_tp
                rows = source_rows[replica*mesh.ep:(replica+1)*mesh.ep]
                if not replica_tokens[replica] or len(members) == 1:
                    continue
                matrix = moe_payload_matrix(b.p, operation, [r*b.wire_factor for r in rows], width, b.m['top_k'])
                meta = moe_wire_metadata(b.p, operation, width, top_k=b.m['top_k'], num_experts=b.m['num_experts'])
                ends.extend(b.collective(f'{self.name}.{operation}{i}', 'all_to_all_v', members, 0,
                    predecessors, 'MoE', self.layer, matrix, True, phase=self.phase, wire_metadata={**meta, 'parallel_axis': 'ep'}))
            return ends or predecessors
        buffers, assignments = [], []
        for i, group in enumerate(self.groups['expert_tp']):
            total = replica_tokens[i // mesh.ep]
            assignments.append(total / mesh.ep * b.m['top_k'] * b.wire_factor)
            if assignments[-1]:
                for rank in group:
                    buffers.append(b.allocation(f'{self.name}.{rank}.expert_buffer', rank,
                        assignments[-1]*width*DTYPE_BYTES[b.p['activation']], 'MoE', 'expert_buffer', self.layer, b.p['activation']))
        deps = exchange('dispatch', [b.barrier(self.name+'.experts_ready', deps, buffers)])
        ends = []
        for i, group in enumerate(self.groups['expert_tp']):
            total = replica_tokens[i // mesh.ep]
            for rank in group:
                ends.extend(self.kernel('MoE', rank, total/mesh.ep, deps, replica_records[i // mesh.ep],
                                        part='routed', total=total))
        deps = self.reduce('expert_tp', 'MoE', self.groups['expert_tp'], assignments,
                           ends or deps, 'expert_tp', width=width)
        deps = [b.barrier(self.name+'.experts_done', exchange('combine', deps), releases=[a.id for a in buffers])]
        deps = self.regroup(origins, deps, True)
        if shared and self.phase == 'backward':
            deps = local('shared', deps)
            if any(t.tp_shard for t in shared):
                deps = self.reduce('shared_tp', 'MoE', self.tp_groups, self.tokens, deps, 'tp')
        if has_up and self.phase != 'backward':
            deps = local('latent_up', deps)
        return deps

    def kv_exchange(self, deps, *, gradients=False):
        """All-gather actual cyclic KV rows; reverse the edges for summed KV gradients."""
        b, mesh = self.b, self.mesh
        model = {**b.m, 'kv_heads': b.active_kv[self.layer]} if b.active_kv and b.active_kv[self.layer] else b.m
        elems = attention_state(model, self.layer, 1, 1, tp=mesh.tp)['elements_per_token']
        dtype = b.p['gradient'] if gradients else b.p['compute']
        width = elems * DTYPE_BYTES[dtype]
        ends, allocations = [], []
        for index, group in enumerate(self.groups['cp']):
            first = index // mesh.tp * mesh.cp
            rows = [sum(self.batch.context(first+c)) for c in range(mesh.cp)]
            active = [self.tokens[first+c] > 0 for c in range(mesh.cp)]
            matrix = [[n * width if i != j and active[j] else 0 for j in range(mesh.cp)]
                      for i, n in enumerate(rows)]
            if gradients:
                matrix = list(map(list, zip(*matrix)))
            if not any(any(row) for row in matrix):
                continue
            name = self.name + f'.cp_{"kv_grad" if gradients else "kv"}.{index}'
            buffers = []
            for j, rank in enumerate(group):
                size = sum(row[j] for row in matrix)
                if size:
                    buffers.append(b.allocation(name+f'.{rank}.workspace', rank, size, self.kind, 'cp_workspace', self.layer, dtype))
            start = b.barrier(name+'.allocate', deps, buffers)
            allocations.extend(buffers)
            ends.extend(b.topo.collective(b.g, name, 'all_to_all_v', group, 0, (start,),
                payload_matrix=matrix, metadata={'component': self.kind, 'layer': self.layer, 'phase': self.phase,
                    'parallel_axis': 'cp', 'operation': 'cp_kv_gradient' if gradients else 'cp_kv_gather', 'buffer_dtype': dtype}))
        if allocations:
            ends = [b.barrier(self.name + ('.kv_grad_ready' if gradients else '.kv_ready'), ends)]
            if gradients:
                sums = []
                for a in allocations:
                    rank = a.pool.split('.')[0]
                    adds = a.size_bytes / DTYPE_BYTES[dtype]
                    sums.append(b.add(Event(a.id+'.sum', tuple(ends),
                        {b.scalar(rank): adds, rank+'.hbm': 3*a.size_bytes}, slots=(rank+'.compute_stream',),
                        metadata={'kernel': 'cp_sum_kv_gradients', 'component': self.kind, 'layer': self.layer,
                                  'phase': self.phase, 'flops': adds})))
                ends = sums
        return ends or deps, [a.id for a in allocations]

    def cache_io(self, deps, *, writing):
        b = self.b
        offload = b.f['kv_offload_fraction'] if b.f['kv_offload'] != 'none' else 0
        convert = b.p['kv'] != b.p['compute']
        if b.training or self.kind in ('KDA', 'CSA', 'HCA') or not (offload or convert):
            return deps, []
        width = attention_state(b.m, self.layer, 1, 1, tp=b.tp)['elements_per_token']
        outputs, retained = [], []
        for i, group in enumerate(self.tp_groups):
            rows = self.tokens[i] if writing else sum(self.batch.context(i))-self.tokens[i]
            if not rows:
                continue
            elems = width * rows
            packed = b.kv_bytes(elems)
            raw = elems * DTYPE_BYTES[b.p['compute']]
            for rank in group:
                name = f'{self.name}.{rank}.kv_{"write" if writing else "read"}'
                host = b.host(rank) if b.f['kv_offload'] == 'host' else 'remote0'
                current = list(deps)
                buffers = []
                if offload and not writing:
                    event = b.topo.transfer(name+'.fetch', host, rank, packed*offload, tuple(current),
                        {'component': self.kind, 'operation': 'kv_read', 'layer': self.layer, 'phase': self.phase})
                    a = b.allocation(name+'.received', rank, packed*offload, self.kind, 'kv_fetch_buffer', self.layer, b.p['kv'])
                    event.allocs = (a,)
                    buffers.append(a.id)
                    current = [b.add(event)]
                if convert:
                    size, dtype = (packed, b.p['kv']) if writing else (raw, b.p['compute'])
                    a = b.allocation(name+'.buffer', rank, size, self.kind, 'kv_conversion_buffer', self.layer, dtype)
                    current = [b.add(Event(name+'.convert', tuple(current), {b.scalar(rank): 2*elems, rank+'.hbm': raw+packed},
                        slots=(rank+'.compute_stream',), allocs=(a,), releases=tuple(buffers),
                        metadata={'component': self.kind, 'operation': 'precision_conversion', 'layer': self.layer,
                                  'phase': self.phase, 'kernel': 'kv_encode_conversion' if writing else 'kv_decode_conversion'}))]
                    buffers = [a.id]
                if offload and writing:
                    transfer = b.topo.transfer(name+'.store', rank, host, packed*offload, tuple(current),
                        {'component': self.kind, 'operation': 'kv_write', 'layer': self.layer, 'phase': self.phase})
                    current = [b.add(transfer)]
                retained.extend(buffers)
                outputs.extend(current)
        return outputs or deps, retained

    def lookup(self, deps):
        b, spec = self.b, self.b.f['engram']
        if not spec['enabled'] or self.layer not in spec['layers']:
            return deps, []
        ends, held = [], []
        for i, group in enumerate(self.tp_groups):
            queries = self.tokens[i] * spec['queries_per_token'] / b.tp
            for rank in group:
                misses = queries * (1-spec['hit_rate'])
                name = f'{self.name}.{rank}.engram'
                if misses:
                    src = b.host(rank) if spec['location'] == 'host' else 'remote0'
                    size = misses * spec['record_bytes']
                    event = b.topo.transfer(name+'.read', src, rank, size, tuple(deps),
                        {'component': 'Engram', 'operation': 'lookup', 'layer': self.layer, 'phase': self.phase})
                    event.work[src+'.engram_iops'] = misses
                    event.work[src+'.iops'] = misses
                    event.startup_s += spec['latency_s']
                    a = b.allocation(name+'.received', rank, size, 'Engram', 'lookup_buffer', self.layer)
                    event.allocs = (a,)
                    held.append(a.id)
                    ends.append(b.add(event))
                if queries > misses:
                    ends.append(b.add(Event(name+'.hit', tuple(deps), {rank+'.hbm': (queries-misses)*spec['record_bytes']},
                        metadata={'component': 'Engram', 'operation': 'cache_hit', 'phase': self.phase})))
        return ends or deps, held

    def attention(self, deps):
        b, mesh = self.b, self.mesh
        deps, held = self.cache_io(deps, writing=False)
        owner = mesh.cp > 1 and self.kind in ('KDA', 'CSA', 'HCA')
        dense_cp = mesh.cp > 1 and not owner
        window = b.windows[self.layer] if b.windows else None
        def compute(part, predecessors):
            ends = []
            for slot, group in enumerate(self.tp_groups):
                rows = self.batch.rows(slot, owner=owner)
                tokens = sum(q for _, q in rows)
                pairs = 0 if part == 'qkv' else self.batch.pairs(slot, window)
                for rank in group:
                    ends.extend(self.kernel(self.kind, rank, tokens, predecessors, rows, part=part, pairs=pairs))
            return ends or predecessors
        def move_to_owner(predecessors, returning):
            ends = []
            for index, group in enumerate(self.groups['cp']):
                first = index // mesh.tp * mesh.cp
                for c in range(1, mesh.cp):
                    count = self.tokens[first+c]
                    if count:
                        src, dst = (group[0], group[c]) if returning else (group[c], group[0])
                        name = self.name+f'.cp_owner.{int(returning)}.{index}.{c}'
                        ends.extend(self.transfer(name, src, dst, count*b.m['hidden_dim']*DTYPE_BYTES[b.p['activation']],
                                                  predecessors, self.kind, 'cp', b.p['activation']))
            return ends or predecessors
        if dense_cp:
            if self.phase == 'backward':
                # Reassemble K/V for local-query backward, then sum remote dK/dV
                # at the original context owners before projection backward.
                deps, kv = self.kv_exchange(deps)
                deps = compute('context', deps)
                deps, grad = self.kv_exchange(deps, gradients=True)
                deps = compute('qkv', deps)
                buffers = kv + grad
            else:
                deps = compute('qkv', deps)
                deps, buffers = self.kv_exchange(deps)
                deps = compute('context', deps)
            if buffers:
                deps = [b.barrier(self.name+'.cp_release', deps, releases=buffers)]
            if self.cp_intermediates:
                deps = [b.barrier(self.name+'.cp_qkv_release', deps, releases=self.cp_intermediates)]
        else:
            deps = compute(None, move_to_owner(deps, False) if owner else deps)
        if self.kind == 'CSA':
            indexed = []
            for i, group in enumerate(self.tp_groups):
                rows = self.batch.rows(i, owner=owner)
                if rows and self.phase != 'backward':
                    indexed.extend(b.sparse_index_collective(self.name+f'.i{i}', self.layer, [group],
                        [s for s, _ in rows], deps, self.phase, query_lengths=[q for _, q in rows]))
            deps = indexed or deps
        counts = [sum(q for _, q in self.batch.rows(i, owner=owner)) for i in range(len(self.tp_groups))]
        if any(eid in b.attention_tp_outputs for eid in deps):
            reductions = list(deps)
            for i, group in enumerate(self.tp_groups):
                if not counts[i]:
                    continue
                chunks = [b.attention_tp_outputs.pop(f'{self.name}.{rank}.{self.kind}') for rank in group]
                ready = []
                for q, parts in enumerate(zip(*chunks)):
                    projected, rows, buffers = zip(*parts)
                    prefix = f'{self.name}.{self.kind}.tp{i}.q{q}'
                    ready = b.collective(prefix, 'all_reduce', group, rows[0]*b.m['hidden_dim']*DTYPE_BYTES[b.p['communication']],
                        [*projected, *ready], self.kind, self.layer, phase=self.phase, activation=True, wire_metadata={'parallel_axis': 'tp'})
                    ready = [b.barrier(prefix+'.done', ready, releases=buffers)]
                reductions.extend(ready)
            deps = reductions
        elif b.sp and not owner:
            deps = self.reduce('attention_sp', self.kind, self.tp_groups,
                [ceil(n/mesh.tp)*mesh.tp for n in counts], deps, 'tp', 'reduce_scatter')
        else:
            deps = self.reduce('attention_tp', self.kind, self.tp_groups, counts, deps, 'tp')
        deps, buffers = self.cache_io(move_to_owner(deps, True) if owner else deps, writing=True)
        return [b.barrier(self.name+'.cache_release', deps, releases=held+buffers)] if held or buffers else deps

    def execute(self, deps):
        b = self.b
        order = b.component_order(self.layer)
        for component in (reversed(order) if self.phase == 'backward' else order):
            sp_buffers = []
            lookup_buffers = []
            if component == ('PreNorm' if b.m['architecture'] else self.kind):
                deps, lookup_buffers = self.lookup(deps)
            if b.sp and component in (self.kind, 'Dense', 'MoE', 'Output'):
                counts = [self.batch.last_tokens(i) for i in range(len(self.tokens))] if component == 'Output' and self.phase == 'prefill' else self.tokens
                padded = [ceil(n/b.tp)*b.tp for n in counts]
                for i, group in enumerate(self.tp_groups):
                    if padded[i]:
                        for rank in group:
                            sp_buffers.append(b.allocation(f'{self.name}.{component}.{rank}.sp_input', rank,
                                padded[i]*b.m['hidden_dim']*DTYPE_BYTES[b.p['activation']], component, 'sp_gathered_input', self.layer, b.p['activation']))
                start = b.barrier(f'{self.name}.{component}.sp_start', deps, sp_buffers)
                deps = self.reduce(component+'.sp_gather', component, self.tp_groups, padded, [start], 'tp', 'all_gather')
            if component == self.kind:
                deps = self.attention(deps)
            elif component == 'MoE':
                deps = self.moe(deps)
            else:
                counts = self.tokens
                if b.sp and component in ('PreNorm', 'PostNorm', 'AttnRes'):
                    counts = [ceil(n/b.tp) for n in counts]
                if component == 'Output' and self.phase == 'prefill':
                    counts = [self.batch.last_tokens(i) for i in range(len(self.tp_groups))]
                ends = [end for i, group in enumerate(self.tp_groups) for rank in group
                        for end in self.kernel(component, rank, counts[i], deps, self.batch.rows(i))]
                deps = ends or deps
                if b.sp and component in ('Dense', 'WordEmbedding'):
                    deps = self.reduce(component+'.sp', component, self.tp_groups,
                        [ceil(n/b.tp)*b.tp for n in self.tokens], deps, 'tp', 'reduce_scatter')
                elif component == 'Dense':
                    deps = self.reduce('dense_tp', component, self.tp_groups, counts, deps, 'tp')
            if sp_buffers:
                deps = [b.barrier(f'{self.name}.{component}.sp_release', deps, releases=[a.id for a in sp_buffers])]
            if lookup_buffers:
                deps = [b.barrier(self.name+'.lookup_release', deps, releases=lookup_buffers)]
        return deps


def grouped_inference(b, phase, initial_dependencies):
    """Keep the normal token and layer barriers used by reports and streaming."""
    b.initial()
    b.g.metadata.update(token_completion_events={}, sequence_token_completion_events={})
    w, mesh = b.w, b.mesh
    plan = getattr(b, '_inference_plan', None)
    if plan and phase != 'all':
        raise ValueError('Speculative round plans cannot be split into inference phases')
    def state(batch, slot, rank, layer, prefix, step):
        owner = mesh.cp > 1 and b.a['attention_by_layer'][layer] in ('KDA', 'CSA', 'HCA')
        return b.inference_state_allocations(prefix, rank, layer, batch.context(slot, owner=owner), step)
    if phase == 'decode':
        batch = LayerBatch(mesh, ((i, s, s) for i, s in enumerate(w['input_lengths'])))
        seeded = [a for rank, (_, stage, slot, _) in b.coords.items() for layer in b.stage_layers[stage]
                  for a in state(batch, slot, rank, layer, f'infer.seed.l{layer}', 0)]
        b.g.initial_allocations.extend(seeded)
        b.g.metadata['seeded_inference_state_allocations'] = [a.id for a in seeded]
    for d in range(mesh.dp):
        deps = list(initial_dependencies)
        token_ends = []
        sequence_ends = {str(i): [] for i in range(w['batch_size'])}
        count = len(plan) if plan else max(w['output_lengths'])
        for step in range(int(phase == 'decode'), 1 if phase == 'prefill' else count):
            item = plan[step] if plan else None
            current = item['phase'] if item else ('decode' if step else 'prefill')
            active = [i for i, n in enumerate(w['output_lengths']) if plan or step < n]
            records = [(i, item['context_before']+item['query_tokens'] if item else w['input_lengths'][i]+step,
                        item['query_tokens'] if item else (1 if step else w['input_lengths'][i])) for i in active]
            batch = LayerBatch(mesh, records)
            # Completed requests retain their allocated KV pages through the batch.
            committed = LayerBatch(mesh, ((i, item['context_before']+(item['query_tokens'] if current == 'prefill' else item['committed_tokens'])
                                        if item else w['input_lengths'][i]+min(step, n-1), 0)
                                        for i, n in enumerate(w['output_lengths'])))
            prefix = f'infer.d{d}.s{step}'
            before = getattr(b, '_inference_round_before', None)
            if before:
                deps = before(prefix, item, deps)
            receives, residuals = [], []
            for stage in range(mesh.pp):
                for layer in b.stage_layers[stage]:
                    name = prefix+f'.l{layer}'
                    run = GroupedLayer(b, batch, d, stage, layer, name, current)
                    allocations = []
                    for slot, ranks in enumerate(run.tp_groups):
                        for rank in ranks:
                            allocations.extend(state(committed, slot, rank, layer, name, step))
                            block = attnres_state(b.m, layer, ceil(run.tokens[slot]/mesh.tp) if b.sp else run.tokens[slot])
                            if block['new_elements']:
                                a = b.allocation(name+f'.{rank}.attnres_block', rank,
                                    block['new_elements']*DTYPE_BYTES[b.p['activation']], 'AttnRes', 'attnres_residual', layer, b.p['activation'])
                                allocations.append(a)
                                residuals.append(a.id)
                    deps = [b.barrier(name+'.state', deps, allocations)]
                    deps = run.execute(deps)
                    deps = [b.barrier(name+'.layer_done', deps, releases=receives)]
                    receives = []
                if stage+1 < mesh.pp:
                    transfers = []
                    for i, (rank, dest) in enumerate(zip(mesh.ranks(d, stage), mesh.ranks(d, stage+1))):
                        n = batch.tokens(i//mesh.tp)
                        if n:
                            end, received = b.pipeline_transfer(name+f'.{rank}.pp', rank, dest, n*b.m['hidden_dim'], deps)
                            transfers.append(end)
                            receives.append(received)
                    deps = transfers or deps
            after = getattr(b, '_inference_round_after', None)
            if after:
                deps = after(prefix, item, deps)
            end = b.barrier(f'infer.d{d}.token{step}', deps, releases=residuals, phase='token_complete', token_index=step)
            tokens = item['committed_tokens'] if item else 1
            token_ends.extend([end]*tokens)
            for i in active:
                sequence_ends[str(i)].extend([end]*tokens)
            deps = [end]
        b.g.metadata['token_completion_events'][str(d)] = token_ends
        b.g.metadata['sequence_token_completion_events'][str(d)] = sequence_ends
    return b.g


def grouped_training_layer(b, batch, stage, layer, name, phase, deps, step, micro):
    terminals, release, output = [], [], []
    for d in range(b.dp):
        key = (step, micro, d, layer)
        prefix = f'{name}.d{d}' if b.dp > 1 else name
        run = GroupedLayer(b, batch, d, stage, layer, prefix, phase)
        previous = deps
        if phase == 'forward' and b.a.get('attnres_block_size'):
            residuals = []
            for i, group in enumerate(run.tp_groups):
                elems = attnres_state(b.m, layer, batch.tokens(i))['new_elements']
                if elems:
                    for rank in group:
                        a = b.allocation(f'{prefix}.{rank}.attnres', rank, elems*DTYPE_BYTES[b.p['activation']],
                                         'AttnRes', 'attnres_residual', layer, b.p['activation'])
                        residuals.append(a)
                        b.residual_blocks[step, micro, rank].append(a.id)
            if residuals:
                previous = [b.barrier(prefix+'.residuals', previous, residuals)]
        if phase == 'backward' and b.f['activation_checkpointing']:
            replay = GroupedLayer(b, batch, d, stage, layer, prefix+'.recompute', 'recompute')
            previous = replay.execute(previous)
            release.extend(replay.retained)
        terminals.extend(run.execute(previous))
        if phase == 'forward':
            b.saved[key] = run.retained
            output.extend(run.output_retained)
        else:
            release.extend(b.saved.pop(key))
    return list(dict.fromkeys(terminals)), release, output

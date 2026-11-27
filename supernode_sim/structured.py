"""Component/tensor-shape execution path for explicit hybrid architectures.

Storage is tensor-derived. Kernel timing, balanced expert placement and Muon
workspace remain explicit architecture models until calibrated against traces.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import ceil, prod

from .components import (parameter_inventory, inventory_summary, optimizer_states,
                         component_activations, attention_state, attention_training_state,
                         attnres_state)
from .core import Allocation, DTYPE_BYTES, Event, Graph, Resource
from .training import pipeline_schedule
from .training_schedule import gather_dependencies, prefetch_depth, replica_all_reduce
from .sparse_attention import sparse_attention_state, sparse_attention_work, position_bias_rows
from .moe_wire import (aligned_expert_factor, moe_payload_matrix, moe_wire_metadata, training_routed_flops,
                       training_routing_factors, wire_payload_parts)
from .training_attention import causal_pairs
from .calibration import model_shape_signature
from .kernel_workload import kernel_workload_shape
from .rank_layout import RankLayout, independent_groups
from .fsdp import TrainingShards, describe_domains


class Builder:
    def __init__(self, config, topology):
        self.c, self.topo = config, topology
        self.m, self.h, self.w, self.x = (config[k] for k in ('model', 'hardware', 'workload', 'execution'))
        self.p, self.f, self.opt = self.x['precision'], self.x['features'], self.x['optimizer']
        self.a = self.m['architecture']
        self.model_shape_signature = model_shape_signature(self.m)
        self.tp, self.pp, self.ep, self.dp = (self.x[k] for k in ('tp', 'pp', 'ep', 'dp'))
        self.grouped = independent_groups(self.x) or self.x.get('fsdp_domain') is not None
        self.mesh = RankLayout.resolve(self.x, len(self.x['rank_order']) or self.h['npu_count'])
        self.lanes = self.mesh.attention_dp * self.mesh.cp
        self.stage_groups = {}
        self.sp = self.x.get('sequence_parallel', False)
        self.fsdp = self.x['fsdp']
        self.training = self.w['scenario'] == 'training'
        self.routing = self.x.get('training_moe_routing') if self.training else None
        self.wire_factor, self.expert_factor, self.accepted_factor = training_routing_factors(self.routing)
        self.windows = self.x.get('training_attention_windows') if self.training else None
        self.active_kv = self.x.get('training_active_kv_heads') if self.training else None
        self.forward_only = False
        self.train_lengths = [self.w['sequence_length']] * self.w['batch_size']
        self.g = Graph()
        topology.populate(self.g, training_hbm_fraction=self.x['training_hbm_fraction'] if self.training else None)
        self.tensors = parameter_inventory(self.m)
        if self.a is None:
            from .grouped_layers import budget_layers
            self.a, self.tensors = budget_layers(self.m)
        self.by_layer = {l: [t for t in self.tensors if t.layer == l] for l in range(self.m['layers'])}
        self.shards = TrainingShards(self) if self.x.get('fsdp_domain') is not None else None
        self.sharded_training = (any(bucket.domain.degree > 1 for buckets in self.shards.layers.values() for bucket in buckets)
                                 if self.shards else self.fsdp > 1)
        if self.shards:
            self.g.metadata['fsdp_domains'] = describe_domains(self.x, self.shards.domains)
        order = self.x['rank_order'] or list(range(self.h['npu_count']))
        self.ranks = {coord: f'npu{order[i]}' for i, coord in enumerate(product(range(self.dp), range(self.pp), range(self.lanes), range(self.tp)))}
        self.coords = {rank: coord for coord, rank in self.ranks.items()}
        self.stage_layers = {s: [l for l in range(self.m['layers']) if l * self.pp // self.m['layers'] == s] for s in range(self.pp)}
        self.saved = {}
        self.grad_buffers = {}
        self.retained_gathers = {}
        self.attention_tp_outputs = {}
        self.cache = {}
        self.residual_blocks = defaultdict(list)
        self.pipeline_receives = {}
        self.g.metadata.update(component_inventory=inventory_summary(self.m), logical_component_state_bytes={},
            unsupported_production_features=['interleaved_pipeline', 'pipeline_zero2_schedule',
                'activation_offload', 'distributed_muon_orthogonalization'],
            scenario=self.w['scenario'], assumptions=[
                'Parameter and state capacities derive from named tensor shapes, explicit tying and TP/EP/FSDP placement.',
                'Activation formulas assume fused normalization, SiLU/SiTU GLU and streaming attention; actual allocator/kernel traces may differ.',
                'Expert routing and token distribution are balanced expectations; expert skew and capacity padding need workload calibration.',
                'Per-component kernels use roofline timing; pipeline follows non-interleaved 1F1B in training and fixed ragged batch in inference.',
                'KV offload applies to growing attention KV; KDA recurrent matrices and convolution histories stay in device memory.',
                'Muon keeps one matrix momentum state; vectors/norms/embeddings/output use Adam m/v fallback. Newton-Schulz time and scratch are estimates.',
                'FP8/FP4 weight scales use per-tensor block rounding; precision overrides are storage studies unless identified as a released model format.',
                'Ragged inference stops compute and KV reads for finished sequences; their allocated caches remain reserved through the fixed batch.',
                'Pipeline transfers materialize source and receive buffers; the receive is retained through the consuming stage pass.',
                'Prefill projects logits only for the final input position of each live sequence; training projects all sequence positions.',
                'Muon orthogonalization reserves full per-matrix/per-head scratch. Distributed matrix assembly/orthogonalization is not implemented; sharded Muon timing is a proxy, not production validation.',
            ])
        if self.a.get('attnres_block_size'):
            self.g.metadata['assumptions'].append('AttnRes retains residual blocks across layers; FP32 reference attention workspace is included. PP>1 is rejected until residual forwarding is modeled.')
        if any(kind in ('CSA', 'HCA') for kind in self.a['attention_by_layer']):
            self.g.metadata['assumptions'].extend([
                'CSA/HCA model sliding-window/shared latent KV, gated compression and CSA indexing; these components do not reproduce a complete DeepSeek V4 model or mHC.',
                'Sparse latent/index pools are replicated across TP and grow on demand; rolling-window and FP32 compression state reserve per-request buffers. No L2 hit ratio is assumed.',
                'CSA index-score TP all-reduce is a serialized cost at the component boundary, not an exact intra-kernel dependency trace. Top-k operation throughput uses configured FP32 throughput as an analytical scalar proxy.',
                'CSA/HCA training uses the generic component backward multiplier and saved-state model; indexer auxiliary training loss and exact gradient routing are not modeled.',
            ])
        self.g.metadata['sequence_parallel'] = {
            'enabled': self.sp, 'degree': self.tp if self.sp else 1,
            'world_axis': 'shared_with_tp',
            'padding_policy': 'ceil(valid_source_tokens / TP) rows per rank; padding is excluded from attention/FFN kernels and KV state',
        }
        if self.sp:
            self.g.metadata['assumptions'].extend([
                'Megatron-style sequence parallelism shares TP ranks: norms/AttnRes/residuals store sequence shards, attention/FFN gather valid sequences and retain TP head/matrix sharding.',
                'Equal-chunk SP collectives pad each EP source to a TP multiple, including single-token decode and selected final output rows; pointwise kernels process padded local rows, attention/FFN and KV process valid rows only.',
                'SP uses a TP all-gather before attention/FFN and TP reduce-scatter afterwards; gathered full-hidden input remains allocated through the component. Embedding enters SP by reduce-scatter; output gathers selected positions before vocabulary-sharded projection.',
                'SP MoE retains full latent dispatch/combine independently on each TP lane; fused DeepEP dispatch and cross-lane deduplication are not modeled.',
            ])
        if self.grouped:
            self.g.metadata['parallel_groups'] = self.mesh.describe()
            self.g.metadata['assumptions'].append(
                'Whole requests are assigned round-robin to attention replicas. CP owns cyclic context positions. '
                'Dense attention exchanges KV rows for local queries. KDA and sparse attention gather to the first CP '
                'rank of each TP lane. Routed experts use independent EP/TP groups and uniform expected routing.')

    def pool(self, node):
        return node + ('.hbm' if node.startswith('npu') else '.dram' if node.startswith('cpu') else '.memory')

    def host(self, rank):
        return self.topo.host_for(int(rank[3:]))

    def add(self, event):
        self.g.events.append(event)
        return event.id

    def barrier(self, name, deps=(), allocs=(), releases=(), **metadata):
        return self.add(Event(name, tuple(dict.fromkeys(deps)), allocs=tuple(allocs), releases=tuple(releases), metadata=metadata))

    def weight_dtype(self, tensor):
        return (self.p.get('routed_expert_weight') if tensor.ep_shard else None) or self.p['weight']

    def allocation(self, name, node, size, component, state, layer=None, dtype=None, **metadata):
        return Allocation(name, self.pool(node), float(size), state, dtype,
                          metadata={'component': component, 'state': state, 'layer': layer,
                                    'tensor_name': metadata.pop('tensor_name', name), **metadata})

    def scale(self, elements, dtype):
        return ceil(elements / self.f['weight_scale_block']) * self.f['scale_bytes'] if dtype in ('fp4', 'fp8', 'int8') else 0

    def scalar(self, node):
        if not node.startswith('npu'):
            return node + '.compute'
        key = node + '.compute.fp32'
        rate = self.h['npu']['flops'].get('fp32', self.h['npu']['flops'][self.p['compute']]) * self.h['npu']['efficiency']
        self.g.resources[key] = Resource(key, rate, 'FLOP/s', 'compute')
        return key

    def groups(self, stage, axis):
        if axis in ('tp', 'ep', 'cp', 'attention_dp', 'expert_tp', 'expert_dp'):
            name = 'attention_tp' if axis == 'tp' else axis
            return [group for d in range(self.dp) for group in self.layer_groups(d, stage)[name]]
        if self.shards:
            family = 'expert' if axis.startswith('expert_') else 'dense'
            domain = self.shards.domains[family]
            return (domain.shards if axis.endswith('fsdp') else domain.replicas)[stage]
        if axis == 'fsdp':
            return [[self.ranks[d, stage, e, t] for d in range(start, start + self.fsdp)] for start, e, t in product(range(0, self.dp, self.fsdp), range(self.lanes), range(self.tp))]
        return [[self.ranks[d, stage, e, t] for d in range(offset, self.dp, self.fsdp)] for offset, e, t in product(range(self.fsdp), range(self.lanes), range(self.tp))]

    def local_elements(self, tensor, fsdp=1):
        return tensor.local_elements(self.mesh.expert_tp if tensor.ep_shard else self.tp, self.ep, fsdp)

    def resident_elements(self, tensor, rank):
        if self.shards:
            return self.shards.owned_elements(tensor, rank) + self.shards.padding_elements(tensor, rank)
        return self.local_elements(tensor, self.fsdp)

    def layer_groups(self, replica, stage):
        key = replica, stage
        if key not in self.stage_groups:
            self.stage_groups[key] = self.mesh.groups(replica, stage)
        return self.stage_groups[key]

    def collective(self, prefix, kind, ranks, size, deps, component, layer, matrix=None, activation=False,
                   *, phase=None, wire_metadata=None):
        meta = {'component': component, 'layer': layer, 'operation': 'dispatch' if 'dispatch' in prefix else 'combine' if 'combine' in prefix else kind}
        meta.update(wire_metadata or {})
        if phase is not None:
            meta['phase'] = phase
        wire = meta.get('wire_dtype', self.p['communication'])
        moe = component == 'MoE' and meta['operation'] in ('dispatch', 'combine')
        original_deps = tuple(deps)
        algorithm = self.x.get('all_to_all_algorithm', 'pairwise')
        outer = moe
        if moe:
            selection = self.topo.select_collective_timing_scope(kind, ranks, size, matrix, meta, algorithm)
            meta['collective_scope_selection'] = selection
            outer = selection['selected_scope'] == 'activation_collective'

        def finish(terminals):
            if not outer:
                return list(terminals)
            return list(self.topo.complete_collective_profile(self.g, prefix, kind, ranks, size,
                deps=original_deps, terminals=tuple(terminals), payload_matrix=matrix,
                metadata=meta, algorithm=algorithm, timing_scope='activation_collective'))

        packed = []
        buffers = []
        if activation and self.p['activation'] != wire and len(ranks) > 1:
            for i, rank in enumerate(ranks):
                payload = sum(matrix[i]) if matrix is not None else size / len(ranks) if kind == 'all_gather' else size
                aid = f'{prefix}.{rank}.wire'
                buffers.append(aid)
                amount, extra = wire_payload_parts(payload, wire, meta)
                packed.append(self.add(Event(aid + '.pack', tuple(deps),
                    {self.scalar(rank): 2 * amount, rank + '.hbm': amount * DTYPE_BYTES[self.p['activation']] + payload + extra},
                    slots=(rank + '.compute_stream',), allocs=(self.allocation(aid, rank, payload, component, 'communication_buffer', layer, wire),),
                    metadata={**meta, 'kernel': 'communication_pack', 'tensor_elements': amount,
                              'wire_metadata_bytes': extra, 'source_precision': self.p['activation'], 'destination_precision': wire})))
            deps = packed
        terminals = self.topo.collective(self.g, prefix, kind, ranks, size, tuple(deps), payload_matrix=matrix,
            metadata=meta, algorithm=algorithm, apply_profiles=not outer)
        if buffers:
            ends = []
            for i, rank in enumerate(ranks):
                payload = sum(row[i] for row in matrix) if matrix is not None else size / len(ranks) if kind == 'reduce_scatter' else size
                count, extra = wire_payload_parts(payload, wire, meta)
                ends.append(self.add(Event(prefix + f'.{rank}.unpack', tuple(terminals),
                    {self.scalar(rank): 2 * count, rank + '.hbm': payload + count * DTYPE_BYTES[self.p['activation']] + extra},
                    slots=(rank + '.compute_stream',), releases=(buffers[i],), metadata={**meta, 'kernel': 'communication_unpack',
                        'tensor_elements': count, 'wire_metadata_bytes': extra,
                        'source_precision': wire, 'destination_precision': self.p['activation']})))
            return finish(ends)
        return finish(terminals)

    def local_tensors(self, layer, component=None):
        return [t for t in self.by_layer[layer] if not t.alias_of and (component is None or t.component == component)]

    def states(self, tensor):
        return ['weight'] + (['gradient'] + optimizer_states(tensor, self.opt['kind']) if self.training and not self.forward_only else [])

    def state_dtype(self, state, tensor):
        return self.weight_dtype(tensor) if state == 'weight' else self.p['gradient'] if state == 'gradient' else self.p['master'] if state == 'master_weight' else self.p['optimizer']

    def initial(self):
        logical = defaultdict(lambda: defaultdict(float))
        for tensor in self.tensors:
            component = tensor.component if self.m['architecture'] else 'Unassigned'
            for state in self.states(tensor) if not tensor.alias_of else []:
                logical[component][state] += tensor.accounted_elements * DTYPE_BYTES[self.state_dtype(state, tensor)]
        self.g.metadata['logical_component_state_bytes'] = {k: dict(v) for k, v in logical.items()}
        totals = defaultdict(float)
        for states in logical.values():
            for state, amount in states.items():
                totals[state] += amount
        totals['total'] = sum(totals.values())
        self.g.metadata['logical_memory_bytes'] = dict(totals)
        for rank, (d, stage, e, t) in self.coords.items():
            grouped = {}
            for layer in self.stage_layers[stage]:
                for tensor in self.local_tensors(layer):
                    storage = self.resident_elements(tensor, rank)
                    padding = self.shards.padding_elements(tensor, rank) if self.shards else 0
                    elements = storage - padding
                    component = tensor.component if self.m['architecture'] else 'Unassigned'
                    for state in self.states(tensor):
                        dtype = self.state_dtype(state, tensor)
                        node = self.host(rank) if self.f['optimizer_offload'] and state in ('master_weight', 'optimizer_momentum', 'optimizer_variance', 'muon_momentum') else rank
                        key = layer, component, state, dtype, node, self.grouped and tensor.ep_shard
                        size, names = grouped.setdefault(key, [0.0, []])
                        grouped[key][0] += elements * DTYPE_BYTES[dtype]
                        names.append(tensor.name)
                        if padding:
                            aid = f'fsdp_pad.{rank}.{tensor.name}.{state}'
                            self.g.initial_allocations.append(self.allocation(aid, node, padding*DTYPE_BYTES[dtype],
                                component, 'fsdp_flat_padding', layer, dtype, padding_for=state, owner_rank=rank))
                    for state, dtype, size in [('quantization_metadata', None, self.scale(storage, self.weight_dtype(tensor))),
                                               ('extra_weight_copy', self.f['extra_weight_copy_dtype'], storage * DTYPE_BYTES[self.f['extra_weight_copy_dtype']] if self.f['extra_weight_copy_dtype'] else 0)]:
                        if size:
                            key = layer, component, state, dtype, rank, self.grouped and tensor.ep_shard
                            grouped.setdefault(key, [0.0, []])[0] += size
                            grouped[key][1].append(tensor.name)
            for i, ((layer, component, state, dtype, node, routed), (size, names)) in enumerate(grouped.items()):
                domain = self.shards.domains['expert' if routed else 'dense'] if self.shards else None
                self.g.initial_allocations.append(self.allocation(f'state.{rank}.{i}', node, size, component, state, layer, dtype,
                    tensor_name=f'layers.{layer}.{component}.{state}', tensor_members=names, owner_rank=rank,
                    shard={'tp': self.mesh.expert_tp if routed else self.tp,
                           'ep': self.ep if routed or not self.grouped else 1, 'fsdp': domain.degree if domain else self.fsdp,
                           **({'fsdp_index': domain.index[rank], 'fsdp_family': 'expert' if routed else 'dense'} if domain else {}),
                           **({'group': 'expert' if routed else 'attention', 'cp': self.mesh.cp} if self.grouped else {})}, replica=d))
            for state, field in [('runtime', 'runtime_bytes'), ('workspace', 'workspace_bytes')]:
                size = self.h['npu'][field]
                if size:
                    self.g.initial_allocations.append(self.allocation(f'{rank}.{state}', rank, size, 'System', state))
        if self.f['engram']['enabled']:
            engram = self.f['engram']
            nodes = {self.host(r) for r in self.coords} if engram['location'] == 'host' else {'remote0'}
            for node in sorted(nodes):
                self.g.initial_allocations.append(self.allocation(node + '.engram_table', node, engram['table_bytes'], 'Engram', 'lookup_table'))
                self.g.resources[node + '.engram_iops'] = Resource(node + '.engram_iops', engram['iops'], 'operation/s', 'iops')
            for rank in self.coords:
                if engram['cache_bytes']:
                    self.g.initial_allocations.append(self.allocation(rank + '.engram_cache', rank, engram['cache_bytes'], 'Engram', 'lookup_cache'))

    def component_order(self, layer):
        if not self.m['architecture']:
            return [self.a['attention_by_layer'][layer]] + (['MoE'] if self.a['ffn_by_layer'][layer] == 'MoE' else [])
        return ((['WordEmbedding'] if layer == 0 else []) +
                (['AttnRes'] if self.a.get('attnres_block_size') else []) +
                ['PreNorm', self.a['attention_by_layer'][layer], 'PostNorm', self.a['ffn_by_layer'][layer]] +
                (['Output'] if layer == self.m['layers'] - 1 else []))

    def gather(self, name, layer, stage, deps, phase):
        if self.shards:
            return self.shards.gather(self, name, layer, stage, deps, phase)
        allocations = []
        retained = []
        sizes = {}
        for tensor in self.local_tensors(layer):
            elements = self.local_elements(tensor)
            dtype = self.weight_dtype(tensor)
            sizes[tensor.component] = sizes.get(tensor.component, 0) + elements * DTYPE_BYTES[dtype] + self.scale(elements, dtype)
        payload = sum(sizes.values())
        for rank, coord in self.coords.items():
            if coord[1] != stage:
                continue
            for component, amount in sizes.items():
                if amount:
                    aid = name + f'.{rank}.{component}.gathered'
                    allocations.append(self.allocation(aid, rank, amount, component, 'gathered_weight', layer))
                    retained.append(aid)
        ready = self.barrier(name + '.gather_allocate', deps, allocations)
        ends = []
        for i, ranks in enumerate(self.groups(stage, 'fsdp')):
            ends += self.collective(name + f'.gather{i}', 'all_gather', ranks, payload, (ready,), 'System', layer,
                phase=phase, wire_metadata={'parallel_axis': 'fsdp', 'category': 'fsdp_weights'})
        return self.barrier(name + '.gather_done', ends), retained

    def kernel_elements(self, tensor):
        if tensor.alias_of:
            return tensor.elements / self.tp
        count = self.local_elements(tensor)
        if self.active_kv and tensor.component == 'GQA' and tensor.tp_max_shards:
            heads = self.active_kv[tensor.layer] or self.m['kv_heads']
            count *= max(heads, self.tp) / max(self.m['kv_heads'], self.tp)
        return count

    def executed_expert_factor(self, local_tokens):
        return aligned_expert_factor(local_tokens * self.ep * self.m['top_k'], self.m['num_experts'],
                                     self.expert_factor, self.routing.get('expert_row_multiple'))

    def activation_rows(self, layer, tokens, *, checkpoint=False, training=None):
        model = self.m
        if self.active_kv and self.active_kv[layer] and training is not False:
            model = {**model, 'kv_heads': self.active_kv[layer]}
        rows = component_activations(model, layer, tokens, tp=self.tp, ep=self.ep,
                                    training=self.training if training is None else training, checkpointing=checkpoint)
        if not self.m['architecture']:
            rows = [{**row, 'component': self.a['attention_by_layer'][layer]} for row in rows]
            if self.a['ffn_by_layer'][layer] == 'MoE':
                rows.append({'component': 'MoE', 'name': 'routed_gate_up', 'elements':
                    tokens * model['top_k'] * model['intermediate_dim'] / self.tp,
                    'formula': 'routed intermediate budget per tensor shard'})
        if self.mesh.expert_tp != self.tp:
            for row in rows:
                if row['name'] == 'routed_gate_up':
                    row['elements'] *= self.tp / self.mesh.expert_tp
                    row['formula'] = 'routed intermediate elements per expert TP shard'
        if self.routing and training is not False:
            for row in rows:
                if row['name'] == 'routed_gate_up':
                    row['elements'] *= self.executed_expert_factor(tokens)
                    row['formula'] += ' * executed expert rows / requested rows'
        if self.training and self.x['flash_attention_blocks']:
            for row in rows:
                if row['name'] == 'softmax_lse' and row['component'] in ('MHA', 'GQA', 'MQA', 'MLA'):
                    row['dtype'] = 'fp32'
        return rows

    def inference_state_allocations(self, prefix, rank, layer, lengths, step):
        """Grow persistent committed state; speculative candidate buffers are separate."""
        kind = self.a['attention_by_layer'][layer]
        state = (sparse_attention_state(self.m, layer, lengths, tp=self.tp) if kind in ('CSA', 'HCA')
                 else attention_state(self.m, layer, len(lengths), max(lengths, default=0), tp=self.tp))
        component = state['component']
        allocations = []
        if state['kind'] == 'recurrent':
            if step == 0:
                for category, elements, dtype in [('recurrent_state', state['recurrent_elements'], self.p['recurrent']),
                                                   ('convolution_state', state['convolution_elements'], self.p['activation'])]:
                    if elements:
                        allocations.append(self.allocation(prefix + f'.{rank}.{category}', rank,
                            elements * DTYPE_BYTES[dtype], component, category, layer, dtype))
            return allocations
        if state['kind'] == 'compressed_sparse':
            # Latent and index pools are replicated across TP. FP32 compressor
            # history is distinct from the configured KV dtype and metadata.
            values = {key: (state[key] * DTYPE_BYTES[self.p['kv']], self.p['kv'])
                      for key in state['storage_categories']['kv_dtype']}
            values.update({key: (state[key] * 4, 'fp32') for key in state['storage_categories']['fp32']})
            if self.p['kv'] in ('fp8', 'int8'):
                values['kv_scales'] = (sum(ceil(state[key] / self.f['kv_scale_block']) for key in
                    state['storage_categories']['kv_dtype']) * self.f['scale_bytes'], None)
            for category, (size, dtype) in values.items():
                key = rank, layer, category
                growth = max(0, size - self.cache.get(key, 0))
                self.cache[key] = max(size, self.cache.get(key, 0))
                if growth:
                    allocations.append(self.allocation(prefix + f'.{rank}.{category}', rank, growth,
                        component, category, layer, dtype, formula=state['formula']))
            return allocations
        pages = sum(ceil(length / self.f['kv_page_tokens']) for length in lengths)
        elements = pages * self.f['kv_page_tokens'] * state['elements_per_token']
        values = {'kv_cache': elements * DTYPE_BYTES[self.p['kv']],
                  'kv_scales': ceil(elements / self.f['kv_scale_block']) * self.f['scale_bytes'] if self.p['kv'] in ('fp4', 'fp8', 'int8') else 0,
                  'page_table': pages * self.f['kv_page_entry_bytes']}
        for category, size in values.items():
            key = rank, layer, category
            growth = max(0, size - self.cache.get(key, 0))
            self.cache[key] = max(size, self.cache.get(key, 0))
            fraction = self.f['kv_offload_fraction'] if self.f['kv_offload'] != 'none' else 0
            store = self.host(rank) if self.f['kv_offload'] == 'host' else 'remote0'
            for part, node, suffix in [(1 - fraction, rank, 'local'), (fraction, store, 'offload')]:
                if growth * part:
                    allocations.append(self.allocation(prefix + f'.{rank}.{category}.{suffix}', node,
                        growth * part, component, category, layer, self.p['kv'] if category == 'kv_cache' else None))
        return allocations

    def sparse_index_collective(self, prefix, layer, groups, lengths, deps, phase,
                                query_lengths=None, divisor=1):
        """Account for CSA index-score TP reduction in the component schedule.

        Compute is aggregated per component, so this is a serialized communication
        cost, not the exact intra-kernel order of index projection and attention.
        """
        if self.a['attention_by_layer'][layer] != 'CSA' or self.tp == 1:
            return list(deps)
        work = sparse_attention_work(self.m, layer, lengths,
            phase='prefill' if phase in ('forward', 'backward', 'recompute') else phase,
            tp=self.tp, query_lengths=query_lengths)
        amount = work['indexer_reduce_elements'] * DTYPE_BYTES[self.p['compute']] / divisor
        if not amount:
            return list(deps)
        terminals = []
        for i, group in enumerate(groups):
            terminals += list(self.topo.collective(self.g, prefix + f'.index_scores.tp{i}', 'all_reduce', group,
                amount, tuple(deps), metadata={'component': 'CSA', 'layer': layer,
                    'operation': 'index_score_reduce', 'buffer_dtype': self.p['compute']}))
        return terminals or list(deps)

    def kernel(self, name, rank, layer, component, tokens, deps, *, backward=False, phase='prefill', total_tokens=None, allocations=(), releases=(), lengths=None, query_lengths=None, local_work=False, part=None, causal_work=None):
        tensors = self.local_tensors(layer, component)
        if part in ('routed', 'shared', 'latent_up'):
            tensors = [t for t in tensors if t.ep_shard == (part == 'routed')
                       and ('.moe.latent_up.' in t.name) == (part == 'latent_up')]
        elif part in ('qkv', 'context'):
            tensors = [t for t in tensors if ('.output.' in t.name) == (part == 'context')]
        divisor = self.ep if self.training and not local_work else 1
        total_tokens = tokens * self.ep if total_tokens is None else total_tokens
        # Tied output still performs projection, without owning another tensor.
        if component == 'Output' and not tensors:
            tensors = [t for t in self.by_layer[layer] if t.component == 'Output']
        active_elements = routed_elements = 0.0
        projection_elements = 0.0
        weight_reads = 0.0
        conversion_elements = 0.0
        for tensor in tensors:
            elements = self.kernel_elements(tensor)
            touched = 1 - (1 - self.m['top_k'] / self.m['num_experts']) ** total_tokens if tensor.ep_shard else 1.0
            active = tensor.active_fraction * self.ep if tensor.ep_shard else (1 if self.m['architecture'] else tensor.active_fraction)
            active_elements += elements * active
            if tensor.ep_shard:
                routed_elements += elements * active
            projection_elements += elements * active if tensor.role == 'matrix' else 0
            if component in ('CSA', 'HCA') and tensor.name.endswith('.position_bias'):
                a = self.a[component.lower()]
                queries = query_lengths if query_lengths is not None else (
                    [1] * len(lengths or []) if phase == 'decode' else list(lengths or []))
                touched *= position_bias_rows(lengths or [], queries, a['compress_ratio']) / a['compress_ratio']
            if self.f.get('extra_weight_copy_dtype') == self.p['compute']:
                weight_reads += elements * touched * DTYPE_BYTES[self.p['compute']]
            else:
                weight_reads += elements * touched * DTYPE_BYTES[self.weight_dtype(tensor)] + self.scale(elements * touched, self.weight_dtype(tensor))
            if self.weight_dtype(tensor) != self.p['compute']:
                conversion_elements += elements * touched
        flops = 2 * (projection_elements if component in ('CSA', 'HCA') else active_elements) * tokens
        if component == 'WordEmbedding':
            flops = tokens * self.m['hidden_dim'] / self.tp
            weight_reads = tokens * self.m['hidden_dim'] * DTYPE_BYTES[self.p['weight']] / self.tp
            conversion_elements = 0
        if component in ('PreNorm', 'PostNorm'):
            flops = 5 * tokens * sum(t.elements for t in tensors if t.role == 'norm')
        sparse = None
        window = self.windows[layer] if self.windows and self.training else None
        if component in ('CSA', 'HCA'):
            sparse = sparse_attention_work(self.m, layer, lengths or [],
                phase='prefill' if phase in ('forward', 'backward', 'recompute') else phase,
                tp=self.tp, query_lengths=query_lengths)
            flops += sparse['extra_flops'] / divisor
        elif component == 'KDA':
            a = self.a['kda']
            flops += 8 * tokens * a['num_heads'] * a['head_k_dim'] * a['head_v_dim'] / self.tp
        elif component in ('MLA', 'GQA', 'MHA', 'MQA'):
            if phase == 'verify':
                if query_lengths is None or len(query_lengths) != len(lengths or []):
                    raise ValueError('Verification attention requires per-request query lengths')
                pairs = sum(q * (s - q) + q * (q + 1) / 2 for s, q in zip(lengths, query_lengths))
            elif window is not None:
                pairs = sum(causal_pairs(s, window) for s in lengths or [])
            elif phase in ('prefill', 'forward', 'backward', 'recompute'):
                pairs = sum(s * (s + 1) / 2 for s in lengths or [])
            else:
                pairs = sum(lengths or [])
            pairs = (pairs / divisor if causal_work is None else causal_work)
            qk_dim = self.a['mla']['qk_nope_head_dim'] + self.a['mla']['qk_rope_head_dim'] if component == 'MLA' else self.m['head_dim']
            v_dim = self.a['mla']['v_head_dim'] if component == 'MLA' else self.m['head_dim']
            flops += 2 * self.m['num_heads'] * (qk_dim + v_dim) * pairs / self.tp
        elif component == 'AttnRes':
            block = attnres_state(self.m, layer, tokens)
            flops += 10 * block['stored_elements']
        if backward:
            flops *= 2
        routing_work = {}
        if self.routing and component == 'MoE' and part in (None, 'routed') and phase in ('forward', 'backward', 'recompute'):
            requested = 2 * routed_elements * tokens * (2 if backward else 1)
            expert_factor = self.executed_expert_factor(tokens)
            flops += requested * (expert_factor - 1)
            routing_work = training_routed_flops(requested, expert_factor, self.accepted_factor)
        if sparse is not None and sparse['indexer_score_workspace_elements'] and not backward:
            workspace = self.allocation(name + '.index_scores', rank,
                sparse['indexer_score_workspace_elements'] * DTYPE_BYTES[self.p['compute']] / divisor,
                component, 'indexer_score_workspace', layer, self.p['compute'],
                formula='reference indexer pre-mask scoring matrix; local heads plus reduced scores')
            allocations = tuple(allocations) + (workspace,)
            releases = tuple(releases) + (workspace.id,)
        working = sum(a.size_bytes for a in allocations)
        pending = list(deps)
        conversion_id = None
        if conversion_elements and self.f.get('extra_weight_copy_dtype') != self.p['compute']:
            conversion_id = name + '.compute_weights'
            size = conversion_elements * DTYPE_BYTES[self.p['compute']]
            pending = [self.add(Event(name + '.weight_conversion', tuple(pending),
                {self.scalar(rank): 2 * conversion_elements, rank + '.hbm': size + weight_reads},
                slots=(rank + '.compute_stream',), allocs=(self.allocation(conversion_id, rank, size, component, 'conversion_buffer', layer, self.p['compute']),),
                metadata={'kernel': 'weight_conversion', 'component': component, 'layer': layer, 'phase': phase}))]
            weight_reads = size
        cache_reads = 0
        if component in ('MLA', 'GQA', 'MHA', 'MQA', 'KDA') and lengths:
            state = attention_state(self.m, layer, len(lengths), max(lengths), tp=self.tp)
            if state['kind'] == 'recurrent':
                cache_reads = state['recurrent_elements'] * DTYPE_BYTES[self.p['recurrent']] + state['convolution_elements'] * DTYPE_BYTES[self.p['activation']]
                cache_reads /= divisor
            elif phase in ('decode', 'verify'):
                cache_reads = sum(lengths) * state['elements_per_token'] * DTYPE_BYTES[self.p['kv']]
        cache_writes = 0
        if not self.training and component in ('MLA', 'GQA', 'MHA', 'MQA', 'KDA'):
            state = attention_state(self.m, layer, len(lengths or []), max(lengths or [0]), tp=self.tp)
            if state['kind'] == 'recurrent':
                cache_writes = state['recurrent_elements'] * DTYPE_BYTES[self.p['recurrent']] + state['convolution_elements'] * DTYPE_BYTES[self.p['activation']]
            else:
                cache_writes = tokens * state['elements_per_token'] * DTYPE_BYTES[self.p['compute']]
        if sparse is not None:
            cache_reads = ((sparse['attention_read_elements'] + sparse['indexer_read_elements']) * DTYPE_BYTES[self.p['kv']]
                           + (sparse['compression_read_elements'] + sparse['state_read_elements']) * 4) / divisor
            cache_writes = (sparse['cache_write_elements'] * DTYPE_BYTES[self.p['kv']]
                            + sparse['state_write_elements'] * 4) / divisor
        if part == 'qkv':
            cache_reads = 0
        elif part == 'context':
            cache_writes = 0
        work = {rank + '.compute': max(flops, 0), rank + '.hbm': weight_reads + 2 * working + cache_reads + cache_writes}
        if sparse is not None and sparse['selection_operations'] and not backward:
            selection = rank + '.selection'
            scalar_rate = self.h['npu']['flops'].get('fp32', self.h['npu']['flops'][self.p['compute']]) * self.h['npu']['efficiency']
            self.g.resources[selection] = Resource(selection, scalar_rate, 'operation/s', 'selection')
            work[selection] = sparse['selection_operations'] / divisor
        if backward:
            work[rank + '.hbm'] += active_elements * DTYPE_BYTES[self.p['gradient']]
        event = Event(name, tuple(pending), work, slots=(rank + '.compute_stream',),
            allocs=tuple(allocations), releases=tuple(releases) + ((conversion_id,) if conversion_id else ()),
            metadata={'kernel': component.lower() if part is None else f'{component.lower()}_{part}', 'component': component, 'layer': layer, 'phase': phase,
                      'flops': flops, 'working_set_bytes': weight_reads + working + cache_reads, **routing_work,
                      **({'attention_window': window} if window is not None and component in ('GQA', 'MHA', 'MQA', 'MLA') else {}),
                      **({'active_kv_heads': self.active_kv[layer]} if self.active_kv and self.active_kv[layer] and component == 'GQA' else {}),
                      'tokens': tokens, 'sequence_parallel': self.sp,
                      'workload_shape': kernel_workload_shape(phase=phase, tokens=tokens,
                          lengths=list(lengths or []), query_lengths=query_lengths,
                          training=self.training, ep=divisor),
                      'model_shape_signature': self.model_shape_signature,
                      'tensor_parallel_degree': self.mesh.expert_tp if part == 'routed' else self.tp, 'expert_parallel_degree': self.ep,
                       'sequence_parallel_degree': self.tp if self.sp else 1,
                       **({'sparse_attention': sparse, 'selection_rate_assumption': 'Scalar operation throughput proxied by configured FP32 FLOP/s; calibrate with kernel profiles.'} if sparse is not None else {})})
        if local_work:
            shape = event.metadata['workload_shape']
            shape['sequence_scope'] = 'expert_replica' if part == 'routed' else 'attention_replica_local_queries'
            if causal_work is not None:
                shape['dense_causal_pairs_reference'] = causal_work
            if part:
                event.metadata['model_kernel_part'] = part
            if self.mesh.cp > 1 and part != 'routed':
                cp_rank = self.coords[rank][2] % self.mesh.cp
                shape.update(context_parallel_rank=cp_rank, context_parallel_degree=self.mesh.cp)
                shape['sequence_signature'] += f':cp{cp_rank}/{self.mesh.cp}'
            event.metadata.update(context_parallel_degree=self.mesh.cp, expert_tensor_parallel_degree=self.mesh.expert_tp)
        if self.training and self.x['flash_attention_blocks'] and component in ('MLA', 'MHA', 'MQA', 'GQA'):
            from .training_attention import flash_attention
            parameter_bytes = weight_reads + (active_elements * DTYPE_BYTES[self.p['gradient']] if backward else 0)
            return flash_attention(self, event, rank, lengths, tensors, parameter_bytes)
        return self.add(event)

    def annotate_collective_allocations(self):
        for event in self.g.events:
            for allocation in event.allocs:
                allocation.metadata.setdefault('component', event.metadata.get('component', 'System'))
                allocation.metadata.setdefault('state', 'communication_buffer' if allocation.category == 'communication_buffer' else allocation.category)
                allocation.metadata.setdefault('layer', event.metadata.get('layer'))

    def pipeline_transfer(self, name, source, dest, elements, deps):
        """Both endpoints remain resident until transfer/consumption completes."""
        dtype, wire = self.p['activation'], self.p['communication']
        raw_bytes, wire_bytes = elements * DTYPE_BYTES[dtype], elements * DTYPE_BYTES[wire]
        raw = name + '.source'
        start = self.barrier(name + '.source_allocate', deps,
            [self.allocation(raw, source, raw_bytes, 'System', 'pipeline_buffer', dtype=dtype)])
        send = raw
        if dtype != wire:
            send = name + '.packed'
            start = self.add(Event(name + '.pack', (start,),
                {self.scalar(source): 2 * elements, source + '.hbm': raw_bytes + wire_bytes},
                slots=(source + '.compute_stream',),
                allocs=(self.allocation(send, source, wire_bytes, 'System', 'conversion_buffer', dtype=wire),),
                releases=(raw,), metadata={'component': 'System', 'operation': 'precision_conversion', 'kernel': 'pipeline_pack'}))
        received = name + '.receive'
        event = self.topo.transfer(name + '.transfer', source, dest, wire_bytes, (start,),
                                   {'component': 'System', 'operation': 'pipeline'})
        event.allocs = (self.allocation(received, dest, wire_bytes, 'System', 'pipeline_buffer', dtype=wire),)
        event.releases = (send,)
        end = self.add(event)
        if dtype != wire:
            decoded = name + '.decoded'
            end = self.add(Event(name + '.unpack', (end,),
                {self.scalar(dest): 2 * elements, dest + '.hbm': raw_bytes + wire_bytes},
                slots=(dest + '.compute_stream',),
                allocs=(self.allocation(decoded, dest, raw_bytes, 'System', 'pipeline_buffer', dtype=dtype),),
                releases=(received,), metadata={'component': 'System', 'operation': 'precision_conversion', 'kernel': 'pipeline_unpack'}))
            received = decoded
        return end, received

    def kv_bytes(self, elements):
        data = elements * DTYPE_BYTES[self.p['kv']]
        scales = (ceil(elements / self.f['kv_scale_block']) * self.f['scale_bytes']
                  if elements and self.p['kv'] in ('fp4', 'fp8', 'int8') else 0)
        return data + scales

    def training_tp_reduce(self, name, component, layer, stage, ranks, tokens, deps, phase):
        width = self.m['hidden_dim'] * DTYPE_BYTES[self.p['communication']]
        groups = self.groups(stage, 'tp')
        if deps[0] not in self.attention_tp_outputs:
            return [end for i, group in enumerate(groups)
                    for end in self.collective(name + f'.{component}.tp{i}', 'all_reduce', group,
                        tokens * width, deps, component, layer, activation=True, phase=phase)]
        chunks = {rank: self.attention_tp_outputs.pop(eid) for rank, eid in zip(ranks, deps)}
        terminals = list(deps)  # Keep parent releases and any final compute on the layer path.
        for i, group in enumerate(groups):
            previous = []
            for q, parts in enumerate(zip(*(chunks[rank] for rank in group))):
                projected, rows, buffers = zip(*parts)
                prefix = name + f'.{component}.tp{i}.q{q}'
                ends = self.collective(prefix, 'all_reduce', group, rows[0] * width,
                    [*projected, *previous], component, layer, activation=True, phase=phase,
                    wire_metadata={'parallel_axis': 'tp', 'query_block': q})
                previous = [self.barrier(prefix + '.done', ends, releases=buffers)]
            terminals.extend(previous)
        return terminals

    def training_ep(self, name, operation, layer, stage, tokens, deps, phase):
        width = self.a['moe'].get('latent_dim') or self.m['hidden_dim']
        top_k = self.m['top_k']
        matrix = moe_payload_matrix(self.p, operation, [tokens * self.wire_factor] * self.ep, width, top_k)
        metadata = moe_wire_metadata(self.p, operation, width, top_k=top_k, num_experts=self.m['num_experts'])
        ends = []
        for i, group in enumerate(self.groups(stage, 'ep')):
            ends += self.collective(f'{name}.{operation}{i}', 'all_to_all_v', group, 0, deps,
                'MoE', layer, matrix, True, phase=phase, wire_metadata=metadata)
        return ends

    def recompute_layer(self, name, layer, stage, ranks, tokens, deps):
        """Replay the entire forward layer, including TP/EP communication."""
        previous, retained = list(deps), []
        rows = self.activation_rows(layer, tokens, checkpoint=False)
        lengths = self.train_lengths
        for component in self.component_order(layer):
            if component == 'MoE' and self.ep > 1:
                previous = self.training_ep(name + '.recompute', 'dispatch', layer, stage, tokens, previous, 'recompute')
            ends = []
            for rank in ranks:
                allocations, releases = [], []
                for i, row in enumerate(rows):
                    if row['component'] != component:
                        continue
                    dtype = row.get('dtype', self.p['activation'])
                    aid = name + f'.{rank}.recompute.{component}.{i}'
                    allocations.append(self.allocation(aid, rank, row['elements'] * DTYPE_BYTES[dtype],
                        component, 'recompute_workspace', layer, dtype, formula=row['formula']))
                    (releases if row.get('lifetime') == 'operator_working_set' else retained).append(aid)
                if component == 'KDA':
                    state = self.training_attention_state(layer)
                    aid = name + f'.{rank}.recompute.kda_state'
                    allocations.append(self.allocation(aid, rank, state['recurrent_chunk_elements'] * DTYPE_BYTES[self.p['recurrent']],
                        'KDA', 'recompute_workspace', layer, self.p['recurrent']))
                    retained.append(aid)
                if component in ('CSA', 'HCA'):
                    state = self.training_attention_state(layer)
                    for category, field, dtype, byte_width in (
                            ('compressed_attention_pool', 'compressed_pool_elements', self.p['kv'], DTYPE_BYTES[self.p['kv']]),
                            ('attention_selection_indices', 'selection_index_elements', None, 4)):
                        size = state[field] * byte_width
                        if size:
                            aid = name + f'.{rank}.recompute.{component}.{category}'
                            allocations.append(self.allocation(aid, rank, size, component, 'recompute_workspace',
                                layer, dtype, formula=state['formula'], storage_role=category,
                                storage_dtype='int32' if dtype is None else dtype))
                            retained.append(aid)
                ends.append(self.kernel(name + f'.{rank}.recompute.{component}', rank, layer, component,
                    tokens, previous, phase='recompute', allocations=allocations, releases=releases, lengths=lengths))
            previous = ends
            if component == 'CSA':
                previous = self.sparse_index_collective(name + '.recompute', layer,
                    self.groups(stage, 'tp'), lengths, previous, phase='recompute', divisor=self.ep)
            if component == 'MoE' and self.ep > 1:
                previous = self.training_ep(name + '.recompute', 'combine', layer, stage, tokens, previous, 'recompute')
            if component in (self.a['attention_by_layer'][layer], self.a['ffn_by_layer'][layer]) and self.tp > 1:
                previous = self.training_tp_reduce(name + '.recompute', component, layer, stage,
                                                   ranks, tokens, previous, 'recompute')
        return previous, retained

    def inference(self, *, phase='all', initial_dependencies=()):
        if phase not in ('all', 'prefill', 'decode'):
            raise ValueError('Inference phase must be all, prefill or decode')
        if self.grouped:
            from .grouped_layers import grouped_inference
            return grouped_inference(self, phase, initial_dependencies)
        requested_phase = phase
        self.initial()
        self.g.metadata.update(token_completion_events={}, sequence_token_completion_events={})
        plan = getattr(self, '_inference_plan', None)
        if plan is not None and requested_phase != 'all':
            raise ValueError('Speculative round plans cannot be split into inference phases')
        if requested_phase == 'decode':
            seeded = []
            for rank, (d, stage, e, t) in self.coords.items():
                owned = [i for i in range(self.w['batch_size']) if i % self.ep == e]
                lengths = [self.w['input_lengths'][i] for i in owned]
                for layer in self.stage_layers[stage]:
                    allocations = self.inference_state_allocations(f'infer.seed.l{layer}', rank, layer, lengths, 0)
                    self.g.initial_allocations.extend(allocations)
                    seeded.extend(a.id for a in allocations)
            self.g.metadata['seeded_inference_state_allocations'] = seeded
        for d in range(self.dp):
            previous = list(initial_dependencies)
            token_ends = []
            sequence_ends = {str(i): [] for i in range(self.w['batch_size'])}
            steps = (range(1, max(self.w['output_lengths'])) if requested_phase == 'decode' else
                     range(1) if requested_phase == 'prefill' else
                     range(len(plan) if plan is not None else max(self.w['output_lengths'])))
            for step in steps:
                plan_item = plan[step] if plan is not None else None
                phase = plan_item['phase'] if plan_item else ('prefill' if step == 0 else 'decode')
                active = (list(range(self.w['batch_size'])) if plan_item else
                          [i for i, length in enumerate(self.w['output_lengths']) if length > step])
                local = {e: [i for i in active if i % self.ep == e] for e in range(self.ep)}
                tokens = ({e: plan_item['query_tokens'] * len(local[e]) for e in range(self.ep)} if plan_item else
                          {e: sum(self.w['input_lengths'][i] if step == 0 else 1 for i in local[e]) for e in range(self.ep)})
                sp_rows = {e: ceil(tokens[e] / self.tp) for e in range(self.ep)} if self.sp else tokens
                sp_full_tokens = {e: sp_rows[e] * self.tp for e in range(self.ep)} if self.sp else tokens
                if self.sp:
                    self.g.metadata['sequence_parallel'].setdefault('steps', []).append({
                        'replica': d, 'token_index': step, 'phase': phase,
                        'valid_source_tokens_by_ep': dict(tokens),
                        'padded_local_rows_by_ep': dict(sp_rows),
                        'gathered_padded_tokens_by_ep': dict(sp_full_tokens),
                        'selected_output_padded_tokens_by_ep': {e: ceil(len(local[e]) / self.tp) * self.tp for e in range(self.ep)},
                    })
                total_tokens = sum(tokens.values())
                residuals = []
                incoming_receives = []
                round_prefix = f'infer.d{d}.s{step}'
                round_before = getattr(self, '_inference_round_before', None)
                if round_before:
                    previous = round_before(round_prefix, plan_item, previous)
                for layer in range(self.m['layers']):
                    stage = layer * self.pp // self.m['layers']
                    prefix = f'infer.d{d}.s{step}.l{layer}'
                    ranks = [self.ranks[d, stage, e, t] for e, t in product(range(self.ep), range(self.tp))]
                    ready = list(previous)
                    # Persistent cache increments and KDA fixed recurrent state.
                    for rank in ranks:
                        e = self.coords[rank][2]
                        owned = [i for i in range(self.w['batch_size']) if i % self.ep == e]
                        lengths = ([plan_item['context_before'] + (plan_item['query_tokens'] if phase == 'prefill'
                                    else plan_item['committed_tokens'])] * len(owned) if plan_item else
                                   [self.w['input_lengths'][i] + min(step, self.w['output_lengths'][i] - 1) for i in owned])
                        allocations = self.inference_state_allocations(prefix, rank, layer, lengths, step)
                        block = attnres_state(self.m, layer, sp_rows[e] if self.sp else tokens[e])
                        if block['new_elements']:
                            aid = prefix + f'.{rank}.attnres_block'
                            allocations.append(self.allocation(aid, rank, block['new_elements'] * DTYPE_BYTES[self.p['activation']], 'AttnRes', 'attnres_residual', layer, self.p['activation']))
                            residuals.append(aid)
                        ready.append(self.barrier(prefix + f'.{rank}.state', previous, allocations))
                    state_hook = getattr(self, '_inference_state_hook', None)
                    if state_hook:
                        ready = state_hook(prefix, layer, ready, step)
                    for component in self.component_order(layer):
                        before_hook = getattr(self, '_inference_component_before', None)
                        if before_hook:
                            replacement = before_hook(prefix, layer, component, ready, step)
                            if replacement is not None:
                                ready = replacement
                                continue
                        sp_buffers = []
                        if self.sp and component in (self.a['attention_by_layer'][layer], self.a['ffn_by_layer'][layer], 'Output'):
                            selected_output = component == 'Output' and phase != 'verify'
                            gather_rows = ({e: ceil(len(local[e]) / self.tp) * self.tp for e in range(self.ep)}
                                           if selected_output else sp_full_tokens)
                            gathered = []
                            for rank in ranks:
                                e = self.coords[rank][2]
                                if gather_rows[e]:
                                    aid = prefix + f'.{component}.{rank}.sp_gathered_input'
                                    sp_buffers.append(aid)
                                    gathered.append(self.allocation(aid, rank, gather_rows[e] * self.m['hidden_dim'] * DTYPE_BYTES[self.p['activation']],
                                        component, 'sp_gathered_input', layer, self.p['activation'],
                                        valid_source_tokens=len(local[e]) if selected_output else tokens[e], padded_tokens=gather_rows[e], sequence_parallel_degree=self.tp))
                            gather_start = self.barrier(prefix + f'.{component}.sp_gather_allocate', ready, gathered)
                            terminals = []
                            for e in range(self.ep):
                                if gather_rows[e]:
                                    group = [self.ranks[d, stage, e, t] for t in range(self.tp)]
                                    terminals += self.collective(prefix + f'.{component}.e{e}.sp_all_gather', 'all_gather', group,
                                        gather_rows[e] * self.m['hidden_dim'] * DTYPE_BYTES[self.p['communication']], (gather_start,), component, layer, activation=True)
                            ready = terminals or [gather_start]
                        if component == 'MoE':
                            terminals = []
                            for t in range(self.tp):
                                group = [self.ranks[d, stage, e, t] for e in range(self.ep)]
                                width = self.a['moe'].get('latent_dim') or self.m['hidden_dim']
                                matrix = moe_payload_matrix(self.p, 'dispatch', [tokens[e] for e in range(self.ep)], width, self.m['top_k'])
                                terminals += self.collective(prefix + f'.t{t}.dispatch', 'all_to_all_v', group, 0, ready, component, layer, matrix, True,
                                    phase=phase, wire_metadata=moe_wire_metadata(self.p, 'dispatch', width, top_k=self.m['top_k'], num_experts=self.m['num_experts']))
                            ready = terminals
                        ends = []
                        for rank in ranks:
                            e = self.coords[rank][2]
                            count = total_tokens / self.ep if component == 'MoE' else tokens[e]
                            if self.sp and component in ('PreNorm', 'PostNorm', 'AttnRes'):
                                count = sp_rows[e]
                            if component == 'Output' and step == 0:
                                count = len(local[e])
                            if count == 0:
                                continue
                            allocations = []
                            for i, row in enumerate(self.activation_rows(layer, count, training=False)):
                                if row['component'] == component:
                                    dtype = row.get('dtype', self.p['activation'])
                                    allocations.append(self.allocation(prefix + f'.{rank}.{component}.activation{i}', rank, row['elements'] * DTYPE_BYTES[dtype], component, 'activation', layer, dtype, formula=row['formula']))
                            deps = list(ready)
                            active_lengths = ([plan_item['context_before'] + plan_item['query_tokens']] * len(local[e])
                                              if plan_item else [self.w['input_lengths'][i] + step for i in local[e]])
                            kind = self.a['attention_by_layer'][layer]
                            offload = self.f['kv_offload'] != 'none' and self.f['kv_offload_fraction'] > 0
                            if component == kind and kind != 'KDA' and offload:
                                state = attention_state(self.m, layer, len(local[e]), 1, tp=self.tp)
                                fetch_elements = (sum(max(0, length - 1) for length in active_lengths) if step else 0) * state['elements_per_token']
                                amount = self.kv_bytes(fetch_elements) * self.f['kv_offload_fraction']
                                if amount:
                                    aid = prefix + f'.{rank}.kv_fetch'
                                    start = self.barrier(aid + '.allocate', deps, [self.allocation(aid, rank, amount, component, 'kv_fetch_buffer', layer, self.p['kv'])])
                                    source = self.host(rank) if self.f['kv_offload'] == 'host' else 'remote0'
                                    deps = [self.add(self.topo.transfer(aid + '.transfer', source, rank, amount, (start,), {'operation': 'kv_read', 'component': component, 'layer': layer}))]
                                    allocations_to_release = [aid]
                                else:
                                    allocations_to_release = []
                            else:
                                allocations_to_release = []
                            if component == kind and kind != 'KDA' and step and self.p['kv'] != self.p['compute']:
                                state = attention_state(self.m, layer, len(local[e]), 1, tp=self.tp)
                                current_queries = plan_item['query_tokens'] if plan_item else 1
                                old_elements = sum(max(0, length - current_queries) for length in active_lengths) * state['elements_per_token']
                                if old_elements:
                                    aid = prefix + f'.{rank}.kv_decoded'
                                    raw = old_elements * DTYPE_BYTES[self.p['compute']]
                                    end = self.add(Event(aid + '.convert', tuple(deps),
                                        {self.scalar(rank): 2 * old_elements, rank + '.hbm': self.kv_bytes(old_elements) + raw},
                                        slots=(rank + '.compute_stream',),
                                        allocs=(self.allocation(aid, rank, raw, component, 'kv_conversion_buffer', layer, self.p['compute']),),
                                        releases=tuple(allocations_to_release),
                                        metadata={'component': component, 'layer': layer, 'operation': 'precision_conversion', 'kernel': 'kv_decode_conversion'}))
                                    deps = [end]
                                    allocations_to_release = [aid]
                            engram = self.f['engram']
                            if component == 'PreNorm' and engram['enabled'] and layer in engram['layers']:
                                query_tokens = tokens[e] if self.sp else count
                                misses = query_tokens * engram['queries_per_token'] * (1 - engram['hit_rate']) / self.tp
                                amount = misses * engram['record_bytes']
                                if amount:
                                    source = self.host(rank) if engram['location'] == 'host' else 'remote0'
                                    aid = prefix + f'.{rank}.engram_buffer'
                                    start = self.barrier(aid + '.allocate', deps, [self.allocation(aid, rank, amount, 'Engram', 'lookup_buffer', layer)])
                                    event = self.topo.transfer(aid + '.transfer', source, rank, amount, (start,), {'operation': 'lookup', 'component': 'Engram', 'layer': layer})
                                    event.work[source + '.engram_iops'] = misses
                                    event.work[source + '.iops'] = misses
                                    event.startup_s += engram['latency_s']
                                    deps.append(self.add(event))
                                    allocations_to_release.append(aid)
                            raw_kv = None
                            new_elements = 0
                            kv_convert = component == kind and kind != 'KDA' and self.p['kv'] != self.p['compute']
                            if component == kind and kind != 'KDA' and (kv_convert or offload):
                                state = attention_state(self.m, layer, len(local[e]), 1, tp=self.tp)
                                new_elements = count * state['elements_per_token']
                                raw_kv = prefix + f'.{rank}.kv_raw'
                                raw_dtype = self.p['compute'] if kv_convert else self.p['kv']
                                raw_bytes = new_elements * DTYPE_BYTES[raw_dtype] if kv_convert else self.kv_bytes(new_elements)
                                allocations.append(self.allocation(raw_kv, rank, raw_bytes, component, 'kv_write_buffer', layer, raw_dtype))
                            end = self.kernel(prefix + f'.{rank}.{component}', rank, layer, component, count, deps,
                                              phase=phase, total_tokens=total_tokens, allocations=allocations,
                                              releases=[a.id for a in allocations if a.id != raw_kv] + allocations_to_release, lengths=active_lengths,
                                              query_lengths=[plan_item['query_tokens']] * len(local[e]) if plan_item else None)
                            if kv_convert and new_elements:
                                aid = prefix + f'.{rank}.kv_quantized'
                                full_bytes = self.kv_bytes(new_elements)
                                output_alloc = (self.allocation(aid, rank, full_bytes, component, 'kv_write_buffer', layer, self.p['kv']),) if offload else ()
                                end = self.add(Event(aid + '.convert', (end,),
                                    {self.scalar(rank): 2 * new_elements,
                                     rank + '.hbm': new_elements * DTYPE_BYTES[self.p['compute']] + full_bytes},
                                    slots=(rank + '.compute_stream',), allocs=output_alloc, releases=(raw_kv,),
                                    metadata={'component': component, 'layer': layer, 'operation': 'precision_conversion', 'kernel': 'kv_encode_conversion'}))
                                raw_kv = aid if offload else None
                            if component == kind and kind != 'KDA' and offload:
                                amount = self.kv_bytes(new_elements) * self.f['kv_offload_fraction']
                                if amount:
                                    aid = prefix + f'.{rank}.kv_write_buffer'
                                    dest = self.host(rank) if self.f['kv_offload'] == 'host' else 'remote0'
                                    event = self.topo.transfer(aid + '.transfer', rank, dest, amount, (end,), {'operation': 'kv_write', 'component': component, 'layer': layer})
                                    event.releases = (raw_kv,) if raw_kv else ()
                                    end = self.add(event)
                            ends.append(end)
                        ready = ends or ready
                        if component == 'CSA' and self.tp > 1:
                            index_ends = []
                            for e in range(self.ep):
                                lengths = ([plan_item['context_before'] + plan_item['query_tokens']] * len(local[e])
                                           if plan_item else [self.w['input_lengths'][i] + step for i in local[e]])
                                queries = [plan_item['query_tokens']] * len(local[e]) if plan_item else None
                                group = [self.ranks[d, stage, e, t] for t in range(self.tp)]
                                index_ends += self.sparse_index_collective(prefix + f'.e{e}', layer,
                                    [group], lengths, ready, phase, query_lengths=queries)
                            ready = index_ends or ready
                        if component == 'MoE':
                            terminals = []
                            for t in range(self.tp):
                                group = [self.ranks[d, stage, e, t] for e in range(self.ep)]
                                width = self.a['moe'].get('latent_dim') or self.m['hidden_dim']
                                matrix = moe_payload_matrix(self.p, 'combine', [tokens[e] for e in range(self.ep)], width, self.m['top_k'])
                                terminals += self.collective(prefix + f'.t{t}.combine', 'all_to_all_v', group, 0, ready, component, layer, matrix, True,
                                    phase=phase, wire_metadata=moe_wire_metadata(self.p, 'combine', width, top_k=self.m['top_k'], num_experts=self.m['num_experts']))
                            ready = terminals
                        if (component in (self.a['attention_by_layer'][layer], self.a['ffn_by_layer'][layer]) or self.sp and component == 'WordEmbedding') and self.tp > 1:
                            reductions = []
                            for e in range(self.ep):
                                if self.sp and not sp_full_tokens[e]:
                                    continue
                                group = [self.ranks[d, stage, e, t] for t in range(self.tp)]
                                reduction = 'reduce_scatter' if self.sp else 'all_reduce'
                                wire_tokens = sp_full_tokens[e] if self.sp else tokens[e]
                                reductions += self.collective(prefix + f'.{component}.e{e}.tp', reduction, group,
                                    wire_tokens * self.m['hidden_dim'] * DTYPE_BYTES[self.p['communication']], ready, component, layer, activation=True)
                            ready = reductions
                        if sp_buffers:
                            ready = [self.barrier(prefix + f'.{component}.sp_gather_release', ready, releases=sp_buffers)]
                        after_hook = getattr(self, '_inference_component_after', None)
                        if after_hook:
                            ready = after_hook(prefix, layer, component, ready, step)
                    previous = [self.barrier(prefix + '.layer_done', ready, releases=incoming_receives)]
                    incoming_receives = []
                    next_stage = (layer + 1) * self.pp // self.m['layers']
                    if layer + 1 < self.m['layers'] and next_stage != stage:
                        transfers = []
                        for e, t in product(range(self.ep), range(self.tp)):
                            source, dest = self.ranks[d, stage, e, t], self.ranks[d, next_stage, e, t]
                            end, received = self.pipeline_transfer(prefix + f'.{source}.pp', source, dest,
                                tokens[e] * self.m['hidden_dim'], previous)
                            transfers.append(end)
                            incoming_receives.append(received)
                        previous = transfers
                round_after = getattr(self, '_inference_round_after', None)
                if round_after:
                    previous = round_after(round_prefix, plan_item, previous)
                committed = plan_item['committed_tokens'] if plan_item else 1
                end = self.barrier(f'infer.d{d}.token{step}', previous, releases=residuals, phase='token_complete', token_index=step)
                token_ends.extend([end] * committed)
                for i in active:
                    sequence_ends[str(i)].extend([end] * committed)
                previous = [end]
            self.g.metadata['token_completion_events'][str(d)] = token_ends
            self.g.metadata['sequence_token_completion_events'][str(d)] = sequence_ends
        return self.g

    def training_attention_state(self, layer):
        rows = [attention_training_state(self.m, layer, 1 / self.ep, length, tp=self.tp)
                for length in self.train_lengths]
        return {key: sum(row[key] for row in rows) if isinstance(value, (int, float)) else value
                for key, value in rows[0].items()}

    def synchronize_gradients(self, name, layer, stage, previous, phase):
        if self.shards:
            return self.shards.reduce_gradients(self, name, layer, stage, previous)
        tensors = self.local_tensors(layer)
        all_elements = sum(self.local_elements(t) for t in tensors)
        dense_elements = sum(self.local_elements(t) for t in tensors if not t.ep_shard)
        reductions = [('fsdp', 'reduce_scatter', all_elements), ('replica', 'all_reduce', all_elements / self.fsdp)]
        if self.grouped:
            reductions.extend((axis, 'all_reduce', count / self.fsdp) for axis, count in (
                ('attention_dp', dense_elements), ('cp', dense_elements),
                ('expert_dp', all_elements - dense_elements)))
        else:
            reductions.append(('ep', 'all_reduce', dense_elements / self.fsdp))
        for axis, kind, elements in reductions:
            if axis == 'replica':
                previous = replica_all_reduce(self.g, self.topo, name + '.gradient.replica',
                    self.groups(stage, axis), elements * DTYPE_BYTES[self.p['gradient']],
                    previous, self.x, {'phase': phase, 'category': 'data_parallel_gradients',
                                       'component': 'System', 'layer': layer})
                continue
            terminals = []
            for i, group in enumerate(self.groups(stage, axis)):
                terminals += self.collective(name + f'.gradient.{axis}{i}', kind, group, elements * DTYPE_BYTES[self.p['gradient']], previous, 'System', layer, phase=phase,
                    wire_metadata={'parallel_axis': axis, 'buffer_dtype': self.p['gradient']})
            previous = list(dict.fromkeys(terminals))
        return previous

    def training_graph(self):
        from .grouped_layers import grouped_training_layer, LayerBatch
        self.initial()
        if any(kind in ('CSA', 'HCA') for kind in self.a['attention_by_layer']):
            self.g.metadata['assumptions'].append(
                'CSA/HCA training is an analytical forward/backward resource proxy, not complete DeepSeek V4 training: '
                'mHC, indexer auxiliary loss and gradient routing are not modeled. Sparse attention saves compressed '
                'pools and selection indices; checkpointing discards them after forward and reconstructs them before backward.')
        stage_previous = {}
        for step in range(self.w['training_steps']):
            for stage in range(self.pp):
                for phase, micro in pipeline_schedule(stage, self.pp, self.w['gradient_accumulation']):
                    if self.forward_only and phase == 'backward':
                        continue
                    if hasattr(self, 'training_lengths_by_micro'):
                        self.train_lengths = self.training_lengths_by_micro[micro]
                    batch = LayerBatch(self.mesh, ((i, n, n) for i, n in enumerate(self.train_lengths)))
                    exact = self.grouped or any(rows != batch.requests[0] for rows in batch.requests[1:])
                    forward = phase == 'forward'
                    prefix = f'train.s{step}.p{stage}.{phase}.m{micro}'
                    deps = [stage_previous[stage]] if stage in stage_previous else ([f'train.step{step-1}.done'] if step else [])
                    if forward and stage:
                        deps += [f'train.s{step}.p{stage-1}.forward.m{micro}.send']
                    if not forward:
                        deps += [f'train.s{step}.p{stage}.forward.m{micro}.done']
                        if stage + 1 < self.pp:
                            deps += [f'train.s{step}.p{stage+1}.backward.m{micro}.send']
                    previous = [self.barrier(prefix + '.start', deps)]
                    objective_release = []
                    gathers, layer_ends, gradient_ends = [], [], []
                    for layer in self.stage_layers[stage][::1 if forward else -1]:
                        name = prefix + f'.l{layer}'
                        gathered = []
                        if self.sharded_training:
                            if self.x['reshard_after_forward'] or (forward and micro == 0):
                                gather_deps = gather_dependencies(previous, gathers, layer_ends, prefetch_depth(self.x))
                                end, gathered = self.gather(name, layer, stage, gather_deps, phase)
                                gathers.append(end)
                                previous = list(dict.fromkeys([*previous, end]))
                                if not self.x['reshard_after_forward']:
                                    self.retained_gathers[step, layer] = gathered
                                    gathered = []
                        rows_by_rank = {}
                        ranks = [r for r, c in self.coords.items() if c[1] == stage]
                        local_tokens = sum(self.train_lengths) / self.ep
                        for rank in ([] if exact else ranks):
                            # Checkpointing changes retention, not the forward
                            # operator's temporary working set.
                            rows_by_rank[rank] = self.activation_rows(layer, local_tokens, checkpoint=False)
                        layer_release = []
                        if not forward and self.sharded_training and self.shards:
                            previous = self.shards.prepare_gradients(self, name, layer, ranks, step, micro, previous)
                        elif not forward and self.sharded_training:
                            allocations = []
                            for rank in ranks:
                                key = step, layer, rank
                                if key not in self.grad_buffers:
                                    ids = []
                                    for component in self.component_order(layer):
                                        tensors = self.local_tensors(layer, component)
                                        count = sum(self.local_elements(t) for t in tensors)
                                        if count:
                                            aid = f'gradient.s{step}.l{layer}.{rank}.{component}.m{micro}'
                                            allocations.append(self.allocation(aid, rank, count * DTYPE_BYTES[self.p['gradient']],
                                                component, 'unsharded_gradient_buffer', layer, self.p['gradient']))
                                            ids.append(aid)
                                    self.grad_buffers[key] = ids
                            previous = [self.barrier(name + '.gradient_buffers', previous, allocations)]
                        if exact:
                            previous, release, held_logits = grouped_training_layer(self, batch, stage, layer,
                                name, phase, previous, step, micro)
                            layer_release.extend(release)
                            objective_release.extend(held_logits)
                        else:
                            if not forward and self.f['activation_checkpointing']:
                                previous, retained = self.recompute_layer(name, layer, stage, ranks, local_tokens, previous)
                                layer_release.extend(retained)
                            for component in self.component_order(layer)[::1 if forward else -1]:
                                ends = []
                                if component == 'MoE' and self.ep > 1:
                                    previous = self.training_ep(name, 'dispatch', layer, stage, local_tokens, previous, phase)
                                for rank in ranks:
                                    key = step, micro, rank, layer, component
                                    allocations, releases = [], []
                                    if forward:
                                        for i, row in enumerate(rows_by_rank[rank]):
                                            if row['component'] == component:
                                                dtype = row.get('dtype', self.p['activation'])
                                                a = self.allocation(name + f'.{rank}.{component}.a{i}', rank, row['elements'] * DTYPE_BYTES[dtype], component, 'activation', layer, dtype, formula=row['formula'])
                                                allocations.append(a)
                                                checkpoint_temporary = (self.f['activation_checkpointing'] and
                                                    not (component == 'PreNorm' and row['name'] == 'input'))
                                                if row.get('lifetime') == 'operator_working_set' or checkpoint_temporary or self.forward_only:
                                                    if component == 'Output' and getattr(self, '_training_loss', None):
                                                        objective_release.append(a.id)
                                                    else:
                                                        releases.append(a.id)
                                        if component == 'KDA' and not self.f['activation_checkpointing'] and not self.forward_only:
                                            state = self.training_attention_state(layer)
                                            size = state['recurrent_chunk_elements'] * DTYPE_BYTES[self.p['recurrent']]
                                            if size:
                                                allocations.append(self.allocation(name + f'.{rank}.kda_saved', rank, size, 'KDA', 'saved_recurrent_states', layer, self.p['recurrent']))
                                        if component in ('CSA', 'HCA'):
                                            state = self.training_attention_state(layer)
                                            for category, field, dtype, byte_width in (
                                                    ('compressed_attention_pool', 'compressed_pool_elements', self.p['kv'], DTYPE_BYTES[self.p['kv']]),
                                                    ('attention_selection_indices', 'selection_index_elements', None, 4)):
                                                size = state[field] * byte_width
                                                if size:
                                                    allocation = self.allocation(name + f'.{rank}.{component}.{category}', rank,
                                                        size, component, category, layer, dtype,
                                                        formula=state['formula'], storage_dtype='int32' if dtype is None else dtype)
                                                    allocations.append(allocation)
                                                    if self.f['activation_checkpointing'] or self.forward_only:
                                                        releases.append(allocation.id)
                                        self.saved[key] = [a.id for a in allocations if a.id not in releases and a.id not in objective_release]
                                        if component == 'AttnRes':
                                            block = attnres_state(self.m, layer, local_tokens)
                                            if block['new_elements']:
                                                aid = name + f'.{rank}.attnres'
                                                allocations.append(self.allocation(aid, rank, block['new_elements'] * DTYPE_BYTES[self.p['activation']], component, 'attnres_residual', layer, self.p['activation']))
                                                self.residual_blocks[step, micro, rank].append(aid)
                                    else:
                                        releases = self.saved[key]
                                    ends.append(self.kernel(name + f'.{rank}.{component}', rank, layer, component, local_tokens, previous, backward=not forward, phase=phase,
                                        allocations=allocations, releases=releases, lengths=self.train_lengths))
                                previous = ends
                                if forward and component == 'CSA':
                                    previous = self.sparse_index_collective(name, layer, self.groups(stage, 'tp'),
                                        self.train_lengths, previous,
                                        phase=phase, divisor=self.ep)
                                if component == 'MoE' and self.ep > 1:
                                    previous = self.training_ep(name, 'combine', layer, stage, local_tokens, previous, phase)
                                if component in (self.a['attention_by_layer'][layer], self.a['ffn_by_layer'][layer]) and self.tp > 1:
                                    previous = self.training_tp_reduce(name, component, layer, stage,
                                        ranks, local_tokens, previous, phase)
                        if not forward:
                            sync = self.x['sync_each_microbatch'] or micro == self.w['gradient_accumulation'] - 1
                            if sync:
                                compute_done = previous
                                previous = [*previous, *gradient_ends[-1:]]
                                previous = self.synchronize_gradients(name, layer, stage, previous, phase)
                                grad_release = (self.shards.release_gradients(self, layer, ranks, step) if self.shards else
                                                [aid for rank in ranks for aid in self.grad_buffers.pop((step, layer, rank), [])])
                                if self.x['overlap_gradient_sync'] and (self.dp > 1 or self.lanes > 1 or self.mesh.expert_dp > 1):
                                    gradient_ends.append(self.barrier(name + '.gradient.done', previous, releases=grad_release))
                                    previous = compute_done
                                else:
                                    layer_release.extend(grad_release)
                            if micro == self.w['gradient_accumulation'] - 1 and not self.x['reshard_after_forward']:
                                layer_release += self.retained_gathers.pop((step, layer), [])
                        previous = [self.barrier(name + '.done', previous, releases=gathered + layer_release)]
                        layer_ends.append(previous[0])
                    extra_release = []
                    source_stage = stage - 1 if forward else stage + 1
                    if 0 <= source_stage < self.pp:
                        suffix = '.receive' if self.p['activation'] == self.p['communication'] else '.decoded'
                        for d, e, t in product(range(self.dp), range(self.lanes), range(self.tp)):
                            source_rank = self.ranks[d, source_stage, e, t]
                            extra_release.append(f'train.s{step}.p{source_stage}.{phase}.m{micro}.{source_rank}.pipeline' + suffix)
                    if (not forward or self.forward_only) and stage == 0:
                        for rank in ranks:
                            extra_release += self.residual_blocks.pop((step, micro, rank), [])
                    loss_hook = getattr(self, '_training_loss', None)
                    if forward and stage == self.pp - 1 and loss_hook:
                        previous = loss_hook(prefix, stage, micro, previous)
                        extra_release += objective_release
                    end = self.barrier(prefix + '.done', previous + gradient_ends[-1:], releases=extra_release)
                    transfers = []
                    next_stage = stage + (1 if forward else -1)
                    if 0 <= next_stage < self.pp:
                        for d, e, t in product(range(self.dp), range(self.lanes), range(self.tp)):
                            rank, dest = self.ranks[d, stage, e, t], self.ranks[d, next_stage, e, t]
                            transferred, _ = self.pipeline_transfer(prefix + f'.{rank}.pipeline', rank, dest,
                                batch.tokens(e) * self.m['hidden_dim'], previous)
                            transfers.append(transferred)
                    sent = self.barrier(prefix + '.send', transfers or (end,))
                    stage_previous[stage] = end if self.x['overlap_pipeline_p2p'] else self.barrier(
                        prefix + '.stage_ready', (end, sent))
            if self.forward_only:
                self.barrier(f'train.step{step}.done', list(stage_previous.values()))
            else:
                self.optimizer(step, list(stage_previous.values()))
            stage_previous = {s: f'train.step{step}.done' for s in range(self.pp)}
        self.g.metadata['step_end_events'] = [f'train.step{s}.done' for s in range(self.w['training_steps'])]
        return self.g

    def optimizer(self, step, deps):
        ends = []
        depth = self.opt.get('inflight_components_per_rank')
        if depth is not None:
            self.g.metadata['optimizer_inflight_components_per_rank'] = depth
            note = f'Optimizer updates use {depth} independent in-flight component chains per rank; each chain admits the next bucket after weight refresh.'
            if note not in self.g.metadata['assumptions']:
                self.g.metadata['assumptions'].append(note)
        for rank, coord in self.coords.items():
            tails = [None] * depth if depth else []
            bucket = 0
            for layer in self.stage_layers[coord[1]]:
                for component in self.component_order(layer):
                    tensors = self.local_tensors(layer, component)
                    if not tensors:
                        continue
                    prefix = f'optimizer.s{step}.{rank}.l{layer}.{component}'
                    node = self.host(rank) if self.f['optimizer_offload'] else rank
                    elements = sum(self.resident_elements(t, rank) for t in tensors)
                    if self.shards and not elements:
                        continue
                    current = list(deps)
                    lane = bucket % depth if depth else None
                    if depth and tails[lane] is not None:
                        current.append(tails[lane])
                    gradient = elements * DTYPE_BYTES[self.p['gradient']]
                    releases = []
                    if self.f['optimizer_offload']:
                        aid = prefix + '.gradient'
                        event = self.topo.transfer(prefix + '.D2H', rank, node, gradient, tuple(current), {'component': component, 'layer': layer, 'operation': 'optimizer_gradient', 'direction': 'D2H'})
                        event.allocs = (self.allocation(aid, node, gradient, component, 'offload_gradient', layer, self.p['gradient']),)
                        current = [self.add(event)]
                        releases.append(aid)
                    flops, optimizer_bytes, largest_matrix = 0.0, 0.0, 0.0
                    for tensor in tensors:
                        count = self.resident_elements(tensor, rank)
                        if not count:
                            continue
                        states = optimizer_states(tensor, self.opt['kind'])
                        optimizer_bytes += sum(count * DTYPE_BYTES[self.p['master'] if state == 'master_weight' else self.p['optimizer']] for state in states)
                        if 'muon_momentum' in states:
                            rows, columns = tensor.shape[-2:]
                            if self.opt['muon_variant'] == 'per_head' and tensor.component in ('MLA', 'KDA') and ('q_up' in tensor.name or 'kv_up' in tensor.name or '.q.' in tensor.name or '.k.' in tensor.name or '.v.' in tensor.name):
                                rows = max(1, rows // self.m['num_heads'])
                            flops += count * (8 + self.opt['ns_iterations'] * 4 * min(rows, columns))
                            largest_matrix = max(largest_matrix, rows * columns)
                        else:
                            flops += 14 * count
                    scratch = []
                    if largest_matrix:
                        aid = prefix + '.muon_workspace'
                        scratch.append(self.allocation(aid, node, largest_matrix * DTYPE_BYTES[self.p['optimizer']] * self.opt['workspace_copies'], component, 'muon_workspace', layer, self.p['optimizer']))
                        releases.append(aid)
                    current = [self.add(Event(prefix + '.update', tuple(current), {self.scalar(node): flops, self.pool(node): gradient + 2 * optimizer_bytes}, slots=(node + '.compute_stream',), allocs=tuple(scratch), releases=tuple(releases), metadata={'component': component, 'layer': layer, 'kernel': self.opt['kind'], 'phase': 'optimizer', 'flops': flops}))]
                    wire = self.p['weight_transfer'] if self.f['optimizer_offload'] else self.p['weight']
                    amount = elements * DTYPE_BYTES[wire]
                    aid = prefix + '.updated_weights'
                    current = [self.add(Event(prefix + '.pack', tuple(current), {self.scalar(node): 2 * elements, self.pool(node): elements * DTYPE_BYTES[self.p['master']] + amount}, slots=(node + '.compute_stream',), allocs=(self.allocation(aid, node, amount, component, 'conversion_buffer', layer, wire),), metadata={'component': component, 'kernel': 'weight_conversion', 'phase': 'optimizer'}))]
                    if self.f['optimizer_offload']:
                        event = self.topo.transfer(prefix + '.H2D', node, rank, amount, tuple(current), {'component': component, 'layer': layer, 'operation': 'optimizer_weight', 'direction': 'H2D'})
                        received = prefix + '.received'
                        event.allocs = (self.allocation(received, rank, amount, component, 'conversion_buffer', layer, wire),)
                        event.releases = (aid,)
                        current = [self.add(event)]
                        aid = received
                        scales = self.scale(elements, wire)
                        if scales:
                            current = [self.add(self.topo.transfer(prefix + '.scales', node, rank, scales, tuple(current), {'component': component, 'operation': 'weight_scale_transfer', 'layer': layer}))]
                    storage = sum(self.resident_elements(t, rank) * DTYPE_BYTES[self.weight_dtype(t)] for t in tensors)
                    ends.append(self.add(Event(prefix + '.refresh', tuple(current), {self.scalar(rank): 2 * elements, rank + '.hbm': amount + storage}, slots=(rank + '.compute_stream',), releases=(aid,), metadata={'component': component, 'kernel': 'weight_refresh', 'layer': layer, 'phase': 'weight_refresh'})))
                    if depth:
                        tails[lane] = ends[-1]
                        bucket += 1
        self.barrier(f'train.step{step}.done', ends)


def build_structured(config, topology):
    builder = Builder(config, topology)
    graph = builder.training_graph() if builder.training else builder.inference()
    builder.annotate_collective_allocations()
    return graph

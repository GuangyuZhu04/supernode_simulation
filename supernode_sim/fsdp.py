"""FSDP process groups and parameter ownership on an existing training mesh."""
from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from .core import DTYPE_BYTES
from .training_schedule import replica_all_reduce


@dataclass
class ShardDomain:
    degree: int
    shards: list[list[list[str]]]
    replicas: list[list[list[str]]]
    index: dict[str, int]

    def describe(self):
        return {'degree': self.degree, 'stages': [
            {'stage': s, 'shard_groups': groups, 'replica_groups': self.replicas[s]}
            for s, groups in enumerate(self.shards)]}


def resolve_domains(mesh, execution):
    """Group only ranks that own the same tensor slice before FSDP.

    A dense slice fixes pipeline stage and attention TP. A routed slice fixes
    stage, expert ID partition and expert TP. Ordered shard members identify
    the matching gradient slice for the remaining replica all-reduces.
    """
    spec = execution['fsdp_domain']
    owners = {'dense': {}, 'expert': {}}
    outer = {}
    for stage in range(mesh.pp):
        for d in range(mesh.dp):
            ranks = mesh.ranks(d, stage)
            for slot, rank in enumerate(ranks):
                owners['dense'][rank] = (stage, slot % mesh.tp)
                outer[rank] = d
            for slot, index in enumerate(mesh.expert_order):
                owners['expert'][ranks[index]] = (stage, slot % (mesh.ep * mesh.expert_tp))

    domains = {}
    hierarchy = execution.get('dp_domain_size')
    for family, ownership in owners.items():
        families = defaultdict(list)
        for rank, key in ownership.items():
            families[key].append(rank)
        available = len(next(iter(families.values())))
        supplied = spec.get(family + '_groups')
        common = execution.get('fsdp', 1)
        requested = spec.get(family + '_degree')
        within_domain = available // (mesh.dp // hierarchy) if hierarchy else available
        degree = requested or (len(supplied[0]) if supplied else None)
        if degree is None:
            degree = common if common > 1 else within_domain
        if available % degree:
            raise ValueError(f'FSDP {family} degree {degree} must divide the {available} copies of each compatible shard')
        by_family = defaultdict(list)
        if supplied is None:
            for key, members in families.items():
                by_family[key] = [members[i:i+degree] for i in range(0, available, degree)]
        else:
            used = set()
            for physical in supplied:
                group = [f'npu{i}' for i in physical]
                if len(group) != degree:
                    raise ValueError(f'FSDP {family}_groups must all have degree {degree}')
                if any(r not in ownership or r in used for r in group) or len(set(group)) != degree:
                    raise ValueError(f'FSDP {family}_groups must partition physical ranks exactly once')
                key = ownership[group[0]]
                if any(ownership[r] != key for r in group):
                    raise ValueError(f'FSDP {family} group {physical} mixes different pre-FSDP parameter shards or PP stages')
                used.update(group)
                by_family[key].append(group)
            if len(used) != len(ownership):
                raise ValueError(f'FSDP {family}_groups must cover every physical rank')
        shards = [[] for _ in range(mesh.pp)]
        replicas = [[] for _ in range(mesh.pp)]
        indices = {}
        for (stage, _), groups in by_family.items():
            if hierarchy:
                if any(len({outer[r] // hierarchy for r in group}) > 1 for group in groups):
                    raise ValueError(f'Each FSDP {family} shard group must fit inside one configured dp_domain_size')
                groups.sort(key=lambda group: outer[group[0]] // hierarchy)
            shards[stage].extend(groups)
            replicas[stage].extend(list(column) for column in zip(*groups))
            for group in groups:
                indices.update((rank, lane) for lane, rank in enumerate(group))
        domains[family] = ShardDomain(degree, shards, replicas, indices)
    return domains


def describe_domains(execution, domains):
    return {'implementation': execution['fsdp_domain'].get('implementation', 'fsdp2'),
            **{family: domain.describe() for family, domain in domains.items()}}


def legacy_domains(mesh, degree):
    """The scalar contract shards all parameter families along outer DP."""
    stages = []
    for stage in range(mesh.pp):
        ranks = [mesh.ranks(d, stage) for d in range(mesh.dp)]
        shards = [list(group) for start in range(0, mesh.dp, degree) for group in zip(*ranks[start:start+degree])]
        replicas = [list(group) for lane in range(degree) for group in zip(*ranks[lane::degree])]
        stages.append({'stage': stage, 'shard_groups': shards, 'replica_groups': replicas})
    domain = {'degree': degree, 'stages': stages}
    return {'implementation': 'legacy_full_shard', 'replica_scope': 'outer_dp', 'dense': domain, 'expert': domain}


class LayerBucket:
    """One communication bucket per wrapped layer, ownership family and weight dtype."""
    def __init__(self, builder, tensors, family, domain, implementation):
        first = tensors[0]
        self.family, self.domain = family, domain
        self.component = 'MoE' if family == 'expert' else 'System'
        self.owner_layer = first.layer
        self.tied_parameter = None
        self.dtype = builder.weight_dtype(first)
        degree = domain.degree
        counts = [builder.local_elements(t) for t in tensors]
        total = sum(counts)
        # Budget profiles may supply non-integral element counts and no dim-0 geometry.
        flat = implementation == 'fsdp' and all(t.shape and n == int(n) for t, n in zip(tensors, counts))
        chunk = ceil(total / degree) if flat else 0
        offset = 0
        self.partitions = {}
        self.gather_bytes = defaultdict(float)
        self.gradient_elements = defaultdict(float)
        self.replica_elements = [0.0] * degree
        for t, count in zip(tensors, counts):
            if flat:
                stop = offset + count
                owned = [max(0, min(stop, (i + 1) * chunk) - max(offset, i * chunk)) for i in range(degree)]
                # The flat allocation and collective include the final aligned tail.
                wire = count + (degree * chunk - total if t is tensors[-1] else 0)
                offset = stop
            else:
                tp = builder.mesh.expert_tp if t.ep_shard else builder.tp
                rows = t.local_rows(tp, builder.ep)
                if implementation == 'fsdp2' and rows:
                    rows_per_shard = ceil(rows / degree)
                    row_elements = count / rows
                    owned = [max(0, min(rows_per_shard, rows - i * rows_per_shard)) * row_elements for i in range(degree)]
                    wire = degree * rows_per_shard * row_elements
                else:
                    owned = [count / degree] * degree
                    wire = count
            self.partitions[t.name] = tuple(owned)
            component = t.component if builder.m['architecture'] else 'Unassigned'
            # Quantization scales are independently blocked on the sending shards.
            scale_counts = [chunk - n for n in self.replica_elements] if flat and t is tensors[-1] else owned
            scale_wire = degree * max(builder.scale(n, self.dtype) for n in scale_counts)
            self.gather_bytes[component] += wire * DTYPE_BYTES[self.dtype] + scale_wire
            self.gradient_elements[component] += wire
            for i, n in enumerate(owned):
                self.replica_elements[i] += n
        self.flat_padding = [max(0, chunk - n) for n in self.replica_elements] if flat else []
        if flat:
            self.replica_elements = [chunk] * degree
        self.last_tensor = tensors[-1]


class TrainingShards:
    def __init__(self, builder):
        self.domains = resolve_domains(builder.mesh, builder.x)
        self.implementation = builder.x['fsdp_domain']['implementation']
        self.layers = {}
        self.alias_buckets = defaultdict(list)
        self.partitions = {}
        self.padding = {}
        aliases = defaultdict(set)
        for tensor in builder.tensors:
            if tensor.alias_of:
                aliases[tensor.alias_of].add(tensor.layer)
        for layer in builder.by_layer:
            grouped = defaultdict(list)
            for tensor in builder.local_tensors(layer):
                family = 'expert' if tensor.ep_shard else 'dense'
                tied = tensor.name if aliases.get(tensor.name, set()) - {layer} else None
                grouped[family, builder.weight_dtype(tensor), tied].append(tensor)
            buckets = []
            for (family, _, tied), tensors in grouped.items():
                bucket = LayerBucket(builder, tensors, family, self.domains[family], self.implementation)
                buckets.append(bucket)
                if tied:
                    bucket.tied_parameter = tied
                    for consumer in sorted(aliases[tied] - {layer}):
                        self.alias_buckets[consumer].append(bucket)
            self.layers[layer] = buckets
            for bucket in buckets:
                self.partitions.update((name, (bucket.domain, partition)) for name, partition in bucket.partitions.items())
                if any(bucket.flat_padding):
                    self.padding[bucket.last_tensor.name] = bucket.flat_padding

    def owned_elements(self, tensor, rank):
        domain, owned = self.partitions[tensor.name]
        return owned[domain.index[rank]]

    def padding_elements(self, tensor, rank):
        padding = self.padding.get(tensor.name)
        return padding[self.partitions[tensor.name][0].index[rank]] if padding else 0

    def prepare_gradients(self, b, name, layer, ranks, step, micro, deps):
        allocations = []
        for i, bucket in enumerate(self.layers[layer] + self.alias_buckets[layer]):
            if bucket.domain.degree == 1:
                continue
            for rank in ranks:
                key = (step, bucket, rank)
                if key in b.grad_buffers:
                    continue
                buffers = [b.allocation(f'{name}.{rank}.bucket{i}.{comp}.grad', rank,
                    count * DTYPE_BYTES[b.p['gradient']], comp, 'unsharded_gradient_buffer', layer,
                    b.p['gradient'], fsdp_family=bucket.family, owner_layer=bucket.owner_layer, microbatch=micro)
                    for comp, count in bucket.gradient_elements.items() if count]
                b.grad_buffers[key] = [a.id for a in buffers]
                allocations.extend(buffers)
        return [b.barrier(name+'.gradient_buffers', deps, allocations)]

    def release_gradients(self, b, layer, ranks, step):
        return [aid for bucket in self.layers[layer] for rank in ranks
                for aid in b.grad_buffers.pop((step, bucket, rank), [])]

    def gather(self, b, name, layer, stage, deps, phase):
        ends, retained = [], []
        # A retained owner gather already covers use through a later tied alias.
        uses = self.layers[layer] + (self.alias_buckets[layer] if b.x['reshard_after_forward'] else [])
        for i, bucket in enumerate(uses):
            domain = bucket.domain
            if domain.degree == 1:
                continue
            size = sum(bucket.gather_bytes.values())
            prefix = f'{name}.fsdp.{bucket.family}.{i}'
            meta = self.metadata(bucket, 'fsdp_weights', bucket.dtype)
            for j, ranks in enumerate(domain.shards[stage]):
                allocations = [b.allocation(f'{prefix}.{rank}.{component}.gathered', rank, amount,
                    component, 'gathered_weight', layer, bucket.dtype, fsdp_family=bucket.family)
                    for rank in ranks for component, amount in bucket.gather_bytes.items() if amount]
                retained.extend(a.id for a in allocations)
                start = b.barrier(f'{prefix}.g{j}.allocate', deps, allocations)
                ends.extend(b.collective(f'{prefix}.g{j}', 'all_gather', ranks, size, [start],
                                         bucket.component, layer, phase=phase, wire_metadata=meta))
        return b.barrier(name+'.gather_done', ends or deps), retained

    def metadata(self, bucket, category, dtype):
        return {'category': category, 'parallel_axis': 'fsdp', 'fsdp_family': bucket.family,
                'fsdp_implementation': self.implementation, 'fsdp_shard_degree': bucket.domain.degree,
                'fsdp_owner_layer': bucket.owner_layer, 'buffer_dtype': dtype,
                **({'tied_parameter': bucket.tied_parameter} if bucket.tied_parameter else {})}

    def reduce_gradients(self, b, name, layer, stage, deps):
        terminals = []
        dtype = b.p['gradient']
        hierarchy = b.x.get('dp_domain_size')
        for i, bucket in enumerate(self.layers[layer]):
            domain = bucket.domain
            prefix = f'{name}.gradient.{bucket.family}.{i}'
            meta = self.metadata(bucket, 'fsdp_gradients', dtype)
            reduced = []
            if domain.degree > 1:
                full = sum(bucket.gradient_elements.values()) * DTYPE_BYTES[dtype]
                for j, ranks in enumerate(domain.shards[stage]):
                    reduced.extend(b.collective(f'{prefix}.rs{j}', 'reduce_scatter', ranks, full,
                        deps, bucket.component, layer, phase='backward', wire_metadata=meta))
            ready = reduced or deps
            replicas = []
            for j, ranks in enumerate(domain.replicas[stage]):
                elements = bucket.replica_elements[domain.index[ranks[0]]]
                if len(ranks) > 1 and elements:
                    local_width = len(ranks) // (b.dp // hierarchy) if hierarchy else 1
                    replicas.extend(replica_all_reduce(b.g, b.topo, f'{prefix}.replica{j}', [ranks],
                        elements * DTYPE_BYTES[dtype], ready, b.x, {**meta, 'category': 'data_parallel_gradients',
                        'component': bucket.component, 'phase': 'backward', 'layer': layer}, local_width=local_width))
            terminals.extend(replicas or ready)
        return list(dict.fromkeys(terminals or deps))

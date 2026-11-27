"""Training dependencies and communication on physical rank groups."""
from itertools import product


def prefetch_depth(execution):
    depth = execution.get('prefetch_layers')
    return int(execution['prefetch']) if depth is None else depth


def gather_dependencies(previous, gathers, completed_layers, depth):
    """Order gathers and bound the window to the active layer plus lookahead."""
    if not depth or not gathers:
        return previous
    deps = [gathers[-1]]
    if len(completed_layers) > depth:
        deps.append(completed_layers[-depth - 1])
    return deps


def replica_all_reduce(graph, topology, name, groups, size, deps, execution, metadata, *, local_width=None):
    """Reduce corresponding FSDP shards across all replicas, in place.

    Within each domain, RS gives one slice to each lane. Corresponding lanes
    reduce across domains, then local AG reconstructs the entire shard.
    """
    domain = execution.get('dp_domain_size')
    local = local_width if local_width is not None else (domain // execution['fsdp'] if domain else 1)
    terminals = []
    for index, ranks in enumerate(groups):
        prefix = f'{name}.g{index}'

        def collective(suffix, kind, members, amount, predecessors, scope):
            return topology.collective(graph, prefix + suffix, kind, members, amount,
                tuple(predecessors), metadata={**metadata, 'parallel_axis': 'dp_replica',
                    'dp_reduction_scope': scope, 'buffer_dtype': execution['precision']['gradient']})

        if local == 1 or local == len(ranks):
            scope = 'flat' if domain is None else ('within_domain' if local == len(ranks) else 'between_domains')
            terminals.extend(collective('', 'all_reduce', ranks, size, deps, scope))
            continue
        domains = [ranks[start:start + local] for start in range(0, len(ranks), local)]
        scattered = [collective(f'.local{d}.rs', 'reduce_scatter', members, size, deps, 'within_domain')
                     for d, members in enumerate(domains)]
        # Each ring terminal covers the final round, including all member ranks.
        ready = tuple(end for ends in scattered for end in ends)
        reduced = [end for lane, members in enumerate(zip(*domains))
                   for end in collective(f'.lane{lane}.ar', 'all_reduce', list(members),
                                         size / local, ready, 'between_domains')]
        for d, members in enumerate(domains):
            terminals.extend(collective(f'.local{d}.ag', 'all_gather', members, size, reduced, 'within_domain'))
    return list(dict.fromkeys(terminals or deps))


def domain_rank_order(layout, domain_size, model_ranks=None):
    """Map logical [dp, pp, ep, tp] coordinates to contiguous physical DP domains."""
    mp = model_ranks or layout['pp'] * layout['ep'] * layout['tp']
    return [(d // domain_size * mp + model_rank) * domain_size + d % domain_size
            for d, model_rank in product(range(layout['dp']), range(mp))]


def dp_domain_summary(execution, su_size, npu_count=None, fsdp_domains=None):
    dp, fsdp = execution['dp'], execution['fsdp']
    mp = npu_count // dp if npu_count is not None else execution['tp'] * execution['ep'] * execution['pp']
    order = execution['rank_order'] or range(dp * mp)
    domain = execution.get('dp_domain_size')

    def spans(width):
        return [len({order[d * mp + lane] // su_size for d in range(start, start + width)})
                for start in range(0, dp, width) for lane in range(mp)]

    domain_spans = spans(domain or dp)
    if fsdp_domains:
        physical = {}
        for family in ('dense', 'expert'):
            desc = fsdp_domains[family]
            groups = [group for stage in desc['stages'] for group in stage['shard_groups']]
            physical[family] = {'degree': desc['degree'], 'shard_group_count': len(groups),
                'shard_groups_crossing_su': sum(len({int(r[3:]) // su_size for r in group}) > 1 for group in groups)}
        return {'dp': dp, 'dp_domain_size': domain, 'domains_per_dp_group': dp // (domain or dp),
                'dp_group_count': mp, 'max_su_domains_per_dp_domain': max(domain_spans),
                'dp_domains_crossing_su': sum(n > 1 for n in domain_spans), 'fsdp_by_family': physical}
    fsdp_spans = spans(fsdp)
    return {'dp': dp, 'fsdp': fsdp, 'dp_domain_size': domain, 'domains_per_dp_group': dp // (domain or dp),
            'replica_group_size': dp // fsdp, 'dp_group_count': mp,
            'max_su_domains_per_dp_domain': max(domain_spans),
            'dp_domains_crossing_su': sum(n > 1 for n in domain_spans),
            'fsdp_groups_crossing_su': sum(n > 1 for n in fsdp_spans)}

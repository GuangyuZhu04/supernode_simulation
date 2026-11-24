"""Optional transfer-class arbitration on the training fabric."""
from collections import Counter, defaultdict
from itertools import product


def configure_training_qos(graph, execution):
    policy = execution['features']['training_qos']
    weights = policy['weights']
    order = execution['rank_order'] or range(execution['dp'] * execution['pp'] * execution['ep'] * execution['tp'])
    coords = product(*(range(execution[axis]) for axis in ('dp', 'pp', 'ep', 'tp')))
    logical = {f'npu{physical}': coord for physical, coord in zip(order, coords)}
    for event in graph.events:
        meta = event.metadata
        if meta.get('type') != 'transfer' or not ({'SU', 'SO'} & meta['physical_bytes_by_domain'].keys()):
            continue
        src, dst = logical.get(meta['src']), logical.get(meta['dst'])
        if src is None or dst is None:
            continue
        if meta.get('parallel_axis') == 'dp_replica' or meta.get('category') == 'data_parallel_gradients':
            traffic = 'dp'
        elif src[3] != dst[3]:
            traffic = 'tp'
        elif src[2] != dst[2]:
            traffic = 'ep'
        elif src[1] != dst[1]:
            traffic = 'pp'
        else:
            traffic = {'all_gather': 'fsdp_gather', 'reduce_scatter': 'fsdp_reduce_scatter',
                       'all_reduce': 'dp'}.get(meta.get('collective'), 'other')
        meta['traffic_class'] = traffic
        if policy['enabled']:
            meta['qos_weight'] = weights.get(traffic, 1)
    if policy['enabled']:
        graph.metadata['training_qos'] = {'weights': dict(weights), 'default_weight': 1,
            'policy': 'Per-transfer weighted sharing on contended fabric resources; higher weight wins simultaneous collective-slot admission. Active slots are nonpreemptive.'}


def priority_order(ordered, events):
    """Reorder only transfers competing for the same ready collective slots."""
    indices = [i for i, eid in enumerate(ordered) if events[eid].slots and 'qos_weight' in events[eid].metadata]
    counts = Counter(slot for i in indices for slot in events[ordered[i]].slots)
    competing = [i for i in indices if any(counts[s] > 1 for s in events[ordered[i]].slots)]
    winners = sorted((ordered[i] for i in competing), key=lambda eid: -events[eid].metadata['qos_weight'])
    for i, eid in zip(competing, winners):
        ordered[i] = eid
    return ordered


def _fabric(key):
    return key.startswith(('link:', 'shared:', 'io:')) or any(s in key for s in ('.nic.', '.SU.', '.SO.'))


def weighted_fabric_rates(services, capacities, weights, fair_rates):
    """Fill residual capacity with per-resource entitlements, freezing saturated paths.

    Each resource offers the pending flows a fraction of their solo rate.
    Fabric offers are weighted; memory and compute offers use equal weights.
    A flow takes its most restrictive offer. Advance along these proposals
    until a resource fills, then redistribute the remaining capacity.
    """
    incidence = defaultdict(list)
    for eid, (_, demands) in services.items():
        for key, demand in demands:
            incidence[key].append((eid, demand))
    weighted = {key for key, flows in incidence.items() if len(flows) > 1 and _fabric(key)
                and len({weights.get(eid, 1) for eid, _ in flows}) > 1}
    if not weighted:
        return fair_rates(services, capacities)
    fractions = dict.fromkeys(services, 0.0)
    remaining = {key: capacities[key] for key in incidence}
    pending = {eid for eid, (solo, _) in services.items() if solo > 0}
    while pending:
        offers = dict.fromkeys(pending, float('inf'))
        active = {}
        for key, flows in incidence.items():
            flows = [(eid, amount) for eid, amount in flows if eid in pending]
            if not flows:
                continue
            active[key] = flows
            if key in weighted:
                total = sum(amount * weights.get(eid, 1) for eid, amount in flows)
                for eid, _ in flows:
                    offers[eid] = min(offers[eid], remaining[key] / total * weights.get(eid, 1))
            else:
                offer = remaining[key] / sum(amount for _, amount in flows)
                for eid, _ in flows:
                    offers[eid] = min(offers[eid], offer)
        slopes = {key: sum(amount * offers[eid] for eid, amount in flows) for key, flows in active.items()}
        limits = {key: remaining[key] / slope for key, slope in slopes.items() if slope > 0}
        step = min(limits.values())
        for eid in pending:
            fractions[eid] += step * offers[eid]
        for key, slope in slopes.items():
            remaining[key] = max(0.0, remaining[key] - step * slope)
        for key, limit in limits.items():
            if limit <= step * (1 + 1e-12):
                pending.difference_update(eid for eid, _ in active[key])
    return {eid: fraction * services[eid][0] for eid, fraction in fractions.items()}

"""Two independent process meshes over the ranks of one pipeline stage."""
from dataclasses import dataclass
from itertools import product


def independent_groups(execution):
    return (execution.get('cp', 1) != 1 or bool(execution.get('expert_rank_order'))
            or any(execution.get(key) is not None for key in ('expert_tp', 'attention_dp', 'expert_dp')))


def configured_groups(execution, count):
    """Physical group membership for an already validated single- or dual-pool deployment."""
    explicit_fsdp = execution.get('fsdp_domain') is not None
    use_mesh = independent_groups(execution) or explicit_fsdp
    if not use_mesh and execution.get('fsdp', 1) == 1:
        return {}
    pools = execution['features']['prefill_decode_disaggregation']
    if pools['enabled']:
        return {'parallel_groups_by_pool': {phase: RankLayout.resolve(
            {**execution, 'rank_order': pools[phase+'_ranks']}, count//2).describe() for phase in ('prefill', 'decode')}}
    layout = RankLayout.resolve(execution, count)
    result = {'parallel_groups': layout.describe()} if use_mesh else {}
    if explicit_fsdp:
        from .fsdp import describe_domains, resolve_domains
        result['fsdp_domains'] = describe_domains(execution, resolve_domains(layout, execution))
    elif execution.get('fsdp', 1) > 1:
        from .fsdp import legacy_domains
        result['fsdp_domains'] = legacy_domains(layout, execution['fsdp'])
    return result


@dataclass(frozen=True)
class RankLayout:
    tp: int
    cp: int
    attention_dp: int
    expert_tp: int
    ep: int
    expert_dp: int
    pp: int
    dp: int
    order: tuple | range
    expert_order: tuple | range

    @classmethod
    def resolve(cls, execution, count):
        tp, ep, pp, dp = (execution.get(k, 1) for k in ('tp', 'ep', 'pp', 'dp'))
        cp = execution.get('cp', 1)
        independent = independent_groups(execution)
        etp = (execution.get('expert_tp') or 1) if independent else tp
        if not independent and execution.get('fsdp_domain') is None and count != tp * ep * pp * dp:
            raise ValueError('tp * pp * ep * dp must equal hardware.npu_count; fsdp is a DP subgroup')
        width, remainder = divmod(count, pp * dp)
        if remainder or width % (tp * cp) or width % (etp * ep):
            raise ValueError('Each DP replica and PP stage must contain whole attention TP × CP and expert TP × EP groups')
        adp, edp = width // (tp * cp), width // (etp * ep)
        for key, actual in (('attention_dp', adp), ('expert_dp', edp)):
            if execution.get(key) not in (None, actual):
                raise ValueError(f'{key} must be {actual} for the configured physical stage width {width}')
        supplied = execution.get('rank_order')
        order = tuple(supplied) if supplied else range(count)
        if supplied and (len(order) != count or len(set(order)) != count):
            raise ValueError('rank_order must be a permutation of physical NPU indices')
        expert_order = execution.get('expert_rank_order')
        if expert_order and sorted(expert_order) != list(range(width)):
            raise ValueError(f'expert_rank_order must be a permutation of the {width} stage-local rank indices')
        return cls(tp, cp, adp, etp, ep, edp, pp, dp, order, tuple(expert_order) if expert_order else range(width))

    @property
    def width(self):
        return self.tp * self.cp * self.attention_dp

    def ranks(self, d, stage):
        start = (d * self.pp + stage) * self.width
        return [f'npu{n}' if isinstance(n, int) else n for n in self.order[start:start + self.width]]

    def groups(self, d, stage):
        ranks = self.ranks(d, stage)
        def chunks(members, size):
            return [members[i:i+size] for i in range(0, len(members), size)]
        attention = chunks(ranks, self.tp)
        experts = chunks([ranks[i] for i in self.expert_order], self.expert_tp)
        return {
            'attention_tp': attention,
            'cp': [[attention[a*self.cp+c][t] for c in range(self.cp)]
                   for a, t in product(range(self.attention_dp), range(self.tp))],
            'attention_dp': [[attention[a*self.cp+c][t] for a in range(self.attention_dp)]
                             for c, t in product(range(self.cp), range(self.tp))],
            'expert_tp': experts,
            'ep': [[experts[r*self.ep+e][t] for e in range(self.ep)]
                   for r, t in product(range(self.expert_dp), range(self.expert_tp))],
            'expert_dp': [[experts[r*self.ep+e][t] for r in range(self.expert_dp)]
                          for e, t in product(range(self.ep), range(self.expert_tp))],
        }

    def describe(self):
        return {'stage_width': self.width, 'tp': self.tp, 'cp': self.cp,
                'attention_dp': self.attention_dp, 'expert_tp': self.expert_tp,
                'ep': self.ep, 'expert_dp': self.expert_dp, 'pp': self.pp, 'dp': self.dp,
                'stages': [{'replica': d, 'stage': s, 'groups': self.groups(d, s)}
                           for d, s in product(range(self.dp), range(self.pp))]}


def shard_rows(length, shard, degree):
    """Number of positions 0..length-1 whose index modulo degree is shard."""
    return (length + degree - 1 - shard) // degree


def shard_pairs(length, queries, shard, degree, window=None):
    """Exact causal pairs for cyclic query ownership, including a sliding window."""
    def prefix(n):
        rows = shard_rows(n, shard, degree)
        linear = rows if window is None else min(rows, shard_rows(window, shard, degree))
        return linear * (shard + 1) + degree * linear * (linear - 1) // 2 + (rows-linear) * (window or 0)
    return prefix(length) - prefix(length - queries)

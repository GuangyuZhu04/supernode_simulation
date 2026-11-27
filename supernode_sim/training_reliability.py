"""Bind fault calendars to simulated training steps or explicitly supplied cluster profiles."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from math import ceil
from pathlib import Path

from pydantic import Field, StrictInt, model_validator

from .config import FaultCampaign, Source, Spec, TrainingRecovery, read_data
from .engine import simulate as simulate_graph
from .reliability import Placement, fault_trace, simulate_trial, summarize, trace_digest


class ClusterProfile(Spec):
    installed_cards: StrictInt = Field(gt=0)
    su_size: StrictInt = Field(gt=0)
    cards_per_host: StrictInt = Field(gt=0)
    cards_per_replica: StrictInt = Field(gt=0, description='Size of a complete, independently recoverable replica.')
    tokens_per_replica_step: StrictInt = Field(gt=0)
    step_time_s: float = Field(gt=0)
    source: Source

    def placement(self, policy):
        if self.installed_cards % self.su_size or policy.spares_per_su >= self.su_size:
            raise ValueError('Installed cards must form whole SUs, each with active cards after reserving spares')
        active = self.installed_cards - self.installed_cards // self.su_size * policy.spares_per_su
        if active % self.cards_per_replica:
            raise ValueError('Active cards after sparing must fit whole recovery replicas')
        replicas = active // self.cards_per_replica
        return Placement(self.installed_cards, self.su_size, self.cards_per_host, policy.spares_per_su,
                         tuple(slot // self.cards_per_replica for slot in range(active)),
                         (self.tokens_per_replica_step,) * replicas)


class Sensitivity(Spec):
    card_rate_factor: float = Field(default=1, ge=0)
    recovery_time_factor: float = Field(default=1, ge=0)


class ReliabilityStudy(FaultCampaign):
    target_tokens: StrictInt = Field(gt=0)
    clusters: dict[str, ClusterProfile] = Field(min_length=1)
    policies: dict[str, TrainingRecovery] = Field(min_length=1)
    variants: dict[str, Sensitivity] = Field(default_factory=lambda: {'baseline': Sensitivity()}, min_length=1)
    sources: dict[str, Source] = Field(default_factory=dict)

    @model_validator(mode='after')
    def link_timing(self):
        if any(f.scope == 'link' for f in self.faults.values()):
            raise ValueError('Link faults need routed bandwidth resources. Use pretrain.reliability in a native training config.')
        return self


def _case(campaign, policy, layout, target, step_time, variant):
    trials = []
    recovery_times = defaultdict(lambda: defaultdict(list))
    for index in range(campaign.trials):
        events = fault_trace(campaign, layout, index, variant.card_rate_factor)
        row, records = simulate_trial(campaign, policy, layout, target, step_time, events,
                                      recovery_factor=variant.recovery_time_factor)
        trials.append(dict(trial=index, trace_sha256=trace_digest(events), **row))
        if index == 0:
            sample = records
        for record in records:
            for key in ('detected_at_s', 'first_commit_at_s', 'full_capacity_at_s', 'full_throughput_at_s', 'physical_repaired_at_s'):
                if record['effect'] == 'standby_only' and key not in ('detected_at_s', 'physical_repaired_at_s'):
                    continue
                at = record[key]
                recovery_times[record['fault']][key].append(None if at is None else at - record['at_s'])
    by_fault = {}
    for name, milestones in recovery_times.items():
        by_fault[name] = {}
        for key, values in milestones.items():
            observed = sorted(v for v in values if v is not None)
            n = len(observed)
            by_fault[name][key] = {'observed': n, 'pending_at_end': len(values) - n,
                                   'observed_p50_s': observed[ceil(0.5 * n) - 1] if n else None,
                                   'observed_p95_s': observed[ceil(0.95 * n) - 1] if n else None}
    width = sum(layout.tokens_by_unit)
    rates = {}
    for name, fault in campaign.faults.items():
        rate = fault.rate_per_unit_hour * (variant.card_rate_factor if fault.scope == 'card' else 1)
        installed = layout.count(fault.scope)
        rates[name] = {'scope': fault.scope, 'installed_units': installed, 'rate_per_unit_hour': rate,
                       'installed_events_per_hour': installed * rate}
        if fault.scope == 'card':
            active_rate = len(layout.unit_of_slot) * rate
            rates[name]['healthy_active_card_events_per_hour'] = active_rate
            rates[name]['healthy_active_card_mean_interval_h'] = 1 / active_rate if active_rate else None
    summary = summarize(trials)
    exposure_hours = sum(row['wall_s'] for row in trials) / 3600
    summary['observed_physical_events_per_hour'] = sum(row['faults_observed'] for row in trials) / exposure_hours
    summary['observed_global_stall_episodes_per_hour'] = sum(row['global_stall_episodes'] for row in trials) / exposure_hours
    return {'installed_cards': layout.installed, 'active_cards': len(layout.unit_of_slot),
            'installed_su_size': layout.su_size, 'recovery_units': len(layout.tokens_by_unit),
            'target_tokens': target, 'healthy_step_s': step_time(()), 'healthy_tokens_per_step': width,
            'healthy_duration_s': ((target + width - 1) // width) * step_time(()),
            'policy': policy.model_dump(), 'stochastic_fault_rates': rates, 'summary': summary, 'trials': trials,
            'first_trial_incidents': sample, 'recovery_latency_by_fault': by_fault}


def _native_placement(config, graph, policy, step_tokens):
    """Join outer DP replicas sharing any dense or expert FSDP state shard."""
    h, e = config.hardware, config.execution
    active_width = h.npu_count // e.dp
    outer = [0] * h.npu_count
    for rank, physical in enumerate(e.rank_order or range(h.npu_count)):
        outer[physical] = rank // active_width
    parents = list(range(e.dp))

    def root(d):
        while d != parents[d]:
            parents[d] = parents[parents[d]]
            d = parents[d]
        return d

    domains = graph.metadata.get('fsdp_domains', {})
    for family in ('dense', 'expert'):
        for stage in domains.get(family, {}).get('stages', ()):
            for group in stage['shard_groups']:
                owner = root(outer[int(group[0][3:])])
                for rank in group[1:]:
                    parents[root(outer[int(rank[3:])])] = owner
    units = {}
    unit_for_outer = [units.setdefault(root(d), len(units)) for d in range(e.dp)]
    sizes = Counter(unit_for_outer)
    tokens_per_outer, rem = divmod(step_tokens, e.dp)
    if rem or not tokens_per_outer:
        raise ValueError('Pretraining effective tokens must partition into complete data-parallel batches')
    spares = policy.spares_per_su
    installed_width = h.su_size + spares
    if spares and h.npu_count > h.su_size and (
            h.su_size % h.npus_per_host or installed_width % h.npus_per_host):
        raise ValueError('Native local spares must preserve the simulated host partition across SUs')
    return Placement(h.npu_count // h.su_size * installed_width, installed_width, h.npus_per_host, spares,
                     tuple(unit_for_outer[d] for d in outer),
                     tuple(sizes[u] * tokens_per_outer for u in range(len(units))))


def native_reliability(config, graph, metrics):
    pretrain = config.workload.training.pretrain
    campaign = pretrain.reliability
    steps = config.workload.training_steps
    step_tokens, remainder = divmod(metrics['training_task']['effective_tokens'], steps)
    if remainder:
        raise ValueError('A representative pretraining step needs repeatable effective token counts')
    layout = _native_placement(config, graph, campaign.policy, step_tokens)
    for fault in campaign.faults.values():
        for name in fault.resources:
            resource = graph.resources.get(name)
            if resource is None or resource.unit != 'byte/s':
                raise ValueError(f'Unknown or non-bandwidth resource in link fault: {name}')
    timings = []

    @lru_cache(maxsize=None)
    def step_time(links):
        if not links:
            return metrics['step_time_s']
        limits = {}
        for name in links:
            fault = campaign.faults[name]
            for key in fault.resources:
                limits[key] = min(limits.get(key, 1.0), fault.bandwidth_fraction)
        resources = graph.resources.copy()
        for key, fraction in limits.items():
            r = resources[key]
            resources[key] = replace(r, capacity=r.capacity * fraction, capacity_schedule=tuple(
                (t, cap * fraction) for t, cap in r.capacity_schedule))
        execution = simulate_graph(replace(graph, resources=resources), retain_event_payload=False)
        if execution['status'] != 'feasible':
            raise ValueError(f'Degraded training step infeasible for {links}: {execution["diagnostics"]}')
        mean_s = execution['duration_s'] / steps
        timings.append({'faults': list(links), 'resource_capacity_fractions': limits, 'step_time_s': mean_s})
        return mean_s

    case = _case(campaign, campaign.policy, layout, pretrain.dataset_tokens * pretrain.epochs,
                 step_time, Sensitivity())
    case.update(cluster=config.hardware.name, policy_name=campaign.policy.mode, variant='baseline',
                timing_basis='simulated_mean_optimizer_step', degraded_step_profiles=timings,
                first_step_s=metrics['first_step_time_s'], steady_step_s=metrics['steady_step_time_s'])
    return {'schema_version': 1, 'campaign': campaign.model_dump(), 'cases': [case],
            'sources': {k: v.model_dump(exclude_none=True) for k, v in config.sources.items()}}


def run_study(manifest):
    spec = ReliabilityStudy.model_validate(read_data(manifest))
    cases = []
    for name, cluster in spec.clusters.items():
        for policy_name, policy in spec.policies.items():
            layout = cluster.placement(policy)
            for variant_name, variant in spec.variants.items():
                case = _case(spec, policy, layout, spec.target_tokens, lambda _: cluster.step_time_s, variant)
                case.update(cluster=name, policy_name=policy_name, variant=variant_name,
                            timing_basis='supplied_replica_step', step_source=cluster.source.model_dump(exclude_none=True))
                cases.append(case)
    return {'schema_version': 1, 'campaign': spec.model_dump(), 'cases': cases}


def write_reliability_report(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'training_reliability.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    rows = [{**{key: c[key] for key in ('cluster', 'policy_name', 'variant', 'installed_cards',
                                         'active_cards', 'installed_su_size', 'target_tokens', 'healthy_duration_s')},
             **c['summary']} for c in result['cases']]
    with (directory / 'training_reliability.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path

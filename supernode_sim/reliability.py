"""Pretraining progress on a wall-clock fault calendar. Times are in seconds."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil, floor, inf, isclose, isfinite, nextafter
from random import Random

from .config import FaultCampaign, TrainingRecovery


@dataclass(frozen=True)
class Placement:
    installed: int
    su_size: int
    cards_per_host: int
    spares_per_su: int
    # Active slots are in physical order, omitting reserved cards at each SU's end.
    unit_of_slot: tuple[int, ...]
    tokens_by_unit: tuple[int, ...]

    def count(self, scope):
        if scope == 'card':
            return self.installed
        if scope == 'host':
            return ceil(self.installed / self.cards_per_host)
        return self.installed // self.su_size if scope == 'su' else 1

    def members(self, scope, target):
        width = {'card': 1, 'host': self.cards_per_host, 'su': self.su_size}[scope]
        return range(target * width, min(self.installed, (target + 1) * width))

    def unit(self, card):
        domain, offset = divmod(card, self.su_size)
        width = self.su_size - self.spares_per_su
        return self.unit_of_slot[domain * width + offset] if offset < width else None


@dataclass(frozen=True, order=True, slots=True)
class Incident:
    at_s: float
    name: str
    target: int
    detect_s: float
    repair_s: float
    origin: str


def _random_incidents(name, fault, count, horizon, rate, seed, keep=1.0):
    if not isfinite(rate):
        raise ValueError(f'Aggregate fault rate for {name} exceeds numeric range')
    if not rate or not keep:
        return
    rng = Random(seed)
    marks = Random('thin:' + seed)
    at = rng.expovariate(rate)
    while at <= horizon:
        target = rng.randrange(count)
        if keep == 1 or marks.random() < keep:
            yield Incident(at, name, target, fault.detect_s, fault.repair_s, 'poisson')
        following = at + rng.expovariate(rate)
        if following <= at:
            raise ValueError(f'Fault interarrival for {name} is below clock resolution')
        at = following


def fault_trace(spec: FaultCampaign, layout: Placement, trial: int, card_rate_factor=1.0):
    """Sample installed physical units on calendar time, independently of job policy."""
    events = []
    for event in spec.trace:
        fault = spec.faults[event.fault]
        if event.target >= layout.count(fault.scope):
            raise ValueError(f'Fault {event.fault}: physical target {event.target} outside the installed layout')
        events.append(Incident(event.at_s, event.fault, event.target,
                               fault.detect_s if event.detect_s is None else event.detect_s,
                               fault.repair_s if event.repair_s is None else event.repair_s, 'trace'))
    for name, fault in sorted(spec.faults.items()):
        count = layout.count(fault.scope)
        rate = fault.rate_per_unit_hour * count / 3600
        factor = card_rate_factor if fault.scope == 'card' else 1
        seed = f'{spec.seed}/{trial}/{name}'
        # Couple sensitivities by thinning the baseline or superposing an independent stream.
        # The surviving baseline incidents retain both their timestamps and physical targets.
        events.extend(_random_incidents(name, fault, count, spec.horizon_s, rate, seed, min(1, factor)))
        if factor > 1:
            events.extend(_random_incidents(name, fault, count, spec.horizon_s, (factor - 1) * rate, 'extra:' + seed))
    events.sort()
    return events


class Progress:
    """Only finished optimizer steps advance the data cursor. Checkpoints commit atomically."""

    def __init__(self, target, interval, write_s):
        self.target = target
        self.interval = interval
        self.write_s = write_s
        self.tokens = self.steps = self.saved_tokens = self.saved_steps = 0
        self.partial = 0.0
        self.partial_width = 0
        self.write_left = 0.0
        self.checkpoints = 0
        self.writing_s = 0.0
        self.rollback_tokens = 0
        self.partial_lost = 0.0

    def _save(self):
        self.saved_tokens, self.saved_steps = self.tokens, self.steps
        self.checkpoints += 1

    def _commit(self, steps, width):
        self.steps += steps
        self.tokens = min(self.target, self.tokens + steps * width)
        if self.tokens < self.target and self.interval and self.steps % self.interval == 0:
            self.write_left = self.write_s
            if not self.write_left:
                self._save()

    def interrupt(self, *, rollback):
        self.partial_lost += self.partial * self.partial_width
        self.partial = 0.0
        if rollback:
            self.rollback_tokens += self.tokens - self.saved_tokens
            self.tokens, self.steps = self.saved_tokens, self.saved_steps
            self.write_left = 0.0
        elif self.write_left:
            # Healthy replicas retain the training state and retry the interrupted save.
            self.write_left = self.write_s

    def advance(self, now, end, step_s, width):
        """Batch complete steps and checkpoint cycles until the next external transition."""
        first_commit = None
        first_start = None
        while now < end and self.tokens < self.target:
            available = end - now
            if self.write_left:
                spent = min(available, self.write_left)
                self.writing_s += spent
                self.write_left -= spent
                now = min(end, now + spent)
                if not self.write_left:
                    self._save()
                continue
            if self.partial:
                remaining = (1 - self.partial) * step_s
                if available < remaining:
                    self.partial = min(nextafter(1.0, 0.0), self.partial + available / step_s)
                    return end, first_commit, first_start
                now = min(end, now + remaining)
                self._commit(1, self.partial_width)
                self.partial = 0.0
                if first_commit is None:
                    first_commit = now
                continue

            if first_start is None:
                first_start = now
            to_goal = (self.target - self.tokens + width - 1) // width
            if self.interval and self.steps == self.saved_steps:
                cycle_s = self.interval * step_s + self.write_s
                cycles = min(floor(available / cycle_s), (to_goal - 1) // self.interval)
                if cycles:
                    if first_commit is None:
                        first_commit = now + step_s
                    count = cycles * self.interval
                    self.steps += count
                    self.tokens += count * width
                    self.saved_steps, self.saved_tokens = self.steps, self.tokens
                    self.checkpoints += cycles
                    self.writing_s += cycles * self.write_s
                    now = min(end, now + cycles * cycle_s)
                    continue
            count = min(to_goal, floor(nextafter(available / step_s, inf)))
            if self.interval:
                count = min(count, self.interval - self.steps % self.interval)
            if not count:
                self.partial = min(nextafter(1.0, 0.0), available / step_s)
                self.partial_width = width
                return end, first_commit, first_start
            if first_commit is None:
                first_commit = min(end, now + step_s)
            now = min(end, now + count * step_s)
            self._commit(count, width)
        return now, first_commit, first_start


class Recovery:
    """Map physical outages to the currently bound workers and runnable recovery units."""

    def __init__(self, layout: Placement, policy: TrainingRecovery, scale):
        self.layout, self.policy, self.scale = layout, policy, scale
        self.barrier = 0.0
        self.global_restart = False
        self.units = {}  # unavailable recovery unit -> earliest time it can contribute
        self.links = {}  # degraded resource group -> physical repair time
        self.binding = {}  # changes to the initial physical placement after swaps
        self.free_spares = {}
        self.physical_down = {}
        self.swaps = self.spare_misses = 0

    def refresh(self, now):
        self.units = {unit: at for unit, at in self.units.items() if at > now}
        self.links = {name: at for name, at in self.links.items() if at > now}
        if self.barrier <= now:
            self.global_restart = False

    def next_transition(self, now):
        return min((t for t in (self.barrier, *self.units.values(), *self.links.values())
                    if t > now), default=inf)

    def _pool(self, domain):
        if domain not in self.free_spares:
            end = (domain + 1) * self.layout.su_size
            self.free_spares[domain] = set(range(end - self.layout.spares_per_su, end))
        return self.free_spares[domain]

    def mark_outages(self, now, simultaneous, faults):
        """A spare hit by another root cause at this instant cannot be reserved."""
        if self.policy.mode == 'local_spare':
            for event in simultaneous:
                fault = faults[event.name]
                if fault.scope != 'link':
                    repaired = now + event.detect_s * self.scale + event.repair_s * self.scale
                    for card in self.layout.members(fault.scope, event.target):
                        self.physical_down[card] = max(self.physical_down.get(card, 0), repaired)

    def _hard_fault(self, now, detected, repaired, members):
        policy = self.policy
        ready = {}
        for card in members:
            unit = self.binding.get(card, self.layout.unit(card))
            if unit is None:
                continue
            available = repaired
            if policy.mode == 'local_spare':
                available = self.physical_down[card]
                spares = self._pool(card // self.layout.su_size)
                # Reserve within the SU at incident time, then detect and switch.
                # A common-mode outage has already disabled all affected spares.
                spare = min((p for p in spares if self.physical_down.get(p, -inf) <= now
                             and p not in members), default=None)
                switch_at = detected + policy.swap_s * self.scale
                if spare is not None and switch_at < available:
                    self.binding[card], self.binding[spare] = None, unit
                    spares.remove(spare)
                    spares.add(card)
                    self.swaps += 1
                    available = switch_at
                elif spare is None:
                    self.spare_misses += 1
            ready[unit] = max(ready.get(unit, 0), available)
        return ready

    def fault(self, event, fault, progress):
        now = event.at_s
        detected = now + event.detect_s * self.scale
        repaired = detected + event.repair_s * self.scale
        record = {'at_s': now, 'fault': event.name, 'scope': fault.scope, 'target': event.target, 'origin': event.origin,
                  'detection_due_s': detected, 'physical_repair_due_s': repaired,
                  'first_commit_at_s': None, 'full_capacity_at_s': None, 'full_throughput_at_s': None}
        if fault.scope == 'link':
            self.links[event.name] = max(self.links.get(event.name, 0), repaired)
            return {**record, 'effect': 'bandwidth_degraded'}
        ready = self._hard_fault(now, detected, repaired, self.layout.members(fault.scope, event.target))
        if not ready:
            return {**record, 'effect': 'standby_only'}
        newly_out = ready.keys() - self.units.keys()
        rebuild = (self.policy.rebuild_s + self.policy.restore_s) * self.scale
        for unit, at in ready.items():
            self.units[unit] = max(self.units.get(unit, 0), at + rebuild)
        global_restart = (self.policy.mode != 'replica_isolation' or self.global_restart
                          or len(self.units) == len(self.layout.tokens_by_unit))
        if global_restart:
            self.global_restart = True
            self.barrier = max(self.barrier, *self.units.values())
            progress.interrupt(rollback=True)
        elif newly_out:
            self.barrier = max(self.barrier, detected + self.policy.isolate_s * self.scale)
            progress.interrupt(rollback=False)
        available = max(ready.values())
        return {**record, 'workers_available_due_s': available,
                'communication_ready_due_s': available + self.policy.rebuild_s * self.scale,
                'state_loaded_due_s': available + rebuild,
                'effect': 'global_restart' if global_restart else 'replica_offline',
                'affected_units': sorted(ready)}


def simulate_trial(campaign, policy, layout, target, step_time, events, *, recovery_factor=1.0):
    """Replay one installed-fleet trace; repairs and faults continue during job stalls."""
    recovery = Recovery(layout, policy, recovery_factor)
    work = Progress(target, campaign.checkpoint_interval_steps, campaign.checkpoint_write_s)
    healthy_width = sum(layout.tokens_by_unit)
    healthy_step_s = step_time(())
    now = stalled = degraded = 0.0
    stall_episodes = 0
    was_stalled = False
    index = 0
    records, wait_commit, wait_full, wait_step = [], [], [], []
    effects = defaultdict(int)
    while now < campaign.horizon_s and work.tokens < target:
        recovery.refresh(now)
        start = index
        while index < len(events) and events[index].at_s <= now:
            index += 1
        simultaneous = events[start:index]
        recovery.mark_outages(now, simultaneous, campaign.faults)
        for event in simultaneous:
            row = recovery.fault(event, campaign.faults[event.name], work)
            effects[row['effect']] += 1
            records.append(row)
            if row['effect'] != 'standby_only':
                wait_commit.append(row)
                wait_full.append(row)
                wait_step.append(row)
        recovery.refresh(now)
        if not recovery.units and not recovery.links and recovery.barrier <= now:
            for row in wait_full:
                row['full_capacity_at_s'] = now
            wait_full.clear()
        stop = min(campaign.horizon_s, recovery.next_transition(now),
                   events[index].at_s if index < len(events) else inf)
        if recovery.barrier > now:
            if not was_stalled:
                stall_episodes += 1
            was_stalled = True
            stalled += stop - now
            now = stop
            continue
        was_stalled = False
        width = healthy_width - sum(layout.tokens_by_unit[u] for u in recovery.units)
        duration = step_time(tuple(sorted(recovery.links)))
        end, first_commit, first_start = work.advance(now, stop, duration, width)
        if recovery.units or recovery.links:
            degraded += end - now
        elif first_start is not None:
            for row in wait_step:
                row['full_throughput_at_s'] = first_start
            wait_step.clear()
        if first_commit is not None:
            for row in wait_commit:
                row['first_commit_at_s'] = first_commit
            wait_commit.clear()
        now = end
    active = len(layout.unit_of_slot)
    recovery.refresh(now)
    if not recovery.units and not recovery.links and recovery.barrier <= now:
        for incident in wait_full:
            incident['full_capacity_at_s'] = now
    for incident in records:
        for observed, due in (('detected_at_s', 'detection_due_s'), ('physical_repaired_at_s', 'physical_repair_due_s')):
            incident[observed] = incident[due] if incident[due] <= now else None
    ideal_s = work.tokens * healthy_step_s / healthy_width
    idle_spare_hours = (layout.installed - active) * now / 3600
    active_loss_hours = active * (now - ideal_s) / 3600
    if active_loss_hours < 0 and isclose(now, ideal_s, rel_tol=1e-14):
        active_loss_hours = 0.0
    row = {'completed': work.tokens == target, 'wall_s': now, 'net_tokens': work.tokens,
           'net_tokens_s': work.tokens / now, 'effective_training_fraction': ideal_s / now,
           'lost_card_hours': idle_spare_hours + active_loss_hours, 'reserved_card_hours': idle_spare_hours,
           'active_loss_card_hours': active_loss_hours,
           'rollback_tokens': work.rollback_tokens, 'discarded_partial_token_equivalent': work.partial_lost,
           'checkpoint_write_s': work.writing_s, 'durable_checkpoints': work.checkpoints,
           'durable_tokens': work.saved_tokens, 'global_stall_s': stalled, 'global_stall_episodes': stall_episodes,
           'degraded_window_s': degraded,
           'faults_observed': index, 'effects': dict(effects), 'spare_swaps': recovery.swaps,
           'spare_misses': recovery.spare_misses}
    return row, records


def trace_digest(events):
    digest = sha256()
    for e in events:
        digest.update(json.dumps((e.at_s, e.name, e.target, e.detect_s, e.repair_s, e.origin),
                                 separators=(',', ':')).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def summarize(trials):
    """Nearest-rank quantiles. Censored completions sort beyond the observation horizon."""
    result = {'trials': len(trials), 'completed': sum(r['completed'] for r in trials)}
    result['censored'] = result['trials'] - result['completed']
    for key in ('wall_s', 'net_tokens_s', 'effective_training_fraction', 'lost_card_hours', 'rollback_tokens',
                'global_stall_s', 'checkpoint_write_s'):
        values = sorted(r[key] for r in trials if key != 'wall_s' or r['completed'])
        for label, q in (('p50', 0.50), ('p95', 0.95)):
            rank = ceil(q * len(trials))
            result[f'{key}_{label}'] = values[rank - 1] if rank <= len(values) else None
    return result

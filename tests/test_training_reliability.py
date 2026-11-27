"""Recovery semantics, native routed steps and the published FT-HSDP accounting window."""
from pathlib import Path

import pytest

from supernode_sim.analysis import run
from supernode_sim.config import Config, FaultCampaign, TrainingRecovery, load_config
from supernode_sim.reliability import Placement, fault_trace, simulate_trial, summarize
from supernode_sim.training_reliability import run_study


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def replay(events, *, layout=None, faults=None, policy=None, target=4000, horizon=100, **checkpoint):
    layout = layout or Placement(4, 4, 2, 0, (0, 0, 1, 1), (2, 2))
    spec = FaultCampaign(horizon_s=horizon, trace=events, faults=faults or {
        'card': {'scope': 'card', 'repair_s': 5}}, **checkpoint)
    strategy = TrainingRecovery(**(policy or {}))
    return simulate_trial(spec, strategy, layout, target, lambda _: 10, fault_trace(spec, layout, 0))


@pytest.mark.parametrize(('failure_at', 'lost_tokens', 'finish'), [(23, 8, 72), (24, 0, 49)])
def test_checkpoint_is_durable_only_after_write(failure_at, lost_tokens, finish):
    row, events = replay([{'at_s': failure_at, 'fault': 'card'}], target=16,
                         checkpoint_interval_steps=2, checkpoint_write_s=4)
    assert row['completed'] and row['net_tokens'] == 16
    assert row['rollback_tokens'] == lost_tokens
    assert row['wall_s'] == finish
    assert events[0]['first_commit_at_s'] == failure_at + 15
    assert events[0]['full_capacity_at_s'] == failure_at + 5


def test_billion_steps_skip_complete_checkpoint_cycles_and_round_final_batch():
    target = 4_000_000_003
    row, _ = replay([], target=target, horizon=20e9, checkpoint_interval_steps=100, checkpoint_write_s=2)
    assert row['wall_s'] == 10_000_000_010 + 20_000_000
    assert row['net_tokens'] == target
    assert row['durable_tokens'] == 4_000_000_000


def test_su_spares_stay_local_and_repaired_device_returns_to_pool():
    layout = Placement(8, 4, 2, 1, (0, 0, 0, 1, 1, 1), (3, 3))
    row, events = replay([{'at_s': t, 'fault': 'card', 'target': card}
                          for t, card in [(1, 0), (2, 1), (22, 2)]],
                         faults={'card': {'scope': 'card', 'repair_s': 20}},
                         layout=layout, policy={'mode': 'local_spare', 'spares_per_su': 1, 'swap_s': 2})
    assert row['spare_swaps'] == 2 and row['spare_misses'] == 1
    assert [r['workers_available_due_s'] for r in events] == [3, 22, 24]
    assert events[0]['physical_repaired_at_s'] == 21
    # The second failure prolongs the first interruption.
    assert events[0]['first_commit_at_s'] == 34
    assert row['global_stall_s'] == 23


def test_shared_fault_and_overlapping_replica_loss_roll_back_when_last_state_is_lost():
    policy = {'mode': 'replica_isolation', 'isolate_s': 2}
    # Host 0 contains ranks from both replicas. One root cause, two unavailable units.
    layout = Placement(4, 4, 2, 0, (0, 1, 0, 1), (2, 2))
    row, events = replay([{'at_s': 20, 'fault': 'host'}], layout=layout, policy=policy,
                         faults={'host': {'scope': 'host', 'repair_s': 5}})
    assert row['effects'] == {'global_restart': 1} and row['rollback_tokens'] == 8
    assert events[0]['affected_units'] == [0, 1]
    row, events = replay([{'at_s': 20, 'fault': 'card'}, {'at_s': 32, 'fault': 'card', 'target': 2}],
                         policy=policy, faults={'card': {'scope': 'card', 'repair_s': 30}})
    assert row['effects'] == {'replica_offline': 1, 'global_restart': 1}
    assert row['rollback_tokens'] == 10
    assert events[0]['first_commit_at_s'] == 32
    assert events[1]['full_capacity_at_s'] == 62


def test_simultaneous_fault_on_spare_and_step_boundary_for_rejoin():
    layout = Placement(4, 4, 2, 1, (0, 0, 0), (3,))
    row, incidents = replay([{'at_s': 1, 'fault': 'card', 'target': p} for p in (0, 3)],
                            layout=layout, policy={'mode': 'local_spare', 'spares_per_su': 1})
    assert row['spare_swaps'] == 0 and row['spare_misses'] == 1
    assert incidents[1]['effect'] == 'standby_only'
    # A replica becomes available five seconds into a surviving replica's step.
    row, (event,) = replay([{'at_s': 0, 'fault': 'card'}], policy={'mode': 'replica_isolation'})
    assert event['full_capacity_at_s'] == 5 and event['full_throughput_at_s'] == 10
    assert event['first_commit_at_s'] == 10 and row['net_tokens'] == 38


def test_overlapping_degraded_links_keep_each_limit_until_its_own_repair():
    layout = Placement(4, 4, 2, 0, (0, 0, 1, 1), (2, 2))
    spec = FaultCampaign(horizon_s=65, faults={
        name: {'scope': 'link', 'repair_s': 20, 'resources': [name], 'bandwidth_fraction': 0.5}
        for name in ('a', 'b')}, trace=[{'at_s': 0, 'fault': 'a'}, {'at_s': 10, 'fault': 'b'}])
    step_times = {(): 10, ('a',): 20, ('b',): 20, ('a', 'b'): 40}
    row, (a, b) = simulate_trial(spec, TrainingRecovery(), layout, 1000, step_times.__getitem__,
                                 fault_trace(spec, layout, 0))
    assert a['full_capacity_at_s'] == b['full_capacity_at_s'] == 30
    assert a['first_commit_at_s'] == b['first_commit_at_s'] == 25
    assert a['full_throughput_at_s'] == 37.5
    assert row['global_stall_s'] == 0 and row['degraded_window_s'] == 30


def test_rates_use_installed_physical_units_and_censoring_keeps_denominator():
    layout = Placement(8, 4, 2, 0, (0, 0, 0, 0, 1, 1, 1, 1), (4, 4))
    reserved = Placement(8, 4, 2, 1, (0, 0, 0, 1, 1, 1), (3, 3))
    spec = FaultCampaign(horizon_s=3600, seed=17, faults={'card': {'scope': 'card', 'rate_per_unit_hour': 4,
                                                                    'repair_s': 60}})
    trace = fault_trace(spec, layout, 0)
    assert trace and trace == fault_trace(spec, reserved, 0) and trace != fault_trace(spec, layout, 1)
    assert set(fault_trace(spec, layout, 0, 0.5)) < set(trace) < set(fault_trace(spec, layout, 0, 1.5))
    assert all(0 <= event.target < 8 for event in trace)
    completed, _ = replay([], target=12, horizon=40)
    incomplete, _ = replay([], target=24, horizon=40)
    summary = summarize([completed, incomplete])
    assert summary['wall_s_p50'] == 30 and summary['wall_s_p95'] is None
    assert summary['censored'] == 1


def test_trace_durations_override_the_cause_and_unobserved_milestones_stay_missing():
    _, (event,) = replay([{'at_s': 2, 'fault': 'card', 'detect_s': 7, 'repair_s': 40}], horizon=20)
    assert event['origin'] == 'trace' and event['detected_at_s'] == 9
    assert event['physical_repair_due_s'] == 49 and event['physical_repaired_at_s'] is None
    assert event['first_commit_at_s'] is None


def test_ft_hsdp_fixed_window():
    sync, local = run_study(EXAMPLES / 'training_reliability_ft_hsdp.yaml')['cases']
    assert sync['installed_cards'] == 98_304 and local['recovery_units'] == 12
    assert sync['trials'][0]['trace_sha256'] == local['trials'][0]['trace_sha256']
    assert sync['trials'][0]['global_stall_s'] == 600 and local['trials'][0]['global_stall_s'] == 180
    assert local['trials'][0]['degraded_window_s'] == 420
    assert sync['summary']['effective_training_fraction_p50'] == pytest.approx(8 / 18)
    assert local['summary']['effective_training_fraction_p50'] == pytest.approx((8 + 7 * 11 / 12) / 18)
    assert local['summary']['wall_s_p50'] is None
    assert local['first_trial_incidents'][0]['first_commit_at_s'] == 200


def test_native_link_profile_matches_actual_resource_schedule_and_net_budget():
    example = load_config(EXAMPLES / 'training_reliability_atlas950.yaml')
    result = run(example)
    case = result['training_reliability']['cases'][0]
    profile, = case['degraded_step_profiles']
    assert profile['step_time_s'] > case['healthy_step_s']
    data = example.model_dump()
    budget = data['workload']['training']['pretrain']
    del budget['reliability']
    data['hardware']['bandwidth_schedules'] = {
        key: [(0, result['requirements']['per_resource'][key]['capacity'] * factor)]
        for key, factor in profile['resource_capacity_fractions'].items()}
    assert run(data)['metrics']['step_time_s'] == pytest.approx(profile['step_time_s'])
    # With writes and faults disabled the original rounded pretraining projection is recovered.
    data['hardware'].pop('bandwidth_schedules')
    budget.update(dataset_tokens=130, reliability={'horizon_s': 10})
    healthy = run(data)['metrics']['training_task']['budget_projection']
    assert healthy['reliability']['wall_s_p50'] == pytest.approx(healthy['duration_s'])
    assert healthy['reliability']['net_tokens_s_p50'] == pytest.approx(130 / healthy['duration_s'])


def test_native_rank_placement_and_shared_fsdp_state():
    data = load_config(EXAMPLES / 'training_reliability_atlas950.yaml').model_dump()
    campaign = data['workload']['training']['pretrain']['reliability']
    campaign.update(policy={'mode': 'replica_isolation'}, trace=[{'at_s': 0, 'fault': 'card_loss', 'target': 1}])
    case = run(data)['training_reliability']['cases'][0]
    assert case['first_trial_incidents'][0]['affected_units'] == [1]
    assert case['first_trial_incidents'][0]['effect'] == 'replica_offline'
    data['execution']['fsdp'] = 2
    case = run(Config.model_validate(data))['training_reliability']['cases'][0]
    assert case['recovery_units'] == 1 and case['first_trial_incidents'][0]['effect'] == 'global_restart'

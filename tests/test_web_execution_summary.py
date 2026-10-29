"""Display summaries count the complete trace rather than sampled UI rows."""
from supernode_sim.web_worker import compact_result, execution_summary


def test_phase_summary_counts_full_trace_and_preserves_event_context():
    trace = [{
        'id': f'event-{i}', 'start_s': i / 1000, 'end_s': i / 1000 + .1,
        'duration_s': .1, 'slots': [],
        'metadata': {'phase': 'prefill' if i < 320 else 'decode',
                     'token_index': i, 'step': 2, 'microbatch': 3,
                     'src': 'npu0', 'dst': 'npu1', 'kernel': 'attention'},
    } for i in range(601)]
    result = compact_result({'mode': 'simulate', 'estimate': {'event_count': 601},
        'simulation': {'status': 'feasible', 'completed_events': 601, 'trace': trace}})
    assert len(result['trace']['events']) == 300
    assert result['trace']['sampled'] is True
    summary = result['execution_summary']
    assert summary['complete'] is True
    assert summary['planned_events'] == summary['completed_events'] == summary['trace_events'] == 601
    phases = {row['phase']: row for row in summary['phases']}
    assert phases['prefill']['event_count'] == 320
    assert phases['decode']['event_count'] == 281
    assert phases['prefill']['start_s'] == 0
    assert phases['prefill']['end_s'] == trace[319]['end_s']
    assert phases['decode']['start_s'] < phases['prefill']['end_s']  # Concurrent windows stay overlapping.
    assert phases['decode']['end_s'] == trace[-1]['end_s']
    for row in result['trace']['events']:
        source = trace[int(row['id'].split('-')[1])]
        assert row['metadata'] == source['metadata']


def test_phase_summary_retains_unclassified_events_and_bounds_many_phases():
    trace = [{'start_s': i, 'end_s': i + 1, 'duration_s': 1,
              'metadata': {'phase': f'phase-{i}'}} for i in range(120)]
    trace.insert(0, {'start_s': 0, 'end_s': 1, 'duration_s': 1})
    summary = execution_summary({'mode': 'simulate', 'estimate': {'event_count': 200},
        'simulation': {'status': 'infeasible', 'completed_events': 121, 'trace': trace}})
    assert summary['complete'] is False
    assert summary['planned_events'] == 200
    assert summary['completed_events'] == 121
    assert summary['grouped_phases'] is True
    assert len(summary['phases']) <= summary['phase_limit']
    assert sum(row['event_count'] for row in summary['phases']) == 121
    assert sum(row['event_duration_sum_s'] for row in summary['phases']) == 121
    assert next(row for row in summary['phases'] if row['phase'] == 'unclassified')['event_count'] == 1
    assert max(row['end_s'] for row in summary['phases']) == 120


def test_estimate_has_no_observed_execution_phases():
    result = compact_result({'mode': 'estimate', 'estimate': {'event_count': 89}})
    summary = result['execution_summary']
    assert summary['basis'] == 'estimate_only'
    assert summary['complete'] is None
    assert summary['planned_events'] == 89
    assert summary['completed_events'] is None
    assert summary['trace_events'] == 0
    assert summary['phases'] == []

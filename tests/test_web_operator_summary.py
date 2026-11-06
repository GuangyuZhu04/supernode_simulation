"""Web operator summaries retain exact ledgers and bounded observed chains."""
from copy import deepcopy

from supernode_sim.web_worker import compact_result, compact_event


def child(parent, index):
    return {'id': f'{parent}.op{index}', 'start_s': index, 'end_s': index + 1, 'duration_s': 1,
            'startup_s': .01, 'dependency_ready_s': index - .1, 'slot_wait_s': .1, 'memory_wait_s': 0,
            'slots': ['npu0.compute_stream'], 'work': {'npu0.compute': 48, 'npu0.hbm': 128},
            'metadata': {'kernel': 'dense.up', 'phase': 'backward', 'component': 'Dense',
                         'trace_parent_event': parent, 'original_kernel': 'dense', 'trace_id': 'synthetic',
                         'operation_index': index, 'matmul': {'m': 2, 'n': 3, 'k': 4},
                         'implementation': {'kernel_name': 'test'}, 'trace_source': 'synthetic'}}


def test_training_ledger_is_preserved_without_truncating_operator_rows():
    ledger = {'step': {'operator_breakdown': {'operators': [{'operator': f'op{i}', 'total_s': i / 100}
                                                         for i in range(120)]}},
              'steady_step': None, 'steady_step_unavailable_reason': 'At least two steps'}
    source = {'mode': 'simulate', 'metrics': {'training_latency_breakdown': ledger}}
    assert compact_result(source)['metrics']['training_latency_breakdown'] == ledger


def test_chain_sample_counts_full_trace_and_keeps_timing_shape_and_source():
    trace = [child(f'parent{parent}', operation) for parent in range(30) for operation in range(70)]
    source = {'mode': 'simulate', 'simulation': {'status': 'feasible', 'trace': trace}}
    original = deepcopy(source)
    compact = compact_result(source)
    execution = compact['operator_execution']
    assert execution['parent_count'] == 30
    assert execution['expanded_operations'] == 2100
    assert execution['event_counts']['compute'] == 2100
    assert len(execution['groups']) == execution['group_limit'] == 24
    assert execution['groups_truncated'] is True
    for group in execution['groups']:
        assert group['truncated'] is True
        assert group['event_count'] == 70
        assert len(group['events']) == execution['operations_per_group'] == 64
        assert [event['metadata']['operation_index'] for event in group['events']] == list(range(64))
    first = execution['groups'][0]['events'][0]
    assert first['startup_s'] == .01 and first['slot_wait_s'] == .1
    assert first['metadata']['implementation']['kernel_name'] == 'test'
    assert first['metadata']['trace_source'] == 'synthetic'
    assert first['work'] == trace[0]['work']
    assert len(compact['trace']['events']) == 300
    assert source == original


def test_event_details_bound_optional_work_and_metadata_without_inventing_missing_work():
    event = child('parent', 0)
    event['work'] = {f'resource{i}': i for i in range(50)}
    event['metadata']['implementation']['long'] = 'x' * 3000
    compact = compact_event(event)
    assert len(compact['work']) == 32
    assert compact['details_truncated'] is True
    assert len(compact['metadata']['implementation']['long']) == 2000
    assert 'work' not in compact_event({'id': 'no-work'})


def test_completion_and_communication_counts_do_not_misclassify_pack_kernels():
    trace = [child('parent', 0), {'id': 'done', 'metadata': {'operator_trace_completion': True}},
             {'id': 'transfer', 'metadata': {'type': 'transfer'}},
             {'id': 'pack', 'metadata': {'kernel': 'communication_pack', 'semantic': 'all_reduce'}}]
    execution = compact_result({'mode': 'simulate', 'simulation': {'trace': trace}})['operator_execution']
    assert execution['event_counts'] == {'compute': 2, 'completion': 1, 'communication': 1, 'other': 0}
    estimate = compact_result({'mode': 'estimate'})['operator_execution']
    assert estimate['basis'] == 'estimate_only' and not estimate['groups']

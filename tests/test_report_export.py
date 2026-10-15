"""Large-run export is a reversible presentation filter, never a numeric filter."""
import contextlib
import csv
import gzip
import json
from pathlib import Path
import shutil
import uuid

import pytest

from supernode_sim.analysis import run
from supernode_sim.cli import parser
from supernode_sim.core import Allocation, Event, Graph
from supernode_sim.memory_breakdown import build_memory_breakdown
import supernode_sim.report as report


@contextlib.contextmanager
def artifact_directory():
    base = Path(__file__).resolve().parent / '.export_tmp'
    base.mkdir(exist_ok=True)
    path = base / ('report_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        target = path.resolve()
        if target.parent != base.resolve() or not target.name.startswith('report_'):
            raise RuntimeError('Unsafe export test cleanup')
        shutil.rmtree(target)
        try:
            base.rmdir()
        except OSError:
            pass


def sample_result():
    result = run({'model': {'name': 'synthetic-export', 'total_params': 1000, 'layers': 1,
                            'hidden_dim': 4, 'num_heads': 2, 'kv_heads': 1, 'head_dim': 2},
                  'hardware': {'npu_count': 1, 'su_size': 1},
                  'workload': {'scenario': 'training', 'sequence_length': 2}})
    trace = [
        {'id': 'compute', 'metadata': {'kernel': 'attention'}},
        {'id': 'token', 'metadata': {}},
        {'id': 'allocation_marker', 'metadata': {}},
        {'id': 'collective.transfer', 'metadata': {'operation': 'all_reduce'}},
        {'id': 'collective.done', 'metadata': {}},
    ]
    for index, row in enumerate(trace):
        row.update(start_s=index, end_s=index + 1, slots=[])
    result['simulation']['trace'] = trace
    result['simulation']['memory_log'] = [{'sequence': 0, 'time_s': 0, 'action': 'allocate', 'allocation_id': 'w', 'pool': 'npu0.hbm', 'bytes': 10}]
    result['model_metadata']['sequence_token_completion_events'] = {'0': {'0': ['token']}}
    result['model_metadata']['step_end_events'] = []
    model_states = ['weights', 'gradients', 'master_weights', 'muon_momentum', 'activation',
                    'kv_cache', 'recurrent_state', 'quantization_metadata', 'attnres_residual']
    initial = [Allocation('initial_buffer', 'npu0.hbm', 1, 'buffers', metadata={'state': 'buffers'})]
    dynamic = tuple(Allocation(state, 'npu0.hbm', 2, state,
                               metadata={'component': 'Model', 'state': state}) for state in model_states)
    ephemeral = Allocation('ephemeral_buffer', 'npu0.hbm', 100, 'buffers', metadata={'state': 'buffers'})
    graph = Graph(memory_caps={'npu0.hbm': 1000}, initial_allocations=initial, events=[
        Event('allocation_marker', allocs=dynamic), Event('collective.transfer', allocs=(ephemeral,))])
    result['memory_breakdown'] = build_memory_breakdown(graph)
    return result


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_large_default_exports_exact_counts_and_preserves_numeric_summaries(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    result = sample_result()
    before = json.dumps(result, sort_keys=True)
    with artifact_directory() as directory:
        report.write_report(result, directory)
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        export = persisted['artifact_export']
        assert export['compacted'] is True
        assert export['events']['total_rows'] == 5
        assert export['events']['exported_rows'] == 3
        assert export['events']['omitted_rows'] == 2
        assert export['tensors']['total_rows'] == 11
        assert export['tensors']['exported_rows'] == 10
        assert export['tensors']['omitted_rows'] == 1
        assert export['memory_log']['total_rows'] == export['memory_log']['omitted_rows'] == 1
        assert export['memory_log']['exported_rows'] == 0
        assert persisted['simulation']['trace']['artifact'] == 'events.csv'
        assert persisted['memory_breakdown']['tensors']['artifact'] == 'tensors.csv'
        for key in ('metrics', 'estimate', 'model_metadata', 'memory_capacities_bytes', 'requirements'):
            assert persisted[key] == json.loads(json.dumps(result[key]))
        for key in ('pools', 'cluster', 'component_state_rows', 'state_rows', 'logical_component_state_bytes'):
            assert persisted['memory_breakdown'][key] == result['memory_breakdown'][key]
        events = csv_rows(directory / 'events.csv')
        assert {row['id'] for row in events} == {'compute', 'token', 'allocation_marker'}
        timeline = json.loads((directory / 'timeline.json').read_text(encoding='utf-8'))
        assert {row['name'] for row in timeline['traceEvents']} == {'compute', 'token', 'allocation_marker'}
        tensors = csv_rows(directory / 'tensors.csv')
        assert {row['allocation_id'] for row in tensors} == {row['allocation_id'] for row in result['memory_breakdown']['tensors']} - {'ephemeral_buffer'}
        assert any(row['allocation_id'] == 'initial_buffer' for row in tensors)
        text = (directory / 'report.md').read_text(encoding='utf-8')
        assert '大图使用精简导出' in text
        assert '--full-detail' in text
        assert '原始日志未写入其他隐藏文件' in text
    assert json.dumps(result, sort_keys=True) == before


def test_full_detail_restores_all_raw_arrays_and_all_artifact_rows(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    result = sample_result()
    with artifact_directory() as directory:
        report.write_report(result, directory, full_detail=True)
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert persisted['artifact_export']['compacted'] is False
        assert persisted['simulation']['trace'] == result['simulation']['trace']
        assert persisted['simulation']['memory_log'] == result['simulation']['memory_log']
        assert persisted['memory_breakdown']['tensors'] == result['memory_breakdown']['tensors']
        assert len(csv_rows(directory / 'events.csv')) == 5
        assert len(csv_rows(directory / 'tensors.csv')) == 11
        assert len(json.loads((directory / 'timeline.json').read_text(encoding='utf-8'))['traceEvents']) == 5
        assert all(persisted['artifact_export'][key]['omitted_rows'] == 0 for key in ('events', 'tensors', 'memory_log'))


def test_small_runs_preserve_previous_complete_json_and_artifacts(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 100)
    result = sample_result()
    with artifact_directory() as directory:
        report.write_report(result, directory)
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert persisted == json.loads(json.dumps(result))
        assert len(csv_rows(directory / 'events.csv')) == 5
        assert len(csv_rows(directory / 'tensors.csv')) == 11


def test_large_estimate_exports_tensors_without_invented_timeline_reference(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    result = sample_result()
    result.pop('simulation')
    result['mode'] = 'estimate'
    result['status'] = 'estimated'
    with artifact_directory() as directory:
        report.write_report(result, directory)
        persisted = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert persisted['artifact_export']['compacted']
        assert persisted['artifact_export']['events']['artifact'] is None
        assert persisted['artifact_export']['events']['timeline_artifact'] is None
        assert not (directory / 'timeline.json').exists()
        assert len(csv_rows(directory / 'tensors.csv')) == 10


def test_full_detail_cli_is_explicitly_available_only_for_complete_run_commands():
    for command in ('estimate', 'simulate'):
        arguments = parser().parse_args([command, 'case.yaml', '--full-detail'])
        assert arguments.full_detail
    for arguments in (['sweep', 'case.yaml', '--axes', 'axes.yaml', '--full-detail'],
                      ['size-su', 'case.yaml', '--full-detail']):
        with pytest.raises(SystemExit):
            parser().parse_args(arguments)


def test_human_pool_labels_use_natural_numeric_order():
    group = {'pools': ['npu0.hbm', 'npu1.hbm', 'npu10.hbm', 'npu63.hbm', 'npu9.hbm'], 'pool_label': 'old lexical label'}
    assert report._display_pool_label(group) == 'npu0.hbm … npu63.hbm (5 pools)'


def test_reference_hardware_and_native_muon_limit_are_prominent():
    result = sample_result()
    result['config']['hardware']['name'] = 'Ascend 950DT public roadmap profile'
    result['config']['hardware']['npu_count'] = 64
    result['config']['hardware']['npu']['hbm_bytes'] = 144e9
    result['model_metadata']['fidelity'] = {'parameter_capacity': 'shape_derived', 'distributed_muon': 'not_implemented'}
    text = report.markdown_report(result)
    lead = text.split('## 性能与约束')[0]
    assert 'Ascend 950DT' in lead and '144 GB' in lead and '64' in lead
    assert '尚未通过整机实测校准' in lead
    assert '尚未实现分布式 Muon 正交化' in lead
    assert '不代表真实整机部署验证' in lead


def test_prefiltered_lean_rows_keep_original_manifest_counts(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 100)
    result = sample_result()
    breakdown = result['memory_breakdown']
    breakdown['tensors'] = [row for row in breakdown['tensors'] if row['allocation_id'] != 'ephemeral_buffer']
    breakdown['tensor_detail_export'] = {'total_rows': 11, 'retained_rows': 10, 'omitted_rows': 1,
                                        'include_ephemeral_details': False}
    with artifact_directory() as directory:
        report.write_report(result, directory)
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        counts = saved['artifact_export']['tensors']
        assert counts['total_rows'] == 11
        assert counts['available_rows'] == counts['exported_rows'] == 10
        assert counts['prefiltered_rows'] == counts['omitted_rows'] == 1
        assert len(csv_rows(directory / 'tensors.csv')) == 10
        assert saved['memory_breakdown']['pools'] == breakdown['pools']
        with pytest.raises(ValueError, match='Full tensor details were not materialized'):
            report.write_report(result, directory / 'full', full_detail=True)


def test_full_detail_rejects_event_payloads_not_retained_by_engine(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    result = sample_result()
    result['simulation']['trace_payload_fields_omitted'] = ['work', 'allocations', 'releases']
    with artifact_directory() as directory:
        report.write_report(result, directory)
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert saved['artifact_export']['trace_payload_fields_omitted'] == ['work', 'allocations', 'releases']
        with pytest.raises(ValueError, match='Full event payloads were not retained'):
            report.write_report(result, directory / 'full', full_detail=True)


def test_compact_checkpoint_trace_is_read_once_for_counts_csv_and_timeline(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    result = sample_result()

    class CountingRecords:
        def __init__(self, rows):
            self.rows, self.passes = rows, 0

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            self.passes += 1
            yield from self.rows

    trace = CountingRecords(result['simulation']['trace'])
    result['simulation']['trace'] = trace
    with artifact_directory() as directory:
        report.write_report(result, directory)
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert saved['artifact_export']['events']['exported_rows'] == 3
        assert len(csv_rows(directory / 'events.csv')) == 3
        assert len(json.loads((directory / 'timeline.json').read_text(encoding='utf-8'))['traceEvents']) == 3
        assert trace.passes == 1
        assert '\n' not in (directory / 'result.json').read_text(encoding='utf-8')


def test_large_tensor_csv_is_losslessly_compressed_only_in_default_export(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 2)
    monkeypatch.setattr(report, 'COMPRESS_TABLE_THRESHOLD', 2)
    result = sample_result()
    with artifact_directory() as directory:
        report.write_report(result, directory)
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        details = saved['artifact_export']['tensors']
        assert details['artifact'] == 'tensors.csv.gz'
        assert details['compression'] == 'gzip'
        assert not (directory / 'tensors.csv').exists()
        with gzip.open(directory / details['artifact'], 'rt', encoding='utf-8-sig', newline='') as stream:
            compressed_rows = list(csv.DictReader(stream))
        assert len(compressed_rows) == details['exported_rows'] == 10
        text = (directory / 'report.md').read_text(encoding='utf-8')
        assert 'tensors.csv.gz' in text and 'gzip 无损压缩' in text
        report.write_report(result, directory / 'full', full_detail=True)
        all_rows = csv_rows(directory / 'full' / 'tensors.csv')
        assert compressed_rows == [row for row in all_rows if row['allocation_id'] != 'ephemeral_buffer']
        assert not (directory / 'full' / 'tensors.csv.gz').exists()


def test_large_memory_log_alone_triggers_compact_export(monkeypatch):
    monkeypatch.setattr(report, 'AUTO_DETAIL_THRESHOLD', 20)
    result = sample_result()
    result['simulation']['memory_log'] *= 30
    with artifact_directory() as directory:
        report.write_report(result, directory)
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        assert saved['artifact_export']['compacted']
        assert saved['artifact_export']['memory_log']['omitted_rows'] == 30

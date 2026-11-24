import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from supernode_sim.analysis import run
from supernode_sim.config import load_config
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.report import write_report
from supernode_sim.training_summary import rank_activity


def test_rank_time_partition_overlap_and_concurrent_operators():
    graph = Graph(resources={'npu0.compute': Resource('npu0.compute', 1, 'FLOP/s', 'compute')})
    compute = Event('kernel', work={'npu0.compute': 1}, metadata={'kernel': 'gemm'})
    def transfer(name):
        return Event(name, metadata={'type': 'transfer', 'src': 'npu0', 'dst': 'npu1', 'collective': name})
    graph.events = [compute, transfer('all_reduce'), transfer('all_gather')]
    trace = [dict(id=e.id, start_s=start, end_s=end)
             for e, start, end in zip(graph.events, (1, 2, 3), (4, 6, 5))]
    totals, compute_ops, exposed_ops, idle, ranks = rank_activity(graph, {'trace': trace, 'duration_s': 8}, 3)
    assert totals == {'compute_only': 1, 'compute_and_transfer': 2, 'exposed_transfer': 6,
                      'other_local': 0, 'idle': 15}
    assert compute_ops == {'gemm': 3}
    assert exposed_ops == {'all_reduce': 4.5, 'all_gather': 1.5}
    assert idle == {'before_first_activity': 3, 'after_last_activity': 4, 'between_activities': 0,
                    'inactive_rank': 8}
    assert all(sum(states.values()) == 8 for states in ranks.values())


def test_export_complete_and_estimate_to_same_directory(tmp_path):
    spec = load_config(Path(__file__).resolve().parents[1] / 'examples/training_overlap_qos.yaml')
    result = run(spec, full_detail=False)
    write_report(result, tmp_path)
    summary = json.loads((tmp_path / 'training_summary.json').read_text(encoding='utf-8'))
    assert summary == result['training_summary']
    groups = summary['groups']
    assert groups['rank_time']['total'] == pytest.approx(spec.hardware.npu_count * summary['duration_s'])
    assert groups['hbm']['total'] == pytest.approx(summary['peak_hbm']['peak_bytes'])
    assert groups['step_latency']['total'] == pytest.approx(result['metrics']['step_time_s'])
    for group in groups.values():
        assert sum(r['percent'] for r in group['segments']) == pytest.approx(100)
    assert ElementTree.parse(tmp_path / 'training_breakdown.svg').getroot().tag.endswith('}svg')
    assert 'training_breakdown.svg' in (tmp_path / 'report.md').read_text(encoding='utf-8')
    write_report(run(spec, mode='estimate'), tmp_path)
    assert json.loads((tmp_path / 'training_summary.json').read_text(encoding='utf-8'))['status'] == 'unavailable'
    assert not (tmp_path / 'training_breakdown.svg').exists()

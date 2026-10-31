"""An imported operator decomposition remains bound to the actual compiler input."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from supernode_sim.analysis import compile_graph, device_requirements
from supernode_sim.config import Config, deep_merge, load_config
from supernode_sim.engine import simulate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'skills/supernode-modeling/scripts/build_operator_trace.py'
ASSET = SCRIPT.parents[1] / 'assets/operator-trace.synthetic.json'
SPEC = importlib.util.spec_from_file_location('operator_trace_builder', SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
EVENT = 'train.s0.p0.backward.m0.l0.npu0.Dense'


def test_build_overlay_refines_real_training_graph_without_claiming_measurement():
    config = load_config(ROOT / 'examples/features/synthetic-training.yaml')
    overlay = builder.build_calibration(config, EVENT, json.loads(ASSET.read_text()))
    assert overlay['source']['status'] == 'assumed'
    trace = overlay['hardware']['kernel_traces']['dense'][0]
    assert trace['metadata_match']['workload_shape']['query_tokens'] == 8
    assert len(trace['operations']) == 6
    refined = Config.model_validate(deep_merge(config.model_dump(), {'hardware': overlay['hardware']}))
    graph = compile_graph(refined)
    original = simulate(compile_graph(config))
    result = simulate(graph)
    assert result['status'] == 'feasible'
    assert graph.metadata['operator_trace']['matched_events'] == 1
    assert graph.metadata['fidelity']['hardware_performance'] == 'uncalibrated'
    assert result['resource_work']['npu0.compute'] == original['resource_work']['npu0.compute']
    assert result['resource_work']['npu0.hbm'] > original['resource_work']['npu0.hbm']
    assert result['memory_final_bytes'] == original['memory_final_bytes']
    parent = next(e for e in compile_graph(config).events if e.id == EVENT)
    operators = [e for e in graph.events if e.metadata.get('trace_parent_event') == EVENT]
    assert operators
    assert all(e.metadata['working_set_bytes'] == parent.metadata['working_set_bytes'] + 512 for e in operators)
    assert all(e.metadata['working_set_scope'] == 'parent_component_proxy_plus_local_workspace' for e in operators)
    requirements = device_requirements(graph, 1, result['resource_work'])
    assert requirements['npu0']['max_operator_working_set_bytes'] >= parent.metadata['working_set_bytes'] + 512


def test_wrong_operator_work_and_unknown_events_are_not_imported():
    config = load_config(ROOT / 'examples/features/synthetic-training.yaml')
    document = json.loads(ASSET.read_text())
    with pytest.raises(ValueError, match='Unknown aggregate'):
        builder.build_calibration(config, 'not-an-event', document)
    document['operations'][0]['matmul']['m'] = 1
    with pytest.raises(ValueError, match='matmul FLOPs'):
        builder.build_calibration(config, EVENT, document)


def test_budget_event_without_shape_identity_is_not_silently_bound_by_layer():
    config = load_config(ROOT / 'examples/demo_training.yaml')
    events = [event for event in builder.aggregate_graph(config).events if event.metadata.get('kernel')]
    assert events
    assert all(builder.describe_event(event, config) is None for event in events)
    with pytest.raises(ValueError, match='shape identity'):
        builder.build_calibration(config, events[0].id, json.loads(ASSET.read_text()))


def test_cli_inspection_and_build_use_the_same_event_and_shape():
    config_path = str(ROOT / 'examples/features/synthetic-training.yaml')
    inspected = subprocess.run([sys.executable, str(SCRIPT), config_path,
        '--list', '--kernel', 'dense', '--phase', 'backward'], capture_output=True, text=True, check=True)
    event = json.loads(inspected.stdout)['events'][0]
    built = subprocess.run([sys.executable, str(SCRIPT), config_path,
        '--event-id', event['event_id'], '--operators', str(ASSET)], capture_output=True, text=True, check=True)
    trace = json.loads(built.stdout)['hardware']['kernel_traces']['dense'][0]
    assert trace['metadata_match'] == event['metadata_match']
    assert trace['flops'] == event['flops']
    assert trace['memory_bytes'] == event['memory_bytes']

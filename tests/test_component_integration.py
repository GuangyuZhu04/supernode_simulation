"""Architecture/schema/compiler/report checks with independent small anchors."""
import contextlib
import csv
import json
from pathlib import Path
import shutil
import uuid

import pytest

from supernode_sim.analysis import compile_graph, run
from supernode_sim.config import Config, deep_merge, load_config
from supernode_sim.core import Allocation, Graph
from supernode_sim.memory_breakdown import build_memory_breakdown
from supernode_sim.report import markdown_report, write_report


@contextlib.contextmanager
def artifact_directory():
    base = Path(__file__).resolve().parent / '.component_integration_tmp'
    base.mkdir(exist_ok=True)
    path = base / ('report_' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        target = path.resolve()
        if target.parent != base.resolve() or not target.name.startswith('report_'):
            raise RuntimeError('Unsafe component test cleanup')
        shutil.rmtree(target)
        try:
            base.rmdir()
        except OSError:
            pass


def shape_config(scenario='training', patch=None):
    raw = {
        'model': {'name': 'synthetic-small-shape', 'layers': 1, 'hidden_dim': 4,
                  'intermediate_dim': 8, 'num_heads': 2, 'kv_heads': 1, 'head_dim': 2,
                  'architecture': {'vocab_size': 16, 'attention_by_layer': ['GQA'], 'ffn_by_layer': ['Dense']}},
        'hardware': {'npu_count': 1, 'su_size': 1, 'npus_per_host': 1,
                     'npu': {'hbm_bytes': 1e7, 'runtime_bytes': 0, 'workspace_bytes': 0}},
        'workload': {'scenario': scenario, 'batch_size': 1, 'sequence_length': 4,
                     'gradient_accumulation': 2, 'input_length': 3, 'output_length': 2},
    }
    return Config.model_validate(deep_merge(raw, patch or {}))


def test_shape_schema_derives_small_gqa_and_preserves_embedding_activity_semantics():
    spec = shape_config()
    # Embedding/output: 64 each; 3 norms: 12; Q/K/V/O: 16+8+8+16;
    # SwiGLU gate/up/down: 32 each. Embedding lookup activates one row (4).
    assert spec.model.total_params == 284
    assert spec.model.active_params == 224
    assert Config.model_validate(spec.model_dump()) == spec
    tied = shape_config(patch={'model': {'architecture': {'tie_embeddings': True}}})
    assert tied.model.total_params == 220
    assert tied.model.active_params == 220


def test_shape_schema_rejects_conflicting_budget_but_accepts_reported_rounding():
    with pytest.raises(ValueError, match='conflicts with tensor shapes'):
        shape_config(patch={'model': {'total_params': 300}})
    with pytest.raises(ValueError, match='conflicts with tensor shapes'):
        shape_config(patch={'model': {'active_params': 200}})
    spec = shape_config(patch={'model': {'reported_total_params': 300, 'reported_active_params': 200}})
    assert (spec.model.total_params, spec.model.active_params) == (284, 224)
    assert (spec.model.reported_total_params, spec.model.reported_active_params) == (300, 200)


@pytest.mark.parametrize('scenario', ['training', 'inference'])
def test_actual_component_report_reconciles_simultaneous_pool_and_cluster_peaks(scenario):
    result = run(shape_config(scenario))
    assert result['status'] == 'feasible', result.get('simulation', {}).get('diagnostics')
    memory = result['memory_breakdown']
    assert memory['mode'] == 'simulated'
    components = memory['component_inventory']['components']
    assert {'WordEmbedding', 'PreNorm', 'PostNorm', 'GQA', 'Dense', 'Output'} <= set(components)
    for pool in memory['pools']:
        assert sum(row['bytes'] for row in pool['composition_at_peak']) == pytest.approx(pool['peak_bytes'])
        assert pool['peak_bytes'] == pytest.approx(result['simulation']['memory_peaks_bytes'][pool['pool']])
    assert sum(row['bytes'] for row in memory['cluster']['composition_at_peak']) == pytest.approx(memory['cluster']['peak_bytes'])
    assert memory['cluster']['peak_bytes'] <= sum(pool['peak_bytes'] for pool in memory['pools']) + 1e-8
    report = markdown_report(result)
    assert '组件 × 状态' in report
    assert '全模型逻辑参数' in report
    assert '不可相加' in report
    # The unused CPU pool is a legitimate zero-memory row, not a missing value.
    cpu = next(pool for pool in memory['pools'] if pool['pool'] == 'cpu0.dram')
    assert cpu['peak_bytes'] == 0
    with artifact_directory() as path:
        write_report(result, path)
        persisted = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        assert any(p['pool'] == 'cpu0.dram' and p['peak_bytes'] == 0 for p in persisted['memory_breakdown']['pools'])
        with (path / 'model_parameters.csv').open(encoding='utf-8-sig', newline='') as stream:
            logical = list(csv.DictReader(stream))
        assert sum(float(row['accounted_elements']) for row in logical) == 284


def test_muon_fallback_state_assignment_is_visible_logically_and_physically():
    result = run(shape_config(patch={'execution': {'optimizer': {'kind': 'muon'}}}))
    assert result['status'] == 'feasible', result.get('simulation', {}).get('diagnostics')
    logical = result['memory_breakdown']['logical_component_state_bytes']
    muon = sum(states.get('muon_momentum', 0) for states in logical.values())
    # GQA 48 and SwiGLU 96 matrix elements use Muon; embedding/output/norms
    # (64+64+12 elements) retain Adam first and second moments.
    assert muon == (48 + 96) * 4
    momentum = sum(states.get('momentum', 0) + states.get('adam_momentum', 0) for states in logical.values())
    variance = sum(states.get('variance', 0) + states.get('adam_variance', 0) for states in logical.values())
    assert momentum == variance == (64 + 64 + 12) * 4
    assert 'muon_momentum' not in logical['WordEmbedding']
    report = markdown_report(result)
    assert 'Muon 动量' in report
    assert '全模型逻辑状态容量' in report


def test_tp_replication_is_not_confused_with_unreplicated_logical_capacity():
    config = shape_config(patch={'hardware': {'npu_count': 2, 'su_size': 2}, 'execution': {'tp': 2}})
    result = run(config, 'estimate')
    logical = result['memory_breakdown']['logical_component_state_bytes']
    assert sum(states.get('weights', 0) for states in logical.values()) == 284 * 2
    tensors = result['memory_breakdown']['tensors']
    physical_weights = sum(row['bytes'] for row in tensors if row['is_initial'] and row['state'] == 'weights')
    # Replicated 12 norm elements plus both 8-element K/V matrices each add
    # one extra copy when TP=2 exceeds one KV head.
    assert physical_weights == 284 * 2 + (12 + 8 + 8) * 2
    assert result['memory_breakdown']['mode'] == 'resident_only'


def test_legacy_parameter_budget_keeps_unassigned_attribution():
    result = run({'model': {'name': 'legacy-unknown-architecture', 'total_params': 1000,
                           'layers': 1, 'hidden_dim': 4, 'num_heads': 2, 'kv_heads': 1, 'head_dim': 2},
                  'hardware': {'npu_count': 1, 'su_size': 1},
                  'workload': {'scenario': 'training', 'sequence_length': 2}}, 'estimate')
    assert result['status'] == 'estimated'
    assert set(result['memory_breakdown']['component_inventory']['components']) == {'Unassigned'}
    assert {row['component'] for row in result['memory_breakdown']['tensors']} <= {'Unassigned', 'System'}
    assert 'Unassigned' in markdown_report(result)


def test_report_caps_thousands_of_distinct_device_groups():
    graph = Graph(memory_caps={f'npu{i}.hbm': 1e6 for i in range(1100)}, initial_allocations=[
        Allocation(f'w{i}', f'npu{i}.hbm', i + 1, 'weights',
                   metadata={'component': 'Attention', 'state': 'weights'}) for i in range(1100)])
    memory = build_memory_breakdown(graph)
    template = run(shape_config(), 'estimate')
    template['memory_breakdown'] = memory
    report = markdown_report(template)
    assert len(memory['equivalent_pool_groups']) == 1100
    assert len(report.splitlines()) < 300
    assert '最多 16 组' in report


def test_official_k3_source_has_declared_hybrid_topology():
    source = Path(__file__).resolve().parents[1] / 'docs' / 'sources' / 'kimi_k3_config.json'
    model = json.loads(source.read_text(encoding='utf-8'))['text_config']
    assert model['hidden_size'] == 7168
    assert model['num_hidden_layers'] == 93
    assert len(model['linear_attn_config']['kda_layers']) == 69
    assert len(model['linear_attn_config']['full_attn_layers']) == 24
    assert set(model['linear_attn_config']['kda_layers']).isdisjoint(model['linear_attn_config']['full_attn_layers'])
    assert sorted(model['linear_attn_config']['kda_layers'] + model['linear_attn_config']['full_attn_layers']) == list(range(1, 94))
    assert model['num_experts'] == 896
    assert model['num_experts_per_token'] == 16


@pytest.mark.parametrize('variant,weight,routed', [
    ('fp8', 'fp8', None), ('fp16', 'fp16', None), ('mxfp4', 'bf16', 'fp4'),
])
def test_public_k3_ascend950_cases_preserve_real_hardware_and_shape_counts(variant, weight, routed):
    path = Path(__file__).resolve().parents[1] / 'examples' / f'kimi_k3_ascend950_{variant}.yaml'
    config = load_config(path)
    assert config.model.total_params == 2_779_484_476_000
    assert config.model.active_params == 104_189_619_808
    assert config.model.architecture.attention_by_layer.count('KDA') == 69
    assert config.model.architecture.attention_by_layer.count('MLA') == 24
    assert config.model.architecture.moe.latent_dim == 3584
    assert config.hardware.npu_count == config.hardware.su_size == 64
    assert config.hardware.npu.hbm_bytes == 144e9
    assert config.hardware.npu.hbm_bytes * config.hardware.npu_count == 9.216e12
    assert config.execution.precision.weight == weight
    assert config.execution.precision.routed_expert_weight == routed
    assert config.sources['model.architecture'].status == 'public'
    assert config.sources['hardware.npu.hbm_bytes'].status == 'public'
    assert config.sources['hardware.npu.efficiency'].status == 'assumed'
    assert config.sources['hardware.cpu'].status == 'assumed'
    assert not config.hardware.kernel_profiles  # No fabricated measured throughput.


def test_training_reference_keeps_capacity_failure_visible_and_scales_domains_explicitly():
    base = Path(__file__).resolve().parents[1] / 'examples'
    single = load_config(base / 'kimi_k3_ascend950_training_muon.yaml')
    dual = load_config(base / 'kimi_k3_ascend950_training_muon_128.yaml')
    assert single.execution.optimizer.kind == dual.execution.optimizer.kind == 'muon'
    assert single.hardware.npu.hbm_bytes == dual.hardware.npu.hbm_bytes == 144e9
    assert (single.hardware.npu_count, single.hardware.su_size) == (64, 64)
    assert (dual.hardware.npu_count, dual.hardware.su_size) == (128, 64)
    assert (single.execution.dp, single.execution.fsdp) == (1, 1)
    assert (dual.execution.dp, dual.execution.fsdp) == (2, 2)
    assert single.model.total_params == dual.model.total_params

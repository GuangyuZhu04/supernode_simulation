from copy import deepcopy
from pathlib import Path

import pytest

from supernode_sim.components import attention_state
from supernode_sim.config import Config, import_hf_config, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_official_nested_hf_import_derives_text_backbone():
    result = import_hf_config(ROOT / 'docs/sources/kimi_k3_config.json')
    assert result['missing_fields'] == []
    assert result['model']['total_params'] == 2779484476000
    assert result['model']['active_params'] == 104189619808
    assert result['model']['architecture']['attention_by_layer'].count('KDA') == 69
    assert 'vision' in result['scope']


def test_mla_runtime_cache_layout_changes_memory_not_parameters():
    spec = load_config(ROOT / 'examples/kimi_k3_ascend950_fp8.yaml')
    model = spec.model.model_dump()
    compressed = attention_state(model, 3, 1, 2048, tp=8)
    expanded = deepcopy(model)
    expanded['architecture']['mla']['cache_layout'] = 'expanded'
    assert compressed['elements'] == 2048 * (512 + 64)
    assert attention_state(expanded, 3, 1, 2048, tp=8)['elements'] == 2048 * 12 * (128 + 64 + 128)
    assert attention_state(model, 0, 1, 2048, tp=8)['elements'] == attention_state(model, 0, 1, 1048576, tp=8)['elements']
    assert expanded['total_params'] == model['total_params']


def test_unsupported_fp4_states_and_structured_prefetch_are_explicit():
    data = load_config(ROOT / 'examples/kimi_k3_ascend950_fp8.yaml').model_dump()
    data['execution']['precision']['gradient'] = 'fp4'
    with pytest.raises(ValueError, match='storage only'):
        Config.model_validate(data)
    data['execution']['precision']['gradient'] = 'fp16'
    data['execution']['prefetch'] = True
    with pytest.raises(ValueError, match='lookahead'):
        Config.model_validate(data)

"""Strict optional feature inputs and an auditable simulator capability plan.

Selections expand to existing canonical fields before configuration validation.
They never multiply a runtime by a nominal feature speedup. Resolved configuration
serialization omits the input shorthand so sweeps can change canonical fields.
"""
from __future__ import annotations

import copy
from typing import Any


_SUPPORTED = {
    'precision': (['training', 'inference'], ['execution.precision'],
        'Storage, communication and compute dtypes remain separate; hardware must provide the selected compute throughput.'),
    'activation_checkpointing': (['training'], ['execution.features.activation_checkpointing'],
        'Retained activation memory and recomputation work are modeled explicitly.'),
    'optimizer_offload': (['training'], ['execution.features.optimizer_offload'],
        'Optimizer state placement, CPU work and host-device transfers are modeled.'),
    'kv_offload': (['inference'], ['execution.features.kv_offload', 'execution.features.kv_offload_fraction',
        'execution.features.kv_page_tokens', 'execution.features.kv_page_entry_bytes'],
        'KV placement, page rounding, metadata and transfers are modeled; this is not a paged-attention scheduler implementation.'),
    'sequence_parallel': (['inference'], ['execution.sequence_parallel'],
        'Requires structured inference, TP>1 and PP=1; sequence parallelism shares TP ranks.'),
    'fsdp': (['training'], ['execution.fsdp', 'execution.reshard_after_forward', 'execution.sync_each_microbatch', 'execution.prefetch'],
        'FSDP is a DP subgroup. Structured lookahead prefetch is unsupported.'),
    'parallelism': (['training', 'inference'], ['execution.tp', 'execution.pp', 'execution.ep', 'execution.dp', 'execution.rank_order'],
        'Physical rank count, placement and collective work remain subject to configuration validation.'),
    'optimizer': (['training'], ['execution.optimizer'],
        'Adam and analytical Muon component work are modeled; distributed Muon is not implemented.'),
    'engram': (['inference'], ['execution.features.engram'],
        'Lookup state, explicit hit-rate assumptions and lookup transfers/IOPS are modeled.'),
    'quantization': (['training', 'inference'], ['execution.features.weight_scale_block', 'execution.features.kv_scale_block',
        'execution.features.scale_bytes', 'execution.features.extra_weight_copy_dtype'],
        'Quantization metadata and optional weight copies are counted; numerical quality and runtime kernel support are not inferred.'),
    'attention': (['training', 'inference'], ['model.architecture.attention_by_layer', 'model.architecture.mla',
        'model.architecture.kda', 'model.architecture.csa', 'model.architecture.hca'],
        'MLA/KDA/MHA/GQA/MQA/CSA/HCA use explicit architecture dimensions; changing attention is a model-architecture hypothesis, not a drop-in quality-preserving speedup.'),
    'speculative_decode': (['inference'], ['execution.features.speculative_decode'],
        'Methods dspark/generic require explicit acceptance counts, draft cost and draft residency. This does not reproduce or validate a named serving implementation.'),
    'kernel_profiles': (['training', 'inference'], ['hardware.kernel_profiles'],
        'Measured or assumed solo-kernel profiles may represent fusion/graph/runtime implementations; actual matching coverage is reported separately.'),
}

_UNSUPPORTED = {
    'continuous_batching': 'Arrival-driven scheduling and dynamic admission are not implemented.',
    'prefix_cache': 'Dynamic prefix lookup, eviction and cross-request reuse are not implemented.',
    'chunked_prefill': 'Serving-level chunk scheduling and prefill/decode interleaving are not implemented.',
    'prefill_decode_disaggregation': 'Separate prefill/decode pools and request handoff are not implemented.',
    'mhc': 'Manifold-constrained hyper-connections are not implemented.',
    'flash_attention': 'No universal boolean speedup is modeled; select explicit kernel_profiles for a concrete kernel implementation.',
    'cuda_graphs': 'No universal graph-capture speedup is modeled; use explicit kernel_profiles and measured startup where appropriate.',
}


def feature_catalog() -> dict[str, Any]:
    """Return finite capabilities, limitations and exact accepted feature names."""
    entries = [
        {'name': name, 'support_status': 'supported', 'modeling': 'profile_driven' if name == 'kernel_profiles' else 'analytical',
         'scenarios': scenarios, 'canonical_paths': paths, 'notes': note}
        for name, (scenarios, paths, note) in _SUPPORTED.items()
    ]
    entries.extend({'name': name, 'support_status': 'unsupported', 'modeling': 'none',
                    'scenarios': [], 'canonical_paths': [], 'notes': note}
                   for name, note in _UNSUPPORTED.items())
    return {
        'schema_version': '1.0', 'features': entries,
        'input_format': 'Top-level features object, or --features JSON/YAML file containing that object directly.',
        'performance_evidence': 'supported means a simulator model exists. analytical/profile_driven never imply measured serving accuracy or a guaranteed speedup.',
    }


def _merge_without_conflict(base: dict, patch: dict, label: str) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        path = f'{label}.{key}' if label else key
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_without_conflict(result[key], value, path)
        elif result[key] != value:
            raise ValueError(f'Conflicting feature selection at {path}; remove the duplicate canonical setting or provide the same value')
    return result


def merge_feature_selections(base: dict | None, supplied: dict) -> dict:
    """Combine inline and file selections without silently overriding either."""
    if base is not None and not isinstance(base, dict):
        raise ValueError('inline features must be an object')
    if not isinstance(supplied, dict):
        raise ValueError('feature selection file must contain a feature object')
    return _merge_without_conflict(base or {}, supplied, 'features')


def _set_canonical(data: dict, path: str, value: Any) -> None:
    parts = path.split('.')
    node = data
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ValueError(f'Cannot apply feature to {path}: {part} is not a configuration object')
        node = node[part]
    leaf = parts[-1]
    if leaf not in node:
        node[leaf] = copy.deepcopy(value)
    elif isinstance(node[leaf], dict) and isinstance(value, dict):
        node[leaf] = _merge_without_conflict(node[leaf], value, path)
    elif node[leaf] != value:
        raise ValueError(f'Conflicting feature selection at {path}; remove the duplicate canonical setting or provide the same value')


def resolve_feature_input(data: dict) -> dict:
    """Expand only explicitly supplied fields, retaining source shorthand in memory."""
    from .config import FeatureSelectionSpec

    selection = FeatureSelectionSpec.model_validate(data['features'])
    supplied = selection.model_dump(exclude_unset=True)
    result = copy.deepcopy(data)
    scenario = result.get('workload', {}).get('scenario')
    # Reject inactive scenario-only dictionaries too: an inference optimizer
    # selection would otherwise validate but have no execution effect.
    for name, value in supplied.items():
        if value is None:
            continue
        scenarios = _SUPPORTED[name][0]
        disabled = value is False or value == {}
        if name == 'engram' and isinstance(value, dict):
            disabled = value.get('enabled') is False
        if name == 'kv_offload' and isinstance(value, dict):
            disabled = value.get('location') == 'none' and not (set(value) - {'location'})
        if name == 'speculative_decode' and isinstance(value, dict):
            disabled = value.get('enabled') is False
        if name == 'fsdp' and isinstance(value, dict):
            disabled = value == {'degree': 1}
        if scenario is not None and scenario not in scenarios and not disabled:
            raise ValueError(f'features.{name} is only supported for {"/".join(scenarios)}')

    direct = {
        'precision': 'execution.precision', 'optimizer': 'execution.optimizer',
        'sequence_parallel': 'execution.sequence_parallel',
        'activation_checkpointing': 'execution.features.activation_checkpointing',
        'optimizer_offload': 'execution.features.optimizer_offload',
        'engram': 'execution.features.engram',
        'speculative_decode': 'execution.features.speculative_decode',
        'kernel_profiles': 'hardware.kernel_profiles',
        'attention': 'model.architecture',
        'parallelism': 'execution', 'quantization': 'execution.features',
    }
    for name, path in direct.items():
        if supplied.get(name) is not None:
            patch = supplied[name]
            if name == 'attention':
                patch = {key: value for key, value in patch.items() if value is not None}
                if not patch:
                    continue
            _set_canonical(result, path, patch)
    kv = supplied.get('kv_offload')
    if kv is not None:
        for key, value in kv.items():
            target = {'location': 'kv_offload', 'fraction': 'kv_offload_fraction',
                      'page_tokens': 'kv_page_tokens', 'page_entry_bytes': 'kv_page_entry_bytes'}[key]
            _set_canonical(result, 'execution.features.' + target, value)
    fsdp = supplied.get('fsdp')
    if fsdp is not None:
        for key, value in fsdp.items():
            _set_canonical(result, 'execution.' + ('fsdp' if key == 'degree' else key), value)
    return result


def build_feature_plan(config, calibration: dict | None = None) -> dict[str, Any]:
    """Explain resolved options; supplied profiles do not prove measured fidelity."""
    data = config.model_dump() if hasattr(config, 'model_dump') else copy.deepcopy(config)
    execution = data['execution']
    options = execution['features']
    architecture = data['model'].get('architecture')
    selected = {
        'precision': True,
        'activation_checkpointing': options['activation_checkpointing'],
        'optimizer_offload': options['optimizer_offload'],
        'kv_offload': options['kv_offload'] != 'none',
        'sequence_parallel': execution['sequence_parallel'],
        'fsdp': execution['fsdp'] > 1,
        'parallelism': any(execution[name] > 1 for name in ('tp', 'pp', 'ep', 'dp')),
        'optimizer': data['workload']['scenario'] == 'training',
        'engram': options['engram']['enabled'],
        'quantization': any(value in ('fp4', 'fp8', 'int8') for value in execution['precision'].values()) or options['extra_weight_copy_dtype'] is not None,
        'attention': architecture is not None,
        'speculative_decode': options['speculative_decode']['enabled'],
        'kernel_profiles': bool(data['hardware']['kernel_profiles']),
    }

    def value_at(path):
        value = data
        for part in path.split('.'):
            if value is None:
                return None
            value = value.get(part)
        return value

    rows = []
    for item in feature_catalog()['features']:
        if item['support_status'] != 'supported':
            continue
        row = dict(item)
        name = row['name']
        row['selected'] = selected[name]
        row['parameters'] = {path: value_at(path) for path in row['canonical_paths']}
        if name == 'speculative_decode' and row['selected']:
            speculative = options['speculative_decode']
            costs = [speculative['draft']['cost'], speculative['draft'].get('prefill_cost'),
                     speculative.get('commit_cost'), speculative.get('replay_cost')]
            if any(cost and cost.get('duration_s') is not None for cost in costs):
                row['modeling'] = 'profile_driven'
        rows.append(row)
    return {
        'schema_version': '1.0', 'scenario': data['workload']['scenario'], 'features': rows,
        'kernel_calibration_coverage': copy.deepcopy(calibration),
        'coverage_reference': 'model_metadata.kernel_calibration',
        'notes': [
            'Supported denotes simulator coverage, not a quality-preserving substitution, measured acceleration or runtime compatibility guarantee.',
            'Analytical costs use explicit work, state and topology. Profile-driven costs remain dependent on their provenance and matching coverage.',
            'No measured serving claim follows from a configured profile or a source.status value alone; inspect actual kernel coverage and independent workload measurements.',
            'Resolved configuration contains canonical fields only; feature input shorthand is omitted to permit stable reruns and canonical-field sweeps.',
        ],
    }

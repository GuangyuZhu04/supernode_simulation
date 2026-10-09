"""Versioned, strict configuration and transparent profile import."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer

DType = Literal['fp4', 'fp8', 'fp16', 'bf16', 'fp32', 'int8']
Positive = float


class Spec(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, validate_default=True)


class Source(Spec):
    status: Literal['assumed', 'public', 'user', 'measured'] = 'assumed'
    description: str = 'User scenario; uncalibrated architecture assumptions.'
    uri: str | None = None
    revision: str | None = None


class MLASpec(Spec):
    q_lora_rank: int = Field(default=0, ge=0)
    kv_lora_rank: int = Field(default=512, gt=0)
    qk_nope_head_dim: int = Field(default=128, gt=0)
    qk_rope_head_dim: int = Field(default=64, ge=0)
    v_head_dim: int = Field(default=128, gt=0)
    use_output_gate: bool = False
    use_nope: bool = False
    cache_layout: Literal['compressed', 'expanded'] = 'compressed'


class KDASpec(Spec):
    num_heads: int = Field(default=32, gt=0)
    head_k_dim: int = Field(default=128, gt=0)
    head_v_dim: int = Field(default=128, gt=0)
    conv_kernel: int = Field(default=4, gt=0)
    gate_low_rank_dim: int = Field(default=128, gt=0)
    use_full_rank_gate: bool = False
    state_dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp32'
    chunk_size: int = Field(default=64, gt=0)
    conv_state_tokens: int | None = Field(default=None, ge=0)


class MoESpec(Spec):
    num_experts: int = Field(default=1, gt=0)
    top_k: int = Field(default=1, gt=0)
    intermediate_dim: int = Field(gt=0)
    num_shared_experts: int = Field(default=0, ge=0)
    shared_intermediate_dim: int | None = Field(default=None, gt=0)
    latent_dim: int | None = Field(default=None, gt=0)
    latent_norm: bool = False
    shared_input_dim: int | None = Field(default=None, gt=0)
    router_bias: bool = False


class ArchitectureSpec(Spec):
    vocab_size: int = Field(gt=0)
    tie_embeddings: bool = False
    attention_by_layer: list[Literal['MLA', 'KDA', 'MHA', 'GQA', 'MQA']]
    ffn_by_layer: list[Literal['Dense', 'MoE']]
    mla: MLASpec = Field(default_factory=MLASpec)
    kda: KDASpec = Field(default_factory=KDASpec)
    moe: MoESpec | None = None
    dense_intermediate_dim: int | None = Field(default=None, gt=0)
    linear_bias: bool = False
    norm_bias: bool = False
    final_norm: bool = True
    attnres_block_size: int | None = Field(default=None, gt=0)


class ModelSpec(Spec):
    name: str
    total_params: float | None = Field(default=None, gt=0)
    active_params: float | None = Field(default=None, gt=0)
    layers: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    intermediate_dim: int | None = Field(default=None, gt=0)
    num_heads: int = Field(default=32, gt=0)
    kv_heads: int = Field(default=8, gt=0)
    head_dim: int = Field(default=128, gt=0)
    attention_kind: Literal['mha', 'gqa', 'mqa', 'latent'] = 'gqa'
    kv_latent_dim: int = Field(default=512, gt=0)
    kv_rope_dim: int = Field(default=64, ge=0)
    num_experts: int = Field(default=1, gt=0)
    top_k: int = Field(default=1, gt=0)
    moe_layer_fraction: float = Field(default=0, ge=0, le=1)
    expert_parameter_fraction: float = Field(default=0, ge=0, le=1)
    architecture: ArchitectureSpec | None = None
    reported_total_params: float | None = Field(default=None, gt=0)
    reported_active_params: float | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def coherent(self):
        if self.architecture is not None:
            from .components import inventory_summary
            architecture = self.architecture
            if architecture.moe is not None:
                self.num_experts, self.top_k = architecture.moe.num_experts, architecture.moe.top_k
            if 'MoE' in architecture.ffn_by_layer and architecture.moe is None:
                raise ValueError('MoE layers require architecture.moe')
            summary = inventory_summary(self.model_dump())
            for field in ('total_params', 'active_params'):
                provided = getattr(self, field)
                derived = summary[field]
                if provided is not None and not math.isclose(provided, derived, rel_tol=1e-9):
                    raise ValueError(f'{field} conflicts with tensor shapes: derived {derived}; use reported_{field} for rounded public counts')
                setattr(self, field, derived)
            self.expert_parameter_fraction = summary['expert_params'] / self.total_params
            self.moe_layer_fraction = architecture.ffn_by_layer.count('MoE') / self.layers
        if self.total_params is None:
            raise ValueError('total_params is required without model.architecture')
        if self.active_params is None:
            self.active_params = self.total_params * (1 - self.expert_parameter_fraction + self.expert_parameter_fraction * self.top_k / self.num_experts)
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim
        if self.active_params > self.total_params:
            raise ValueError('active_params must not exceed total_params')
        if self.architecture is None and self.active_params < self.total_params * (1 - self.expert_parameter_fraction) * (1 - 1e-12):
            raise ValueError('active_params must include all dense parameters')
        if self.top_k > self.num_experts:
            raise ValueError('top_k must not exceed num_experts')
        if self.kv_heads > self.num_heads or self.num_heads % self.kv_heads:
            raise ValueError('kv_heads must divide num_heads')
        if self.attention_kind == 'mha' and self.kv_heads != self.num_heads:
            raise ValueError('MHA requires kv_heads == num_heads')
        if self.attention_kind == 'mqa' and self.kv_heads != 1:
            raise ValueError('MQA requires kv_heads == 1')
        if self.num_experts == 1 and (self.moe_layer_fraction or self.expert_parameter_fraction):
            raise ValueError('MoE fractions require num_experts > 1')
        if bool(self.moe_layer_fraction) != bool(self.expert_parameter_fraction):
            raise ValueError('MoE layer and parameter fractions must both be positive or both zero')
        return self


class NPUSpec(Spec):
    hbm_bytes: float = Field(default=80e9, gt=0)
    hbm_Bps: float = Field(default=3e12, gt=0)
    flops: dict[str, float] = Field(default_factory=lambda: {'fp8': 2e15, 'fp16': 1e15, 'bf16': 1e15, 'fp32': 5e13})
    efficiency: float = Field(default=0.6, gt=0, le=1)
    runtime_bytes: float = Field(default=2e9, ge=0)
    workspace_bytes: float = Field(default=64e6, ge=0)

    @model_validator(mode='after')
    def finite_flops(self):
        if not self.flops or any(v <= 0 or not math.isfinite(v) for v in self.flops.values()):
            raise ValueError('flops capacities must be finite and positive')
        return self


class CPUSpec(Spec):
    dram_bytes: float = Field(default=2e12, gt=0)
    dram_Bps: float = Field(default=4e11, gt=0)
    flops: float = Field(default=1e12, gt=0)
    efficiency: float = Field(default=0.7, gt=0, le=1)
    iops: float = Field(default=1e8, gt=0)


class LinkSpec(Spec):
    bandwidth_Bps: float = Field(default=9e11, gt=0)
    latency_s: float = Field(default=1e-6, ge=0)
    efficiency: float = Field(default=0.85, gt=0, le=1)
    injection_Bps: float | None = Field(default=None, gt=0)
    uplink_Bps: float | None = Field(default=None, gt=0)

    @model_validator(mode='after')
    def injection(self):
        if self.injection_Bps is None:
            self.injection_Bps = self.bandwidth_Bps
        return self


class ExternalMemorySpec(Spec):
    enabled: bool = False
    capacity_bytes: float = Field(default=1e14, gt=0)
    bandwidth_Bps: float = Field(default=1e11, gt=0)
    latency_s: float = Field(default=5e-6, ge=0)
    iops: float = Field(default=1e7, gt=0)


class ExplicitLink(Spec):
    id: str
    src: str
    dst: str
    bandwidth_Bps: float = Field(gt=0)
    latency_s: float = Field(default=1e-6, ge=0)
    domain: Literal['SU', 'SO', 'H2D', 'D2H', 'remote', 'ssd']
    efficiency: float = Field(default=1, gt=0, le=1)
    shared_group: str | None = None
    shared_Bps: float | None = Field(default=None, gt=0)


class KernelProfile(Spec):
    flops: float = Field(ge=0)
    memory_bytes: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    dtype: DType | None = None
    phase: str | None = None


class HardwareSpec(Spec):
    name: str = 'unspecified hardware profile'
    npu_count: int = Field(default=8, gt=0)
    su_size: int = Field(default=8, gt=0)
    npus_per_host: int = Field(default=8, gt=0)
    topology: Literal['full_mesh', 'mesh', 'clos', 'fat_tree', 'explicit'] = 'full_mesh'
    routing_policy: Literal['shortest_hop', 'su_so'] = 'shortest_hop'
    mesh_width: int | None = Field(default=None, gt=0)
    leaf_size: int | None = Field(default=None, gt=0)
    npu: NPUSpec = Field(default_factory=NPUSpec)
    cpu: CPUSpec = Field(default_factory=CPUSpec)
    scale_up: LinkSpec = Field(default_factory=LinkSpec)
    scale_out: LinkSpec = Field(default_factory=lambda: LinkSpec(bandwidth_Bps=50e9, latency_s=5e-6))
    host_link: LinkSpec = Field(default_factory=lambda: LinkSpec(bandwidth_Bps=64e9, latency_s=2e-6))
    remote: ExternalMemorySpec = Field(default_factory=ExternalMemorySpec)
    ssd: ExternalMemorySpec = Field(default_factory=lambda: ExternalMemorySpec(bandwidth_Bps=10e9, latency_s=100e-6, iops=1e6))
    explicit_links: list[ExplicitLink] = Field(default_factory=list)
    routes: dict[str, list[str]] = Field(default_factory=dict)
    kernel_profiles: dict[str, list[KernelProfile]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def topology_valid(self):
        if self.su_size > self.npu_count or self.npu_count % self.su_size:
            raise ValueError('su_size must divide npu_count')
        if self.topology == 'explicit' and not self.explicit_links:
            raise ValueError('explicit topology requires directed explicit_links')
        ids = [e.id for e in self.explicit_links]
        if len(ids) != len(set(ids)):
            raise ValueError('explicit link IDs must be unique')
        return self


class WorkloadSpec(Spec):
    scenario: Literal['training', 'inference']
    batch_size: int = Field(default=1, gt=0)
    sequence_length: int = Field(default=2048, gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    training_steps: int = Field(default=1, gt=0)
    input_length: int = Field(default=1024, gt=0)
    output_length: int = Field(default=16, gt=0)
    input_lengths: list[int] = Field(default_factory=list)
    output_lengths: list[int] = Field(default_factory=list)

    @model_validator(mode='after')
    def lengths(self):
        self.input_lengths = self.input_lengths or [self.input_length] * self.batch_size
        self.output_lengths = self.output_lengths or [self.output_length] * self.batch_size
        for key in ('input_lengths', 'output_lengths'):
            values = getattr(self, key)
            if len(values) != self.batch_size or any(x < 1 for x in values):
                raise ValueError(f'{key} must contain batch_size positive lengths')
        return self


class PrecisionSpec(Spec):
    weight: DType = 'fp16'
    compute: Literal['fp8', 'fp16', 'bf16', 'fp32'] = 'fp16'
    gradient: DType = 'fp16'
    master: DType = 'fp32'
    optimizer: DType = 'fp32'
    kv: DType = 'fp16'
    weight_transfer: DType = 'fp16'
    activation: DType = 'fp16'
    communication: DType = 'fp16'
    recurrent: Literal['fp16', 'bf16', 'fp32'] = 'fp32'
    routed_expert_weight: DType | None = None


class OptimizerSpec(Spec):
    kind: Literal['adam', 'muon'] = 'adam'
    muon_variant: Literal['matrix', 'per_head'] = 'matrix'
    ns_iterations: int = Field(default=5, gt=0)
    workspace_copies: int = Field(default=2, ge=1)
    inflight_components_per_rank: int | None = Field(default=None, gt=0)

    @model_serializer(mode='wrap')
    def omit_unspecified_pipeline(self, handler):
        data = handler(self)
        if self.inflight_components_per_rank is None:
            data.pop('inflight_components_per_rank', None)
        return data


class EngramSpec(Spec):
    enabled: bool = False
    table_bytes: float = Field(default=1e12, gt=0)
    queries_per_token: float = Field(default=1, ge=0)
    record_bytes: int = Field(default=4096, gt=0)
    hit_rate: float = Field(default=0, ge=0, le=1)
    cache_bytes: float = Field(default=0, ge=0)
    location: Literal['host', 'remote'] = 'host'
    iops: float = Field(default=1e7, gt=0)
    latency_s: float = Field(default=1e-6, ge=0)
    layers: list[int] = Field(default_factory=lambda: [0])

    @model_validator(mode='after')
    def cache_valid(self):
        if self.enabled and self.hit_rate > 0 and self.cache_bytes == 0:
            raise ValueError('Engram hit_rate > 0 requires explicit cache_bytes')
        return self


class FeatureSpec(Spec):
    optimizer_offload: bool = False
    activation_checkpointing: bool = False
    kv_offload: Literal['none', 'host', 'remote'] = 'none'
    kv_offload_fraction: float = Field(default=1, ge=0, le=1)
    kv_page_tokens: int = Field(default=16, gt=0)
    kv_page_entry_bytes: int = Field(default=8, gt=0)
    weight_scale_block: int = Field(default=128, gt=0)
    scale_bytes: int = Field(default=4, gt=0)
    kv_scale_block: int = Field(default=128, gt=0)
    extra_weight_copy_dtype: DType | None = None
    engram: EngramSpec = Field(default_factory=EngramSpec)


class ExecutionSpec(Spec):
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)
    dp: int = Field(default=1, gt=0)
    fsdp: int = Field(default=1, gt=0)
    rank_order: list[int] = Field(default_factory=list)
    reshard_after_forward: bool = True
    sync_each_microbatch: bool = False
    prefetch: bool = False
    precision: PrecisionSpec = Field(default_factory=PrecisionSpec)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)


class ParallelCandidate(Spec):
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)
    dp: int = Field(default=1, gt=0)
    fsdp: int = Field(default=1, gt=0)
    rank_order: list[int] = Field(default_factory=list)


class TargetSpec(Spec):
    throughput_tokens_s: float | None = Field(default=None, gt=0)
    ttft_s: float | None = Field(default=None, gt=0)
    tpot_s: float | None = Field(default=None, gt=0)
    step_time_s: float | None = Field(default=None, gt=0)
    candidate_npu_counts: list[int] = Field(default_factory=list)
    candidate_su_sizes: list[int] = Field(default_factory=list)
    parallel_candidates: list[ParallelCandidate] = Field(default_factory=list)
    total_npus: int | None = Field(default=None, gt=0)
    max_total_npus: int | None = Field(default=None, gt=0)
    max_candidates: int = Field(default=128, gt=0)

    @model_validator(mode='after')
    def ranges(self):
        if any(x < 1 for x in self.candidate_npu_counts + self.candidate_su_sizes):
            raise ValueError('candidate counts must be positive')
        return self


class Config(Spec):
    schema_version: Literal['1.0'] = '1.0'
    model: ModelSpec
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    workload: WorkloadSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    targets: TargetSpec = Field(default_factory=TargetSpec)
    seed: int = 0
    sources: dict[str, Source] = Field(default_factory=lambda: {'*': Source()})
    assumptions: list[str] = Field(default_factory=list)
    calibration: str | None = None

    @model_validator(mode='after')
    def consistency(self):
        e, h, m = self.execution, self.hardware, self.model
        if e.tp * e.pp * e.ep * e.dp != h.npu_count:
            raise ValueError('tp * pp * ep * dp must equal hardware.npu_count; fsdp is a DP subgroup')
        if e.dp % e.fsdp:
            raise ValueError('fsdp must divide dp')
        if e.optimizer.kind == 'muon' and m.architecture is None:
            raise ValueError('Muon requires explicit component tensor shapes in model.architecture')
        if m.architecture and m.architecture.tie_embeddings and e.pp > 1:
            raise ValueError('Tied embedding/output across PP stages requires a shared-gradient protocol; use pp=1 or untied weights')
        if m.architecture and m.architecture.attnres_block_size and e.pp > 1:
            raise ValueError('AttnRes residual-block forwarding across PP is not implemented; use pp=1 for this architecture')
        if m.architecture is None and (e.precision.weight == 'fp4' or e.precision.routed_expert_weight is not None):
            raise ValueError('FP4/mixed routed expert storage requires model.architecture')
        unsupported_fp4 = [key for key, value in e.precision.model_dump().items() if value == 'fp4' and key not in ('weight', 'routed_expert_weight')]
        if unsupported_fp4:
            raise ValueError(f'FP4 is currently supported for component weight storage only, not {unsupported_fp4}')
        if m.architecture:
            if e.prefetch:
                raise ValueError('Component architecture path does not yet model FSDP lookahead prefetch; set prefetch=false')
            if m.num_heads % e.tp:
                raise ValueError('Shape-derived attention requires num_heads divisible by TP')
            if 'KDA' in m.architecture.attention_by_layer and m.architecture.kda.num_heads % e.tp:
                raise ValueError('Shape-derived KDA heads must be divisible by TP')
        if e.pp > m.layers:
            raise ValueError('pp must not exceed model.layers')
        if m.num_experts % e.ep:
            raise ValueError('ep must divide num_experts')
        if e.rank_order and sorted(e.rank_order) != list(range(h.npu_count)):
            raise ValueError('rank_order must be a permutation of physical NPU indices')
        if e.precision.compute not in h.npu.flops:
            raise ValueError('NPU profile must specify FLOPs for the selected compute dtype')
        f = e.features
        if f.kv_offload == 'remote' or (f.engram.enabled and f.engram.location == 'remote'):
            if not h.remote.enabled:
                raise ValueError('remote memory feature requires hardware.remote.enabled')
        if any(i < 0 or i >= m.layers for i in f.engram.layers) or len(set(f.engram.layers)) != len(f.engram.layers):
            raise ValueError('engram.layers must be unique valid layer indices')
        if self.workload.scenario == 'inference':
            if f.optimizer_offload or f.activation_checkpointing or e.fsdp != 1:
                raise ValueError('optimizer offload, activation checkpointing and FSDP are training-only')
            if self.targets.step_time_s is not None:
                raise ValueError('step_time_s is a training target')
            if self.targets.tpot_s is not None and max(self.workload.output_lengths) == 1:
                raise ValueError('TPOT is undefined for one output token; request at least two')
        else:
            if f.kv_offload != 'none' or f.engram.enabled:
                raise ValueError('KV offload and Engram are inference-only in v1')
            if self.targets.ttft_s is not None or self.targets.tpot_s is not None:
                raise ValueError('TTFT and TPOT are inference targets')
        return self


def read_data(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    try:
        data = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f'{path}: malformed YAML: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a mapping/object')
    return data


def deep_merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = read_data(path)
    if data.get('calibration'):
        source = (path.parent / data['calibration']).resolve()
        calibration = read_data(source)
        if set(calibration) - {'hardware', 'source'}:
            raise ValueError('calibration accepts hardware overrides and source only')
        data['hardware'] = deep_merge(data.get('hardware', {}), calibration.get('hardware', {}))
        data.setdefault('sources', {})['hardware.calibration'] = calibration.get('source', {'status': 'user', 'description': 'External calibration overrides', 'uri': str(source)})
    return Config.model_validate(data)


def import_hf_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Import known structural fields; never invent parameter counts from a model name."""
    raw = read_data(path)
    if raw.get('model_type') == 'kimi_k3' and isinstance(raw.get('text_config'), dict):
        text = raw['text_config']
        linear = text['linear_attn_config']
        layers = text['num_hidden_layers']
        full = set(linear['full_attn_layers'])
        kda = set(linear['kda_layers'])
        if full & kda or full | kda != set(range(1, layers + 1)):
            raise ValueError('Kimi K3 attention layer lists must partition the 1-based layer indices')
        model = {
            'name': 'Kimi-K3-text-backbone', 'layers': layers,
            'hidden_dim': text['hidden_size'], 'intermediate_dim': text['intermediate_size'],
            'num_heads': text['num_attention_heads'], 'kv_heads': text['num_attention_heads'],
            'head_dim': text['v_head_dim'], 'attention_kind': 'latent',
            'kv_latent_dim': text['kv_lora_rank'], 'kv_rope_dim': text['qk_rope_head_dim'],
            'architecture': {
                'vocab_size': text['vocab_size'], 'tie_embeddings': text['tie_word_embeddings'],
                'attention_by_layer': ['MLA' if i in full else 'KDA' for i in range(1, layers + 1)],
                'ffn_by_layer': ['Dense' if i < text['first_k_dense_replace'] else 'MoE' for i in range(layers)],
                'dense_intermediate_dim': text['intermediate_size'],
                'attnres_block_size': text.get('attn_res_block_size'),
                'mla': {'q_lora_rank': text['q_lora_rank'], 'kv_lora_rank': text['kv_lora_rank'],
                        'qk_nope_head_dim': text['qk_nope_head_dim'], 'qk_rope_head_dim': text['qk_rope_head_dim'],
                        'v_head_dim': text['v_head_dim'], 'use_output_gate': text['mla_use_output_gate'], 'use_nope': text['mla_use_nope']},
                'kda': {'num_heads': linear['num_heads'], 'head_k_dim': linear['head_dim'], 'head_v_dim': linear['head_dim'],
                        'conv_kernel': linear['short_conv_kernel_size'], 'gate_low_rank_dim': linear['head_dim'],
                        'use_full_rank_gate': linear['use_full_rank_gate']},
                'moe': {'num_experts': text['num_experts'], 'top_k': text['num_experts_per_token'],
                        'intermediate_dim': text['moe_intermediate_size'], 'num_shared_experts': text['num_shared_experts'],
                        'latent_dim': text['routed_expert_hidden_size'], 'latent_norm': text['latent_moe_use_norm'],
                        'shared_input_dim': text['hidden_size'], 'router_bias': True},
            },
        }
        model = deep_merge(model, overrides or {})
        validated = ModelSpec.model_validate(model).model_dump()
        missing = ['architecture.MTP (not supported)'] if text.get('num_nextn_predict_layers', 0) else []
        return {'schema_version': '1.0', 'model': validated, 'missing_fields': missing,
                'scope': 'Text backbone only; vision tower/projector excluded. Storage precision is selected separately from source mixed checkpoint.',
                'sources': {'model': {'status': 'user', 'description': 'Kimi K3 architecture imported from explicit nested HF text_config and checked layer partition', 'uri': str(Path(path).resolve())}}}
    mapping = {
        'num_hidden_layers': 'layers', 'hidden_size': 'hidden_dim',
        'intermediate_size': 'intermediate_dim', 'num_attention_heads': 'num_heads',
        'num_key_value_heads': 'kv_heads', 'head_dim': 'head_dim',
        'num_local_experts': 'num_experts', 'n_routed_experts': 'num_experts',
        'num_experts_per_tok': 'top_k', 'kv_lora_rank': 'kv_latent_dim',
        'qk_rope_head_dim': 'kv_rope_dim',
    }
    model = {target: raw[key] for key, target in mapping.items() if raw.get(key) is not None}
    model['name'] = raw.get('_name_or_path') or raw.get('model_type') or Path(path).stem
    if 'num_heads' in model and 'hidden_dim' in model and 'head_dim' not in model:
        if model['hidden_dim'] % model['num_heads'] == 0:
            model['head_dim'] = model['hidden_dim'] // model['num_heads']
    if 'kv_latent_dim' in model:
        model['attention_kind'] = 'latent'
    elif 'kv_heads' in model and 'num_heads' in model:
        model['attention_kind'] = 'mha' if model['kv_heads'] == model['num_heads'] else ('mqa' if model['kv_heads'] == 1 else 'gqa')
    model.update(overrides or {})
    required = ['total_params', 'active_params', 'layers', 'hidden_dim', 'num_heads', 'kv_heads', 'head_dim', 'attention_kind']
    if model.get('num_experts', 1) > 1:
        required += ['moe_layer_fraction', 'expert_parameter_fraction', 'top_k']
    missing = [key for key in required if key not in model]
    return {'schema_version': '1.0', 'model': model, 'missing_fields': missing,
            'sources': {'model': {'status': 'user', 'description': 'Imported HF structural config; parameter counts require explicit verification', 'uri': str(Path(path).resolve())}}}

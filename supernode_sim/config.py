"""Versioned, strict configuration and transparent profile import."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator, model_serializer

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


class SparseAttentionSpec(Spec):
    q_lora_rank: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    rope_head_dim: int = Field(ge=0)
    o_groups: int = Field(gt=0)
    o_lora_rank: int = Field(gt=0)
    window_size: int = Field(gt=0)

    @model_validator(mode='after')
    def rotary_width(self):
        if self.rope_head_dim > self.head_dim:
            raise ValueError('sparse attention rope_head_dim must not exceed head_dim')
        return self


class CSASpec(SparseAttentionSpec):
    compress_ratio: Literal[4] = 4
    index_num_heads: int = Field(gt=0)
    index_head_dim: int = Field(gt=0)
    index_topk: int = Field(gt=0)

    @model_validator(mode='after')
    def index_width(self):
        if self.index_head_dim < self.rope_head_dim:
            raise ValueError('CSA index_head_dim must be at least rope_head_dim')
        return self


class HCASpec(SparseAttentionSpec):
    compress_ratio: int = Field(default=128, gt=1)

    @model_validator(mode='after')
    def nonoverlapping_ratio(self):
        if self.compress_ratio == 4:
            raise ValueError('HCA compress_ratio=4 is unsupported; use CSA for the overlapping ratio-4 layout')
        return self


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
    output_head_dim: int | None = Field(default=None, gt=0,
        description='Defaults to vocab_size. Use 1 for a reward/value head while retaining the tokenizer embedding vocabulary.')
    tie_embeddings: bool = False
    attention_by_layer: list[Literal['MLA', 'KDA', 'MHA', 'GQA', 'MQA', 'CSA', 'HCA']]
    ffn_by_layer: list[Literal['Dense', 'MoE']]
    mla: MLASpec = Field(default_factory=MLASpec)
    kda: KDASpec = Field(default_factory=KDASpec)
    csa: CSASpec | None = None
    hca: HCASpec | None = None
    moe: MoESpec | None = None
    dense_intermediate_dim: int | None = Field(default=None, gt=0)
    linear_bias: bool = False
    norm_bias: bool = False
    final_norm: bool = True
    attnres_block_size: int | None = Field(default=None, gt=0)

    @model_serializer(mode='wrap')
    def omit_default_output_head(self, handler):
        data = handler(self)
        if self.output_head_dim is None:
            data.pop('output_head_dim', None)
        return data

    @model_validator(mode='after')
    def sparse_dimensions(self):
        if self.tie_embeddings and self.output_head_dim not in (None, self.vocab_size):
            raise ValueError('Tied embeddings require output_head_dim equal to vocab_size')
        for kind in ('CSA', 'HCA'):
            if kind in self.attention_by_layer:
                if getattr(self, kind.lower()) is None:
                    raise ValueError(f'{kind} layers require explicit architecture.{kind.lower()} dimensions')
                if self.linear_bias or self.norm_bias:
                    raise ValueError('CSA/HCA currently require linear_bias=false and norm_bias=false')
        return self


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


class ScaleOutLinkSpec(LinkSpec):
    host_injection_Bps: float | None = Field(default=None, gt=0,
        description='Aggregate scale-out NIC bandwidth per host and direction, before efficiency. '
                    'Omit to retain the per-device injection_Bps shared-host limit.')


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


class IOGroupSpec(Spec):
    """Optional shared endpoint I/O capacity, independent of routed link bytes."""
    id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
    members: list[StrictInt] = Field(min_length=1,
        description='Physical NPU indices, unaffected by execution.rank_order. Overlap between groups expresses nested constraints.')
    domains: list[Literal['SO', 'H2D', 'D2H']] = Field(min_length=1,
        description='Apply only to transfers whose actual route uses one of these domains; H2D is destination/rx, D2H is source/tx.')
    bandwidth_Bps: float = Field(gt=0, strict=True)
    efficiency: float = Field(default=1.0, gt=0, le=1, strict=True)
    duplex: Literal['full', 'shared'] = 'full'
    latency_s: float = Field(default=0.0, ge=0, strict=True,
        description='Additional latency for each participating NPU endpoint, independent of physical link startup.')

    @model_validator(mode='after')
    def members_and_domains(self):
        if any(member < 0 for member in self.members):
            raise ValueError('I/O group members must be nonnegative physical NPU indices')
        if len(self.members) != len(set(self.members)):
            raise ValueError('I/O group members must be unique within each group')
        if len(self.domains) != len(set(self.domains)):
            raise ValueError('I/O group domains must be unique within each group')
        return self


def validate_io_groups(groups: list[IOGroupSpec], npu_count: int) -> None:
    ids = [group.id for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('I/O group IDs must be unique')
    if any(member >= npu_count for group in groups for member in group.members):
        raise ValueError('I/O group members must reference existing physical NPU indices')


CALIBRATION_CONTEXT_FIELDS = (
    'device_sku', 'cann_version', 'hccl_version', 'driver_version', 'runtime',
    'runtime_version', 'transport', 'mode', 'timer_boundary',
)


def validate_calibration_context(context: dict, *, require_complete: bool = False) -> None:
    if not isinstance(context, dict):
        raise ValueError('calibration_context must be an object of known nonempty strings')
    unknown = {'unknown', 'unspecified', 'not_reported', 'not reported', 'unavailable', 'tbd', '?', 'n/a'}
    for key, value in context.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip() or value.strip().lower() in unknown:
            raise ValueError('calibration_context keys and values must be known nonempty strings')
    if require_complete:
        missing = set(CALIBRATION_CONTEXT_FIELDS) - set(context)
        if missing:
            raise ValueError('calibration_context is missing required identity fields: ' + ', '.join(sorted(missing)))
        if context['timer_boundary'] not in ('device_only', 'whole_call'):
            raise ValueError('calibration_context.timer_boundary must be device_only or whole_call')


def validate_json_metadata(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError('metadata_match object keys must be nonempty strings')
        for item in value.values():
            validate_json_metadata(item)
        return
    if isinstance(value, list):
        for item in value:
            validate_json_metadata(item)
        return
    raise ValueError('metadata_match must contain only finite JSON values')


class KernelProfile(Spec):
    flops: float = Field(ge=0)
    memory_bytes: float = Field(ge=0)
    duration_s: float = Field(gt=0, description='Solo kernel service time, excluding startup_s.')
    startup_s: float | None = Field(default=None, ge=0,
        description='Optional measured event startup; replaces rather than adds to existing startup.')
    dtype: DType | None = None
    phase: str | None = None
    hardware_name: str | None = Field(default=None, min_length=1)
    calibration_context: dict[str, StrictStr] = Field(default_factory=dict)
    metadata_match: dict[str, Any] = Field(default_factory=dict)
    source: str | None = Field(default=None, min_length=1)
    evidence_kind: Literal['measurement', 'simulator', 'synthetic', 'unspecified'] = 'unspecified'
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def applicability(self):
        if self.calibration_context:
            validate_calibration_context(self.calibration_context, require_complete=True)
        validate_json_metadata(self.metadata_match)
        validate_json_metadata(self.provenance)
        return self


class MatmulShape(Spec):
    m: StrictInt = Field(gt=0)
    n: StrictInt = Field(gt=0)
    k: StrictInt = Field(gt=0)
    batch: StrictInt = Field(default=1, gt=0)


class KernelResourceDemand(Spec):
    """Normalized service demand, not a complete kernel timing measurement."""
    resource: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    service_s: float = Field(ge=0,
        description='Service demand normalized to the declared physical resource capacity and scope.')
    kind: Literal['compute', 'memory', 'issue']
    source: str = Field(min_length=1)
    evidence_kind: Literal['measured_reference', 'architecture', 'inferred', 'assumed'] = 'assumed'

    @model_validator(mode='after')
    def nonblank_source(self):
        if not self.source.strip():
            raise ValueError('resource demand source must be nonempty')
        return self


class KernelTraceOperation(Spec):
    kernel: str = Field(min_length=1)
    flops: float = Field(ge=0)
    memory_bytes: float = Field(ge=0)
    startup_s: float = Field(default=0.0, ge=0)
    workspace_bytes: float = Field(default=0.0, ge=0)
    resource_demands: list[KernelResourceDemand] = Field(default_factory=list)
    dependency_floor_s: float = Field(default=0.0, ge=0,
        description='Required serial dependency path inside the kernel, excluding startup; a lower bound, not an added delay.')
    matmul: MatmulShape | None = None
    implementation: dict[str, Any] = Field(default_factory=dict,
        description='Explicit kernel binary, tiling and tensor layout identity; never inferred from FLOPs.')

    @model_validator(mode='after')
    def explicit_work(self):
        if not self.kernel.strip():
            raise ValueError('kernel trace operation kernel must be nonempty')
        validate_json_metadata(self.implementation)
        demand_kinds = {}
        demand_values = {}
        for demand in self.resource_demands:
            if demand.resource in demand_kinds and demand_kinds[demand.resource] != demand.kind:
                raise ValueError('resource demand kind must agree for a shared resource')
            demand_kinds[demand.resource] = demand.kind
            demand_values.setdefault(demand.resource, []).append(demand.service_s)
        try:
            totals = [math.fsum(values) for values in demand_values.values()]
        except OverflowError as exc:
            raise ValueError('resource demand totals must be finite') from exc
        if any(not math.isfinite(total) for total in totals):
            raise ValueError('resource demand totals must be finite')
        if self.matmul is not None:
            shape = self.matmul
            expected = 2 * shape.batch * shape.m * shape.n * shape.k
            try:
                matches = math.isclose(self.flops, expected, rel_tol=1e-9, abs_tol=1e-6)
            except OverflowError:
                matches = False
            if not matches:
                raise ValueError('matmul FLOPs must equal 2 * batch * m * n * k')
        return self


class KernelTrace(Spec):
    """A scoped serial operator trace refining one aggregate kernel event."""
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    hardware_name: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    dtype: DType
    flops: float = Field(ge=0, description='Original aggregate event FLOPs; exact-match guard.')
    memory_bytes: float = Field(ge=0, description='Original aggregate HBM traffic; exact-match guard.')
    metadata_match: dict[str, Any] = Field(min_length=1)
    calibration_context: dict[str, StrictStr] = Field(default_factory=dict)
    operations: list[KernelTraceOperation] = Field(min_length=1)

    @model_validator(mode='after')
    def conserved_work(self):
        if any(not getattr(self, key).strip() for key in ('id', 'source', 'hardware_name', 'phase')):
            raise ValueError('kernel trace identity, phase and source must be nonempty')
        validate_json_metadata(self.metadata_match)
        if self.calibration_context:
            validate_calibration_context(self.calibration_context, require_complete=True)
        for operation in self.operations:
            if 'david_v100_service_model' not in operation.implementation:
                continue
            reference = operation.implementation['david_v100_service_model']
            if not isinstance(reference, dict):
                raise ValueError('david_v100_service_model must contain the explicit reference workload')
            identity = {'device_sku': 'DavidV100', 'cann_version': '9.1.T560'}
            if any(self.calibration_context.get(key) != value or reference.get(key) != value
                   for key, value in identity.items()):
                raise ValueError('DavidV100 resource references require matching DavidV100 / CANN 9.1.T560 trace context')
            if type(reference.get('enabled_ai_cores')) is not int or reference['enabled_ai_cores'] != 32:
                raise ValueError('DavidV100 resource references require the measured 32 AI Core configuration')
            cube = reference.get('cube')
            if cube is not None:
                if not isinstance(cube, dict):
                    raise ValueError('DavidV100 Cube reference must contain its explicit mode')
                mode = cube.get('mode')
                if mode != self.dtype and not (mode == 'hf32' and self.dtype == 'fp32'):
                    raise ValueError('DavidV100 Cube mode must match the trace dtype; HF32 requires FP32')
        try:
            total_flops = math.fsum(op.flops for op in self.operations)
            total_bytes = math.fsum(op.memory_bytes for op in self.operations)
        except OverflowError as exc:
            raise ValueError('kernel trace operation totals must be finite') from exc
        if not math.isfinite(total_flops) or not math.isclose(total_flops, self.flops, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('kernel trace operations must conserve aggregate FLOPs')
        if not math.isfinite(total_bytes) or total_bytes < self.memory_bytes - max(1e-6, self.memory_bytes * 1e-9):
            raise ValueError('kernel trace operation HBM bytes must not undercount aggregate memory_bytes')
        return self


class CollectiveProfile(Spec):
    """Exact byte-vector applicability; call duration includes launch/startup."""
    id: str = Field(min_length=1)
    collective: Literal['all_gather', 'reduce_scatter', 'all_reduce', 'all_to_all_v']
    semantic: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    world_size: StrictInt = Field(ge=2)
    dtype: DType
    algorithm: Literal['pairwise', 'direct', 'ring']
    timing_scope: Literal['transport', 'activation_collective'] = 'transport'
    total_payload_bytes: float = Field(gt=0)
    send_payload_bytes: list[float] = Field(min_length=2)
    recv_payload_bytes: list[float] = Field(min_length=2)
    duration_s: float = Field(gt=0, description='Call wall time including startup; excludes warmup and external queueing.')
    source: str = Field(min_length=1)
    hardware_name: str = Field(min_length=1)
    applicability_signature: str | None = Field(default=None, pattern=r'^v1:sha256:[0-9a-f]{64}$')

    @model_validator(mode='after')
    def payload_vectors(self):
        for values in (self.send_payload_bytes, self.recv_payload_bytes):
            if len(values) != self.world_size or any(value < 0 for value in values):
                raise ValueError('collective payload vectors must contain world_size nonnegative values')
            if not math.isclose(math.fsum(values), self.total_payload_bytes, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('collective payload vector sum must equal total_payload_bytes')
        if self.collective != 'all_to_all_v' and self.algorithm != 'ring':
            raise ValueError('all_gather/reduce_scatter/all_reduce profiles require ring algorithm')
        if self.collective == 'all_to_all_v' and self.algorithm == 'ring':
            raise ValueError('all_to_all_v profiles require pairwise or direct algorithm')
        return self


class CollectiveModelSample(Spec):
    payload_bytes: float = Field(gt=0, strict=True, description='Total off-diagonal bytes across all ranks and modeled rounds.')
    duration_s: float = Field(gt=0, strict=True)


class CollectiveModel(Spec):
    id: str = Field(min_length=1)
    collective: Literal['all_gather', 'reduce_scatter', 'all_reduce', 'all_to_all_v']
    semantic: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    world_size: StrictInt = Field(ge=2)
    dtype: DType
    algorithm: Literal['pairwise', 'direct', 'ring']
    timing_scope: Literal['transport', 'activation_collective'] = 'transport'
    hardware_name: str = Field(min_length=1)
    family_signature: str = Field(pattern=r'^collective-family-v1:sha256:[0-9a-f]{64}$')
    metadata_match: dict[str, Any] = Field(default_factory=dict)
    samples: list[CollectiveModelSample] = Field(min_length=2)
    interpolation: Literal['linear'] = 'linear'
    extrapolation: Literal['none'] = 'none'
    source: str = Field(min_length=1)

    @model_validator(mode='after')
    def valid_curve(self):
        validate_json_metadata(self.metadata_match)
        if self.timing_scope == 'activation_collective' and self.semantic in ('dispatch', 'combine'):
            required = ('moe_hidden_width', 'moe_tensor_bytes_per_token', 'moe_metadata_bytes_per_token',
                        'moe_top_k', 'moe_num_experts')
            if any(key not in self.metadata_match for key in required):
                raise ValueError('activation MoE models require five shape/format metadata_match keys: ' + ', '.join(required))
            for key in required:
                value = self.metadata_match[key]
                if type(value) is not int or value < (0 if key == 'moe_metadata_bytes_per_token' else 1):
                    raise ValueError('activation MoE metadata_match shape/format fields must be valid integer counts')
            from .core import DTYPE_BYTES
            if self.metadata_match['moe_top_k'] > self.metadata_match['moe_num_experts']:
                raise ValueError('moe_top_k cannot exceed moe_num_experts')
            if self.metadata_match['moe_tensor_bytes_per_token'] != self.metadata_match['moe_hidden_width'] * DTYPE_BYTES[self.dtype]:
                raise ValueError('moe_tensor_bytes_per_token must equal hidden width times wire dtype bytes')
        if any(right.payload_bytes <= left.payload_bytes for left, right in zip(self.samples, self.samples[1:])):
            raise ValueError('collective model samples must have strictly increasing payload_bytes')
        if any(not getattr(self, key).strip() for key in ('id', 'semantic', 'phase', 'hardware_name', 'source')):
            raise ValueError('collective model identity and source strings must be nonempty')
        if (self.collective == 'all_to_all_v') == (self.algorithm == 'ring'):
            raise ValueError('all_to_all_v models require pairwise/direct; other collectives require ring')
        return self


class HardwareSpec(Spec):
    bandwidth_schedules: dict[str, list[tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(gt=0)]]]] = Field(
        default_factory=dict, description='Resource ID to ordered (simulation seconds, effective byte/s) changes.')
    name: str = 'unspecified hardware profile'
    npu_count: int = Field(default=8, gt=0)
    su_size: int = Field(default=8, gt=0)
    npus_per_host: int = Field(default=8, gt=0)
    topology: Literal['full_mesh', 'mesh', 'clos', 'fat_tree', 'explicit'] = 'full_mesh'
    routing_policy: Literal['shortest_hop', 'su_so'] = 'shortest_hop'
    routing_algorithm: Literal['shortest_hop', 'ecmp_hash'] = 'shortest_hop'
    routing_seed: StrictInt = 0
    mesh_width: int | None = Field(default=None, gt=0)
    leaf_size: int | None = Field(default=None, gt=0)
    npu: NPUSpec = Field(default_factory=NPUSpec)
    cpu: CPUSpec = Field(default_factory=CPUSpec)
    scale_up: LinkSpec = Field(default_factory=LinkSpec)
    scale_out: ScaleOutLinkSpec = Field(default_factory=lambda: ScaleOutLinkSpec(bandwidth_Bps=50e9, latency_s=5e-6))
    host_link: LinkSpec = Field(default_factory=lambda: LinkSpec(bandwidth_Bps=64e9, latency_s=2e-6))
    remote: ExternalMemorySpec = Field(default_factory=ExternalMemorySpec)
    ssd: ExternalMemorySpec = Field(default_factory=lambda: ExternalMemorySpec(bandwidth_Bps=10e9, latency_s=100e-6, iops=1e6))
    explicit_links: list[ExplicitLink] = Field(default_factory=list)
    io_groups: list[IOGroupSpec] = Field(default_factory=list)
    routes: dict[str, list[str]] = Field(default_factory=dict)
    kernel_profiles: dict[str, list[KernelProfile]] = Field(default_factory=dict)
    kernel_traces: dict[str, list[KernelTrace]] = Field(default_factory=dict)
    collective_profiles: list[CollectiveProfile] = Field(default_factory=list)
    calibration_context: dict[str, StrictStr] = Field(default_factory=dict)
    collective_models: list[CollectiveModel] = Field(default_factory=list)

    @model_serializer(mode='wrap')
    def omit_empty_bandwidth_schedules(self, handler):
        data = handler(self)
        if not self.bandwidth_schedules:
            data.pop('bandwidth_schedules', None)
        return data

    @model_validator(mode='after')
    def topology_valid(self):
        for key, points in self.bandwidth_schedules.items():
            if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
                raise ValueError(f'{key}: bandwidth schedule times must be strictly increasing')
        if self.su_size > self.npu_count or self.npu_count % self.su_size:
            raise ValueError('su_size must divide npu_count')
        if self.topology == 'explicit' and not self.explicit_links:
            raise ValueError('explicit topology requires directed explicit_links')
        ids = [e.id for e in self.explicit_links]
        if len(ids) != len(set(ids)):
            raise ValueError('explicit link IDs must be unique')
        validate_io_groups(self.io_groups, self.npu_count)
        validate_calibration_context(self.calibration_context, require_complete=bool(self.collective_models))
        model_ids = [model.id for model in self.collective_models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError('collective model IDs must be unique')
        trace_ids = [trace.id for traces in self.kernel_traces.values() for trace in traces]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError('kernel trace IDs must be unique')
        if any(not kernel.strip() or not traces for kernel, traces in self.kernel_traces.items()):
            raise ValueError('kernel_traces requires nonempty kernel names and trace lists')
        return self


class TrainingDataSpec(Spec):
    batching: Literal['padded', 'packed'] = 'padded'
    sequence_lengths: list[StrictInt] = Field(default_factory=list,
        description='Valid tokens per sequence in one microbatch; packed preserves document boundaries.')


class PretrainingSpec(Spec):
    dataset_tokens: int | None = Field(default=None, gt=0)
    epochs: int = Field(default=1, gt=0)

    @model_validator(mode='after')
    def budget(self):
        if self.epochs != 1 and self.dataset_tokens is None:
            raise ValueError('pretrain.epochs requires dataset_tokens')
        return self


class SFTSpec(Spec):
    loss_mask: Literal['all', 'response'] = 'response'
    supervised_tokens: list[StrictInt] = Field(default_factory=list,
        description='Actual labeled positions after shifting/masking; one count per sequence, including multi-turn assistant spans.')


class DPOSpec(Spec):
    chosen_lengths: list[StrictInt]
    rejected_lengths: list[StrictInt]
    reference_log_probs: Literal['online', 'precomputed'] = 'online'


class RLSpec(Spec):
    algorithm: Literal['grpo', 'ppo'] = 'grpo'
    prompts_per_step: int = Field(default=1, gt=0, description='Prompts per DP replica per RL iteration.')
    samples_per_prompt: int = Field(default=2, gt=0)
    prompt_length: int = Field(default=128, gt=0)
    response_length: int = Field(default=64, gt=0)
    prompt_lengths: list[StrictInt] = Field(default_factory=list)
    response_lengths: list[StrictInt] = Field(default_factory=list)
    rollout_batch_size: int = Field(default=1, gt=0)
    update_epochs: int = Field(default=1, gt=0)
    reference_policy: bool = True
    old_log_probs: Literal['rollout', 'recompute'] = 'rollout'
    reward: Literal['rule', 'model'] = 'rule'
    reward_model: ModelSpec | None = None
    critic_model: ModelSpec | None = None
    rule_reward_latency_s: float = Field(default=0, ge=0, description='Per response latency of an external serialized reward worker.')
    reward_workers: int = Field(default=1, gt=0)
    advantage_flops_per_token: float = Field(default=10, gt=0)
    policy_loss_flops_per_token: float = Field(default=20, gt=0)
    experience_bytes_per_token: int = Field(default=24, ge=24,
        description='At least token ID, old/reference log probabilities, reward, advantage and return in 4-byte slots; PPO adds value storage.')

    @model_validator(mode='after')
    def roles(self):
        if self.algorithm == 'grpo' and (self.samples_per_prompt < 2 or self.critic_model is not None):
            raise ValueError('GRPO requires samples_per_prompt >= 2 and has no critic_model')
        if self.algorithm == 'ppo' and self.critic_model is None:
            raise ValueError('PPO requires an explicit critic_model with a scalar output head')
        if (self.reward == 'model') != (self.reward_model is not None):
            raise ValueError('reward=model requires reward_model; rule reward must not supply a model')
        if self.reward == 'model' and (self.rule_reward_latency_s or self.reward_workers != 1):
            raise ValueError('rule reward latency/workers only apply to reward=rule')
        for model in (self.reward_model, self.critic_model):
            if model and (model.architecture is None or model.architecture.output_head_dim != 1):
                raise ValueError('reward_model/critic_model require explicit architecture with output_head_dim=1 (scalar head)')
        for name, count in (('prompt_lengths', self.prompts_per_step),
                            ('response_lengths', self.prompts_per_step * self.samples_per_prompt)):
            values = getattr(self, name)
            if values and (len(values) != count or any(value < 1 for value in values)):
                raise ValueError(f'rl.{name} requires {count} positive lengths')
        return self


class TrainingTaskSpec(Spec):
    kind: Literal['pretrain', 'sft', 'dpo', 'rl'] = 'pretrain'
    data: TrainingDataSpec = Field(default_factory=TrainingDataSpec)
    pretrain: PretrainingSpec | None = None
    sft: SFTSpec | None = None
    dpo: DPOSpec | None = None
    rl: RLSpec | None = None

    @model_validator(mode='after')
    def task_parameters(self):
        for name in ('pretrain', 'sft', 'dpo', 'rl'):
            if name != self.kind and getattr(self, name) is not None:
                raise ValueError(f'training.{name} is only valid for kind={name}')
        if self.kind != 'pretrain' and getattr(self, self.kind) is None:
            raise ValueError(f'training kind={self.kind} requires its task parameters')
        if self.kind in ('dpo', 'rl') and self.data.sequence_lengths:
            raise ValueError('DPO/RL sequence lengths come from the task-specific arrays')
        return self


class WorkloadSpec(Spec):
    scenario: Literal['training', 'inference']
    batch_size: int = Field(default=1, gt=0)
    sequence_length: int = Field(default=2048, gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    training_steps: int = Field(default=1, gt=0)
    training: TrainingTaskSpec | None = None
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
    moe_dispatch: Literal['fp8', 'fp16', 'bf16', 'fp32', 'int8'] | None = None
    moe_combine: Literal['fp8', 'fp16', 'bf16', 'fp32', 'int8'] | None = None
    moe_dispatch_metadata_bytes_per_token: StrictInt = Field(default=0, ge=0,
        description='Explicit extra wire bytes per routed token copy; includes declared scales/indices. No scale format is inferred.')
    moe_combine_metadata_bytes_per_token: StrictInt = Field(default=0, ge=0,
        description='Explicit extra wire bytes per routed token copy; includes declared scales/indices. No scale format is inferred.')
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


class SpeculativeCostSpec(Spec):
    flops: float = Field(default=0, ge=0)
    memory_bytes: float = Field(default=0, ge=0)
    duration_s: float | None = Field(default=None, gt=0)
    startup_s: float = Field(default=0, ge=0)
    compute_dtype: Literal['fp8', 'fp16', 'bf16', 'fp32'] | None = None
    source: Source = Field(default_factory=Source)

    @model_validator(mode='after')
    def supplied_cost(self):
        if self.duration_s is None and self.flops == 0 and self.memory_bytes == 0:
            raise ValueError('speculative cost requires positive duration_s, flops or memory_bytes')
        return self


class SpeculativeDraftSpec(Spec):
    cost: SpeculativeCostSpec
    cost_unit: Literal['per_step', 'per_round'] = 'per_step'
    prefilled: StrictBool = False
    prefill_cost: SpeculativeCostSpec | None = None
    resident_bytes_per_rank: float = Field(gt=0,
        description='Complete draft weights, KV and runtime residency per participating rank for this scenario.')
    workspace_bytes_per_rank: float = Field(default=0, ge=0)

    @model_validator(mode='after')
    def explicit_prefill_boundary(self):
        if self.prefilled and self.prefill_cost is not None:
            raise ValueError('draft.prefilled and draft.prefill_cost are mutually exclusive')
        return self


class SpeculativeDecodeSpec(Spec):
    enabled: StrictBool = False
    method: Literal['dspark', 'generic'] = 'dspark'
    draft_tokens: int = Field(default=7, ge=1)
    accepted_lengths: list[int] = Field(default_factory=list)
    accepted_length: int | None = Field(default=None, ge=1)
    acceptance_source: Source = Field(default_factory=Source)
    draft: SpeculativeDraftSpec | None = None
    commit_cost: SpeculativeCostSpec | None = None
    replay_cost: SpeculativeCostSpec | None = None
    snapshot_bytes_per_rank: float = Field(default=0, ge=0,
        description='Extra temporary snapshot storage beyond automatic target recurrent/compression-state snapshots.')

    @model_validator(mode='after')
    def acceptance_and_draft(self):
        if any(value < 1 or value > self.draft_tokens + 1 for value in self.accepted_lengths):
            raise ValueError('accepted_lengths must contain committed token counts in [1, draft_tokens + 1], including bonus tokens')
        if self.accepted_length is not None and self.accepted_length > self.draft_tokens + 1:
            raise ValueError('accepted_length must not exceed draft_tokens + 1')
        if self.accepted_lengths and self.accepted_length is not None:
            raise ValueError('use accepted_lengths or accepted_length, not both')
        if self.enabled and (self.draft is None or not (self.accepted_lengths or self.accepted_length is not None)):
            raise ValueError('enabled speculative_decode requires draft cost/residency and accepted_lengths or accepted_length')
        if self.enabled and not self.draft.prefilled and self.draft.prefill_cost is None:
            raise ValueError('enabled speculative_decode requires draft.prefill_cost or explicit draft.prefilled=true; draft prompt preparation cannot be free')
        return self


class MegakernelGroupSpec(Spec):
    name: str = Field(min_length=1)
    kernels: list[str] = Field(min_length=2,
        description='Ordered exact metadata.kernel pattern; repeated names are allowed.')
    phases: list[str] = Field(default_factory=list)
    layers: list[int] = Field(default_factory=list)
    ranks: list[str] = Field(default_factory=list)
    fused_startup_s: float = Field(ge=0,
        description='Explicit fused launch startup; no automatic free-launch assumption.')
    saved_hbm_bytes: float = Field(default=0, ge=0)
    workspace_bytes: float = Field(default=0, ge=0)
    duration_s: float | None = Field(default=None, gt=0,
        description='Optional fused solo service time excluding fused_startup_s; physical work is retained.')
    source: Source = Field(default_factory=Source)

    @model_validator(mode='after')
    def selectors(self):
        import re
        if not self.name.strip() or any(not item.strip() for item in self.kernels + self.phases):
            raise ValueError('megakernel group names, kernel names and phases must be nonempty')
        if any(layer < 0 for layer in self.layers):
            raise ValueError('megakernel layer selectors must be nonnegative')
        if any(re.fullmatch(r'npu[0-9]+', rank) is None for rank in self.ranks):
            raise ValueError('megakernel rank selectors must match npu[0-9]+')
        return self


class MegakernelSpec(Spec):
    enabled: StrictBool = False
    groups: list[MegakernelGroupSpec] = Field(default_factory=list)

    @model_validator(mode='after')
    def explicit_groups(self):
        if self.enabled and not self.groups:
            raise ValueError('enabled megakernel requires explicit groups and fused startup costs')
        names = [group.name for group in self.groups]
        if len(names) != len(set(names)):
            raise ValueError('megakernel group names must be unique')
        return self


class PrefillDecodeDisaggregationSpec(Spec):
    enabled: StrictBool = False
    prefill_ranks: list[StrictInt] = Field(default_factory=list,
        description='Physical NPU indices for the prefill pool; empty with decode_ranks derives consecutive disjoint pools.')
    decode_ranks: list[StrictInt] = Field(default_factory=list)
    token_id_bytes: Literal[4, 8] = Field(default=8,
        description='Bytes per transferred first-token ID; no implicit RPC service time is added.')

    @model_validator(mode='after')
    def rank_lists(self):
        ranks = self.prefill_ranks + self.decode_ranks
        if any(rank < 0 for rank in ranks):
            raise ValueError('prefill/decode rank indices must be nonnegative')
        if len(set(ranks)) != len(ranks):
            raise ValueError('prefill/decode rank lists must be unique and disjoint')
        if self.enabled and bool(self.prefill_ranks) != bool(self.decode_ranks):
            raise ValueError('provide both prefill_ranks and decode_ranks or leave both empty for automatic placement')
        return self


class TrainingQoSSpec(Spec):
    enabled: StrictBool = False
    weights: dict[Literal['pp', 'tp', 'ep', 'fsdp_gather', 'fsdp_reduce_scatter', 'dp', 'other'],
                  Annotated[StrictInt, Field(gt=0, le=65535)]] = Field(default_factory=dict,
        description='Relative fabric service weights per transfer. Unspecified classes have weight 1.')


class FeatureSpec(Spec):
    training_qos: TrainingQoSSpec = Field(default_factory=TrainingQoSSpec)
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
    speculative_decode: SpeculativeDecodeSpec = Field(default_factory=SpeculativeDecodeSpec)
    megakernel: MegakernelSpec = Field(default_factory=MegakernelSpec)
    prefill_decode_disaggregation: PrefillDecodeDisaggregationSpec = Field(default_factory=PrefillDecodeDisaggregationSpec)


class TrainingMFUReferenceSpec(Spec):
    name: str = Field(min_length=1)
    sequence_length: StrictInt = Field(gt=0)
    model_flops_per_token: float = Field(gt=0, description='Published forward plus backward FLOPs per input token.')
    dtype: DType


class TrainingMoERoutingSpec(Spec):
    mode: Literal['ragged', 'fixed_capacity']
    sender_capacity_factor: float = Field(default=1.0, ge=1)
    expert_capacity_factor: float = Field(default=1.0, ge=1)
    assignment_drop_fraction: float = Field(default=0.0, ge=0, lt=1)
    routed_width: StrictInt | None = Field(default=None, gt=0,
        description='Wire row width for parameter-budget models. Structured models derive it from the latent projection.')

    @model_validator(mode='after')
    def capacities(self):
        if self.mode == 'ragged' and (self.sender_capacity_factor != 1 or self.expert_capacity_factor != 1):
            raise ValueError('Ragged routing uses accepted rows; capacity factors apply to fixed_capacity')
        return self


class ExecutionSpec(Spec):
    tp: int = Field(default=1, gt=0)
    all_to_all_algorithm: Literal['pairwise', 'direct'] = 'pairwise'
    sequence_parallel: bool = False
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)
    dp: int = Field(default=1, gt=0)
    fsdp: int = Field(default=1, gt=0)
    rank_order: list[int] = Field(default_factory=list)
    reshard_after_forward: bool = True
    sync_each_microbatch: bool = False
    prefetch: bool = False
    prefetch_layers: StrictInt | None = Field(default=None, ge=0,
        description='FSDP lookahead within a stage and pass. When omitted, prefetch=true selects one layer.')
    overlap_gradient_sync: bool = False
    overlap_pipeline_p2p: bool = True
    overlap_attention_tp: bool = False
    training_attention_windows: list[Annotated[StrictInt, Field(gt=0)] | None] | None = Field(default=None,
        description='Per-layer causal window including the current token; null denotes full causal attention.')
    training_active_kv_heads: list[Annotated[StrictInt, Field(gt=0)] | None] | None = Field(default=None,
        description='Active GQA KV heads per layer within the stored KV projection; null uses the stored width.')
    training_moe_routing: TrainingMoERoutingSpec | None = None
    training_mfu_reference: TrainingMFUReferenceSpec | None = None
    flash_attention_blocks: StrictInt = Field(default=0, ge=0,
        description='Causal sequence partitions for the structured training FlashAttention I/O pipeline. Zero uses the component kernel.')
    training_hbm_fraction: float = Field(default=0.90, gt=0, le=1,
        description='Fraction of physical HBM available to all modeled training allocations, including runtime and workspaces.')
    dp_domain_size: StrictInt | None = Field(default=None, gt=0,
        description='Logical DP ranks per gradient-reduction domain, including FSDP shards. Omit for a flat replica all-reduce.')
    precision: PrecisionSpec = Field(default_factory=PrecisionSpec)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)

    @model_serializer(mode='wrap')
    def omit_optional_training_models(self, handler):
        data = handler(self)
        for key in ('training_attention_windows', 'training_active_kv_heads', 'training_moe_routing', 'training_mfu_reference'):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data


class ParallelCandidate(Spec):
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)
    dp: int = Field(default=1, gt=0)
    fsdp: int = Field(default=1, gt=0)
    dp_domain_size: StrictInt | None = Field(default=None, gt=0)
    rank_order: list[int] = Field(default_factory=list)


class KVOffloadSelection(Spec):
    location: Literal['none', 'host', 'remote'] = 'none'
    fraction: float = Field(default=1, ge=0, le=1)
    page_tokens: int = Field(default=16, gt=0)
    page_entry_bytes: int = Field(default=8, gt=0)


class FSDPSelection(Spec):
    degree: int = Field(default=1, gt=0)
    reshard_after_forward: StrictBool = True
    sync_each_microbatch: StrictBool = False
    prefetch: StrictBool = False
    prefetch_layers: StrictInt | None = Field(default=None, ge=0)
    overlap_gradient_sync: StrictBool = False


class ParallelismSelection(Spec):
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)
    dp: int = Field(default=1, gt=0)
    dp_domain_size: StrictInt | None = Field(default=None, gt=0)
    rank_order: list[int] = Field(default_factory=list)


class QuantizationSelection(Spec):
    weight_scale_block: int = Field(default=128, gt=0)
    kv_scale_block: int = Field(default=128, gt=0)
    scale_bytes: int = Field(default=4, gt=0)
    extra_weight_copy_dtype: DType | None = None


class AttentionSelection(Spec):
    attention_by_layer: list[Literal['MLA', 'KDA', 'MHA', 'GQA', 'MQA', 'CSA', 'HCA']] | None = None
    mla: MLASpec | None = None
    kda: KDASpec | None = None
    csa: CSASpec | None = None
    hca: HCASpec | None = None


class FlashAttentionSelection(Spec):
    blocks: StrictInt = Field(ge=0)
    overlap_tp: StrictBool = False


class FeatureSelectionSpec(Spec):
    training_moe_routing: TrainingMoERoutingSpec | None = None
    training_attention_windows: list[Annotated[StrictInt, Field(gt=0)] | None] | None = None
    training_qos: TrainingQoSSpec | None = None
    flash_attention: FlashAttentionSelection | None = None
    training_task: TrainingTaskSpec | None = None
    precision: PrecisionSpec | None = None
    activation_checkpointing: StrictBool | None = None
    optimizer_offload: StrictBool | None = None
    kv_offload: KVOffloadSelection | None = None
    sequence_parallel: StrictBool | None = None
    fsdp: FSDPSelection | None = None
    parallelism: ParallelismSelection | None = None
    optimizer: OptimizerSpec | None = None
    engram: EngramSpec | None = None
    quantization: QuantizationSelection | None = None
    attention: AttentionSelection | None = None
    speculative_decode: SpeculativeDecodeSpec | None = None
    megakernel: MegakernelSpec | None = None
    prefill_decode_disaggregation: PrefillDecodeDisaggregationSpec | None = None
    kernel_profiles: dict[str, list[KernelProfile]] | None = None

    @model_serializer(mode='wrap')
    def omit_optional_training_models(self, handler):
        data = handler(self)
        for key in ('training_moe_routing', 'training_attention_windows'):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data


class TargetSpec(Spec):
    throughput_tokens_s: float | None = Field(default=None, gt=0)
    ttft_s: float | None = Field(default=None, gt=0)
    tpot_s: float | None = Field(default=None, gt=0)
    step_time_s: float | None = Field(default=None, gt=0)
    candidate_npu_counts: list[int] = Field(default_factory=list)
    candidate_su_sizes: list[int] = Field(default_factory=list)
    candidate_dp_domain_sizes: list[StrictInt] = Field(default_factory=list)
    training_global_batch_size: StrictInt | None = Field(default=None, gt=0,
        description='In size-su training searches, keep this many microbatch samples per optimizer step by adjusting accumulation.')
    parallel_candidates: list[ParallelCandidate] = Field(default_factory=list)
    total_npus: int | None = Field(default=None, gt=0)
    max_total_npus: int | None = Field(default=None, gt=0)
    max_candidates: int = Field(default=128, gt=0)

    @model_validator(mode='after')
    def ranges(self):
        if any(x < 1 for x in self.candidate_npu_counts + self.candidate_su_sizes + self.candidate_dp_domain_sizes):
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
    features: FeatureSelectionSpec | None = Field(default=None, exclude=True,
        description='Optional feature input expanded into canonical execution/model/hardware fields; omitted from resolved output.')

    @model_validator(mode='before')
    @classmethod
    def expand_feature_selection(cls, data):
        if isinstance(data, dict) and data.get('features') is not None:
            from .features import resolve_feature_input
            return resolve_feature_input(data)
        return data

    @model_validator(mode='after')
    def consistency(self):
        e, h, m = self.execution, self.hardware, self.model
        if self.workload.training is not None:
            from .training_tasks import validate_training_task
            validate_training_task(self)
        pd = e.features.prefill_decode_disaggregation
        if pd.enabled:
            if self.workload.scenario != 'inference' or m.architecture is None:
                raise ValueError('prefill_decode_disaggregation requires structured inference')
            if e.dp != 1 or e.pp != 1:
                raise ValueError('prefill_decode_disaggregation currently requires DP=PP=1 and identical TP/EP per stage')
            pool_size = e.tp * e.ep
            if h.npu_count != 2 * pool_size:
                raise ValueError('prefill_decode_disaggregation requires hardware.npu_count = 2 * TP * EP; TP/EP describe each stage pool')
            if e.rank_order:
                raise ValueError('prefill_decode_disaggregation uses prefill_ranks/decode_ranks; execution.rank_order must be empty')
            if not pd.prefill_ranks and not pd.decode_ranks:
                pd.prefill_ranks = list(range(pool_size))
                pd.decode_ranks = list(range(pool_size, h.npu_count))
            if len(pd.prefill_ranks) != pool_size or len(pd.decode_ranks) != pool_size:
                raise ValueError('each prefill/decode rank pool must contain exactly TP * EP physical NPUs')
            if sorted(pd.prefill_ranks + pd.decode_ranks) != list(range(h.npu_count)):
                raise ValueError('prefill/decode rank pools must cover every physical NPU exactly once')
            if e.features.speculative_decode.enabled:
                raise ValueError('prefill_decode_disaggregation with speculative_decode is not implemented')
            if e.features.kv_offload != 'none' or e.features.engram.enabled:
                raise ValueError('prefill_decode_disaggregation currently requires kv_offload=none and Engram disabled')
        elif e.tp * e.pp * e.ep * e.dp != h.npu_count:
            raise ValueError('tp * pp * ep * dp must equal hardware.npu_count; fsdp is a DP subgroup')
        if e.sequence_parallel:
            if e.tp == 1:
                raise ValueError('sequence_parallel requires TP > 1; SP shares the TP ranks')
            if self.workload.scenario != 'inference' or m.architecture is None:
                raise ValueError('sequence_parallel currently requires structured inference')
            if e.pp != 1:
                raise ValueError('sequence_parallel currently requires PP=1; SP pipeline forwarding is not modeled')
        if e.dp % e.fsdp:
            raise ValueError('fsdp must divide dp')
        if e.dp_domain_size is not None and (e.dp % e.dp_domain_size or e.dp_domain_size % e.fsdp):
            raise ValueError('dp_domain_size must divide dp and contain whole FSDP groups')
        if e.flash_attention_blocks and (m.architecture is None or not any(
                kind in ('MHA', 'GQA', 'MQA', 'MLA') for kind in m.architecture.attention_by_layer)):
            raise ValueError('flash_attention_blocks requires structured MHA, GQA, MQA or MLA layers')
        if e.overlap_attention_tp and (e.flash_attention_blocks < 2 or e.tp < 2):
            raise ValueError('overlap_attention_tp requires flash_attention_blocks >= 2 and TP > 1')
        reference = e.training_mfu_reference
        if reference is not None:
            task = self.workload.training
            if (self.workload.scenario != 'training' or self.workload.sequence_length != reference.sequence_length
                    or (task and (task.kind != 'pretrain' or task.data.sequence_lengths))):
                raise ValueError('training_mfu_reference requires uniform pretraining at the declared sequence length')
            if reference.dtype != e.precision.compute or reference.dtype not in h.npu.flops:
                raise ValueError('training_mfu_reference requires a matching compute dtype and hardware peak')
        active_kv = e.training_active_kv_heads
        if active_kv is not None:
            if self.workload.scenario != 'training' or not m.architecture or len(active_kv) != m.layers:
                raise ValueError('training_active_kv_heads requires one entry per structured training layer')
            for kind, heads in zip(m.architecture.attention_by_layer, active_kv):
                if heads is not None and (kind != 'GQA' or heads > m.kv_heads or m.num_heads % heads
                        or max(heads, e.tp) % min(heads, e.tp)):
                    raise ValueError('Active GQA KV heads must fit the stored heads and divide query heads with even TP placement')
        windows = e.training_attention_windows
        if windows is not None:
            if self.workload.scenario != 'training' or len(windows) != m.layers:
                raise ValueError('training_attention_windows requires one entry per layer in training')
            if m.architecture and any(window is not None and kind not in ('MHA', 'GQA', 'MQA', 'MLA')
                    for window, kind in zip(windows, m.architecture.attention_by_layer)):
                raise ValueError('A causal training window requires an MHA, GQA, MQA or MLA layer')
        if e.training_moe_routing is not None:
            if self.workload.scenario != 'training' or not m.moe_layer_fraction:
                raise ValueError('training_moe_routing requires a training workload with routed experts')
            if m.architecture and e.training_moe_routing.routed_width is not None:
                if e.training_moe_routing.routed_width != (m.architecture.moe.latent_dim or m.hidden_dim):
                    raise ValueError('routed_width conflicts with the structured MoE projection')
        if self.workload.scenario != 'training' and (e.prefetch or e.prefetch_layers or
                e.flash_attention_blocks or e.overlap_gradient_sync or e.dp_domain_size is not None or
                self.targets.candidate_dp_domain_sizes or self.targets.training_global_batch_size is not None):
            raise ValueError('FSDP lookahead, training overlap and DP domain search require a training workload')
        if self.workload.training and self.workload.training.kind == 'rl' and self.targets.training_global_batch_size:
            raise ValueError('training_global_batch_size describes optimizer steps; RL search uses rollout/update batches')
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
            if m.num_heads % e.tp:
                raise ValueError('Shape-derived attention requires num_heads divisible by TP')
            if 'KDA' in m.architecture.attention_by_layer and m.architecture.kda.num_heads % e.tp:
                raise ValueError('Shape-derived KDA heads must be divisible by TP')
            for kind in ('CSA', 'HCA'):
                if kind not in m.architecture.attention_by_layer:
                    continue
                sparse = getattr(m.architecture, kind.lower())
                if sparse is None:
                    raise ValueError(f'{kind} layers require explicit architecture.{kind.lower()} dimensions')
                if m.architecture.linear_bias or m.architecture.norm_bias:
                    raise ValueError('CSA/HCA currently require linear_bias=false and norm_bias=false')
                if m.num_heads % sparse.o_groups or sparse.o_groups % e.tp:
                    raise ValueError(f'{kind} requires num_heads divisible by o_groups and o_groups divisible by TP')
                if kind == 'CSA' and sparse.index_num_heads % e.tp:
                    raise ValueError('CSA index_num_heads must be divisible by TP')
        if e.pp > m.layers:
            raise ValueError('pp must not exceed model.layers')
        if m.num_experts % e.ep:
            raise ValueError('ep must divide num_experts')
        if e.rank_order and sorted(e.rank_order) != list(range(h.npu_count)):
            raise ValueError('rank_order must be a permutation of physical NPU indices')
        if e.precision.compute not in h.npu.flops:
            raise ValueError('NPU profile must specify FLOPs for the selected compute dtype')
        f = e.features
        if f.megakernel.enabled:
            for group in f.megakernel.groups:
                if any(layer >= m.layers for layer in group.layers):
                    raise ValueError('megakernel layer selectors must be valid model layer indices')
                if any(int(rank[3:]) >= h.npu_count for rank in group.ranks):
                    raise ValueError('megakernel rank selectors must reference existing physical NPUs')
        if m.architecture and any(kind in ('CSA', 'HCA') for kind in m.architecture.attention_by_layer):
            if e.precision.kv != e.precision.compute:
                raise ValueError('CSA/HCA currently require KV dtype equal to compute dtype; mixed-precision sparse cache conversion is not modeled')
            if f.kv_offload != 'none':
                raise ValueError('CSA/HCA currently require kv_offload=none; sparse window/pool offload is not modeled')
        speculative = f.speculative_decode
        if speculative.enabled:
            if self.workload.scenario != 'inference' or m.architecture is None:
                raise ValueError('speculative_decode currently requires structured inference')
            if e.dp != 1 or e.pp != 1:
                raise ValueError('speculative_decode currently requires DP=PP=1')
            if len(set(self.workload.input_lengths)) != 1 or len(set(self.workload.output_lengths)) != 1:
                raise ValueError('speculative_decode currently requires uniform input and output lengths')
            if min(self.workload.output_lengths) < 2:
                raise ValueError('speculative_decode requires output lengths >=2')
            if f.kv_offload != 'none':
                raise ValueError('speculative_decode currently requires kv_offload=none; block verification offload is not modeled')
            for cost in (speculative.draft.cost, speculative.draft.prefill_cost, speculative.commit_cost, speculative.replay_cost):
                if cost is not None and cost.compute_dtype is not None and cost.compute_dtype not in h.npu.flops:
                    raise ValueError('NPU profile must specify FLOPs for speculative cost compute_dtype')
        if f.kv_offload == 'remote' or (f.engram.enabled and f.engram.location == 'remote'):
            if not h.remote.enabled:
                raise ValueError('remote memory feature requires hardware.remote.enabled')
        if any(i < 0 or i >= m.layers for i in f.engram.layers) or len(set(f.engram.layers)) != len(f.engram.layers):
            raise ValueError('engram.layers must be unique valid layer indices')
        if self.workload.scenario == 'inference':
            if f.training_qos.enabled:
                raise ValueError('training_qos requires a training workload')
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


def load_config(path: str | Path, features: str | Path | dict | None = None) -> Config:
    path = Path(path)
    data = read_data(path)
    if data.get('calibration'):
        source = (path.parent / data['calibration']).resolve()
        calibration = read_data(source)
        if set(calibration) - {'hardware', 'source'}:
            raise ValueError('calibration accepts hardware overrides and source only')
        data['hardware'] = deep_merge(data.get('hardware', {}), calibration.get('hardware', {}))
        data.setdefault('sources', {})['hardware.calibration'] = calibration.get('source', {'status': 'user', 'description': 'External calibration overrides', 'uri': str(source)})
    if features is not None:
        from .features import merge_feature_selections
        supplied = read_data(features) if isinstance(features, (str, Path)) else features
        data['features'] = merge_feature_selections(data.get('features'), supplied)
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

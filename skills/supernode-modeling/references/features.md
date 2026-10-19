# 可选 feature 输入、能力清单与结果核对

在场景顶层写 `features`，或将 feature 对象放入独立 JSON/YAML 文件，通过 `--features` 传入。两种输入都展开为已有的 `execution`、`model.architecture`、`hardware` 配置。feature 不附带固定提速倍率；耗时来自实际建模的工作量、内存、通信、调度依赖和显式 profile。

以下命令在仓库根目录运行。示例目录 `examples/features/` 中所有模型尺寸、硬件能力、draft 成本和 acceptance 都是 **synthetic / assumed**，用于说明接口和检查小规模运行，不能引用为 Kimi K3、DeepSeek V4、DSpark 或 950DT 的真实性能。

## 查看可输入选项

```text
python -m supernode_sim list-features
python -m supernode_sim schema -o outputs/features/schema.json
```

`list-features` 返回 `support_status`、`modeling`、适用场景、canonical 路径与限制。`supported` 表示已有仿真模型；`analytical` 表示公式建模；`profile_driven` 表示使用显式 profile 驱动部分耗时。它们均不代表已在目标软件与硬件上验证提速。profile 的 `source.status: measured` 也不能替代实际匹配覆盖率或独立的端到端验证。

| feature 输入 | 展开的主要字段 | 范围与要求 |
| --- | --- | --- |
| `precision` | `execution.precision` | 分别设置 weight、compute、activation、kv、communication 等；不能把存储低精度等同低精度计算 |
| `activation_checkpointing` | `execution.features.activation_checkpointing` | 训练；保存 activation 与重算相权衡 |
| `optimizer_offload` | `execution.features.optimizer_offload` | 训练；CPU 更新及主机链路可能使 step time 增加 |
| `kv_offload: {location, fraction, page_tokens, page_entry_bytes}` | `execution.features.kv_*` | 推理；remote 还要求 `hardware.remote.enabled` |
| `sequence_parallel` | `execution.sequence_parallel` | structured inference、TP > 1、PP = 1；与 TP 共用 ranks |
| `fsdp: {degree, reshard_after_forward, sync_each_microbatch, prefetch}` | `execution.fsdp` 等 | 训练，degree 整除 DP；structured 路径不支持 lookahead prefetch |
| `parallelism: {tp, pp, ep, dp, rank_order}` | `execution` 对应字段 | 保留卡数、专家数、heads 与拓扑约束 |
| `optimizer` | `execution.optimizer` | 训练；Muon 需要显式张量结构，未实现 distributed Muon |
| `engram` | `execution.features.engram` | 推理；cache、hit rate、IOPS、位置与延迟均为显式输入 |
| `quantization` | `execution.features` 中 scale block、scale bytes、extra weight copy | 描述存储元数据；不推断量化质量或硬件 kernel 兼容性 |
| `attention` | `model.architecture.attention_by_layer` 及 `mla/kda/csa/hca` | 修改结构；必须有形状依据，不能称为质量等价的开关 |
| `speculative_decode` | `execution.features.speculative_decode` | `method: dspark` 或 `generic`，必须提供 draft 资源与 acceptance |
| `kernel_profiles` | `hardware.kernel_profiles` | 使用具体 kernel/profile 表达实现差异，并检查实际匹配覆盖 |

独立 `--features` 文件的内容就是上表对象本身，不再嵌套一层 `features:`。内联 feature、独立 feature 文件和显式 canonical 字段若给同一路径设置不同值，将报冲突；相同值允许。选择一个位置管理每个字段，不依赖覆盖顺序。未知 feature、`speculative_decode: true`、`kernel_profiles: true` 等捷径会被拒绝。

已解析的结果配置只保留 canonical 字段，不重复保存激活用的顶层 `features`，便于重跑和按 canonical 路径做 sweep。复制已有完整配置再改变选项时，应修改其 canonical 字段，或先移除与 feature 文件重复的字段。

## 训练：checkpoint 与 optimizer offload

```text
python -m supernode_sim validate examples/features/synthetic-training.yaml
python -m supernode_sim simulate examples/features/synthetic-training.yaml -o outputs/features/training
```

该场景使用一层小型 GQA、Adam 与显式 BF16 字段，开启 activation checkpointing 和 optimizer offload。对照实验应复制场景并把同一份 `features` 内的两个布尔值改为 `false`，保持模型、batch、sequence length、硬件与其他精度不变。比较 `metrics.step_time_s`、NPU/主机内存峰值、重算工作与 H2D/D2H 流量。offload 的目的可以是降低 NPU 驻留；CPU 更新和链路传输可能使训练变慢。

## attention：CSA、HCA、KDA 显式结构

```text
python -m supernode_sim validate examples/features/synthetic-attention-base.yaml --features examples/features/synthetic-attention-features.yaml
python -m supernode_sim simulate examples/features/synthetic-attention-base.yaml --features examples/features/synthetic-attention-features.yaml -o outputs/features/attention
```

`synthetic-attention-base.yaml` 有意将 `attention_by_layer` 留给独立 feature 文件，因此不能单独运行。两个文件合在一起指定三层 `[CSA, HCA, KDA]`。这是一种人为架构假设，不是完整 V4/K3 模型，也不证明与全 attention 基线具有相同质量。

CSA/HCA 必填公共参数：`q_lora_rank`、`head_dim`、`rope_head_dim`、`o_groups`、`o_lora_rank`、`window_size`。CSA 还需 `index_num_heads`、`index_head_dim`、`index_topk`，当前压缩比为 4。HCA 的 `compress_ratio` 默认为 128，必须大于 1 且不为 4；小示例显式选 8 仅为 synthetic 设置。`num_heads` 必须能被 `o_groups` 整除，`o_groups` 及 CSA 的 `index_num_heads` 必须能被 TP 整除。当前 CSA/HCA 组合不接受 linear/norm bias，要求 `execution.precision.kv == execution.precision.compute` 且 `execution.features.kv_offload: none`；尚未实现稀疏池的混合精度转换与 offload。

KDA 通过 `kda.num_heads/head_k_dim/head_v_dim/conv_kernel/gate_low_rank_dim/...` 描述 recurrent 与 convolution 状态。CSA/HCA 的滑窗、压缩池和 FP32 压缩状态与普通每 token KV 不同，应查看组件内存与状态分类，不能仅按一个 KV 压缩倍数处理。

本路径的 attention FLOPs、访问量和训练工作是 analytical 估计，CSA/HCA 不包含 mHC。对比真实模型时使用实际逐层映射和张量形状，并明确模型质量约束；只有结构名称相同仍不足以判定等价。

## DSpark：显式 draft / verify / commit 成本

```text
python -m supernode_sim validate examples/features/synthetic-dspark.yaml
python -m supernode_sim simulate examples/features/synthetic-dspark.yaml -o outputs/features/dspark
```

最小输入使用 `features.speculative_decode`。`enabled: true` 后要求 structured inference、DP = PP = 1、同 batch 内所有 input lengths 相同且 output lengths 相同、output length 至少为 2、batch size 能被 EP 整除、`execution.features.kv_offload: none`。compute dtype 仍需出现在硬件 FLOPs 表中。不要把不同请求的 acceptance 独立序列直接塞入当前同步 batch 接口。

- `draft_tokens` 是每轮草稿位置预算。`accepted_lengths` 是逐轮**最终提交的输出 token 数，包含 bonus/replacement token**，不是“通过验证的 draft token 数”。每项在 1 到 `draft_tokens + 1` 之间；同轮作用于 batch 内每个请求。也可用固定假设 `accepted_length`，两者互斥。
- prefill 已输出第一个 token，acceptance 列表覆盖之后所有轮次。示例 output length = 7，随后 `[2, 1, 3]` 提交六个 token。末轮按剩余输出长度截断；检查输出中的 unused trace entries，而非假定列表每项都使用。
- `draft.resident_bytes_per_rank` 包含该场景完整 draft weights、KV/state、runtime 常驻预算；`workspace_bytes_per_rank` 单独表示临时空间。不要只填 draft 权重大小。
- draft prompt 准备不能免费省略：提供 `draft.prefill_cost`，或明确设 `draft.prefilled: true` 声明 prompt 状态已在计时区间外准备好。两者互斥。示例给出 assumed prefill cost，并将其计入 target prefill 之前；使用 `prefilled: true` 时必须声明 TTFT 不含这部分外部准备。
- `draft.cost` 是每参与 rank 上整个 local batch 的成本。`cost_unit: per_step` 对应一个草稿位置；`per_round` 对应整轮。末轮变短时，整轮 profile 不会自动按比例缩短。
- cost 可给 `flops`、`memory_bytes` 或显式 `duration_s`，至少一种为正；`duration_s` 不含 `startup_s`，且不能小于物理算力/带宽下界。`source` 记录来源，示例全部为 `assumed`。
- `commit_cost`、`replay_cost` 分别是每 rank 每轮的额外接受/提交、拒绝后的状态 replay 成本。含 KDA/CSA/HCA 且存在拒绝轮时必须给 `replay_cost`；仅做普通 KV 截断的路径可不提供。其他允许省略的成本会报告 `not_modeled`，不能把缺失成本描述为真实零成本。`snapshot_bytes_per_rank` 是自动 recurrent/compression 状态快照以外的额外临时空间。

目标 verify 的所有 query 工作量均计入，拒绝 token 也消耗计算。draft 与 target 共用相应 NPU 计算资源和 HBM；显式统计 draft 驻留、candidate KV、快照和回滚。`method: dspark` 选择这个调度/资源模型，不会自动执行 DSpark 的接受率预测、验证输出正确性或复现某个 serving 实现。

查看 `model_metadata.speculative_decode` 的轮次、draft/verify/commit token 数、acceptance 来源及 commit/replay 是否建模。同轮提交 token 共用模型侧完成时间，不能将零轮内间隔误当作逐 token 客户端实测 ITL，也不能直接与普通 decode 的单步耗时混用。

## 融合、图执行与实测 profile

融合算子、CUDA/NPU 图执行或不同运行时 kernel 通过 `features.kernel_profiles` 输入具体成本，格式与 `hardware.kernel_profiles` 相同。需要给出 profile 对应的操作标识、`flops`、`memory_bytes`、`duration_s`，并按需限定 `dtype`、`phase`、`startup_s`。先按 [runtime-calibration.md](runtime-calibration.md) 确认标识匹配与计时口径。

一个局部 profile 只能改变匹配事件的服务时间；不能据此宣称整个运行时的融合、图捕获、调度或编译行为都已复现。prefill、decode、speculative verify 不同工作量不能共享一个未经核对的整网 TPOT。对实际实现声明 `profile_driven` 时，同时报告匹配事件数、未匹配部分、来源条件和保留的物理工作量。

## 结果位置与尚未建模的能力

普通 `simulate` 输出目录包含 `result.json`、`report.md` 和 CSV 明细。检查这些 JSON 路径：

- `config`：展开后的 canonical 场景，适合保存后重新运行；不会回放顶层 feature shorthand。
- `model_metadata.feature_plan`：实际选项、参数、`analytical/profile_driven` 口径及限制。
- `model_metadata.kernel_calibration`：实际 kernel 匹配覆盖率；feature plan 也引用这份覆盖信息。
- `model_metadata.speculative_decode`：启用投机解码时的轮次及未建模成本。
- `metrics`、`memory_breakdown`、`simulation.memory_peaks_bytes`：本次固定负载的耗时、状态容量及运行峰值。

`validate` 的 stdout 直接包含 `feature_plan`，只表示配置已展开和通过校验，不包含执行后的匹配覆盖率。示例是小图，可先 `validate` 再 `simulate`；不要为了展示选项启动大型实验。

能力清单明确列出尚不支持的 continuous batching、动态 prefix cache、chunked-prefill 调度、prefill/decode 分离服务、mHC，以及通用 `flash_attention: true` / `cuda_graphs: true` 提速开关。KV 分页存储预算不等同完整 paged-attention serving 调度。固定 batch 结果也不提供生产排队、负载到达过程或 P99 SLO 保证；仍须按 [runtime-calibration.md](runtime-calibration.md) 对齐真实请求 trace。

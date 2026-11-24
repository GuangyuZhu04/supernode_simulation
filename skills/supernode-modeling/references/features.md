# 可选 feature 输入、能力清单与结果核对

在场景顶层写 `features`，或将 feature 对象放入独立 JSON/YAML 文件，通过 `--features` 传入。两种输入展开为 `execution`、`model.architecture`、`hardware` 或 `workload.training` 配置。feature 不附带固定提速倍率；耗时来自实际建模的工作量、内存、通信、调度依赖和显式 profile。

以下命令在仓库根目录运行。示例目录 `examples/features/` 中所有模型尺寸、硬件能力、draft 成本和 acceptance 都是 **synthetic / assumed**，用于说明接口和检查小规模运行，不能引用为 Kimi K3、DeepSeek V4、DSpark 或 950DT 的真实性能。

## 查看可输入选项

```text
python -m supernode_sim list-features
python -m supernode_sim schema -o outputs/features/schema.json
```

`list-features` 返回 `support_status`、`modeling`、适用场景、canonical 路径与限制。`supported` 表示已有仿真模型；`analytical` 表示公式建模；`profile_driven` 表示使用显式 profile 驱动部分耗时。它们均不代表已在目标软件与硬件上验证提速。profile 的 `source.status: measured` 也不能替代实际匹配覆盖率或独立的端到端验证。

| feature 输入 | 展开的主要字段 | 范围与要求 |
| --- | --- | --- |
| `training_task` | `workload.training` | 预训练/SFT/DPO/PPO/GRPO 的参数和执行阶段；要求 architecture，见 [训练任务](training-tasks.md) |
| `precision` | `execution.precision` | 分别设置 weight、compute、activation、kv、communication 等；不能把存储低精度等同低精度计算 |
| `activation_checkpointing` | `execution.features.activation_checkpointing` | 训练；保存 activation 与重算相权衡 |
| `optimizer_offload` | `execution.features.optimizer_offload` | 训练；CPU 更新及主机链路可能使 step time 增加 |
| `kv_offload: {location, fraction, page_tokens, page_entry_bytes}` | `execution.features.kv_*` | 推理；remote 还要求 `hardware.remote.enabled` |
| `sequence_parallel` | `execution.sequence_parallel` | structured inference、TP > 1、PP = 1；与 TP 共用 ranks |
| `fsdp: {degree, reshard_after_forward, sync_each_microbatch, prefetch_layers, overlap_gradient_sync}` | `execution.fsdp` 等 | 训练，degree 整除 DP。预算和结构化路径支持逐层预取与异步梯度归约。 |
| `training_qos: {enabled, weights}` | `execution.features.training_qos` | 训练共享链路按流量类别加权，同时就绪的 collective 按权重竞争通信槽。配置与对照方法见 [训练](training.md)。 |
| `flash_attention: {blocks, overlap_tp}` | `execution.flash_attention_blocks`, `execution.overlap_attention_tp` | 结构化训练的因果分块与双缓冲。可选的 TP 输出分块归约适用于前向与重算。 |
| `parallelism: {tp, pp, ep, dp, rank_order}` | `execution` 对应字段 | 保留卡数、专家数、heads 与拓扑约束 |
| `optimizer` | `execution.optimizer` | 训练；Muon 需要显式张量结构，未实现 distributed Muon |
| `engram` | `execution.features.engram` | 推理；cache、hit rate、IOPS、位置与延迟均为显式输入 |
| `quantization` | `execution.features` 中 scale block、scale bytes、extra weight copy | 描述存储元数据；不推断量化质量或硬件 kernel 兼容性 |
| `attention` | `model.architecture.attention_by_layer` 及 `mla/kda/csa/hca` | 修改结构；必须有形状依据，不能称为质量等价的开关 |
| `speculative_decode` | `execution.features.speculative_decode` | `method: dspark` 或 `generic`，必须提供 draft 资源与 acceptance |
| `megakernel` | `execution.features.megakernel` | 对精确 kernel 序列做同 rank/layer/phase 的局部融合；启动成本显式输入，工作量与生命周期保守保留 |
| `prefill_decode_disaggregation` | `execution.features.prefill_decode_disaggregation` | 两个独立物理池，相同 TP/EP；通过真实拓扑搬运缓存状态与首 token ID |
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

最小输入使用 `features.speculative_decode`。`enabled: true` 后要求 structured inference、DP = PP = 1、同 batch 内所有 input lengths 相同且 output lengths 相同、output length 至少为 2、`execution.features.kv_offload: none`。batch 不必被 EP 整除，目标状态按 `i % EP` 的实际请求归属分配；标量 draft/commit/replay 预算必须覆盖最大 local batch，并在所有 rank（含空闲 lane）计费。普通 `simulate` 与 `run_streaming` API 使用相同轮次；后者以组件边界执行，支持单列 warm decode（TTFT 为空）。compute dtype 仍需出现在硬件 FLOPs 表中。不要把不同请求的 acceptance 独立序列直接塞入当前同步 batch 接口。

- `draft_tokens` 是每轮草稿位置预算。`accepted_lengths` 是逐轮**最终提交的输出 token 数，包含 bonus/replacement token**，不是“通过验证的 draft token 数”。每项在 1 到 `draft_tokens + 1` 之间；同轮作用于 batch 内每个请求。也可用固定假设 `accepted_length`，两者互斥。
- prefill 已输出第一个 token，acceptance 列表覆盖之后所有轮次。示例 output length = 7，随后 `[2, 1, 3]` 提交六个 token。末轮按剩余输出长度截断；检查输出中的 unused trace entries，而非假定列表每项都使用。
- `draft.resident_bytes_per_rank` 包含该场景完整 draft weights、KV/state、runtime 常驻预算；`workspace_bytes_per_rank` 单独表示临时空间。不要只填 draft 权重大小。
- draft prompt 准备不能免费省略：提供 `draft.prefill_cost`，或明确设 `draft.prefilled: true` 声明 prompt 状态已在计时区间外准备好。两者互斥。示例给出 assumed prefill cost，并将其计入 target prefill 之前；使用 `prefilled: true` 时必须声明 TTFT 不含这部分外部准备。
- `draft.cost` 是每参与 rank 上整个 local batch 的成本。`cost_unit: per_step` 对应一个草稿位置；`per_round` 对应整轮。末轮变短时，整轮 profile 不会自动按比例缩短。
- cost 可给 `flops`、`memory_bytes` 或显式 `duration_s`，至少一种为正；`duration_s` 不含 `startup_s`，且不能小于物理算力/带宽下界。`source` 记录来源，示例全部为 `assumed`。
- `commit_cost`、`replay_cost` 分别是每 rank 每轮的额外接受/提交、拒绝后的状态 replay 成本。含 KDA/CSA/HCA 且存在拒绝轮时必须给 `replay_cost`；仅做普通 KV 截断的路径可不提供。其他允许省略的成本会报告 `not_modeled`，不能把缺失成本描述为真实零成本。`snapshot_bytes_per_rank` 是自动 recurrent/compression 状态快照以外的额外临时空间。

目标 verify 的所有 query 工作量均计入，拒绝 token 也消耗计算。draft 与 target 共用相应 NPU 计算资源和 HBM；显式统计 draft 驻留、candidate KV、快照和回滚。`method: dspark` 选择这个调度/资源模型，不会自动执行 DSpark 的接受率预测、验证输出正确性或复现某个 serving 实现。

查看 `model_metadata.speculative_decode` 的轮次、draft/verify/commit token 数、acceptance 来源及 commit/replay 是否建模。同轮提交 token 共用模型侧完成时间，不能将零轮内间隔误当作逐 token 客户端实测 ITL，也不能直接与普通 decode 的单步耗时混用。

## Megakernel：局部图变换与显式融合成本

```text
python -m supernode_sim validate examples/features/synthetic-megakernel.yaml
python -m supernode_sim simulate examples/features/synthetic-megakernel.yaml -o outputs/features/megakernel
```

`features.megakernel` 使用 `enabled` 和 `groups`。示例匹配一层 GQA 的 `[prenorm, gqa]`，只选择 prefill/decode，并给出 assumed 启动时间、32 字节 HBM 节省和 64 字节 workspace。示例用于观察实际融合组；默认未融合 kernel 的启动时间为零，因此显式加入 1 微秒融合启动成本后也可能更慢。

每个 group 需要唯一 `name`、有序的 `kernels` 精确标识列表和显式 `fused_startup_s`。`phases`、`layers`、`ranks` 可精确限制作用范围，空列表表示不限；rank 使用 `npu0` 这类物理编号。`kernels` 至少包含两个名称，可重复，以匹配连续的同名 kernel。所有条件均与现有事件匹配，不自动按相似名称猜测。

只有同 rank、同 layer、同 phase 的独占串行链才会变为一个融合事件。不能跨 collective、同步分支或外部依赖强行融合，跳过原因与实际覆盖率写入 `model_metadata.megakernel`。未匹配的 group 不能算作已经生效。

- FLOPs 和其他物理资源工作量保留。`saved_hbm_bytes` 是每个匹配组明确给出的 HBM 访问节省，默认为 0；不能仅因图捕获就删除访存。它不得大于原 HBM 访问总量，也不得超过组内真实临时 buffer 可对应的读写上界，越界会报错。
- `workspace_bytes` 表示融合 kernel 的额外临时存储，默认 0；仍应按实际实现补齐。
- 可给 `duration_s` 作为融合后的独占服务时间，**不含** `fused_startup_s`。未给时保留串行子 kernel 服务时间下界，避免凭空让 compute 和 memory 重叠而产生提速。显式时长仍受保留的物理工作量下界约束。
- 当前采用 `conservative_union`：所有成员 allocation 前移到融合开始，release 延后到融合结束。不会删除供后续使用的 saved activation；该策略可能高估内存峰值，应在报告中注明。
- `source` 记录每组假设或实测来源。名称中含 Megakernel/SuperKernel 不代表复现某个后端；构造图、成本来源和真实 kernel 实现仍须分别核对。

融合先于普通 kernel profile 匹配。原成员的 profile 不会被当作融合后 profile 自动继承；使用融合标识 `megakernel:<group name>` 单独校准，并检查报告中的 member-profile 提示和匹配覆盖。不要把原成员耗时再次加到显式融合服务时间上。

## Prefill/decode 分离：两个物理池与状态 handoff

```text
python -m supernode_sim validate examples/features/synthetic-pd.yaml
python -m supernode_sim simulate examples/features/synthetic-pd.yaml -o outputs/features/pd
```

`features.prefill_decode_disaggregation.enabled: true` 启用固定 batch 的分离执行。`hardware.npu_count` 始终是总物理卡数；当前 P/D 使用相同 TP/EP，且 PP = DP = 1，因此要求 **`npu_count = 2 × TP × EP`**。例如 TP = 2、EP = 1 时，需要总计 4 张卡，每池 2 张。启用 feature 不会替用户自动扩大设备数量。

`prefill_ranks` 和 `decode_ranks` 按相同 TP/EP 坐标一一配对，每个列表必须有 TP × EP 个物理编号，互不重叠，并覆盖所有卡。两列表同时为空时，解析阶段生成前后连续两个池并写入 resolved 配置；只给一个列表会报错。此时 `execution.rank_order` 必须为空，以免两套映射含义冲突。

两池各自驻留完整模型的对应分片、runtime 与 workspace，容量不是共用一份权重。P 池完成 prefill 并产生首 token 后，将 decode 所需持久状态和首 token ID 沿真实拓扑发送到 D 池；D 池在 handoff 完成前不能开始 decode。`token_id_bytes` 可为 4 或 8，默认 8，表示显式整数传输布局。

handoff 包括匹配状态布局的 KV、分页/量化元数据、KDA recurrent/conv state、CSA/HCA 的滑窗、压缩与 index 状态，保守计入已分配 padding。它消耗端点 HBM、链路、注入和共享 NIC 资源。源状态直到相应传输完成才释放；接收状态在传输准入时分配，所以传输期间两池的副本都占容量。

TTFT 截止 P 池首 token 的模型侧完成；首个 decode 间隔包含 handoff，因此带宽或延迟变化应影响这段等待及 TPOT。没有额外猜测客户端首 token 传输、RPC framing、分页表重映射或服务排队成本。所有请求只输出一个 token 时不做 handoff，但 D 池模型仍驻留。

当前允许 SP 和不同请求长度；要求 structured inference，并禁止与 speculative decode、KV offload、Engram 同时启用。尚不支持不同 P/D 并行或精度布局、状态重分片、异构设备池、池内独立副本数、动态请求分派或多 batch 在线流水吞吐。把固定 batch 的结果当作服务集群 goodput 会超出模型范围。Megakernel 可同时使用，但只能在各物理 rank 的合法局部链生效。

组合示例使用独立 feature 文件，`ranks: []` 覆盖两个池。不要把单卡示例中的 `ranks: [npu0]` 不加核对地复制过来：该选择器在此 PD 布局只会作用于 P 池。

```text
python -m supernode_sim simulate examples/features/synthetic-pd.yaml --features examples/features/synthetic-megakernel-features.yaml -o outputs/features/pd-megakernel
```

查看 `model_metadata.prefill_decode_disaggregation` 中的 `state_payload_bytes`、`token_id_payload_bytes`、`handoff_payload_bytes`，以及 `handoffs` 内的源/目标、路径、SU/SO 域和逐跳字节量。结合 `prefill_completion_event`、`handoff_completion_event`、`first_decode_completion_event` 核对等待关系。示例将两卡放在不同 SU，并用较慢的 assumed SO 链路使 handoff 可观察；这不是实际设备性能参数。

## DeepInfer：P/D 独立选型与加速特性组合

参考 DeepInfer 官方部署 Q&A 和 benchmark（核对日期 2026-10-21），把以下策略作为 feature 组合与扫描起点。完整来源、可运行示例和逐项支持范围见 [deepinfer.md](deepinfer.md)。这些是选型策略，不是新的顶层 `features` 字段；上面的输入表与 `list-features` 仍是当前引擎可接受的接口。

| 阶段/特性 | 优先比较的配置 | 使用条件 |
| --- | --- | --- |
| P：小并行组 | EP2/4/8；本项目另比较 TP2/SP2、TP4/SP4 + 小 EP | 在 HBM 能容纳权重、KV、activation 和 workspace 时选择较小组；保留 TP1、SP 关闭的官方风格基线 |
| D：Wide-EP | EP64/128，优先在同一 SU 域；TP1、SP 关闭作为基线 | 检查专家整除、每 rank batch、HBM 和 dispatch/combine 通信，不能只按 EP 倍数推算提速 |
| P/D 独立请求 batch | P `bs=1`；D `bs=128,256`，可加 32/64 为延迟对照 | 这是本项目选型起点，需声明每 rank/每池口径；P 的 token chunk budget 与请求 bs 是不同量 |
| 算子与数据路径 | 按组件低精度、局部 megakernel、具体 kernel profile、DSpark draft/verify | 复用现有接口并检查覆盖率；Mega MoE 跨通信融合和置信度调度不等同现有局部融合/固定验证模型 |
| serving 层优化 | EPLB、异步调度、前缀复用与 SWA bounded replay、动态 chunk 和请求均衡 | 纳入方案分析及扩展需求；当前未实现的机制不能作为已启用的仿真开关 |

官方当前生产配置是 P 小 **EP/Attention DP**、D 大 EP/Attention DP，说明中暂不使用 TP/CP。这里的小 SP+EP 保留为用户提出的候选；本引擎 SP 是与 TP 共用 ranks 的 Megatron 式 sequence parallel。不要把官方 Attention DP、context parallel 和本引擎 `execution.dp` 混为同一参数。

现有 PD feature 仍要求相同 TP/EP 与同一固定 batch。小 P、大 D 及不同 batch 先分别调用 `run_streaming(..., phase='prefill'/'decode')`，报告为独立阶段选型；没有跨布局 KV 重分片、请求汇聚和队列模型时，不拼接成端到端 TTFT/TPOT 或在线吞吐。P/D 总资源是各池实例数乘以各自组大小之和，不能沿用同布局示例的 `2 × TP × EP`。

## 图执行与实测 profile

CUDA/NPU 图执行或不同运行时 kernel 的服务时间可以通过 `features.kernel_profiles` 输入，格式与 `hardware.kernel_profiles` 相同。需要给出 profile 对应的操作标识、`flops`、`memory_bytes`、`duration_s`，并按需限定 `dtype`、`phase`、`startup_s`。先按 [runtime-calibration.md](runtime-calibration.md) 确认标识匹配与计时口径。若要改变局部 kernel 边界，使用上面的显式 megakernel group；单个 timing profile 本身不会执行图变换。

一个局部 profile 只能改变匹配事件的服务时间；不能据此宣称整个运行时的图捕获、调度或编译行为都已复现。prefill、decode、speculative verify 不同工作量不能共享一个未经核对的整网 TPOT。对实际实现声明 `profile_driven` 时，同时报告匹配事件数、未匹配部分、来源条件和保留的物理工作量。

## 结果位置与尚未建模的能力

普通 `simulate` 输出目录包含 `result.json`、`report.md` 和 CSV 明细。检查这些 JSON 路径：

- `config`：展开后的 canonical 场景，适合保存后重新运行；不会回放顶层 feature shorthand。
- `model_metadata.feature_plan`：实际选项、参数、`analytical/profile_driven` 口径及限制。
- `model_metadata.kernel_calibration`：实际 kernel 匹配覆盖率；feature plan 也引用这份覆盖信息。
- `model_metadata.speculative_decode`：启用投机解码时的轮次及未建模成本。
- `model_metadata.megakernel`：融合组数、匹配事件、访存节省、启动时间、跳过原因和保守生命周期口径。
- `model_metadata.prefill_decode_disaggregation`：两池卡数、状态/首 token handoff 字节、真实路由与阶段完成事件。
- `metrics`、`memory_breakdown`、`simulation.memory_peaks_bytes`：本次固定负载的耗时、状态容量及运行峰值。

`validate` 的 stdout 直接包含 `feature_plan`，只表示配置已展开和通过校验，不包含执行后的匹配覆盖率。示例是小图，可先 `validate` 再 `simulate`；不要为了展示选项启动大型实验。

能力清单明确列出尚不支持的 continuous batching、动态 prefix cache、chunked-prefill 调度、mHC，以及通用 `flash_attention: true` / `cuda_graphs: true` 提速开关。已有的固定 batch P/D handoff 不包含动态 serving 调度；KV 分页存储预算也不等同完整 paged-attention serving 调度。固定 batch 结果不提供生产排队、负载到达过程或 P99 SLO 保证；仍须按 [runtime-calibration.md](runtime-calibration.md) 对齐真实请求 trace。

机制参考：[CANN SuperKernel](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)说明二进制算子调度融合；[vLLM disaggregated prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill/)说明独立阶段与 KV connector。它们支持机制解释，不提供本仓库的精度或提速保证；本模型保留上述较窄的执行范围。

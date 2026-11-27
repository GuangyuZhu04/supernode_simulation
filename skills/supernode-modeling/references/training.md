# 训练任务、FSDP2 与 optimizer offload

先区分预训练、SFT、DPO、后训练 RL（PPO/GRPO）。任务参数、特定要求、feature 矩阵、算子校准和完整示例见 [training-tasks.md](training-tasks.md)。使用 `workload.training.kind` 或 `features.training_task` 激活任务路径，要求 `model.architecture`。省略该字段保留原训练模型，不能据此称已模拟 SFT 监督掩码或 RL 生成/奖励/更新闭环。

从 `examples/demo_training.yaml` 开始做小规模完整仿真；`examples/training_10t_fp8.yaml` 与 `examples/training_10t_fp16.yaml` 是 10T 合成容量场景，不是公开真实模型。更改配置后重新 `validate`。

## 参数和执行策略

- `workload.scenario: training`；`batch_size` 是每 DP 副本、每个 microbatch 的序列数。
- 普通固定长度训练每步 tokens 为 `batch_size * sequence_length * gradient_accumulation * dp`。任务路径按有效长度、padding、DPO 双序列或 RL 实际更新次数分别统计；RL 的 `training_steps` 表示整轮迭代，不是 optimizer 步数。
- 每个外层 DP 副本先把完整训练序列分给 Attention 副本，再在副本内按 CP 分配上下文位置。CP=1 且 microbatch 只有一条序列时，一个 Attention 副本承担该序列的全部计算。专家工作量另按路由到各专家副本的 token 及 EP 分片计算。[并行组与 rank 放置](configuration.md#attention-和专家并行组)
- 单独填写 `execution.fsdp` 时，其数值是外层 DP 子组的分片度。使用 `execution.fsdp_domain` 可单独指定稠密参数和路由专家的物理分片组。`reshard_after_forward` 控制前向后是否保留完整权重，`sync_each_microbatch` 控制梯度同步频率。预算与结构化训练都支持 `prefetch_layers`。取 0 时按需 gather，取 N 时最多提前 N 层。省略该字段时，旧配置 `prefetch: true` 对应提前一层。
- `execution.features.optimizer_offload` 将主权重和所选优化器状态放在主机，生成梯度 D2H、CPU 更新和权重 H2D；`activation_checkpointing` 用重计算换 activation 驻留量。`execution.optimizer.kind: muon` 使用矩阵 momentum 与辅助 Adam 参数组，详情见 [memory-breakdown.md](memory-breakdown.md)。
- PP 调度为非交错 1F1B。MoE 使用均匀路由和均衡专家计算；不能用合成结果证明热点专家场景可行。

训练 SP 尚未实现。反向组件默认使用前向 FLOPs 的 2 倍近似。可为 forward、backward、recompute 分别绑定 operator trace，描述 dgrad/wgrad 的形状、启动、访存和 workspace。[子算子校准](operator-traces.md)

预算训练的标准 attention 使用每序列 `S*(S+1)/2` 个因果 query-key 对和实际 `num_heads*head_dim`，与推理 prefill 保持相同口径。该修正不改变 checkpoint 重算次数，也不取消 backward 的现有近似。

并行模式影响不同张量：Dense 权重会在 EP 间复制，expert 权重按 EP 分片；FSDP 并非把所有状态除以总卡数。实际 HBM 峰值还包括 gathered 权重、通信缓冲、activation、预取窗口和 workspace。

## 无流水线的 CP、EP 与 FSDP 训练

8 张卡可以组成一个流水段并处理一份 batch。下面的配置将每条序列的上下文切为两份，将路由专家分给 4 个 EP 位置。每套 EP 分片有两份副本，可将同一专家的训练状态继续分片。[完整训练算例](../../../examples/training_cp_ep_fsdp.yaml)

```yaml
execution:
  pp: 1
  dp: 1
  tp: 1
  cp: 2
  attention_dp: 4
  ep: 4
  expert_tp: 1
  expert_dp: 2
  fsdp_domain:
    implementation: fsdp2
    dense_degree: 4
    expert_groups: [[0, 4], [1, 5], [2, 6], [3, 7]]
```

| 参数 | 分片成员 | 反向同步 |
|---|---|---|
| 稠密层、router 和共享专家 | `[0,1,2,3]`、`[4,5,6,7]` | 组内 reduce-scatter，再由两组中持有同一分片的卡成对 all-reduce |
| 路由专家 | `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]` | 每对保存同一 EP 位置的专家，组内 reduce-scatter 完成同步 |

两类参数均在计算前按自身分片组 all-gather。权重、梯度、主权重和优化器状态按实际本地分片计量。在这个例子中 `DP=1`，`batch_size=4`，梯度累积 2 次，每步处理 8 条序列。FSDP 复用现有 rank，卡数和全局 batch 单独由计算布局与负载决定。

- `implementation: fsdp` 按层、参数归属与权重精度将同类张量展平成一个缓冲区后分片，末尾计入等长通信所需的填充。`implementation: fsdp2` 按每个 TP/EP 本地参数的第 0 维分片，常驻状态按实际拥有的行计量，集合通信按组内最大行数补齐。同一层内兼容的张量合并通信。跨层绑定的 embedding 单独成组并共享存储。此处模拟参数所有权、载荷及生命周期。[PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- `dense_groups`、`expert_groups` 可分别填写完整物理 NPU 编号列表。每类都须恰好覆盖所有卡，组内下标决定分片编号。稠密组固定流水段和 Attention TP 位置，专家组固定流水段、EP 位置和专家 TP 位置。`rank_order`、`expert_rank_order` 生效后再校验物理成员。
- 省略成员时按 `dense_degree`、`expert_degree` 从同一参数分片的副本中自动组网。显式成员可推导分片度。两者均省略时，优先用大于 1 的 `execution.fsdp` 作为公共分片度，否则使用所有兼容副本。显式指定某类度数为 1 时，该类参数完整驻留并按其副本同步梯度。`dp_domain_size` 限定外层梯度归约域，配置时每个 FSDP 分片组必须完整落在其中。
- 仅提供总参数预算时，分片采用均衡元素量估计。缺失 TP 本地整数形状的张量也使用这一估计。具有显式形状的批量专家权重按库存中的 `[本地专家数, …]` 切第 0 维。实际运行时的通信带宽、分桶边界和内存分配仍需按目标框架校准。

```powershell
python -m supernode_sim validate examples/training_cp_ep_fsdp.yaml
python -m supernode_sim simulate examples/training_cp_ep_fsdp.yaml -o outputs/training-cp-ep-fsdp
```

`validate` 返回 `fsdp_domains`。训练结果的 `training_summary.json` 同时列出实际分片组、对应分片的副本组，以及按参数类别和阶段汇总的跨卡发送字节 `fsdp_sent_bytes`。

## 通信重叠、容量与排布

从仓库根目录运行合成案例，单独比较执行策略和排布。

```powershell
python -m supernode_sim simulate examples/training_overlap_qos.yaml -o outputs/training-overlap
python -m supernode_sim size-su examples/training_overlap_qos.yaml --mode multi -o outputs/training-placement
```

| 配置 | 执行与结果口径 |
|---|---|
| `execution.prefetch_layers` | 前向按层序、反向按逆序启动 FSDP all-gather，计算等待本层权重。每次 gather 仍占用链路、HBM 端口和完整权重缓冲。窗口限定在一个 PP stage 的一次 pass 内。 |
| `execution.overlap_gradient_sync` | 开启后，本层梯度就绪即可归约，下一层反传可以继续。每个 stage 按层提交同步，缓冲保留至归约完成，每个 backward 微批结束前排空。 |
| `execution.overlap_pipeline_p2p` | 默认开启。接收 stage 等待对应微批的发送完成，发送 stage 可继续执行已就绪的工作。关闭可对比同步发送。 |
| `execution.flash_attention_blocks` | 0 使用原组件。正整数指定每条序列最多划分的块数。MHA/GQA/MQA 和展开 K/V 的 MLA 分别建立前向、反向、重算的投影、因果 tile 搬入、计算和写回事件。双缓冲允许下一块 HBM 读取与当前块计算重叠。稀疏与循环注意力继续使用各自的组件模型。 |
| `execution.training_hbm_fraction` | 默认 0.90，可设 0.85。每张卡的物理容量乘以该比例得到准入上限。常驻态、runtime、workspace、预取、未分片梯度和临时通信缓冲共享这个上限。设为 1.0 可复跑旧容量假设。CPU DRAM 采用自身容量。 |
| `execution.overlap_attention_tp` | 当 TP 和块数都大于 1 时，可将前向及重算的输出投影按已完成的 query 块送入 TP all-reduce，同时继续计算后面的块。每块都承担通信启动成本，局部输出缓冲保留至本块归约结束。反向等待完整 dKV 累积后归约。 |
| `execution.dp_domain_size` | 每个逻辑域内的 DP rank 数，需整除 DP 并容纳完整 FSDP 组。对同一 FSDP 分片，先做域内 reduce-scatter，再按对应分片跨域 all-reduce，最后域内 all-gather。单域或单 lane 情形退化为一次 all-reduce。 |
| `targets.candidate_dp_domain_sizes` | `size-su` 比较候选域大小、模型并行优先和 DP 域连续的 rank 排布。省略时比较平坦归约与从 SU 推导的域大小。显式的 `rank_order` 固定物理排布。实际跨 SU 程度见 `dp_domains`。 |
| `targets.training_global_batch_size` | 普通训练搜索按 `global_batch / (microbatch × DP)` 派生累积步数，不能整除的候选保留诊断。DPO 按偏好对计 batch。每次搜索的 `highest_throughput.candidate_index` 指向可行候选中集群吞吐最高的配置，`best` 保留最小 SU 的目标。 |

FlashAttention 使用真实序列长度计算每块的有效因果对数，按访问次数计 HBM 流量，分别核算 KV 头分片和复制。块数表示聚合的解析粒度，片上缓冲容量、硬件实际 tile、流水指令和内部启动开销需通过平台 profile 校准。反向的 FLOPs 沿用现有倍数，dKV 的分块累积会产生读改写。拆分后的 kernel 使用独立名字，可以按阶段和块形状校准。

调度依赖参考 TorchTitan 仿真器的计算与通信执行槽，阶段 FLOPs 与步时口径沿用项目对 SimuMax 的映射。FSDP 多层预取参考 FSDP2 的显式预取语义，attention 搬运和计算重叠参考 FlashAttention 的分块 I/O。[已有仿真器源码与版本](zhanlu-training.md) · [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) · [FlashAttention-3](https://arxiv.org/abs/2407.08608)

## 训练流量优先级

```yaml
features:
  training_qos:
    enabled: true
    weights: {pp: 4, tp: 2, ep: 2, fsdp_gather: 2, fsdp_reduce_scatter: 1, dp: 1}
```

默认关闭，未指定类别权重为 1。权重分配到每条活跃传输。在共享 SU/SO 链路、注入端口、NIC 或共享上联上，同等条件下权重 3 和 1 的两条流按 3∶1 分享带宽。同时争用同一 collective 收发槽时，高权重先准入，已占用的槽执行至本次传输结束。计算、HBM、链路启动时延和每条消息的字节数仍按原硬件参数计费。该模型描述算子级仲裁，设备 DSCP/PCP、PFC/ECN、硬件队列和包级抢占需另行建模。

固定模型、全局 batch、排布、预取深度和 HBM 水位，分别运行 QoS 开关。比较 step time、峰值 HBM 和 `metrics.training_communication.by_class` 中的 payload、逐跳字节和事件槽位等待。通信与同卡计算的重叠率先对每张卡的传输区间取并集，再与同卡计算区间求交，分母为所有卡的通信活跃 rank·s。该比例包含事件启动时间，不能直接换算为步时长节省。硬件拟合用同一软件栈的独立 step trace、逐类通信和逐卡 HBM 峰值验证。

## 10T FP8/FP16 交叉检查

令参数数目为 `1e13`，以下为十进制 TB 的**全模型逻辑容量**。采用 FP16 梯度、FP32 master 与 FP32 Adam m/v：

| 组成 | FP8 权重 | FP16 权重 |
|---|---:|---:|
| 权重 | 10 | 20 |
| 梯度 | 20 | 20 |
| master | 40 | 40 |
| Adam m/v | 80 | 80 |
| 合计 | 150 | 160 |
| offload 到 CPU 的 master 与 m/v | 120 | 120 |

这些锚点不含 scale、额外高精度副本、activation、通信缓冲及运行时开销。不能直接当作集群物理总容量或某设备峰值；DP/EP 复制和不完全分片可能增加实际值。

所有参数每优化步更新、梯度传输为 FP16 时，梯度 D2H payload 为 20 TB；`weight_transfer: fp8` 的 H2D 数值 payload 为 10 TB，`weight_transfer: fp16` 为 20 TB。双向合计分别为 30/40 TB，**不含 FP8 元数据与其他 collective**。某权重驻留为 FP8、回传 FP16 时仍使用 20 TB 的回传锚点，并单独计转换事件。

训练权重 FP8 scale 在当前参数预算模型中使用连续块数量近似；真实张量分块、对齐和更新实现需要校准。用 `extra_weight_copy_dtype` 表示明确需要的额外驻留副本，不默认假设所有 FP8 训练都没有高精度副本。

## 结果解释

完整训练新增 `metrics.training_accounting`：分别记录模型 F+B、重计算、optimizer 和辅助工作，按 dtype 使用裸峰值归一。`model_mfu` 排除重算与辅助工作，`executed_compute_utilization` 包含 NPU 所有计算，CPU 独立核算；二者不是实测 busy fraction。SimuMax-Zhanlu 特定 revision 的同名 MFU 包含重算，比较前必须对齐分子，详见 [训练参考](zhanlu-training.md)。

`step_time_s` 是总时间除以训练步数（RL 为迭代数）；`throughput_tokens_s` 为整个集群 actor 处理 tokens/s，另报每副本值。有效/监督/生成 token、优化步和阶段时间见 `metrics.training_task` 与 `training_task_stages.csv`；冻结评分和 rollout 不进入训练 MFU 分子。搜索若改变 DP，会改变整体 batch 和每步 tokens，必须同时展示这一变化。

先核对逻辑锚点和每个 CPU/NPU 内存池，再检查 D2H/H2D 的方向性、主机内存带宽、CPU Adam 计算、FSDP all-gather/reduce-scatter 和 SO 跨域路径。`payload / 可用窗口` 是必要带宽下界；双向链路不能把合计字节量机械地除以单向带宽当成实际时间，重叠与依赖由事件仿真决定。

只有总参数预算的旧模型按层/专家近似分布；配置 `model.architecture` 后，参数、优化器状态和注意力类型按明确张量形状生成。activation 检查点、梯度同步和预取行为以图中的事件及 `model_metadata.assumptions` 为准。验证真实平台预测时，使用独立于校准样本的测试配置报告误差。

10T 表仍是 Adam 锚点，不能直接用于 Muon。Kimi K3 / Ascend 950 的真实维度示例与生产训练差异见 [ascend950-kimi-k3.md](ascend950-kimi-k3.md)。CPU optimizer offload 不保证 HBM 可容纳权重、梯度和工作集；不可行案例保留失败结果。

## Marin 训练校准

704 张 GB200 分布在 11 个 EP64 域，每步 46,137,344 个 token。公开训练先在 4K 上下文切换 EP 实现，后期才扩展到 16K 和 65K。比较阶段时，记录软件版本、上下文、有效 batch、BF16 单卡稠密峰值与 MFU 的分子。Marin 日志对局部和全局层使用不同的注意力跨度与 KV 头数，修正了 Latent MoE 的矩阵宽度，并按三倍前向工作估计前反向 FLOPs。[训练记录](https://github.com/marin-community/marin/issues/8435) · [训练代码](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe_hero_ep/train.py)

| 配置 | 对训练仿真的作用 |
| --- | --- |
| `execution.training_moe_routing` | `fixed_capacity` 分别用 `sender_capacity_factor` 和 `expert_capacity_factor` 控制两向传输的行数和专家执行的行数。`ragged` 按接收的有效行紧凑传输和执行。 |
| `assignment_drop_fraction` | 显式声明丢弃的 token–expert 分配比例。共享专家、路由器、原始 token 数维持全量。预算模型可用 `routed_width` 声明传输宽度，结构化模型从 latent 投影导出。 |
| `execution.training_attention_windows` | 每层填窗口宽度（含当前 token）或 `null`（全因果）。前向、反向、重计算与 FlashAttention 分块共享同一组文档边界和可见位置。 |
| `execution.training_active_kv_heads` | 结构化 GQA 每层实际运行的 KV 头数。可按层切片 K/V 投影及激活，同时按存储宽度保留参数、梯度缓冲和优化器状态。 |
| `execution.training_mfu_reference` | 在定长预训练中填 `name`、`dtype`、`sequence_length` 和公开的前反向 `model_flops_per_token`。报告同时输出外部口径 MFU、原生有效模型 MFU、全部执行计算的峰值归一比例。 |

固定容量按均衡路由的期望值计费，发送容量和专家容量各自乘以请求分配数。ragged 默认保留全部分配，有丢弃时假设在发送前完成筛选。两种模式都沿用拓扑上的实际传输和竞争。容量概算将多波次合计，逐波启动、整数对齐、接收端负载偏斜及 XLA 具体调度需要算子或集合通信 trace。专家填充列覆盖前向、反向及重计算，丢弃分配的前反向工作另列为未执行 FLOPs。上述 EP 与滑窗配置也支持顶层 `features.training_moe_routing` 和 `features.training_attention_windows`。

在仓库根目录运行下面的命令。JSON 保存 704 卡形状下的两组 MFU 对照，以及 16 卡、4 层的完整调度结果。每个切片目录另有显存、关键路径、通信暴露和空泡的原生报告。默认物理模型采用有效因果位置对，窗口长度随文档边界截断。更换模型、设备或精度时应使用对应的日志计数器和算子测量。[实验结果与复现说明](../../../docs/examples/marin-training/README.md)

```bash
python -m examples.marin_training --output outputs/marin/comparison.json --slice-reports outputs/marin --full-config outputs/marin/704-fixed.json
```


## 跨规模训练验证

从 64 卡校准点预测 128、256 卡或跨超节点的训练时，在同一份模型、精度、优化器、设备速率、kernel/collective profile 和软件环境上重跑各规模的事件图。每次扩卡后检查各自实际命中的 profile 数、守卫条件和计时证据类型。配置的 `sources` 记录参数声明的来源。报告的 `coverage`、`fidelity` 记录本次运行实际使用了哪些计时依据。迁移到 Ascend、新 GPU 或不同软件栈时，建立对应的新基线。不同 MoE 模型可共用行数和字节公式，latent 宽度、专家数、top-k、线上精度及元数据取目标模型的值。

```bash
python -m supernode_sim training-scaling examples/training_scaling.yaml -o outputs/training-scaling
```

清单的 `config` 相对于清单所在目录解析。命名一个 `baseline` 和一个或多个 `holdouts`。每个点的 `changes` 使用点分路径，可以调整 TP、PP、EP、DP、FSDP、DP 域、rank 排布、物理拓扑、microbatch 和梯度累积。每个点都从相同基线配置独立生成。强扩展固定每步全局处理 token 数，弱扩展固定每卡每步处理 token 数。吞吐扩展效率等于吞吐倍率除以设备数倍率。跨节点后，应同时检查 DP 的 SO 通信和 EP 的局部收益。

在对应规模的条目中填写可选 `observation`。实测来源应标明运行 ID、设备/软件版本和采样区间。下列数值仅演示字段，用 `synthetic` 标记。录入真机观测时填写 `kind: measured`，只填已有且边界相同的指标。可录入的四个字段已全部列在示例中。

```yaml
observation:
  kind: synthetic
  source: example-schema-only
  step_time_s: 0.080
  steady_step_time_s: 0.075
  exposed_comm_s_per_rank_per_step: 0.012
  peak_hbm_bytes: 21474836480
```

| 指标 | 用途 | 采样口径 |
| --- | --- | --- |
| `step_time_s` / `steady_step_time_s` | 可录入实测 | 前者为完整模拟窗口的平均步时。后者跳过第一步，至少需要两个完整训练步。只有稳态日志时填后者。 |
| `ep_sent_bytes_per_rank_per_step_by_phase` | 仿真诊断 | 跨 rank 的端点发送 payload，含显式配置的 metadata。分开列前向、反向、重计算的 dispatch 和 combine，分母为全部配置卡数 × 步数。多跳链路负载另见单次运行的 `training_communication.by_class`。 |
| `exposed_comm_s_per_rank_per_step` | 可录入实测 | 同卡端点传输区间去重，扣除同卡并行计算，横跨全体配置卡及完整窗口取每卡每步平均。包括主机卸载。 |
| `critical_path_comm_service_s_per_step` | 仿真诊断 | 平均训练步关键路径上的直接通信服务时长。队列等待、资源争用等另列在分段图。 |
| `peak_hbm_bytes` | 可录入实测 | 完整窗口中分配占用最高的单卡瞬时峰值，取这一时刻的各类分配。对比实测时对齐 allocator、runtime 常驻和缓存计数口径。 |

结果索引 `training_scaling.json` 给出各点预测、观测来源、逐项有符号误差和相对误差，以正值表示高估。实测为零时保留绝对量差值。基线若有平均步时实测，另列条件预测 `baseline_adjusted_step_time_s = 基线实测步时 × 目标模拟步时 / 基线模拟步时`，原始预测仍可单独检验。未附观测的点标记为 `prediction_only`，附合成观测的点标记为 `synthetic_reference`。两者均不计入已比较的实测留出点数。重复观测保留在各行，已比较的留出配置数按解析配置哈希去重。显存不可行或强弱扩展工作量不匹配时保留诊断。通过各点的 `summary_file` 和 `chart_file` 可查看 HBM、关键路径、通信暴露、空泡及算子占比。[64/128/256 卡合成示例结果](../../../docs/examples/training-scaling/training_scaling.json) · [128 卡分段图](../../../docs/examples/training-scaling/case_001/training_breakdown.svg)

### MoE 小批量下的专家行对齐

对确实将 grouped GEMM 输入逐专家补齐并物化中间激活的实现，可设 `execution.training_moe_routing.expert_row_multiple`。例如 28 个有效分配均衡送到 8 个专家，有 4 个专家收 3 行、4 个收 4 行。若要求 8 行对齐，则专家端执行 64 行，ragged dispatch 仍按 28 行发送（含本地份额）。固定容量路径先应用 `expert_capacity_factor`，再对齐本地执行行。每次前向、反向和重计算按各自 microbatch 计算，填充计入执行 FLOPs 和相应激活工作区。有效模型 FLOPs 继续按接受的分配计数。空专家保留零行。

这是均衡路由假设下的估计。需要从目标 kernel 的 grouped GEMM 描述或 tensor shape 核实实际物化的行对齐，单凭底层指令 tile 大小无法推断分配的 HBM。仅观察到 MFU 时保留默认连续行模型。路由倾斜仍需逐专家 token 直方图或逐 peer 流量验证。

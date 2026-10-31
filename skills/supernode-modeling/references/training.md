# 训练、FSDP2 与 optimizer offload

从 `examples/demo_training.yaml` 开始做小规模完整仿真；`examples/training_10t_fp8.yaml` 与 `examples/training_10t_fp16.yaml` 是 10T 合成容量场景，不是公开真实模型。更改配置后重新 `validate`。

## 参数和执行策略

- `workload.scenario: training`；`batch_size` 是每 DP 副本、每个 microbatch 的序列数。
- `sequence_length`, `gradient_accumulation`, `training_steps` 决定每步/整次工作量。每步训练 tokens 为 `batch_size * sequence_length * gradient_accumulation * dp`。
- `execution.fsdp` 是 DP 子组大小；`reshard_after_forward` 控制前向后是否保留 gathered 权重；`sync_each_microbatch` 控制梯度同步频率；`prefetch` 表示有限的下一层预取。
- `execution.features.optimizer_offload` 将主权重和所选优化器状态放在主机，生成梯度 D2H、CPU 更新和权重 H2D；`activation_checkpointing` 用重计算换 activation 驻留量。`execution.optimizer.kind: muon` 使用矩阵 momentum 与辅助 Adam 参数组，详情见 [memory-breakdown.md](memory-breakdown.md)。
- PP 调度为非交错 1F1B。MoE 使用均匀路由和均衡专家计算；不能用合成结果证明热点专家场景可行。

结构化训练当前不支持 `prefetch`，训练 SP 也未实现；不要将预算路径的预取或推理 SP 能力套入结构化训练。反向组件默认仍采用前向计算量的 2 倍近似。需要细分时使用 [operator trace](operator-traces.md)，分别绑定 forward/backward/recompute，显式描述 dgrad/wgrad 等子算子、矩阵形状、访存、启动与局部 workspace，再提供相应实测 profile。拆分保持原依赖，梯度同步/优化器不能提前启动；原预算守恒不证明真实 autograd 已完整复现。

预算训练的标准 attention 使用每序列 `S*(S+1)/2` 个因果 query-key 对和实际 `num_heads*head_dim`，与推理 prefill 保持相同口径。该修正不改变 checkpoint 重算次数，也不取消 backward 的现有近似。

并行模式影响不同张量：Dense 权重会在 EP 间复制，expert 权重按 EP 分片；FSDP 并非把所有状态除以总卡数。实际 HBM 峰值还包括 gathered 权重、通信缓冲、activation、预取窗口和 workspace。

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

`step_time_s` 是总时间除以训练步数；`throughput_tokens_s` 为整个集群训练 tokens/s，另报每副本值。搜索若改变 DP，会改变整体 batch 和每步 tokens，必须同时展示这一变化。

先核对逻辑锚点和每个 CPU/NPU 内存池，再检查 D2H/H2D 的方向性、主机内存带宽、CPU Adam 计算、FSDP all-gather/reduce-scatter 和 SO 跨域路径。`payload / 可用窗口` 是必要带宽下界；双向链路不能把合计字节量机械地除以单向带宽当成实际时间，重叠与依赖由事件仿真决定。

只有总参数预算的旧模型按层/专家近似分布；配置 `model.architecture` 后，参数、优化器状态和注意力类型按明确张量形状生成。activation 检查点、梯度同步和预取行为以图中的事件及 `model_metadata.assumptions` 为准。验证真实平台预测时，使用独立于校准样本的测试配置报告误差。

10T 表仍是 Adam 锚点，不能直接用于 Muon。Kimi K3 / Ascend 950 的真实维度示例与生产训练差异见 [ascend950-kimi-k3.md](ascend950-kimi-k3.md)。CPU optimizer offload 不保证 HBM 可容纳权重、梯度和工作集；不可行案例保留失败结果。

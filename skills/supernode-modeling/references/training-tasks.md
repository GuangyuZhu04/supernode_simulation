# 训练任务分类与算子级参数

任务入口是 `workload.training`，也可用顶层 `features.training_task` / `--features` 输入同一对象。保持 `workload.scenario: training`；`kind` 为 `pretrain/sft/dpo/rl`，RL 通过 `rl.algorithm` 选择 `grpo/ppo`。要求显式 `model.architecture`。省略任务字段时保留原训练路径，不自动加入 loss 或 RL 阶段。

## 任务选择与特定要求

| 任务 | 必须区分的负载 | 实际执行 | 主要指标 |
|---|---|---|---|
| 预训练 / 继续预训练 | 数据预算、有效长度、packing、全参状态 | F+B、全位置交叉熵、重算、梯度同步、optimizer | 有效 tokens/s、平均/稳态步时、MFU、数据预算外推 |
| SFT / 指令微调 | 上下文与回复、多轮助手标签、padding | 完整上下文 F+B；mask 不剪掉主干和输出 GEMM | 处理/有效/监督 tokens/s、activation 峰值 |
| DPO / 离线偏好优化 | chosen/rejected 成对负载、冻结或预计算 reference | 双序列 actor F+B、可选 reference 前向、pairwise loss | pairs 数、参考成本、actor 步时、双模型驻留 |
| GRPO / 后训练 RL | 每提示多样本、变长生成、组相对优势、更新复用 | rollout→old-log-prob（可选）→reference（可选）→reward→group advantage→actor update | 迭代时间、生成与更新 token、经验缓冲、各阶段占比 |
| PPO / RLHF | actor/reference/reward/critic 独立角色、GAE、value loss | 上述生成/评分流程，加 critic value 前向和 critic 更新 | actor/critic 优化步、价值阶段成本、全部角色驻留 |

这是资源执行代理，不运行真实策略学习或预测质量、收敛。DPO 没有 online rollout；PPO/GRPO 不能用普通训练步乘一个倍率代替。任务未指定时根据数据和训练目标判别；仍无法区分时先完成通用参数收集，再询问会改变执行流程的任务信息。

## 通用参数与 token 口径

下文 `training.*` 都指 `workload.training.*`。

| 参数 | 定义与约束 | 执行影响 |
|---|---|---|
| `workload.batch_size` | 每 DP 副本每 microbatch；DPO 为偏好对，其他为序列 | GEMM 行数、activation、通信；RL 为更新 batch，与 rollout 分开 |
| `workload.gradient_accumulation` | 每优化步 microbatch 数 | 前后向与梯度同步边界 |
| `workload.training_steps` | 普通任务为 actor 优化步；RL 为迭代数 | 重复完整流程，不能将 update epoch 当 rollout 次数 |
| `workload.sequence_length` | 正整数长度上限 | padded 的计算长度；RL 必须覆盖 prompt+response |
| `training.data.batching` | `padded`（默认）/`packed` | padded 计算固定长度；packed 去除 padding 并保留独立序列边界 |
| `training.data.sequence_lengths` | 仅预训练/SFT；每 microbatch 的有效长度数组，数量为 batch；缺省采用上限 | 有效 token 统计及 packed 的 GEMM、attention、activation 和重算形状 |

`packed` 表示已去 padding 的独立因果片段，不模拟离线装箱算法。Attention 使用 `Σ Sᵢ(Sᵢ+1)/2`，不能用拼接后总长度的平方。各 DP 副本重放相同长度模板，尚不描述跨副本长度不均衡。

例：batch=2，上限16，有效长度 `[8,14]`，助手标签 `[3,5]`。每 microbatch，padded 处理32 token，packed 处理22；二者有效 token=22、监督 token=8。再乘 accumulation、DP 和优化步数。多轮 SFT 要计所有参加 loss 的助手 span，标签计数必须是 shift、截断、mask 后的实际位置；不能把标签占比乘到反向成本上。

### 预训练

`training.pretrain` 可省略：`dataset_tokens` 为每 epoch 的有效数据 token，`epochs` 默认1；epochs>1 必须提供数据量。二者只产生 `metrics.training_task.budget_projection`：

`steps = ceil(dataset_tokens × epochs / 每优化步有效token)`；`duration_s = steps × 当前仿真平均步时`。

基础预算用无故障平均步时外推。设置 `training.pretrain.reliability` 后，`budget_projection.reliability` 另外给出周期性保存、故障、恢复与回滚后的工期分布。数据加载和收敛需单独分析。不同上下文长度的 curriculum 应分别运行各阶段并汇总预算。[故障配置与指标](training-reliability.md)

### SFT

- `training.sft.loss_mask`：`response`（默认）/`all`。
- `training.sft.supervised_tokens`：response 模式必填，每序列0…有效长度，总和>0；all 模式禁止再提供该数组。

标签数影响监督吞吐；模型和输出投影仍处理完整上下文，loss 使用全位置 log-softmax 算量代理。没有据此启用选择性 lm_head、fused linear cross entropy、LoRA 或冻结层。长尾长度通过 sequence_lengths 和 padded/packed 对照表达。

### DPO

- `training.dpo.chosen_lengths/rejected_lengths`：各有 batch_size 个正整数，含 prompt+response，均不超过上限。实际训练序列数为 batch 的两倍。
- `training.dpo.reference_log_probs`：`online`（默认）增加同架构冻结参考前向和权重驻留；`precomputed` 使用预计算参考值，省去本次参考执行。

两种模式均保留 FP32 成对参考值缓冲。预计算成本在本任务窗口之外，数据读取没有另加 I/O 模型。`preference_pairs` 单列；DPO 的 `supervised_tokens` 为不可用，不能伪报为 token 标签数。Pairwise loss 使用每对约20次标量操作的工程预算，token log-softmax 另计；实际 prompt mask 索引不在此性能代理内运行。

## RL 参数与流程

当前为**同一物理池、各角色串行执行**：PP=FSDP=1，支持合法 TP/EP/DP、rank 放置和拓扑。Actor rollout/update 共享权重，不额外复制 rollout 权重，也不虚构跨池同步。Reference/reward/critic 独立权重及 actor/critic 优化器状态从开始同时驻留；KV、经验、activation 按生命周期计算实际峰值。

| `training.rl` 字段 | 默认 / 约束 | 仿真影响 |
|---|---|---|
| `algorithm` | `grpo` / `ppo` | GRPO 无 critic；PPO 必须给 critic_model，增加价值前向、GAE 和 critic 更新 |
| `prompts_per_step` | 1，每 DP 副本 | 每迭代新提示数 |
| `samples_per_prompt` | 2；GRPO≥2 | 生成、打分、更新样本与组统计；不假设共享 prefix KV |
| `prompt_length/response_length` | 128/64 | 数组缺省时的长度模板 |
| `prompt_lengths` | 可选，长度为 prompts_per_step | 每提示输入长度；每项复制 samples_per_prompt 次 |
| `response_lengths` | 可选，长度为 prompts_per_step×samples_per_prompt | 实际生成长度（有 EOS 则包含）；固定 batch 内按长度停止计算 |
| `rollout_batch_size` | 1，每 DP 副本 | 生成分批，末批可小；影响 GEMM、decode 与 KV 峰值 |
| `update_epochs` | 1 | 同一经验重复更新，PPO 同时更新 critic；不重复 rollout |
| `reference_policy` | true | 冻结参考 log-prob 前向和驻留；关闭为无 reference 目标，不保证质量等价 |
| `old_log_probs` | `rollout` / `recompute` | 使用生成保留的行为概率，或额外 actor teacher-forcing 前向；均在首次更新前完成 |
| `reward` | `rule` / `model` | 外部规则/环境服务或独立奖励模型 |
| `reward_model` | model 奖励必填，rule 禁止 | 完整 ModelSpec，显式 architecture、`output_head_dim: 1`，保留正常 vocab_size 输入嵌入 |
| `critic_model` | PPO 必填，GRPO 禁止 | 同上，独立可训练模型、梯度/master/optimizer/重算 |
| `rule_reward_latency_s` | 0，秒/回复 | 外部奖励阶段耗时 `ceil(集群样本数/reward_workers)×latency` |
| `reward_workers` | 1 | 外部规则奖励理想并发；model 奖励禁止规则延迟和非默认 workers |
| `advantage_flops_per_token` | 10 | 每回复 token 的优势算量工程预算，分别代表组优势或 GAE |
| `policy_loss_flops_per_token` | 20 | 每处理位置策略目标标量算量，包含 mask 的保守预算；log-softmax 另计 |
| `experience_bytes_per_token` | 24 B，至少24 | 完整经验 token 的 ID、old/reference log-prob、reward、advantage、return 六个4B槽；PPO 再加4B value；是声明的保守缓冲布局 |

数组顺序为 prompt0 的全部样本、prompt1 的全部样本，依此类推。每副本每迭代样本数 `N=prompts_per_step×samples_per_prompt`；要求 `N % (batch_size×gradient_accumulation)==0`。Actor 每迭代优化步为 `N/(batch_size×gradient_accumulation)×update_epochs`；PPO critic 同样多，不再乘 DP 计成额外优化步。不同 microbatch 分别使用实际长度，不取平均长度平方。

Reference 使用 actor 结构及精度。Reward/critic 可有不同尺寸、attention/FFN，但共用硬件、布局和精度并通过形状校验。标量 head 在所有位置计算，尚未做仅末位置奖励投影。TP 下计标量结果归约、词表 max/sum 归约和采样成本；GRPO 跨 EP 的组统计增加归约。MoE 仍采用均衡路由期望。

`rule_reward_latency_s: 0` 是零外部延迟假设，不能称真实奖励免费。涉及代码执行、工具或环境交互时提供观测服务时间；未提供则在结论中保留此假设。

## 各任务共用 feature

| canonical 参数 | 预训练 / SFT / DPO | RL |
|---|---|---|
| `execution.tp/ep/dp/pp/rank_order` | 既有结构化训练范围；DPO reference 共布局 | TP/EP/DP，PP=1 |
| `execution.fsdp/reshard_after_forward/sync_each_microbatch` | actor all-gather、梯度分片、驻留；reference 冻结不分片 | FSDP=1；累积和梯度同步仍计费 |
| `execution.features.activation_checkpointing` | actor 保存/重算及通信 | 只作用于 actor/critic update，不作用于生成和冻结评分 |
| `execution.features.optimizer_offload` | master/optimizer 到 CPU，显式 D2H/H2D | actor/critic 同配置 offload，冻结模型无 optimizer |
| `execution.optimizer.kind/ns_iterations/workspace_copies/inflight_components_per_rank` | Adam/Muon 算量、scratch、并发桶 | actor/critic 更新；分布式 Muon 限制不变 |
| `execution.precision.*`、scale block、extra weight copy | 存储/计算/梯度/通信/回传分别计 | 额外包含 rollout KV，不能省略 reference/reward |
| `hardware.kernel_traces/kernel_profiles`、megakernel | forward/backward/recompute/loss 分别匹配 | 额外区分 task/role/stage 及 rollout prefill/decode |

未接入：LoRA/QLoRA、任意冻结层、训练 SP/CP、选择性 checkpoint、activation offload、异步 rollout、独立池/异构并行/权重版本传输、动态样本过滤、tool 多轮交互、奖励服务器排队。不要添加无效布尔开关或固定速度倍率；按用户需求扩展实际事件模型，或交付明确标注范围的局部阶段实验。学习率、温度、top-p、KL/clip、GAE gamma/lambda 等数值超参不预测质量；它们改变长度/执行路径时，应输入相应分布和算子观测。

## 算子展开与校准

先构造无 trace 的任务基线，再列出实际事件：

```powershell
python skills/supernode-modeling/scripts/build_operator_trace.py examples/training_task_grpo.yaml --list --kernel dense --phase backward
```

事件 ID 为 `task.i<迭代>.<阶段>.<编号>.<原事件ID>`。元数据保留 `training_task/training_role/training_stage/training_update`、模型签名和 workload shape。绑定工具与 NPUSim importer 将任务/角色/阶段加入 scope；不能无依据把 actor backward 的记录用于 reference forward 或 rollout decode。手工 profile 同样应保留这些条件。operations、矩阵、启动、HBM、workspace 和严格守恒规则见 [operator-traces.md](operator-traces.md)。

先组合任务图，再统一 trace→融合→profile。冻结评分仍为 `phase: forward`，需结合 role/stage 判别，不进入模型 F+B 的 MFU 分子。Loss/log-prob/sampling/advantage 是独立辅助算子；trace 展开和融合保留角色身份。默认 backward 仍是前向2倍代理，拆细不证明完整 autograd 精度。

核对 `model_metadata.operator_trace`、`kernel_calibration` 的实际覆盖，同一 backward 阶段可含 actor/critic，需再按角色检查。先验证工作量、依赖及峰值，再用未参加校准的真实 step/iteration 数据验证预测；synthetic 测试不能作为精度提升证明。

## 示例、对照与报告

```powershell
python -m supernode_sim validate examples/training_task_sft.yaml
python -m supernode_sim simulate examples/training_task_sft.yaml -o outputs/training-sft
python -m supernode_sim simulate examples/training_task_grpo.yaml -o outputs/training-grpo
python skills/supernode-modeling/scripts/evaluate_training_tasks.py --output outputs/training-tasks
```

五份完整配置是 `examples/training_task_{pretrain,sft,dpo,grpo,ppo}.yaml`，网页也可选择。评估脚本另跑 SFT padded/减少标签、DPO 预计算 reference、GRPO 单 epoch 对照，保存完整报告及 comparison.json。全部 synthetic，无真实平台性能主张。复制安装的 skill 可用 `--project-root` 指向示例所在仓库。

每次交付包含以下信息：

- 配置、来源、命令、任务/算法、batch 定义、样本预算和长度分布。
- `metrics.training_task` 的处理/有效/监督 token、偏好对或 rollout token、actor/critic 优化步和阶段时间；逐次记录在 `training_task_stages.csv`。
- `metrics.training_accounting` 的 F+B、重算、optimizer、辅助工作、dtype 和角色；RL 分母用完整迭代时间，rollout 不混入模型 F+B。
- `model_metadata.training_task.role_logical_memory_bytes` 和实际物理池峰值；单模型参数表主要描述 actor，多角色张量用 `metadata.training_role` 区分。
- 一项与任务相关的敏感性对照及剩余支持边界；不同监督/样本预算不能只比较原始吞吐。

`throughput_tokens_s` 为 actor **处理 token/s**，effective/supervised/rollout 吞吐单列。RL `step_time_s` 是整轮平均迭代时间，不是单次 optimizer 时间。估算或执行失败时不补造阶段耗时和吞吐；预算外推不表示完整训练已模拟，更不表示达到某种质量。

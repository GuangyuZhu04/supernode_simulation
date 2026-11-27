# 配置与可复现运行

请求级服务使用独立 `serve <模型配置> --serving <请求调度配置>`；`serving-schema` 输出请求 schema，不能混入原模型 Config。字段、SLO 和适用范围见 [serving.md](serving.md)。训练工作量/利用率在 `metrics.training_accounting`，口径见 [zhanlu-training.md](zhanlu-training.md)。

调用 skill 时默认先按 [前端启动流程](frontend.md) 自动准备并打开可编辑网页。网页编辑与以下 CLI/API 使用同一 `Config` 契约；启动网页不等于提交仿真，数值任务仍按用户请求执行。

## 配置来源

版本为 `schema_version: '1.0'`；完整、严格的字段定义以项目 `supernode_sim/config.py` 和 `python -m supernode_sim schema` 为准。未知字段会报错。JSON 与 YAML 使用同一配置契约，不把本文表格当作完整 schema。

| 对象 | 常用字段与口径 |
|---|---|
| `model` | `name`, `total_params`, `active_params`, `layers`, `hidden_dim`, `intermediate_dim`, `num_heads`, `kv_heads`, `head_dim`, `attention_kind`；MoE 另设 `num_experts`, `top_k`, `moe_layer_fraction`, `expert_parameter_fraction` |
| `hardware` | `npu_count`, `su_size`, `npus_per_host`, `topology`, `routing_policy`, `npu`, `cpu`, `scale_up`, `scale_out`, `host_link`, `remote`, `ssd` |
| `workload` | `scenario: training/inference`, `batch_size`；`training.kind: pretrain/sft/dpo/rl` 及专用参数见 [训练任务](training-tasks.md)，其他字段见训练/推理参考 |
| `execution` | `tp`, `cp`, `attention_dp`, `expert_tp`, `ep`, `expert_dp`, `sequence_parallel`, `pp`, `dp`, `fsdp`, `rank_order`, `precision`, `features` |
| `targets` | `throughput_tokens_s`, `ttft_s`, `tpot_s`, `step_time_s`，及搜索候选字段 |
| `sources` | 以字段路径或对象名称为 key，记录 `status`, `description`, `uri`, `revision` |
| `assumptions` | 记录用户提供或分析所需的假设，随输出保存 |
| `features` | 可选统一输入，展开为 canonical 配置；具体选项与冲突规则见 [features.md](features.md) |

内部单位是 byte、second、FLOP。容量示例用十进制 TB/GB，吞吐用 byte/s。单独配置标量 `fsdp` 时要求 `dp % fsdp == 0`。训练时可用 `execution.fsdp_domain` 选择 FSDP 或 FSDP2 并分别布置稠密和专家参数的分片组，支持 `pp=1, dp=1`。[域配置及 8 卡示例](training.md#无流水线的-cpep-与-fsdp-训练)

### Attention 和专家并行组

单池中，每个外层 DP 副本的流水段宽度为 `W = npu_count / (pp × dp)`。同一段的两套分解都必须覆盖这 W 张卡。

| 执行位置 | 段内分解 | 各轴的含义 |
|---|---|---|
| Attention | `tp × cp × attention_dp = W` | TP 切投影矩阵及注意力头，CP 切上下文位置，attention_dp 分配不同请求 |
| 路由专家 | `expert_tp × ep × expert_dp = W` | expert_tp 切单个专家矩阵，EP 切专家集合，expert_dp 复制整套专家分片 |

设置 `cp > 1`、给出 `expert_rank_order` 排列，或显式设置 `expert_tp`、`attention_dp`、`expert_dp` 中任一项即使用独立布局。此时专家 TP 默认 1。两个段内副本数可省略，由 W 整除推导。旧配置仍使用 `cp=1, attention_dp=ep, expert_tp=tp, expert_dp=1`。外层 `dp` 表示各自执行一份完整 batch 的副本数。段内副本共享这份 batch。

典型 8 卡配置为 `execution: {tp: 2, cp: 2, attention_dp: 2, expert_tp: 1, ep: 8, expert_dp: 1}`。默认放置时 Attention TP 组为 `[0,1], [2,3], [4,5], [6,7]`，第一份请求副本内的 CP 组为 `[0,2], [1,3]`，EP 组覆盖 `[0..7]`。设置 `rank_order` 会对两套组一起重映射。运行后可在 `model_metadata.parallel_groups.stages[].groups` 查看每个通信组的实际成员。

请求按序号轮转分配给 Attention 副本。CP 按位置对 CP 度取模分配 token 和 KV。MHA、MQA、GQA、MLA 为本地查询交换 KV，反向将 KV 梯度归还原持有者。KDA、CSA、HCA 在每条 TP 通道上将输入聚到 CP 的第一个 rank，完成有序状态更新后发回输出。FlashAttention 分块调度当前要求 CP=1。路由专家通过自己的 EP 组 dispatch/combine，在专家 TP 组归约矩阵分片结果。两种 TP 布局之间需要的隐藏向量复制按实际端点计费。专家负载沿用均匀 top-k 的期望值。

PD 分离时 P、D 各占一半物理卡，各池独立应用上述公式，且 PP=DP=1。两份物理映射互斥并覆盖所有设备。[PD 配置](features.md)

`expert_rank_order` 可以单独指定专家放置。其值是 W 个段内下标的排列，先按 `rank_order` 取到这一段的物理卡，再按该排列构建专家网格。相同排列用于每个流水段和外层 DP 副本。例如 W=8、专家 TP2 时，`[0,4,1,5,2,6,3,7]` 将第一组专家 TP 放在段内位置 0、4，Attention TP2 仍使用位置 0、1。两组之间的激活重分布进入通信账本。

从仓库根目录运行完整示例。

```powershell
python -m supernode_sim simulate examples/independent_parallel_groups.yaml -o outputs/independent-parallel-groups
```

`sources` 的状态为 `assumed/public/user/measured`。实测校准配置应保留测量来源。引擎默认值是合成假设，不表示某厂商的硬件或某命名模型的已验证规格。用户只有总参数和活跃参数时，可先做条件容量分析；时间、KV、拓扑可行性需要额外结构和执行信息。

`execution.precision` 包含 `weight`, `compute`, `gradient`, `master`, `optimizer`, `kv`, `weight_transfer`, `activation`, `communication`, `recurrent`，以及可选的 `routed_expert_weight`。通常先锁定独立梯度/优化器精度，再比较权重 FP8/FP16，避免把两种权重形态变成两种不同训练算法。

组件建模使用 `model.architecture`：显式 `attention_by_layer` / `ffn_by_layer` 列表与 MLA/KDA/MoE 维度。此模式不需要手填总参数预算；保留 `reported_total_params` / `reported_active_params` 作为公开约数。Muon 通过 `execution.optimizer` 选择，参数分组需要张量形状。字段详情和参数账本以 CLI schema 与 [memory-breakdown.md](memory-breakdown.md) 为准。

## 独立拓扑输入

自然语言或拓扑文件可先经 [supernode-topology 子 skill](../subskills/supernode-topology/SKILL.md) 转为 topology spec，显式声明 SU 结构、SO 域成员及域间链路，再生成原生 `hardware` 片段或合入完整场景。该规范独立于引擎 schema；`scale_out.domains`、`inter_domain_links` 等编译输入不能直接写入 `hardware`。输入契约与域的口径见 [topology.md](topology.md)。

生成的显式图使用 `hardware.routing_policy: su_so`，同 SU 的 NPU 通信限走 SU 边，跨 SU 限走 SO 边；原有显式图默认 `shortest_hop` 保持可使用混合路径。编译到新文件，保留模型/负载/执行/SLO 与来源。用 `--base` 合并时校验并行乘积与连通性，再 `validate` / `simulate`。仅有 `hardware` 的输出片段不是完整场景，不能单独送入 `snmodel simulate`。

## CLI

以下命令均从项目根目录运行；用户配置可以是绝对路径。

```text
python -m supernode_sim validate examples/demo_training.yaml
python -m supernode_sim estimate examples/training_10t_fp8.yaml --output outputs/10t-fp8-estimate
python -m supernode_sim simulate examples/demo_inference.yaml --output outputs/demo-inference
python -m supernode_sim sweep examples/demo_training.yaml --axes examples/sweep_precision.yaml --mode simulate --output outputs/precision-sweep
python -m supernode_sim size-su examples/demo_training.yaml --mode both --output outputs/su-search
python -m supernode_sim feedback examples/demo_inference.yaml --search examples/feedback_inference.yaml --output outputs/demo-feedback
```

`sweep` 的 axes 文件是“点分字段路径 → 非空候选列表”的映射，例如：

```yaml
execution.precision.weight: [fp8, fp16]
execution.precision.weight_transfer: [fp8, fp16]
```

这会形成四种组合；若只想比较“驻留与回传同时变化”的两种预设，分别运行两份完整配置，避免误读笛卡尔积。扫描 `active_params` 时记录结构不变的假设，不据此自动推断专家数或 top-k。

`sweep`、`size-su`、`feedback` 支持 `--workers 4` 并行评估候选；默认 1，候选排序和结果口径保持一致。该参数控制运行仿真器的主机进程数，与所模拟的 TP/SP/EP/DP 无关。内存预算、Python 多进程入口和计时验证见 [resource-usage.md](resource-usage.md)。

`feedback` 固定模型、硬件和拓扑，联合遍历 TP/SP/EP、输入长度与固定 batch，在满足 TTFT/TPOT 后最大化 Decode 吞吐。`--search` 接受独立的 YAML/JSON 搜索规范，不能把其字段写入原生场景顶层；可用 `--ttft-s`、`--tpot-s`（别名 `--tpop-s`）、`--prefill-lengths`、`--decode-batch-sizes`、`--output-length`、`--max-candidates`、`--engine auto/simulate/streaming` 覆盖相应设置。两个时延目标以秒输入，省略时可沿用场景中的目标。详细搜索字段、默认候选范围、吞吐口径和复跑方式见 [feedback.md](feedback.md)。

输出包括 `report.md`, `result.json`, `config.resolved.json`, `latency_breakdown.csv`, `operator_breakdown.csv`；估算/仿真包含资源与内存 CSV，仿真另有事件 CSV 与 Chrome/Perfetto `timeline.json`。时延 CSV 保存 TTFT / TPOT（TPOP）的 kernel 服务、分域通信与等待分项，单位秒。算子 CSV 保存推理窗口和训练步的逐算子耗时、占窗口比例及算子内部计算、通信、访存、启动等分项和比例（秒 / %）。未定义或未执行的指标留空，详见 [latency-breakdown.md](latency-breakdown.md)。实际文件以命令结果为准。`status=estimated` 不是可行性证明；`infeasible` 表示资源/执行约束失败，`slo_failed` 表示完成执行但不满足 SLO。搜索根状态可为 `completed`，仍须检查每个子搜索和候选的状态。

Feedback 另输出 `candidates.csv`；存在可行最优候选时输出 `best.config.json`，可直接交给 `simulate` 复核。`config.resolved.json` 保存解析后的基础场景，不能当作最优候选配置；搜索范围与所有候选的结果保存在 `result.json` 中。

大型默认导出的张量表可能是无损压缩的 `tensors.csv.gz`，以 `artifact_export` 为准；`--full-detail` 保留全部明细和普通 CSV。主机内存紧张、批量实验或检查点压缩选项见 [resource-usage.md](resource-usage.md)。

## 外部 profile 与 HF 导入

```text
python -m supernode_sim import-hf path/to/config.json --overrides overrides.yaml --output outputs/model-import.json
```

HF 导入输出结构字段和 `missing_fields`，不是可直接执行的完整场景。将已核实的结构放入完整配置的 `model`，补充硬件、负载和并行策略。HF 文件通常不能证明总参数、活跃参数和 MoE 参数占比；不要由模型名字填入这些数字。

Kimi K3 的适配器识别嵌套 `text_config`，验证 1-based KDA/MLA 层号完整分区，导入 LatentMoE、gated MLA/KDA 与 AttnRes，并从形状推导文本 backbone 参数。`scope` 明确排除 vision tower/projector；非零 MTP 会进入 `missing_fields`。不要把外层多模态模型的名称当作已计入视觉组件的证明。发布 checkpoint 的混合精度不会自动成为运行方案，需独立选择。

MLA 的 `architecture.mla.cache_layout` 可设 `compressed` / `expanded`。导入后的默认 `compressed` 是优化存储假设；HF 参考实现的 expanded K/V 与优化 latent-cache 内核不同。以样例中的按字段来源说明为模板，明确所模拟的缓存布局和未校准的 kernel 耗时。

配置顶层 `calibration` 可引用相对于场景文件的 YAML/JSON；校准文件支持 `hardware` 覆盖和 `source`。例如使用实测 `npu.efficiency`、`hbm_Bps`、链路效率和 CPU 有效 FLOPs。kernel 专用校准字段先检查当前 schema 的支持范围，不能默默将未支持字段忽略。复跑优先使用 `config.resolved.json`，避免外部校准文件改变结果。

`hardware.kernel_profiles` 的条目按 kernel 名、FLOPs、HBM bytes 及可选 dtype/phase 匹配；`duration_s` 不含启动，新增可选 `startup_s` 显式替换原启动时间。检查实际匹配覆盖率与 prefill/decode 分项，不以配置了 profile 作为校准成功证明。实测数据契约、缓存命中口径及校准限制见 [runtime-calibration.md](runtime-calibration.md)。

`hardware.kernel_traces` 可在校准前把匹配的组件展开为串行子算子，显式声明矩阵维度、FLOPs/HBM、启动和局部工作区；FLOPs 守恒、访存不得低估原事件，必填阶段/精度/硬件与元数据约束。使用 `scripts/build_operator_trace.py` 从真实编译事件绑定来源明确的算子列表；字段、命令与普通/流式报告口径见 [operator-traces.md](operator-traces.md)。这不是任意 profiler 文件的直接导入器。

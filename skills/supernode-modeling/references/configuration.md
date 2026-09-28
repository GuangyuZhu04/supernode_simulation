# 配置与可复现运行

## 配置来源

版本为 `schema_version: '1.0'`；完整、严格的字段定义以项目 `supernode_sim/config.py` 和 `python -m supernode_sim schema` 为准。未知字段会报错。JSON 与 YAML 使用同一配置契约，不把本文表格当作完整 schema。

| 对象 | 常用字段与口径 |
|---|---|
| `model` | `name`, `total_params`, `active_params`, `layers`, `hidden_dim`, `intermediate_dim`, `num_heads`, `kv_heads`, `head_dim`, `attention_kind`；MoE 另设 `num_experts`, `top_k`, `moe_layer_fraction`, `expert_parameter_fraction` |
| `hardware` | `npu_count`, `su_size`, `npus_per_host`, `topology`, `npu`, `cpu`, `scale_up`, `scale_out`, `host_link`, `remote`, `ssd` |
| `workload` | `scenario: training/inference`, `batch_size`；场景专用字段见训练/推理参考 |
| `execution` | `tp`, `pp`, `ep`, `dp`, `fsdp`, `rank_order`, `precision`, `features` |
| `targets` | `throughput_tokens_s`, `ttft_s`, `tpot_s`, `step_time_s`，及搜索候选字段 |
| `sources` | 以字段路径或对象名称为 key，记录 `status`, `description`, `uri`, `revision` |
| `assumptions` | 记录用户提供或分析所需的假设，随输出保存 |

内部单位是 byte、second、FLOP。容量示例用十进制 TB/GB；吞吐用 byte/s。配置中的 `80e9` 表示 80 GB，不是 80 GiB。`tp * pp * ep * dp == npu_count`；FSDP 是 DP 子组，`dp % fsdp == 0`。更换卡数后要同时校验并行分解与 rank 放置。

`sources` 的状态为 `assumed/public/user/measured`。实测校准配置应保留测量来源。引擎默认值是合成假设，不表示某厂商的硬件或某命名模型的已验证规格。用户只有总参数和活跃参数时，可先做条件容量分析；时间、KV、拓扑可行性需要额外结构和执行信息。

`execution.precision` 包含 `weight`, `compute`, `gradient`, `master`, `optimizer`, `kv`, `weight_transfer`, `activation`, `communication`, `recurrent`，以及可选的 `routed_expert_weight`。通常先锁定独立梯度/优化器精度，再比较权重 FP8/FP16，避免把两种权重形态变成两种不同训练算法。

组件建模使用 `model.architecture`：显式 `attention_by_layer` / `ffn_by_layer` 列表与 MLA/KDA/MoE 维度。此模式不需要手填总参数预算；保留 `reported_total_params` / `reported_active_params` 作为公开约数。Muon 通过 `execution.optimizer` 选择，参数分组需要张量形状。字段详情和参数账本以 CLI schema 与 [memory-breakdown.md](memory-breakdown.md) 为准。

## CLI

以下命令均从项目根目录运行；用户配置可以是绝对路径。

```text
python -m supernode_sim validate examples/demo_training.yaml
python -m supernode_sim estimate examples/training_10t_fp8.yaml --output outputs/10t-fp8-estimate
python -m supernode_sim simulate examples/demo_inference.yaml --output outputs/demo-inference
python -m supernode_sim sweep examples/demo_training.yaml --axes examples/sweep_precision.yaml --mode simulate --output outputs/precision-sweep
python -m supernode_sim size-su examples/demo_training.yaml --mode both --output outputs/su-search
```

`sweep` 的 axes 文件是“点分字段路径 → 非空候选列表”的映射，例如：

```yaml
execution.precision.weight: [fp8, fp16]
execution.precision.weight_transfer: [fp8, fp16]
```

这会形成四种组合；若只想比较“驻留与回传同时变化”的两种预设，分别运行两份完整配置，避免误读笛卡尔积。扫描 `active_params` 时记录结构不变的假设，不据此自动推断专家数或 top-k。

输出包括 `report.md`, `result.json`, `config.resolved.json`；估算/仿真包含资源与内存 CSV，仿真另有事件 CSV 与 Chrome/Perfetto `timeline.json`。实际文件以命令结果为准。`status=estimated` 不是可行性证明；`infeasible` 表示资源/执行约束失败，`slo_failed` 表示完成执行但不满足 SLO。搜索根状态可为 `completed`，仍须检查每个子搜索和候选的状态。

## 外部 profile 与 HF 导入

```text
python -m supernode_sim import-hf path/to/config.json --overrides overrides.yaml --output outputs/model-import.json
```

HF 导入输出结构字段和 `missing_fields`，不是可直接执行的完整场景。将已核实的结构放入完整配置的 `model`，补充硬件、负载和并行策略。HF 文件通常不能证明总参数、活跃参数和 MoE 参数占比；不要由模型名字填入这些数字。

Kimi K3 的适配器识别嵌套 `text_config`，验证 1-based KDA/MLA 层号完整分区，导入 LatentMoE、gated MLA/KDA 与 AttnRes，并从形状推导文本 backbone 参数。`scope` 明确排除 vision tower/projector；非零 MTP 会进入 `missing_fields`。不要把外层多模态模型的名称当作已计入视觉组件的证明。发布 checkpoint 的混合精度不会自动成为运行方案，需独立选择。

MLA 的 `architecture.mla.cache_layout` 可设 `compressed` / `expanded`。导入后的默认 `compressed` 是优化存储假设；HF 参考实现的 expanded K/V 与优化 latent-cache 内核不同。以样例中的按字段来源说明为模板，明确所模拟的缓存布局和未校准的 kernel 耗时。

配置顶层 `calibration` 可引用相对于场景文件的 YAML/JSON；校准文件支持 `hardware` 覆盖和 `source`。例如使用实测 `npu.efficiency`、`hbm_Bps`、链路效率和 CPU 有效 FLOPs。kernel 专用校准字段先检查当前 schema 的支持范围，不能默默将未支持字段忽略。复跑优先使用 `config.resolved.json`，避免外部校准文件改变结果。

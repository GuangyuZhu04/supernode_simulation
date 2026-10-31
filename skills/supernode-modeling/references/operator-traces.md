# 训练与推理的子算子执行描述

当组件级 roofline 无法表达多个串行算子的启动、访存和临时工作区时，使用 `hardware.kernel_traces`。它细化既有组件事件，随后由原有 `kernel_profiles` 校准子算子耗时。外部方法与参数映射见 [reference-simulators.md](reference-simulators.md)。

这是调用者提供、受适用范围约束的串行算子描述，不会自动解析 Chakra/AICB/框架 profiler，也不根据模型名称补全真实 autograd。未配置时保留原路径。仅配置 trace 不会将 `hardware_performance` 标为已校准。

## 先检查实际编译事件

从项目根目录运行；复制安装的 skill 需要可导入 `supernode_sim`，示例路径仍属于原项目：

```powershell
python skills/supernode-modeling/scripts/build_operator_trace.py examples/features/synthetic-training.yaml --list --kernel dense --phase backward
```

输出事件 ID、精度、原 FLOPs/HBM bytes、启动成本和可用于匹配的元数据。工具只构图，检查时不应用 kernel 耗时 profile；不执行仿真，不改原文件。默认最多显示 20 条，`total_matching_events` 和 `truncated` 说明是否截断。已有 trace 或 megakernel 的配置会被拒绝，应先使用未拆分的基线。

结构化组件具有 `model_shape_signature`、TP/EP/SP 与 `workload_shape`。优先同时匹配结构和负载；同 FLOPs/bytes 不足以证明运行形状相同。工具默认还锁定 layer，可在有跨层一致性证据时手动放宽，不按序号猜测重复层等价。

该绑定工具仅接受同时具备结构签名和 workload shape 的结构化组件。预算模型、缺少身份的转换事件不会仅凭 layer 或同工作量被自动导入；这类路径须先扩展编译元数据，再提供有依据的匹配。底层 trace schema 的非空 `metadata_match` 仍由调用者选择，随意放宽选择器会丢失相应的适用范围保护。

## 输入契约

`hardware.kernel_traces.<原kernel名>` 为条目列表，每条包含：

| 字段 | 作用 |
|---|---|
| `id`, `source` | 唯一标识和拆分来源；synthetic 必须明确标注 |
| `hardware_name`, `dtype`, `phase` | 必填且精确匹配；训练 `forward/backward/recompute`，推理 `prefill/decode/verify` 分开 |
| `flops`, `memory_bytes` | 原聚合事件的工作量选择器，不是子算子新总量 |
| `metadata_match` | 非空精确匹配字典；建议包括结构签名、并行度和 `workload_shape` |
| `calibration_context` | 可选完整运行环境约束；省略不意味着环境已验证 |
| `operations` | 按执行顺序排列的子算子列表 |

每个 operation 必填 `kernel`、`flops`、`memory_bytes`。可选 `startup_s` 和 `workspace_bytes` 默认零；`matmul` 提供正整数 `m/n/k` 和可选 `batch=1`，校验 `flops = 2 × batch × m × n × k`。dtype 沿用原事件，不支持在一条 trace 内切换计算精度。

子算子 FLOPs 总和必须与原组件相等；HBM 访问字节总和不得低于原值，允许显式增加未融合中间读写。GEMM 的 operand bytes、真实 HBM transaction bytes 和张量驻留容量不是同一项。trace 不用于用更少的 FLOPs/bytes 隐藏原模型误差；确有额外算子计算、融合节省或错误预算时，应修正编译公式或使用既有 megakernel，并提供验证。

原组件 ID 保留为末尾完成点，后继通信/更新仍等待它。原 allocation 从首子算子开始，保守保留到原释放点；`workspace_bytes` 仅在对应子算子执行期间申请并释放，**不描述跨子算子的中间张量生命周期**。当前仅支持单 NPU compute/HBM 和本地 compute stream；带 selection 等其他资源的组件会保留原路径，并报告原因。

原 `startup_s` 仅折入首子算子一次，operation 的启动时间为显式新增成本。子算子的 `kernel_profile.startup_s` 若提供，替换该子事件的完整启动值（包括首项原启动），不叠加；`duration_s` 仍不含启动。父组件 profile 不自动转移到子算子，否则会重复计时。

trace 先展开，再处理 megakernel 和 kernel profiles。融合只对当前可匹配的子算子链生效；检查两种覆盖率，不能假定原组件融合组仍会命中。

## 绑定来源明确的算子列表

工具输入是 JSON，且仅有 `id`、`source`、`operations` 三个字段；`source` 使用项目的 `{status, description, uri?, revision?}` 形式。给定示例将 8-token、16×32 Dense backward 的原 GEMM 预算拆成六个 dgrad/wgrad GEMM；非线性反向没有新增计算，不代表完整真实 autograd。每算子 1 μs 启动和 512 B workspace 均为合成假设。

```powershell
python skills/supernode-modeling/scripts/build_operator_trace.py examples/features/synthetic-training.yaml --event-id train.s0.p0.backward.m0.l0.npu0.Dense --operators skills/supernode-modeling/assets/operator-trace.synthetic.json -o outputs/operator-traces/training.calibration.json
```

输出为可合并的 `hardware.kernel_traces` 校准片段及来源，不是完整场景。可将 hardware 片段合并到基线，或在场景顶层 `calibration` 引用该文件（相对路径从场景文件所在目录解析）。不要覆盖已有校准条目；先合并所需条目，再检查 ID 唯一及适用条件无歧义。多个 trace 同时命中会报错，不以列表顺序择一。

无需手动合并的两场景复现实验：

```powershell
python skills/supernode-modeling/scripts/evaluate_operator_traces.py --output outputs/operator-traces
python -m supernode_sim validate outputs/operator-traces/training.operators.config.json
python -m supernode_sim simulate outputs/operator-traces/inference.operators.config.json -o outputs/operator-traces/inference-report
```

实验保存训练/推理的 baseline 与展开后配置、完整结果、流式推理结果和 `comparison.json`。训练示例细分 backward；推理示例分别绑定 prefill 与每个 decode 上下文。全部为 synthetic；额外启动与访存可能令时延增加，这表示执行成本进入模型，不表示真实硬件变慢或预测误差已下降。

## 核对结果和校准

检查 `model_metadata.operator_trace` 中 `matched_events`、`generated_operations`、`by_phase`、`skip_reasons`、`additional_memory_bytes`、`additional_startup_s`。这里的 memory bytes 是额外 HBM **访问量**，不是额外显存容量；峰值容量仍看 simulation 的 memory pools。覆盖率按原组件事件数计，不能与按子算子事件计的 kernel profile 覆盖率直接相加。

`max_operator_working_set_bytes` 保留父组件原工作集代理值并加上局部 workspace；事件标记 `working_set_scope=parent_component_proxy_plus_local_workspace`。这避免拆分后漏报原工作集，但不是从 HBM 访问字节推导的子算子精确 footprint，也不是硬件缓存容量建议。

普通事件明细中可见 `trace_id`、`original_kernel`、`trace_parent_event`、`operation_index` 和可选 `matmul`。没有 trace 匹配时保持解析组件；不能仅因输入带有配置就称它已生效。流式运行合并展开/校准统计，缓存复用不得绕过形状和环境约束。

对新增子算子采集独占服务时间，按其 kernel 名、phase、dtype、工作量、`matmul`、`workload_shape`、SKU 和运行环境填写 `hardware.kernel_profiles`。推理必须覆盖实际上下文，训练必须区分 forward/dgrad/wgrad/recompute。工作量与形状不匹配时保留解析回退，当前没有 kernel 形状插值或跨域外推。

先核对 FLOPs、真实流量、依赖及峰值，再用未参与校准的实测配置比较 step time/TTFT/TPOT。没有同条件实测时只交付可重跑预测、覆盖率和剩余误差来源，不给出“准确率提升百分比”。在线到达、continuous batching、动态 KV 淘汰、专家倾斜、任意算子内部并行 DAG 仍需独立建模。

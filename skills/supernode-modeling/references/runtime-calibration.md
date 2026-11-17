# 推理实测对齐与校准

用于 TTFT、TPOT/TPOP、decode 性能、仿真与真机差异。先读本文件确定比较是否成立；Kimi K3/950DT 的公开性能点见 [证据与适用范围](kimi-k3-runtime-evidence.md)。这里的测量流程是本项目的工程方法，不是厂商已对本引擎做过验证。

DeepInfer、NVIDIA、Ascend 的固定来源数值、可执行比较工具及本轮引擎修正见 [公开 benchmark 对比](public-benchmark-validation.md)。使用其证据目录时区分 `status`（来源）与 `evidence_kind`（实测/估算/规格）；条件缺失时比较器报告 `not_comparable`，不自动拟合硬件效率。

需要改善预测时，执行 [运行拟合流程](runtime-fitting.md)：分别设置 MoE dispatch/combine wire 格式、NIC 聚合与 collective 服务时间，并运行公开通信数据的拟合/留出实验。检查边界是准备工作，最终应交付实际改变的执行配置与前后数值。

Ascend 多环境使用 [响应曲线流程](ascend-runtime-adaptation.md) 与 [真实数据采集指南](ascend-measurement-sources.md)。环境和软件条件独立分组，消息量可在训练范围内插值；不将 H800 DeepEP 系数作为其他 Ascend 部署的通用修正。

DavidV100 的算子微基准用于建立有适用范围的 [微架构服务需求](david-v100-microarchitecture.md)：分别约束 Cube、搬运和指令供给，保留物理共享及依赖下界。它不是其他 Ascend 机型的硬件参数表；报告实测参考也不等于当前目标算子已完成真机校准。

## 先确定测量的对象

| 对象 | 计时边界 | 当前引擎可比较程度 |
|---|---|---|
| 单 kernel | 相同工作量的独占服务时间；启动开销另列 | `hardware.kernel_profiles` 支持精确工作量匹配 |
| 固定 batch 模型执行 | 权重已加载，从执行开始到每个 token 完成 | 普通 `simulate`；结构化 DP=PP=1 可用 streaming |
| 客户端在线服务 | 请求发出到 token 到达，包含排队及前后处理/传输 | 需要真实请求 trace；当前引擎未模拟完整服务调度 |
| 投机 decode | 草稿、验证、接受/回滚及状态提交，按实际产出 token 折算 | [可选轮次模型](features.md)支持显式成本与接受数；不自动复现具体 DSpark 实现或预测接受率 |
| PD 分离 | P 首 token、状态交接、D 后续 token | 同布局双池固定 batch，按实际拓扑传输；TTFT 未含客户端转发或在线排队 |
| Megakernel | 安全的同 rank/层/阶段局部融合组 | 显式启动、受限 HBM 节省与可选 fused service；核对匹配与保守生命周期 |

“支持某模型”“测过某场景”不等于公开了性能数值。不能把同名模型、相同卡数或相同 FLOPs 当作同一个执行路径。缺失值使用 null/未知，不从其他指标反推。

每个比较保留这些条件：模型/checkpoint/tokenizer revision、文本/多模态范围，芯片 SKU 与数量、HBM/拓扑/rank 放置，框架/CANN/驱动/kernel 版本、图模式与融合，权重/activation/KV 精度，Attention/FFN/expert 各自并行布局，输入输出长度分布，实际 batch 与并发、到达率/突发性，prefix cache 冷热状态、KV 存储/卸载、调度 token budget，以及投机开关和草稿版本。未公开的条件保留未知。

真实框架的 Attention DP、MoE EP 可能共用 rank。先映射 communicator 再检查本引擎 `TP × PP × EP × DP = NPU` 的独立维度契约；不能直接照抄数字相乘。无法表达的布局标为不完全可比。

PD 的总物理卡数是两个池的和，当前相同布局时为 `2 × TP × EP`。不得把 P/D 当成两倍 DP batch，或遗漏第二份模型驻留。

## 指标定义

对请求 i，令 a 为所选边界的开始时刻，t1…tn 为实际 token 的完成/到达时刻，f 为最后响应结束时刻：

- TTFT = t1 − a。引擎 a=执行开始；客户端 a=请求发出，二者不是同一指标。
- TPOT = (tn − t1)/(n−1)。n≤1 时为空。用户写 TPOP 时注明这里采用 TPOT 定义，不新增拼写不同的引擎字段。
- ITL 是每个相邻 token 的间隔；请求 TPOT 的 P99 与 pooled ITL 的 P99 是不同统计量。服务可能一次返回多个 token；只有 chunk 时间戳时不能伪造内部 token 时刻，另报 ICL 和所用工具的折算公式。
- E2E 可以是 f−a 或 tn−a，必须标明。若 benchmark 的 TPOT 用 `(E2E−TTFT)/(n−1)`，确认 E2E 终点是否包含终止消息的额外延迟。
- 输出吞吐 = 测量窗口中按明确计数规则产生的 output tokens / 窗口秒数。不能用 1/平均 TPOT 代表多请求整机吞吐。输入吞吐、请求吞吐、满足联合 SLO 的 goodput 分开。

现有引擎 `metrics.ttft_s` 是批内最大 TTFT，`metrics.tpot_s` 是批内最大的“逐请求平均 TPOT”，不是全局均值或 P99。单 token 请求不判断 TPOT SLO。固定 batch 的最大值不能当成生产尾延迟；decode-only 的 TTFT 为空，只有一次 decode 间隔也不能叫稳态均值。

## decode 中的三种缓存问题

| 含义 | 如何影响时延 | 本项目当前处理 |
|---|---|---|
| 前缀缓存复用 | 跨请求复用已计算的 prompt KV，主要减少 prefill；未缓存后缀仍需计算，并读取已有前缀 | 未实现自动 prefix reuse；不能将“命中 90%”直接变成 decode 少读 90% |
| KV 页驻留 | 需要的历史 KV 位于 HBM、host 或 remote；搬入、淘汰、抢占、重算会阻塞 decode | `kv_offload`/`kv_offload_fraction` 表示静态分层放置；不是动态命中率模型 |
| 硬件缓存 | L2 等层级可能减少 HBM 实际流量，效果取决于工作集、tile、复用距离及并发 | HBM 是共享带宽资源，没有 L2 命中模拟；子算子可用 `resource_demands` 显式约束已知 L2/UB 服务量，仍需独立的实际 HBM 字节证据；未知命中行为单列敏感性假设 |

普通全注意力 decode 通常需要历史 KV；在 HBM 中“命中”仍然要付读带宽成本。KV 被丢弃时，不能直接跳过依赖或将缺失部分耗时设为零。Kimi K3 的 MLA KV 按上下文增长，KDA 保存 recurrent/conv state；不能对二者套同一个上下文长度或缓存命中公式。

若有真实 trace，按访问字节记录 `required bytes / device-resident bytes / host fetch / remote fetch / recomputed tokens / eviction & preemption`，区分 byte hit rate 与 page hit rate。动态驻留模型还需要设备 KV 预算、逐步工作集、替换/准入策略、预取完成时刻和共享链路争用；平均命中率无法单独预测 P99。上述是测量字段和未来扩展要求，**不是当前严格配置可接受的新字段**。

对静态卸载可扫描现有字段 `execution.features.kv_offload_fraction`；先选择 host/remote，remote 必须启用。增加卸载可能缓解容量但增加搬运，保留容量不可行结果。不要修改输入长度来假装 prefix hit：这会同时错误缩小 decode 上下文和 KV 容量。

## 分项校准而非整体乘系数

组件聚合边界无法对应真实 kernel 时，先按 [operator trace](operator-traces.md) 展开，再采集和匹配子算子服务时间。结构化 `workload_shape` 保留长度分布、query/context 与 signature；`matmul` 保留声明的 GEMM 维度。使用它们约束 profile，不用一个 token 数或全局效率跨 prefill/decode 复用测量。trace 覆盖与 profile 覆盖是不同指标，前者本身不构成校准证据。

微架构约束与 kernel profile 是两层证据。`resource_demands` 记录资源服务需求，其 `measured_reference` 仅指参考微基准的来源；`kernel_profiles.evidence_kind=measurement` 则要求目标 kernel 的同条件硬件测量。不能把推导的 `max(资源需求, 依赖路径)` 直接包装成 measurement profile。周期转秒必须有对应模块时钟；核内吞吐与整芯片带宽先统一参与核数和范围。

1. **建立 baseline。** 固定模型范围和软件路径，保存 resolved config。先对齐权重/KV/KDA 结构与存储精度，排除遗漏或重复复制。
2. **测量 prefill 与 decode。** 按实际 batch、上下文长度、KV layout、矩阵形状与 dtype 采样。decode 采集多个输出位置；普通 decode 和投机验证分开。
3. **拆开成本。** 算子服务时间、启动/图调度、HBM、collective、小消息延迟、host/remote 搬运、CPU 前后处理各自测量。共享资源重叠由事件依赖与争用决定，不把所有独立峰值耗时相加。公开优化的阶段收益不能重复相乘。
4. **应用支持的参数。** 同平台有效 HBM/链路带宽及效率可经 `calibration` 覆盖；kernel 通过下述精确匹配校准。投机模式使用显式 draft 成本和实际接受数，PD 使用双池和拓扑交接，megakernel 使用合法融合组。仍未建模的 continuous batching、chunked prefill、动态 preemption、PD 在线队列/异构重分片、任意融合多流及具体 DSpark 内核留在差异清单，不吸收到 NPU efficiency。
5. **留出验证。** 对未参与拟合的长度、batch/并发与负载点比较；相同 timer boundary、聚合方式、窗口、缓存状态和软件版本才计算误差。校准覆盖率是事件计数比例，不是时间比例，更不是准确率。

无真机 profile 时，保留明确的假设范围做敏感性分析。不要选择一个默认“现实修正倍数”，也不要为了对齐 10 ms 修改模型规模、硬件峰值或丢弃通信。资料与场景不匹配时，输出 `not_comparable` 的文字结论及原因，不输出伪精度。

### kernel profile 的执行契约

在场景的 `hardware.kernel_profiles.<事件中的kernel名>` 下写条目；或在现有 calibration 文件的 `hardware` 中覆盖。字段以 `python -m supernode_sim schema` 为准：

| 字段 | 意义 |
|---|---|
| `flops`, `memory_bytes` | 匹配事件的 NPU FLOPs 和 HBM 字节量；不匹配则回退解析模型 |
| `dtype`, `phase` | 应显式区分测量精度与 prefill/decode；省略是通配，不意味着跨阶段验证 |
| `duration_s` | 不含启动开销的独占服务时间，仍受真实 FLOPs、HBM 争用及计算流串行约束 |
| `startup_s` | 可选的独立启动时间；给出时替换该事件原启动值，省略时保留原值 |

先从未校准 `compile_graph`/事件明细确定真实 kernel 名与工作量。按来源记录 measurement 方法、形状、平台和版本。不能把 profiler 整个 wall time 同时写入 `duration_s`，又在 `startup_s` 重复计入启动。服务时间不能低于配置的物理计算/访存及显式资源服务下界，增加启动项也不能绕过此校验。

Ascend 短任务保留 `TaskStart→CoreStart→CoreEnd→TaskEnd→下一TaskStart` 的边界：Core Span、Device Task、同流 Gap 和 Host launch 不是同一成本。DavidV100 报告的约 22 ns 仅是供给充分后的 Gap，92 cycle 不是每个 64³ MMAD 的完整执行时间；空任务有 PMU 扰动，不能跨实验机械扣除固定差值。校准时分开缓存冷热、PMU 开关、连续异步与逐次同步，并记录预热、重复次数及离散程度。

默认仍是 FLOPs/bytes/dtype/phase 的工作量匹配，不自动验证未声明的张量形状、融合或运行栈。条目可附 `hardware_name`、完整 `calibration_context`、`metadata_match` 和 `source`；不匹配则回退，报告 scope mismatch，不通过 legacy 同工作量条目绕过。结构化组件事件具有模型结构签名及 TP/EP 元数据，可用于明确的选择器；它们不认证真实 checkpoint、tile 或 fused kernel。同工作量不同形状不能无证据复用，也不能用整网含通信的耗时校准单 kernel 后再重复计算通信。

检查 `model_metadata.kernel_calibration` 的 `eligible_events`, `matched_events`, `coverage_fraction`, `by_phase`, `unmatched_kernels`, `unprofiled_kernels`。只有实际匹配才可标 `partially_kernel_calibrated`；全匹配也不是端到端验证。streaming 结果检查对应 `model_metadata` 与实际执行 phase；只校准 prefill 不能声称 decode 已校准。提前因容量失败的流式执行仅统计已构建部分，不能把它的覆盖率当完整运行覆盖率。

## 处理请求实测 trace

逐请求记录可选 `admission_s`，必须位于 arrival 和首 token/完成之间。工具另报 admission wait 与 server TTFT；缺失保持未知，不推测零排队。`serve` 的模拟 token trace 可用同一工具复核，但仍不能标为实测。有限服务调度的时钟/cohort 见 [serving.md](serving.md)，ServingCast 门外等待排除问题见 [来源核验](msmodeling-servingcast.md)。

使用随 skill 打包、仅依赖 Python 标准库的 [分析脚本](../scripts/analyze_inference_trace.py)。从项目根目录可运行：

```text
python skills/supernode-modeling/scripts/analyze_inference_trace.py skills/supernode-modeling/assets/inference-trace.example.json -o outputs/trace-metrics.example.json
python skills/supernode-modeling/scripts/analyze_inference_trace.py path/to/measured-trace.json -o outputs/measured-metrics.json
```

示例是 synthetic，仅验证指标计算。它不是 950DT benchmark，也不会修改仿真参数。安装到个人 skills 后使用脚本实际所在路径。

输入 JSON：`metadata` 包含 `timer_boundary: client/engine`、`status`（synthetic/public/measured/user/assumed）、`source`、`model`、`hardware`、`runtime`；`measurement_window_s: [start,end]`；`requests` 每项包含唯一 `id`、`arrival_s`、逐 token 的 `token_times_s`，可选 `finish_s`、`status`、`warmup`、`output_tokens`、`input_tokens`。时间为同一单调时钟的秒数。可附 `slo: {ttft_s: ..., tpot_s: ...}`。

脚本将成功且非 warmup 的请求纳入统计，窗口为 `[start,end)`；延迟样本按完成时间落窗选取，arrival 可早于窗口，输出明确的计数。吞吐按窗口内实际 token 计数，不能假装和“完成请求全部 token 数”使用同一分子。失败、取消、未完成及 warmup 分别记录；SLO goodput 为窗口内联合达标的完成请求/秒，单 token 的 TPOT 不适用，缺首 token 的请求不达标。

保留输出的 sample count、分位数算法、排除项与统计说明。小样本 P99 只是样本分位数，不能证明稳定尾延迟。工具只接受真实逐 token 时间戳；只有 SSE/chunk 事件时保留原始日志并使用对应工具的 chunk 指标，禁止均匀插值成 token trace。

## 实测矩阵与报告

从用户部署的实际范围选点；针对 K3 可从输入 8K/32K/128K、输出 1K、并发 1/8/32 开始。这里 K 的精确 token 数必须写明，且“并发”不直接等于本引擎每 DP batch。固定并发与固定到达率测试分开；后者覆盖低负载、接近饱和、突发与过载。记录 warmup、测量时长、请求样本量、随机种子、EOS 行为、重复次数、失败率及未完成队列。

报告同时给出：仿真指标及边界、可比较实测点、绝对/相对误差（有同条件数据时）、prefill/decode 覆盖率、KV 放置/搬运与内存峰值、尚未建模的成本。没有实测 TTFT 就明确尚未校准 TTFT；只有平均 TPOT 就不补造 P50/P99。

## 方法依据

- [NVIDIA AIPerf 指标定义](https://docs.nvidia.com/aiperf/v0.7.0/reference/ai-perf-metrics-reference)：TTFT、请求间隔统计与 chunk 边界。不同版本/工具的命名和计数可能不同，应锁定版本。
- [vLLM 生产指标](https://docs.vllm.ai/en/stable/usage/metrics/)：区分 queue、prefill、decode 等服务阶段；导出对应指标帮助定位差异。
- [vLLM 优化指南](https://docs.vllm.ai/en/v0.20.0/configuration/optimization/)：KV preemption/recompute 与 chunked prefill 调度会影响真实延迟。
- [vLLM 前缀缓存边界](https://docs.vllm.ai/en/v0.10.1/features/automatic_prefix_caching.html#limits)：APC 加速 prefill，不直接缩短生成新 token 的 decode。
- [Vidur profiling](https://github.com/microsoft/vidur/blob/main/docs/profiling.md)：分开采集计算、attention、通信和 CPU 成本；其论文的精度结论不能移作本仓库的精度保证。

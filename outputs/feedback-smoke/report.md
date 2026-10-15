# Feedback：时延约束下的并行与负载搜索

状态：**feasible**；模型：`synthetic-moe-demo`。

目标 TTFT ≤ 1 s；TPOT（兼容输入名 TPOP）≤ 0.01 s。
在满足时延和资源约束的候选中，优先最大化 Decode 吞吐量（token/s）。

模型、拓扑、硬件及其余执行参数沿用基线配置；搜索 TP、SP、EP、Prefill 长度和 Decode batch size。
每个候选采用同一固定 batch 联合执行 Prefill 和 Decode，batch size 同时作用于两个阶段。
Decode 吞吐量只统计首 token 之后的输出 token；首 token 计入 TTFT，不计入 Decode 吞吐量。
计算公式：所有请求的（输出 token 数 − 1）之和 ÷（最晚末 token 完成时间 − 最早首 token 完成时间）。
当 DP 或 PP 大于 1 时，此全局 Decode 窗口可能与其他请求尚未完成的 Prefill 重叠。
每请求输出长度为 4 token（含首 token）。

## 搜索范围与完成情况

搜索完成：是；候选上限：128。
Prefill 长度候选：`[8, 16]`。
Decode batch size 候选：`[1, 2]`。
并行布局数：3；完整布局和默认值来源见 result.json 的 search_space。

| 总候选 | 可行 | 配置无效 | 时延未达标 | 资源不可行 | 错误 |
|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 0 | 0 | 0 | 0 |

结果仅表示已声明有限网格和当前仿真模型下的最优值，不是连续参数空间或真实部署的全局最优值。

## 最优可行候选

候选：`5`；引擎：`simulate`。

| TP | SP | EP | Sequence parallel | Prefill 长度 / token | Decode batch size |
|---:|---:|---:|---|---:|---:|
| 2 | 1 | 2 | 否 | 8 | 2 |

| 指标 | 实际值 |
|---|---:|
| TTFT / s | 0.00109272 |
| TPOT / s | 0.000721437 |
| Decode 吞吐量 / token/s | 2,772.25 |
| Decode 时间窗口 / s | 0.00216431 |
| 全程吞吐量 / token/s | 2,456.22 |

最佳完整配置已导出为 best.config.json，可使用 simulate 命令复现该候选。

## 可行长度与 batch 边界

以下最大值分别在同一布局的可行候选中取值，最大长度和最大 batch 可能来自不同候选，不能直接组合。

| TP | SP | EP | Sequence parallel | 最大 Prefill 长度 | 最大 Decode batch | 可行候选 ID |
|---:|---:|---:|---|---:|---:|---|
| 1 | 1 | 4 | 否 | 16 | 2 | 0, 1, 2, 3 |
| 2 | 1 | 2 | 否 | 16 | 2 | 4, 5, 6, 7 |
| 4 | 1 | 1 | 否 | 16 | 2 | 8, 9, 10, 11 |

Pareto 候选 ID：`5`, `7`。

## 产物与复现

- result.json：完整搜索参数、候选指标、SLO、失败诊断、可行边界与最佳配置。
- candidates.csv：逐候选参数、指标、状态、SLO 与失败诊断，可筛选比较。
- config.resolved.json：应用校准后的基线快照，避免重复加载校准。
- best.config.json：最佳候选的完整可复现配置。

## 假设

- Optimality is limited to the complete declared finite candidate grid and fixed simulator scheduler; no global optimum is claimed.
- Model, hardware/topology, PP, DP, FSDP, rank_order, precision and features are held fixed.
- Prefill length is uniform per request; prefill and decode share the same fixed batch per DP replica and the same parallel layout. No PD disaggregation or chunked prefill is modeled.
- SP shares TP ranks: degree is TP when enabled and 1 otherwise; SP does not multiply the device count.
- TPOP is accepted as an input alias of TPOT. TTFT is the maximum request TTFT; TPOT is the maximum per-request mean token interval.
- Decode throughput = total output tokens after each request first token / (latest last-token completion - earliest first-token completion), aggregated over fixed DP replicas.
- For PP>1 or DP>1, this decode window may overlap other requests remaining prefill; it is not an isolated decode benchmark.
- Both latency targets and any existing throughput target must pass; full prefill/decode memory admission must also succeed.
- Throughput ties prefer longer prefill, then larger batch, then lower TTFT/TPOT, then TP/EP/SP in ascending order.
- Per-layout maximum length and maximum batch can belong to different candidates; only an actual feasible candidate is a jointly validated recommendation.
- The frontier maximizes prefill length, decode batch and measured decode throughput simultaneously.
- Two output tokens measure one decode interval, not steady-state throughput; output length is fixed throughout a search.
- Execution latency excludes request queueing and cold weight loading; results are model predictions, not production P99 guarantees.

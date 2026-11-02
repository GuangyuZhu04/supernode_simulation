# 显式请求到达与 FCFS 成批服务仿真

需要把请求到达、排队、成批和多实例纳入 TTFT/TPOT 时，使用独立 `serve` 命令。普通 `simulate` 仍表示固定 batch 模型执行。参考来源与 ServingCast 实现差异见 [msmodeling-servingcast.md](msmodeling-servingcast.md)。

本版为**非抢占 FCFS 固定成批**：每批调用原算子、通信、内存引擎，实例等整批完成后再接纳下一批。没有 continuous batching、chunked prefill、逐轮 token budget、动态 KV 抢占/重算、prefix reuse、PD 独立队列或投机请求调度；不等于完整 ServingCast。

## 可重跑入口

从项目根目录运行：

```powershell
python -m supernode_sim serving-schema -o outputs/serving.schema.json
python -m supernode_sim serve examples/serving_synthetic_model.yaml --serving examples/serving_synthetic_requests.yaml -o outputs/serving-example
python skills/supernode-modeling/scripts/analyze_inference_trace.py outputs/serving-example/token-trace.json -o outputs/serving-example/trace-check.json
```

示例全部为 synthetic，不能用于 Ascend 实机误差声明。模型场景与服务负载分开输入，不能把服务字段塞进原严格 `Config`。网页已支持“请求级服务”模式：独立请求编辑、成对示例、校验运行、双快照恢复以及服务指标/时间线；详见 [前端使用](frontend.md)。

## 参数与调度

| 字段 | 默认值 / 必填 | 含义 |
|---|---|---|
| `requests` | 必填、非空 | 每项有唯一非空 `id`、`arrival_s`、`input_tokens`、`output_tokens` |
| `arrival_s` | 必填、≥0 | 输入边界到达秒数；从这里计所有后续等待 |
| `input_tokens`, `output_tokens` | 必填、正整数 | 真实请求长度，输出包括首 token |
| `instance_count` | 1 | 完整、隔离硬件实例的份数 |
| `max_batch_size` | 1 | 每批请求上限，不是 token budget 或在途总上限 |
| `max_batch_wait_s` | 0 | 从队头到达起的主动成批等待；不限制繁忙实例引起的排队 |
| `scheduler_overhead_s` | 0 | 每批接纳后、模型开始前收取一次的声明开销 |
| `slo.ttft_s`, `slo.tpot_s` | 可选、≥0 | 客户端 TTFT、请求平均 TPOT 上限；不使用原固定 batch targets |

未提供到达分布生成器；周期或 Poisson 负载应在外部生成显式 arrival 并保留过程、rate、seed。不能只凭 `request_rate` 名称推断 Poisson，也不能混用 requests/s 与 tokens/s。

模型必须 inference、DP=1。总物理卡数为 `instance_count × hardware.npu_count`，每实例独立复制 host/NIC/网络资源，不模拟跨实例争用。TP/EP/PP、精度、拓扑与校准仍由原引擎校验；服务命令拒绝 PD 和 speculative 开关。每批覆盖 workload 的 batch 与长度向量，执行时清空原 targets；保存原基线，服务 SLO 由 `serving.slo` 判断。

调度按 arrival、输入顺序排序；选择最早可用实例，同样可用时选较小 ID。就绪时间为 `max(实例空闲, 队头arrival)`，截止为 `max(就绪时间, 队头arrival + max_batch_wait_s)`。一批提前到齐就提前发出，否则等到截止；未到请求不能被提前消费。admission 后计一次 overhead，再执行真实长度向量。实例等整批结束后释放，短请求完成不会触发批内补位。

仅在同次调用中缓存完全相同的**有序输入/输出长度向量**，不同长度分布不合并成同总 tokens。scheduler overhead 是声明的 wall-time，不模拟 CPU 争用。显存失败不自动缩批/重试，保留已消耗时间后继续下一独立批；采用整批失败口径，不将失败批的部分 token 算作成功产出。

## 指标与输出

- `admission_wait_s = admission_s - arrival_s`，包括繁忙和主动成批等待。
- `model_start_s = admission_s + scheduler_overhead_s`。
- `server_ttft_s = first_token_s - admission_s`，含本批 overhead。
- `client_ttft_s = admission_wait_s + server_ttft_s`。
- `tpot_s = (last_token_s-first_token_s)/(output_tokens-1)`；单 token 为 null。
- `e2e_s = last_token_s - arrival_s`。

“client”指本输入 arrival 边界，不含未建模 HTTP、tokenizer、发送前等待和响应转发。ServingCast 的 LEAVES_CLIENT 在并发门后，比较前必须对齐边界。

内部减去最早 arrival 作为 `clock_origin_s`，所有调度、时延和吞吐使用相对时钟；输出保留绝对时间作展示，并提供 `*_relative_s` 与 `time_window_relative_s`。大时间原点可能令绝对浮点时间看似重合，不能据此重算零延迟。token trace 使用相对时间并在 metadata 记录原点；输入文件已丢失的精度不能恢复。

分位数为有限样本 P50/P90/P95/P99，采用 `(n-1)*p` 线性插值。admission wait 包含失败请求，其他指标仅包含成功且指标可定义的请求。请求平均 TPOT 的 P99 不等于 pooled token ITL 的 P99。

吞吐窗口从最早 arrival 到最后批次结束，含空闲、排队、overhead 与失败尝试；分子仅成功请求/tokens。无正时长时 rate 为 null。goodput 仅计成功且满足所有适用 SLO 的请求；单 token TPOT 为不适用，没有 SLO 不伪填通过数。

| 输出文件 | 内容 |
|---|---|
| `report.md`, `requests.csv` | 指标解释、分位数和逐请求时钟 |
| `result.json` | 请求、批次、失败、执行来源、硬件副本数、缓存/覆盖和边界 |
| `config.resolved.json` | 原模型/硬件基线，保留原 targets 供审查 |
| `serving.resolved.json` | 实际请求、调度与 SLO 输入 |
| `token-trace.json` | 可由 trace 工具复核的模拟 token 时间，明确不是实测 |

token trace 半开窗口上界取最后完成的下一个浮点数，包含末 token；服务吞吐仍使用精确原时长。全失败且零时长时该文件为 null，不造伪窗口。trace 分析的 admission 分位仅用成功完成 cohort，与服务报告的含失败 cohort 分开解释。

独立 trace 工具要求 TTFT、TPOT 两个阈值同时提供；服务只设置一个阈值时，导出 metadata 保留 `serving_slo` 并标 `slo_exported=false`，联合 SLO 结果以服务 `result.json` 为准。

CLI 返回码：0 为完成且适用 SLO 全满足；3 为存在失败或 SLO 不满足；2 为输入错误。模拟 P99 只对声明到达与调度成立，不能承诺生产尾延迟。实测验证必须同时对齐到达 trace、批策略、实例硬件数、长度分布和计时边界。

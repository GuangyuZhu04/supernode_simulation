# Feedback：按延迟目标搜索推理布局与负载

用户给出目标 TTFT、TPOT，要求在现有模型、拓扑和硬件上选择 TP/SP/EP、Prefill 长度及 Decode batch 时，使用 `feedback`。TPOP 是这里接受的 TPOT 别名；结果以 TPOT（TPOP）展示，并保留同值 `tpop_s` 输出字段。优化目标是：在资源可行且满足全部 SLO 的候选中，**优先最大化 Decode 吞吐量**。

## 运行与输入

从项目根目录运行：

```text
python -m supernode_sim feedback examples/demo_inference.yaml --search examples/feedback_inference.yaml --output outputs/demo-feedback
```

也可直接传入目标与候选列表：

```text
python -m supernode_sim feedback examples/demo_inference.yaml --ttft-s 1 --tpop-s 0.05 --prefill-lengths 32 64 128 --decode-batch-sizes 1 2 4 8 --output-length 4 --max-candidates 128 --engine auto --output outputs/demo-feedback
```

`--search` 是独立搜索规范，不是完整场景，也不是 `sweep` 的 dotted-path axes。字段如下；命令行中给出的同名设置覆盖搜索文件。

| 字段 | 含义及省略时行为 |
|---|---|
| `ttft_s` | 正数，TTFT 上限，单位秒；可沿用基础场景 `targets.ttft_s` |
| `tpot_s` | 正数，TPOT 上限，单位秒；兼容 `tpop_s`，可沿用基础场景 `targets.tpot_s` |
| `prefill_lengths` | 正整数列表，每条请求的输入 token 数 |
| `decode_batch_sizes` | 正整数列表，每 DP 副本的固定 batch |
| `tp_sizes`、`ep_sizes` | 可选正整数列表，限制并行度搜索；默认枚举固定 PP/DP 后用满原设备的合法因子 |
| `sp_enabled` | 可选布尔列表，如 `[false, true]`；默认只枚举当前引擎支持的状态 |
| `output_length` | 每条请求的统一输出 token 数，至少 2；省略时继承基础场景统一的输出长度 |
| `max_candidates` | 候选数量上限；默认沿用基础场景 `targets.max_candidates`；超出上限在仿真前拒绝运行，不截断候选 |
| `engine` | `auto`、`simulate` 或 `streaming`；默认 `auto` |

两个延迟目标都须能从搜索参数或基础场景确定。不要把毫秒直接填入秒字段；例如 50 ms 输入 `0.05`。搜索文件同时提供 `tpot_s` 与 `tpop_s` 时两者必须相等。基础场景若有不同的 `output_lengths`，必须显式指定统一 `output_length`，避免在改变 batch 时悄悄丢失输出负载分布。样例的 `[4, 2]` 因而使用 `output_length: 4`。

未提供长度或 batch 列表时，默认分别取 `1, 2, 4, ...` 至基础场景 `max(input_lengths)` 和 `batch_size`，并包含上界端点。它是便于开始运行的有限候选集，不覆盖这些区间的每个整数。用户显式列表可以超出原配置的长度或 batch；按用户意图设置上下界并在结果中披露。原模型没有定义 context 上限，不能据此声称可以无限寻优或证明最大支持上下文。

## 固定项与联合可行性

保持完整 `model` 与 `hardware` 不变，包括 NPU 数、SU/SO 域、显式链路、路由、带宽、容量和 kernel profiles。同时保持 PP、DP、FSDP、`rank_order`、精度、offload、Engram 等执行设置不变。只搜索 TP、SP、EP、齐次输入长度和 batch；输出 horizon 只有显式设置时才改变，TTFT/TPOT 目标取本次输入。原配置若有 `targets.throughput_tokens_s`，仍按完整执行吞吐检查。

普通模式并行约束是 `TP × PP × EP × DP = NPU 数`。启用 PD 分离时为 `2 × TP × EP = NPU 数`、PP=DP=1；搜索按每池卡数枚举布局，保留并重校验已有 P/D 物理映射。SP 是 `execution.sequence_parallel` 布尔开关：开启时 SP degree 等于 TP，关闭时为 1，与 TP 共用 rank。SP 要求结构化推理、TP>1、PP=1，不能再乘入设备数。模型 attention/KDA heads 的 TP 整除与专家数的 EP 整除等约束仍由原生配置校验。

每个候选均执行完整 Prefill 和 Decode，再判断 TTFT、TPOT、完整执行吞吐目标与内存可行性。这里的 Prefill 长度是每条请求的输入长度；Decode batch 同时也是该候选的 Prefill batch。PD 仍使用相同布局与固定 batch，吞吐窗口包含两池交接；没有 chunked prefill 或 continuous batching，不能把两个独立阶段的最优长度、batch 拼成一个通过联合 SLO 的推荐。

`auto` 在结构化模型且 DP=PP=1、未启用投机解码、Megakernel 或 PD 分离时使用有界内存的流式完整运行，其他场景使用普通仿真。上述 feature 保持原输入并在每个候选中执行；接受数、profile 和物理映射不会自动重新校准。`streaming` 显式模式不支持这些 feature 时直接报错；它的完整运行保留 Prefill、Decode 的依赖与内存准入。独立 `phase='decode'` 会跳过 Prefill 和其临时内存，TTFT 为 null，不能用于本搜索的联合可行性判断。

## 吞吐、延迟与排序口径

对所有 DP 副本中的请求，记请求 `i` 的输出 token 数为 `O_i`、首 token 完成时间为 `first_i`、末 token 完成时间为 `last_i`。用于排序的 Decode 吞吐为：

```text
decode_throughput_tokens_s = Σ(O_i - 1) / (max(last_i) - min(first_i))
```

首 token 属于 Prefill 完成结果，不计入 Decode 输出数。DP=PP=1 时，这个分母对应 Decode 阶段的执行时间；DP/PP>1 时，某些请求开始 Decode 时其他请求可能仍在 Prefill，因此该时间窗可能包含与其他请求 Prefill 的重叠。报告这个口径，不把它解释成隔离设备后的纯 Decode 服务率。

TTFT 取批内各请求首 token 完成时间的最大值；TPOT 先计算每个请求首 token 后的平均完成间隔，再取这些平均值的最大值。它们来自同一次完整执行，不含请求到达队列、冷启动加载或 P99 排队延迟。`output_length=2` 只有一次 Decode 间隔；更长 horizon 的均值与缓存增长不同，比较候选时保持 horizon 一致。

普通 `throughput_tokens_s` 仍为所有输出 token 数除以完整执行时间，含 Prefill；它与本搜索的 Decode 吞吐分开报告。不能用 `batch/TPOT` 替代上述整体时间窗口径，尤其在 DP、多阶段或不同时刻完成的请求中。

“最优”只表示声明候选集合内的最高合格 Decode 吞吐。报告该候选自己的 TP/SP/EP、Prefill 长度、Decode batch、实际 TTFT/TPOT、Decode 吞吐及容量峰值。最长可行 Prefill 与最大可行 batch 是另外的容量边界，可能由不同候选达到；它们不自动等于吞吐最优候选的参数。

Decode 吞吐完全相同时，依次优先更长 Prefill、更大 batch、更低 TTFT、更低 TPOT，再按 TP、EP、SP 开关确定稳定顺序，避免输入列表顺序改变推荐。Pareto 同时比较 Prefill 长度、batch 和 Decode 吞吐：只有另一可行候选在三项上均不差且至少一项更好，才支配当前候选。

## 结果与复核

输出目录包含 `result.json`、`report.md`、`candidates.csv`、`latency_breakdown.csv`、`config.resolved.json`；存在可行推荐时另有 `best.config.json`。结果保留实际搜索范围、每个候选的状态和失败原因，以及可行候选的 Pareto 权衡和各并行布局已验证的长度/batch 边界。报告展示最佳候选 TTFT / TPOT（TPOP）的计算、分域通信和等待分项，CSV 保存所有候选的拆解；可加和规则见 [latency-breakdown.md](latency-breakdown.md)。基础场景在 `config.resolved.json`，最优方案在 `best.config.json`；不要混用。

复核最优配置：

```text
python -m supernode_sim simulate outputs/demo-feedback/best.config.json --output outputs/demo-feedback-best
```

大型结构化配置可通过 Python API 使用流式完整运行复核，避免导出完整事件图：

```python
from supernode_sim.config import load_config
from supernode_sim.feedback import feedback
from supernode_sim.streaming import run_streaming

base = load_config("examples/demo_inference.yaml")
result = feedback(base, {
    "ttft_s": 1.0,
    "tpot_s": 0.05,
    "prefill_lengths": [32, 64, 128],
    "decode_batch_sizes": [1, 2, 4, 8],
    "output_length": 4,
    "max_candidates": 128,
})
# 仅在所选模型是结构化推理且 DP=PP=1 时使用：
# verified = run_streaming(load_config("outputs/demo-feedback/best.config.json"))
```

没有可行候选时，保留失败统计与诊断，明确当前范围没有满足目标的方案；不自动增加硬件、改精度、改变拓扑或把 SLO 放宽。若扩大范围，给出新的边界并重新完整遍历；SP padding、路由与内存准入可能产生台阶，不假设长度/batch 单调，不以二分或不可靠下界漏掉候选。

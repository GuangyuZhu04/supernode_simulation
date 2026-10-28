# 公开推理数据对比与真实性改进

核验日期：2026-10-22。用于“与真实推理对比”“参考 DeepInfer / NVIDIA / Ascend 提升仿真真实性”。先按 [runtime-calibration.md](runtime-calibration.md) 对齐对象，再执行本页的比较检查。公开值及缺失条件保存在 [public-inference-benchmarks.json](../assets/public-inference-benchmarks.json)，它是证据目录，不是 `calibration` 文件或硬件 profile。

目标是改善数值预测时，优先使用 [执行模型拟合](runtime-fitting.md)：已接入非对称 MoE wire dtype/metadata、多 NIC 主机和 collective 服务时间，提供 EP16/32 拟合、EP64/128 留出实验。本页的比较器不替代这些执行模型改进。

## 本轮对比结论

现有引擎能计算固定 batch 的算子、HBM、通信和争用，但公开服务测试通常还包含组件并行、连续调度、融合、缓存或投机路径。相同模型名/卡数不足以直接算误差。先修正可由规格和实现确认的资源约束，再对齐微基准；没有同条件数据时保留 `not_comparable`，而不是将整体吞吐拟合成 NPU efficiency。

| 一手公开资料 | 可用证据 | 对当前仿真的约束 |
| --- | --- | --- |
| [DeepInfer 固定版本 Q&A][deepinfer] | EP64/Attention DP64 的固定验证与动态验证对照；公开表为 **estimated throughput** | 保存每 rank 请求数、验证预算和接受数；不把估算吞吐或满接受 benchmark 吞吐当成实测有效输出。当前不支持置信度驱动的跨请求动态预算 |
| [DeepEP 旧版微基准][deepep] | H800 / 每 GPU CX7 400 Gb/s；EP64，128 tokens/batch、hidden7168、top8；dispatch **173 μs**、combine **314 μs** | 两阶段分别 FP8/BF16；本轮可独立配置。测试代码补充 per-rank 字节公式，但原表运行日志未绑定；按声明假设做微基准拟合。旧版不能当作 DeepInfer 当前依赖的新版性能 |
| [NVIDIA TensorRT-LLM R1 固定版本][trtllm] | B200 8 GPU，FP4，输入 1024 / 输出 2048，并发 3072；公开输出吞吐 **35611.5942 token/s** | CLI TP8/EP8 共享 rank，开启 Attention DP，max batch/rank=384；不是本引擎 TP×EP=64。49152 请求的服务测试也不等于同时启动的固定 batch |
| [Ascend 动态 CPP 固定版本][ascend-cpp] | V3.1-W8A8，Atlas 800T A3 16×64GB，TP8/PP2；128k 输入、并发 1、32k chunk，CPP / 静态 PP 的 TTFT **22.5 / 27.0 s** | 当前固定 batch 无动态 chunk/CPP；K 的精确 token 数和测量边界未明确时保持未知。不能以两数之比给全部 prefill 乘折扣 |
| [DGX H100/H200 硬件指南][dgx] | 每主机 8 个集群 ConnectX-7 端口，每卡最高 400 Gb/s | 区分单 GPU 注入和主机聚合能力；本轮新增独立参数解决该表达缺口。端口规格不是实测有效带宽 |
| [NCCL tests 指标说明][nccl] | `algbw=S/t`；AllReduce 的 `busbw=algbw×2(n−1)/n`，AllGather/ReduceScatter 为 `algbw×(n−1)/n` | 先统一 collective、rank 数与 S 的定义；busbw 不等于交换机逐跳字节率，不直接填到每条物理链路 |

NVIDIA 资料采用 synthetic 请求分布，但在真实硬件上运行，证据状态仍为 `public`；这与本仓库 synthetic 硬件示例不同。来源文档的 commit 也不自动证明测试使用相同 runtime/checkpoint commit。未披露的版本、缓存、KV/通信 dtype、计时边界留空。目录不补造 TTFT、TPOT、P99 或置信区间。

Kimi K3 / 950DT 的现有 10.3 ms 参考继续按 [原有证据说明](kimi-k3-runtime-evidence.md) 使用；它的接受长度折算与组件计时边界，不能替代实际客户端逐 token trace。

## 已修正的网络表达缺口

以前 `cpuX.nic.tx/rx` 始终使用单设备 `scale_out.injection_Bps`，同主机多卡被限制在一份注入带宽中。单 NIC 共享主机适用这个假设，多 NIC 主机需要独立聚合上限。

新增 canonical 配置字段 **`hardware.scale_out.host_injection_Bps`**，单位 byte/s，表示每主机每方向聚合 SO 注入能力。以 DGX H100 的 8×400 Gb/s 端口能力为例：

```yaml
hardware:
  npus_per_host: 8
  scale_out:
    bandwidth_Bps: 5.0e10       # 每 GPU 连接：400 Gbit/s / 8
    injection_Bps: 5.0e10      # 单 GPU 每方向注入上限
    host_injection_Bps: 4.0e11 # 8 个端口的每主机聚合上限
    efficiency: 0.8           # 示例假设；须独立测量，非 NVIDIA 规格
```

这是硬件片段，需合并到已有完整场景，不能单独运行。保留已有模型、卡数、SU 域和负载。新字段只对 `scale_out` 有效；未提供时保持旧行为。主机和单卡上限均乘同一个 efficiency，各自限制资源，不对同一次传输重复乘效率。

内置拓扑显式提供该字段时，默认 SO 域上联为各主机 `min(主机聚合能力, 域内该主机设备数 × min(单卡链路, 单卡注入))` 之和；显式 `uplink_Bps` 优先。explicit 图的边能力仍由 `explicit_links` 决定，不自动扩容。它是聚合带宽模型，未模拟具体 GPU↔NIC 亲和性或 rail 路由。高层 topology DSL 尚无此字段；需要时在编译后的 canonical 硬件/场景添加并保存来源，再 validate。网页可用高级 JSON 编辑。

## 可执行的比较检查

脚本 [compare_inference_benchmarks.py](../scripts/compare_inference_benchmarks.py) 仅依赖 Python 标准库：

```text
python skills/supernode-modeling/scripts/compare_inference_benchmarks.py skills/supernode-modeling/assets/benchmark-comparison.synthetic.json -o outputs/realism-validation/synthetic-comparison.json
```

输入为 `schema_version: 1`，包含 `reference` 和 `candidate` 两个记录；示例是 synthetic，只验证计算。每条记录保留 `id/status/evidence_kind/source/context/metrics`；`evidence_kind` 必须为 measurement、estimate、specification、simulation 或 synthetic：

- `context` 对齐模型及 revision、硬件 SKU/数量/拓扑、目标 runtime 及 backend 版本、权重/activation/KV/通信精度、并行组、长度/batch 作用域/并发/到达过程/缓存状态、投机与 chunked prefill。所有额外字段也参与比较。无法确定的条件用 null；两边都未知也不算匹配。
- `runtime` 表示仿真所对标的软件执行路径；仿真器版本、resolved config 哈希及结果路径写入 `source`。不能为了通过检查，直接复制参考的未知或未实现执行路径到 candidate。
- 每个指标分别声明 `value/unit/phase/timer_boundary/aggregation/token_basis/window`。统一单位后比较；脚本不猜测 ms↔s、每卡↔整池，也不将 mean、batch max、P99 混用。补充窗口/采样条件可放在 `window` 对象中，递归参与匹配；指标对象不接受这七项以外的字段。
- 一致且完整时才计算数值差异；reference=0 时相对误差为空。缺值或条件不一致输出逐字段原因。synthetic/spec/assumed 数据即使能算差异，也不构成真实验证；工具始终不自动拟合或判定精度达标。

从目录选择真实参考时，在仓库根目录构建输入：

```python
import json
from pathlib import Path

assets = Path('skills/supernode-modeling/assets')
catalog = json.loads((assets / 'public-inference-benchmarks.json').read_text(encoding='utf-8'))
print([case['id'] for case in catalog['cases']])
# 指定目录中的实际 id；candidate 从自己的配置/结果/测量日志独立整理。
reference_id = catalog['cases'][0]['id']
reference = next(case for case in catalog['cases'] if case['id'] == reference_id)
candidate = json.loads(Path('outputs/candidate-benchmark.json').read_text(encoding='utf-8'))
out = Path('outputs/realism-validation/comparison-input.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'schema_version': 1, 'reference': reference,
                          'candidate': candidate}, ensure_ascii=False, indent=2), encoding='utf-8')
```

再将生成的 `comparison-input.json` 传给比较脚本。未经补充证据，目录里的不完整参考点应产生 `not_comparable`，即使候选数字恰好相等。比较器检查声明的一致性，不认证来源内容；必须人工核对原始日志、实现能力和字段来源。实测逐 token 日志先交给 [trace 分析工具](runtime-calibration.md#处理请求实测-trace)，其 mean/P99 不能直接映射为引擎的 batch max。

## 后续提高数值真实性的顺序

| 优先级 | 当前差异及影响 | 需要的数据与实现 | 验收方式 |
| --- | --- | --- | --- |
| P1 重叠与格式 | 本轮已支持不同 wire dtype/metadata 和服务时间；尚无真实 codec、分块融合计算通信重叠 | 分块 dispatch/GEMM/combine 依赖与目标内核的 scale 格式；更多 payload/batch/EP 微基准 | 核对每 rank payload、逐跳字节、端点与共享资源；先单 collective 后并发 |
| P1 专家负载 | 均匀期望路由对小 batch 给出分数专家工作，缺少热点、冗余专家和 EPLB 迁移 | 逐层 token→expert/rank 分布或明确随机模型、实际副本及负载平衡策略 | 保持 token/top-k 守恒，比较最忙 rank、HBM 与通信长尾；均值不能代替热点 |
| P1 kernel 身份 | 目前 profile 只匹配 FLOPs/bytes/dtype/phase，不保证形状、融合、芯片和 runtime 相同 | tensor shape/layout、SKU/runtime revision、图模式等 profile 身份 | 拒绝不同身份套用；独立 prefill/decode 留出点比较 |
| P2 serving | 固定 batch 未包含连续准入、chunked prefill、动态 CPP 和客户端队列 | 请求到达/完成 trace、token budget、抢占/重算、CPU 调度事件 | 分开 service time 与 queue time；联合 TTFT/TPOT goodput 和过载失败率 |
| P2 非对称 PD / 投机 | 同布局双池和固定接受数未覆盖 DeepInfer 大小 P/D 池、动态验证 | 异构组件 communicator、状态重分片、跨批聚合、实际提交数与 draft 成本 | 首 token、交接、首 decode 间隔及有效输出分别验证 |

上述项目是尚待实现的差距，不是新增的可用 feature。先用容量/字节守恒和共享瓶颈验证，再校准算子/collective，最后才对端到端做留出验证。记录拟合集与留出集、绝对/相对误差和误差随 batch/上下文变化；未验证的场景继续标 `uncalibrated`。硬件规格、数学单元测试和同条件实机留出测试分别回答不同问题。

[deepinfer]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md
[deepep]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/docs/legacy.md
[trtllm]: https://github.com/NVIDIA/TensorRT-LLM/blob/b6261862419c33d6ce2313aff1e7116067d6037d/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
[ascend-cpp]: https://github.com/vllm-project/vllm-ascend/blob/c361fb4990aa9c32a9609e34595d787bb4144f2a/docs/source/tutorials/features/dynamic_chunked_pipeline_parallel.md
[dgx]: https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html
[nccl]: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

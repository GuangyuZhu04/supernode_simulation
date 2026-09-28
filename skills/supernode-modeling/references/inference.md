# 固定 batch 推理、KV 与 Engram

用 `examples/demo_inference.yaml` 验证可运行链路；`examples/inference_10t.yaml` 表示显式假设的 10T 场景。DeepSeek/Kimi/GLM 等名称本身不足以确定 KV 结构；公开结构不完整时保留 synthetic 名称、假设和来源。

## 负载与性能指标

`workload.scenario: inference`；设置每 DP 副本的 `batch_size`, `input_length`, `output_length`，或用 `input_lengths` / `output_lengths` 给出逐序列正整数列表，长度必须与 batch 一致。改变已展开配置的 scalar 长度时，应同时清空或更新对应列表，避免旧列表覆盖新意图。

这是预加载权重后的固定 batch 执行：完成的序列退出 decode，但没有新请求到达，不模拟 continuous batching、服务排队或冷启动权重加载。

- TTFT：执行开始到首个输出 token 完成；报告中聚合 TTFT 取批内序列的最大值。
- TPOT：每序列首 token 之后的平均完成间隔；聚合值取这些序列平均值的最大值，并保留逐 token 间隔。
- `throughput_tokens_s` 为所有 DP 副本输出 tokens/s；输入吞吐另报，分母均为整次执行时间。
- 只生成一个 token 的序列没有 TPOT；不要据此做 TPOT SLO 判断。指标不是 P99 请求延迟。

## KV 结构、精度与放置

`model.attention_kind` 支持 `mha/gqa/mqa/latent`。标准 KV 由 `kv_heads`, `head_dim`, 层数、实际长度决定；latent KV 由 `kv_latent_dim + kv_rope_dim` 表示每 token 保存向量。标准 KV 按 TP 切分 KV heads，不能整除时取整；latent KV 在 TP rank 间复制。不能再给已压缩 latent 结构套一次任意“KV 压缩率”。

混合 MLA/KDA 使用 `model.architecture.attention_by_layer`，不对全部层套用同一个 `attention_kind`。KDA 保存序列长度无关的 recurrent matrix 和短卷积状态，MLA 才按序列长度增加 latent KV。Kimi K3 参数和精度样例见 [ascend950-kimi-k3.md](ascend950-kimi-k3.md)，按组件输出见 [memory-breakdown.md](memory-breakdown.md)。

`execution.precision.kv` 决定元素字节数；`execution.features.kv_page_tokens`, `kv_page_entry_bytes`, `kv_scale_block`, `scale_bytes` 决定分页和 FP8/int8 元数据。逻辑容量、分页填充、page table、scale 分开核算。

`execution.features.kv_offload` 为 `none/host/remote`，`kv_offload_fraction` 为卸载比例。`remote` 要求 `hardware.remote.enabled: true`。KV 增长按增量申请存储，decode 读取仍活跃序列的缓存；取回缓冲需要保持到使用它的 kernel 完成。

## n-gram / Engram

使用 `execution.features.engram`：

```yaml
enabled: true
table_bytes: 1e12
queries_per_token: 1
record_bytes: 4096
hit_rate: 0.0
cache_bytes: 0
location: host
iops: 1e7
latency_s: 1e-6
layers: [0]
```

`hit_rate > 0` 必须有明确的 `cache_bytes`。表放在 host 时，每个参与主机保存一份；缓存容量是每参与 NPU 的容量。remote 表要求启用远程内存。查询率、记录大小、命中率和随机访问 IOPS 是不同约束；不要只用顺序 DDR 带宽估算随机查询性能。

KV offload 与查询会共享实际 host DRAM、host 链路或 remote 资源。评估组合时同时打开 Feature 仿真，不能把独立运行的收益相加。该模型评估通信与资源，不预测 n-gram 表带来的知识能力或模型质量收益。

## MoE 与模型近似

`total_params` 决定驻留权重；`active_params` 决定每 token 参数计算量。200B–800B 是活跃参数范围，不是 activation 容量。专家路由均匀、专家计算均衡，专家权重访存按均匀 top-k 触达专家的期望估算。真实路由倾斜需额外 trace 或实现扩展，不能据现有结果声称覆盖热点。

当前是层级粗粒度算子图，TP reduction 位于层边界，Attention 假设融合/流式执行；不存在常驻的完整二次 attention score 矩阵。读实际输出中的 assumptions，结合 kernel profile 和平台实测校准后再讨论绝对精度。

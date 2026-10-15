# 超节点资源建模报告

状态：**infeasible**；模式：`simulate`。

大图使用精简导出，数值仿真仍包含全部事件。所有容量、通信、性能、组件状态和并发峰值汇总均保持完整。

| 明细 | 原始行数 | 导出行数 | 未导出行数 | 文件/处理 |
|---|---:|---:|---:|---|
| 事件时间线 | 0 | 0 | 0 | events.csv |
| 物理张量 | 2110848 | 360000 | 1750848 | tensors.csv |
| 原始申请/释放日志 | 197184 | 0 | 197184 | 保留精确峰值汇总 |

events.csv / timeline.json 保留模型 kernel、token/step 完成标记和模型张量申请事件，省略 collective 搬运与缓冲控制事件。tensors.csv 省略临时 `buffers` 明细；这些缓冲仍完整计入组件状态和并发峰值。

需要全部原始事件、张量和有序内存日志时，用相同配置在 estimate/simulate 命令添加 `--full-detail` 重新导出。原始日志未写入其他隐藏文件。

模型：`Kimi-K3-text-backbone-FP8-Muon-offload-capacity-study`；场景：`training`；权重：`fp8`。

硬件：`Ascend 950DT / Atlas 950 — 64-NPU reference slice`；NPU 数量：64；每颗 HBM：144 GB。

容量按张量形状推导；激活与耗时仍为解析模型，尚未通过整机实测校准。`feasible` 表示在本模型假设下满足约束，不代表真实整机部署验证。

**Muon 训练限制：尚未实现分布式 Muon 正交化。当前训练步时间是本地更新与通信的代理模型，不能作为已验证的生产训练性能。**

## 性能与约束

| 指标 | 数值 |
|---|---:|
| 理想执行时间必要下界 / s | 13,518.3 |

## 全局逻辑容量

单位为十进制 TB；实际分片、复制和生命周期见设备明细。

| 组成 | TB |
|---|---:|
| weight | 2.77948 |
| gradient | 5.55897 |
| master_weight | 11.1179 |
| optimizer_momentum | 0.00945196 |
| optimizer_variance | 0.00945196 |
| muon_momentum | 11.1085 |
| total | 30.5838 |

## 存储分解：组件 × 状态

口径：**中断前已观测并发峰值**。表格使用十进制 GB（1 GB = 10⁹ byte）；CSV 和 JSON 使用 byte。

### 全模型逻辑参数（分片和复制之前）

结构清单按唯一参数所有权计数；共享权重别名不重复计入。精确形状、参数量、层号、分片标记和别名见 model_parameters.csv。

| 组件 | 总参数 / B | 每 token 活跃参数 / B | 独立参数张量数 | 层数 |
|---|---:|---:|---:|---:|
| WordEmbedding | 1.17441 | 7.168e-06 | 1 | 1 |
| PreNorm | 0.000666624 | 0.000666624 | 93 | 93 |
| PostNorm | 0.000673792 | 0.000673792 | 94 | 93 |
| AttnRes | 0.00268083 | 0.00268083 | 374 | 93 |
| KDA | 30.6181 | 30.6181 | 966 | 69 |
| Dense | 0.726663 | 0.726663 | 3 | 1 |
| MoE | 2,740.21 | 66.0937 | 1,012 | 92 |
| MLA | 5.57271 | 5.57271 | 192 | 24 |
| Output | 1.17441 | 1.17441 | 1 | 1 |

### 全模型逻辑状态容量（分片和复制之前）

以下状态由编译器按具体参数和优化器归属给出；没有从物理副本或合并状态反推 Adam/Muon。单位：GB。

| 组件 | 梯度 | 主权重 | 动量 | Muon 动量 | 二阶矩 | 权重 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AttnRes | 0.00536166 | 0.0107233 | 0.00536166 | 0.00536166 | 0.00536166 | 0.00268083 | 0.0348508 |
| Dense | 1.45333 | 2.90665 | 0 | 2.90665 | 0 | 0.726663 | 7.99329 |
| KDA | 61.2362 | 122.472 | 0.0441512 | 122.428 | 0.0441512 | 30.6181 | 336.843 |
| MLA | 11.1454 | 22.2908 | 0.000196608 | 22.2906 | 0.000196608 | 5.57271 | 61.3 |
| MoE | 5,480.43 | 10,960.9 | 0.00164864 | 10,960.9 | 0.00164864 | 2,740.21 | 30,142.4 |
| Output | 2.34881 | 4.69762 | 4.69762 | 0 | 4.69762 | 1.17441 | 17.6161 |
| PostNorm | 0.00134758 | 0.00269517 | 0.00269517 | 0 | 0.00269517 | 0.000673792 | 0.0101069 |
| PreNorm | 0.00133325 | 0.0026665 | 0.0026665 | 0 | 0.0026665 | 0.000666624 | 0.00999936 |
| WordEmbedding | 2.34881 | 4.69762 | 4.69762 | 0 | 4.69762 | 1.17441 | 17.6161 |
| **总计** | 5,558.97 | 11,117.9 | 9.45196 | 11,108.5 | 9.45196 | 2,779.48 | **30,583.8** |

### 物理内存池容量与峰值

容量和分解均为每个内存池的数值；相同容量与分解的内存池合并展示，完整设备明细保留在 CSV。

| 池组 | 等价内存池 | 数量 | 单池容量 / GB | 单池固定驻留 / GB | 单池峰值或驻留 / GB |
|---|---|---:|---:|---:|---:|
| G1 | cpu0.dram … cpu7.dram (8 pools) | 8 | 4,000 | 3,508.17 | 3,508.17 |
| G2 | npu0.hbm … npu63.hbm (64 pools) | 64 | 144 | 173.963 | 173.963 |

执行未完成：已观测峰值仅覆盖成功申请的内存，不能用来证明完整任务能运行。初始驻留和失败诊断仍须单独检查。

### 每池同一峰值时刻的构成

每组中各组件和状态可以相加得到上表的单池总值。不同池的峰值时刻可能不同。

**G1 · cpu0.dram … cpu7.dram (8 pools) · 单池 GB**

| 组件 | 主权重 | 动量 | Muon 动量 | 二阶矩 | 合计 |
|---|---:|---:|---:|---:|---:|
| AttnRes | 0.0857866 | 0.0428933 | 0.0428933 | 0.0428933 | 0.214467 |
| Dense | 2.90665 | 0 | 2.90665 | 0 | 5.81331 |
| KDA | 124.245 | 0.0443985 | 124.201 | 0.0443985 | 248.535 |
| MLA | 32.4655 | 0.00157286 | 32.4639 | 0.00157286 | 64.9325 |
| MoE | 1,580.18 | 0.0131891 | 1,580.16 | 0.0131891 | 3,160.36 |
| Output | 4.69762 | 4.69762 | 0 | 4.69762 | 14.0929 |
| PostNorm | 0.0215613 | 0.0215613 | 0 | 0.0215613 | 0.064684 |
| PreNorm | 0.021332 | 0.021332 | 0 | 0.021332 | 0.0639959 |
| WordEmbedding | 4.69762 | 4.69762 | 0 | 4.69762 | 14.0929 |
| **总计** | 1,749.32 | 9.54019 | 1,739.78 | 9.54019 | **3,508.17** |

**G2 · npu0.hbm … npu63.hbm (64 pools) · 单池 GB**

| 组件 | 缓冲 | 梯度 | 量化元数据 | 运行时预留 | 权重 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| AttnRes | 0 | 0.00536166 | 8.3776e-05 | 0 | 0.00268083 | 0.00812627 |
| Dense | 0 | 0.181666 | 0.00283853 | 0 | 0.0908329 | 0.275337 |
| KDA | 0 | 7.76533 | 0.121333 | 0 | 3.88266 | 11.7693 |
| MLA | 0 | 2.02909 | 0.0317046 | 0 | 1.01455 | 3.07534 |
| MoE | 0 | 98.761 | 1.54314 | 0 | 49.3805 | 149.685 |
| Output | 0 | 0.293601 | 0.00458752 | 0 | 0.146801 | 0.444989 |
| PostNorm | 0 | 0.00134758 | 2.1056e-05 | 0 | 0.000673792 | 0.00204243 |
| PreNorm | 0 | 0.00133325 | 2.0832e-05 | 0 | 0.000666624 | 0.0020207 |
| System | 0.256 | 0 | 0 | 8 | 0 | 8.256 |
| WordEmbedding | 0 | 0.293601 | 0.00458752 | 0 | 0.146801 | 0.444989 |
| **总计** | 0.256 | 109.332 | 1.70832 | 8 | 54.6661 | **173.963** |

### 各组件独立峰值（不可相加）

下表每项取该组件在整段已观测执行中的最大值，各项可能出现在不同时间；相加不能得到设备峰值。

| 池组 | 组件 | 单池组件独立峰值 / GB |
|---|---|---:|
| G1 | AttnRes | 0.214467 |
| G1 | Dense | 5.81331 |
| G1 | KDA | 248.535 |
| G1 | MLA | 64.9325 |
| G1 | MoE | 3,160.36 |
| G1 | Output | 14.0929 |
| G1 | PostNorm | 0.064684 |
| G1 | PreNorm | 0.0639959 |
| G1 | WordEmbedding | 14.0929 |
| G2 | AttnRes | 0.00812627 |
| G2 | Dense | 0.275337 |
| G2 | KDA | 11.7693 |
| G2 | MLA | 3.07534 |
| G2 | MoE | 149.685 |
| G2 | Output | 0.444989 |
| G2 | PostNorm | 0.00204243 |
| G2 | PreNorm | 0.0020207 |
| G2 | System | 8.256 |
| G2 | WordEmbedding | 0.444989 |

所有内存池同一时刻的合计峰值为 39,199 GB，发生于 0 s。该数值按全局时间线重放得到，未相加各池独立峰值；不同内存类型的容量不能互相替代。

张量、层号、形状、精度、组件来源与申请/释放时刻见 tensors.csv；组件状态交叉表见 component_memory.csv，状态汇总见 state_memory.csv，逐池峰值见 memory_peaks.csv。

缺失组件归属保留为 `Unassigned`；通用缓冲和运行时归为 `System`。已知优化器按其状态语义拆分；无法确定语义的合并状态才保持未拆分，不推定 Adam/Muon 的具体张量。

## 通信

物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。

| 网络域 | 物理传输 / GB |
|---|---:|
| SU | 3,284.09 |
| D2H | 6,997.27 |
| H2D | 3,607.97 |

| 应用语义 | 传输 payload / GB |
|---|---:|
| all_reduce | 2,830.3 |
| dispatch | 226.895 |
| combine | 226.895 |
| optimizer_gradient | 6,997.27 |
| optimizer_weight | 3,498.63 |
| weight_scale_transfer | 109.332 |

应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。

## 主要资源下界

| 资源 | 独占服务时间下界 / s |
|---|---:|
| cpu0.compute | 13,518.3 |
| cpu1.compute | 13,518.3 |
| cpu2.compute | 13,518.3 |
| cpu3.compute | 13,518.3 |
| cpu4.compute | 13,518.3 |
| cpu5.compute | 13,518.3 |
| cpu6.compute | 13,518.3 |
| cpu7.compute | 13,518.3 |
| cpu0.dram | 22.2721 |
| cpu1.dram | 22.2721 |

缓存容量和 MTE 通道数需要 kernel/硬件校准；当前输出工作集与搬运量下界。

## 未满足约束

```json
[
  {
    "code": "initial_memory_exceeded",
    "message": "Resident state exceeds memory capacity",
    "pool": "npu0.hbm",
    "required_bytes": 173962743612.0,
    "capacity_bytes": 144000000000.0
  }
]
```

## 参数来源

- `*`：assumed；Unspecified runtime and workload values are analytical assumptions, not measured Huawei/Kimi performance.
- `model.architecture`：public；Official Kimi K3 config and modeling_kimi_linear.py tensor dimensions. Text backbone only; local source snapshots and hashes in docs/sources.；https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
- `model.reported_total_params`：public；Rounded official 2.8T and 104B counts; used for comparison only, not rescaling derived tensor counts.；https://github.com/MoonshotAI/Kimi-K3
- `model.architecture.kda.state_dtype`：assumed；FP32 recurrent state layout; validate dtype against deployed kernel.
- `model.architecture.kda.chunk_size`：assumed；64-token chunk boundary activation retention; runtime/kernel dependent.
- `model.architecture.kda.conv_state_tokens`：assumed；Four allocated convolution samples; mathematical minimum is three.
- `hardware.npu_count`：public；64 NPUs per compute cabinet; this scenario is one cabinet slice of up to8192-NPU Atlas950, not the full domain.；https://www.huawei.com/en/news/2026/3/mwc-superpod-ai
- `hardware.su_size`：assumed；The simulated SU contains one64-NPU slice; hardware maximum is8192, not64.
- `hardware.npus_per_host`：assumed；Eight NPUs per modeled host; host grouping is not an official BOM.
- `hardware.topology`：assumed；Full-mesh is a logical nonblocking fabric abstraction with shared endpoint injection; not a claim of physical all-to-all cabling.
- `hardware.npu.hbm_bytes`：public；950DT HiZQ2.0:144GB/chip. Here GB is explicitly interpreted as1e9bytes; source unit ambiguity retained.；https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech
- `hardware.npu.hbm_Bps`：public；950DT HiZQ2.0 nominal memory access bandwidth4TB/s; no measured sustained bandwidth supplied.；https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech
- `hardware.npu.flops.fp8`：public；950 series announced1PFLOPS FP8. Source does not specify dense/sparse condition; realized efficiency is assumed.；https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech
- `hardware.npu.flops.fp16`：assumed；0.5PFLOPS FP16 analytical placeholder; not supplied by cited source.
- `hardware.npu.flops.bf16`：assumed；0.5PFLOPS BF16 analytical placeholder; not supplied by cited source.
- `hardware.npu.flops.fp32`：assumed；50TFLOPS FP32 analytical placeholder; not supplied by cited source.
- `hardware.npu.efficiency`：assumed；50 percent effective compute efficiency; no calibration benchmark.
- `hardware.npu.runtime_bytes`：assumed；8GB runtime reservation per NPU, not actual framework allocator measurement.
- `hardware.npu.workspace_bytes`：assumed；256MB additional kernel workspace per NPU, beyond explicit component buffers.
- `hardware.scale_up`：assumed；Official total interconnect2TB/s; model assumes1TB/s per direction and80percent efficiency. Source does not resolve directionality.2.1us is represented once per logical path.
- `hardware.cpu`：assumed；4TB DDR and512GB/s per modeled host;4TFLOPS CPU with50percent efficiency and1e8IOPS; illustrative host configuration, not Ascend950 BOM.
- `hardware.host_link`：assumed；64GB/s per direction shared by eight modeled NPUs per host,85percent efficiency,2us; requires actual host interconnect measurements.
- `hardware.scale_out`：assumed；400Gb/s=50GB/s per modeled host NIC per direction,80percent efficiency,5us; unused in this single-slice case.
- `execution.precision`：assumed；Design reprecision of official model shapes, not the published MXFP4-routed/BF16 checkpoint; storage/compute/KV/activation are independent.
- `execution.parallelism`：assumed；TP8 EP8 PP1 DP1. Dense/shared components replicate across EP; routed experts shard. Uniform routing; no production deployment trace.
- `workload`：assumed；Trainingmicrobatch8,sequence128,oneoptimizerstep; structural capacity case not published trainingbatch.
- `execution.optimizer`：assumed；Muon matrices and per-head QKV match official report;5NSiterations,two workspace copies,FP32 states andAdam fallback for embedding/output/vectors are explicit modeling policies, not disclosed exactK3recipe.
- `model.architecture.mla.cache_layout`：assumed；Compressed MLA cache optimization study: retain 512 latent + 64 additional key elements per token/layer, replicated across TP. Official HF reference instead caches expanded K/V. Kernel compute/runtime is not calibrated to either layout.；https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py
- `hardware.calibration`：assumed；Huawei Ascend 950DT / Atlas 950 64-NPU cabinet slice, based on public roadmap. Nominal chip HBM/FP8 fields sourced; topology direction interpretation, CPU/DDR/H2D/SO and efficiencies are unmeasured assumptions. See docs/sources/README.md.；https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech

## 假设与复现

- Parameter and state capacities derive from named tensor shapes, explicit tying and TP/EP/FSDP placement.
- Activation formulas assume fused normalization, SiLU/SiTU GLU and streaming attention; actual allocator/kernel traces may differ.
- Expert routing and token distribution are balanced expectations; expert skew and capacity padding need workload calibration.
- Per-component kernels use roofline timing; pipeline follows non-interleaved 1F1B in training and fixed ragged batch in inference.
- KV offload applies to growing attention KV; KDA recurrent matrices and convolution histories stay in device memory.
- Muon keeps one matrix momentum state; vectors/norms/embeddings/output use Adam m/v fallback. Newton-Schulz time and scratch are estimates.
- FP8/FP4 weight scales use per-tensor block rounding; precision overrides are storage studies unless identified as a released model format.
- Ragged inference stops compute and KV reads for finished sequences; their allocated caches remain reserved through the fixed batch.
- Pipeline transfers materialize source and receive buffers; the receive is retained through the consuming stage pass.
- Prefill projects logits only for the final input position of each live sequence; training projects all sequence positions.
- Muon orthogonalization reserves full per-matrix/per-head scratch. Distributed matrix assembly/orthogonalization is not implemented; sharded Muon timing is a proxy, not production validation.
- AttnRes retains residual blocks across layers; FP32 reference attention workspace is included. PP>1 is rejected until residual forwarding is modeled.
- Ascend950DT/Atlas950 announced roadmap profile, not a measured installed platform.950DT Q4 2026 availability was stated in2025;2026MWC confirmed cabinet scale, not measured performance.
- 64NPUs are a cabinet slice inside an up-to8192-NPU design. All SI capacity/bandwidth values use decimalbytes; perchip144GB interpretation does not silently adopt inconsistent published1152TBaggregate.
- Text backbone tensor counts are shape-derived, not rounded2.8T budgets. Vision encoder/projector and training MTP are excluded; published training report has1MTP but releasedconfig has0.
- Official model uses SiTU-GLU, gated MLA and KDA+AttnRes; generic kernel timing, fused working sets and uniform expert routing remain uncalibrated.
- Released checkpoint has MXFP4 routed experts with32-element groups anduint8scale; primaryFP8/FP16cases are controlled reprecision studies, not exact checkpoint storage.
- Latency/throughput are analytical predictions, not Kimi production benchmarks. No queueing/continuousbatching/prefixcaching. TTFT begins after weights loaded.
- Detailed productionK3 training uses MoonEP, virtualPP,PipelineZeRO2,activationoffload/remoteoffload andP2PMuon; genericFSdP/optimizer baseline does not reproduce those mechanisms.
- 64-NPU FP8weights+FP16gradients may exceed144GB/NPU despite optimizeroffload. Preserve infeasible outcome; do not inflateHBM to force fit. CPUoffload variant is a designstudy and is not publishedproductionMuondeployment.
- MLA cache_layout=compressed is an explicit optimized cache assumption, not the official HF reference cache implementation. Compare expanded layout using examples/sweep_kimi_k3_kv_layout.yaml. NoPE skips positional encoding but retains the 64-dimensional key projection.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。
- 配置 SHA256：`1b7167b48beea88f9b3126239073859b4eba70f5522b7380f646e6a57a340e0a`。

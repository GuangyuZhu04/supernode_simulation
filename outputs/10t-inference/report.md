# 超节点资源建模报告

状态：**feasible**；模式：`simulate`。

模型：`synthetic-10t-latent-kv-moe`；场景：`inference`；权重：`fp8`。

## 性能与约束

| 指标 | 数值 |
|---|---:|
| 总执行时间 / s | 2.96626 |
| 执行 TTFT / s | 2.21044 |
| 批内最大平均 TPOT / s | 0.293149 |
| 聚合吞吐 / tokens/s | 2.02275 |
| 输入吞吐 / tokens/s | 64.7279 |
| 每副本吞吐 / tokens/s | 2.02275 |
| 理想执行时间必要下界 / s | 2.96585 |

## 全局逻辑容量

单位为十进制 TB；实际分片、复制和生命周期见设备明细。

| 组成 | TB |
|---|---:|
| weight | 10 |
| kv_data | 1.03219e-05 |
| kv_scales | 3.2256e-07 |
| kv_page_tables | 8.96e-09 |
| engram_table | 1 |

## 存储与通信

| 内存池 | 峰值或固定驻留 / GB |
|---|---:|
| npu0.hbm | 718.368 |
| npu1.hbm | 718.368 |
| npu2.hbm | 718.357 |
| npu3.hbm | 718.357 |
| npu4.hbm | 715.846 |
| npu5.hbm | 715.846 |
| npu6.hbm | 715.846 |
| npu7.hbm | 715.846 |
| npu8.hbm | 718.376 |
| npu9.hbm | 718.376 |
| npu10.hbm | 718.361 |
| npu11.hbm | 718.361 |
| npu12.hbm | 715.846 |
| npu13.hbm | 715.846 |
| npu14.hbm | 715.846 |
| npu15.hbm | 715.846 |
| cpu0.dram | 1,000.01 |
| cpu1.dram | 1,000 |
| cpu2.dram | 1,000.01 |
| cpu3.dram | 1,000 |

物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。

| 网络域 | 物理传输 / GB |
|---|---:|
| H2D | 0.0488305 |
| D2H | 0.0186291 |
| SU | 10.276 |
| SO | 0.10276 |

| 应用语义 | 传输 payload / GB |
|---|---:|
| lookup | 0.00321126 |
| kv_write | 0.0186291 |
| dispatch | 3.08281 |
| combine | 3.08281 |
| tp | 4.11042 |
| pipeline | 0.0256901 |
| kv_read | 0.0456192 |

应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。

## 主要资源下界

| 资源 | 独占服务时间下界 / s |
|---|---:|
| npu0.hbm | 1.47981 |
| npu1.hbm | 1.47981 |
| npu8.hbm | 1.47981 |
| npu9.hbm | 1.47981 |
| npu2.hbm | 1.31255 |
| npu3.hbm | 1.31255 |
| npu10.hbm | 1.31255 |
| npu11.hbm | 1.31255 |
| npu4.hbm | 1.14531 |
| npu5.hbm | 1.14531 |

缓存容量和 MTE 通道数需要 kernel/硬件校准；当前输出工作集与搬运量下界。

## 参数来源

- `*`：assumed；Synthetic 10T MoE with latent KV. No DeepSeek/Kimi/GLM architecture or real 2 TB HBM device is claimed.

## 假设与复现

- Fixed ragged batch; finished sequences leave decode but no new requests arrive.
- EP lanes own round-robin sequences; uniform expert routing and balanced expert compute.
- Dense parameters are replicated over EP; expert parameters are sharded over EP.
- Expert HBM reads use expected distinct experts touched by uniform top-k routing; active_params controls FLOPs.
- Per-layer parameters are uniform within dense/expert layer categories.
- Standard KV shards heads (ceil); latent KV is replicated across TP ranks.
- TP boundary hidden states are replicated; PP and MoE messages carry full hidden vectors per TP lane.
- Weights are preloaded; cold-start weight loading and request queueing are excluded.
- Attention is fused/streaming: quadratic score matrices are not resident.
- KV read precision conversion is fused into attention; new KV storage conversion has an explicit kernel and buffer.
- Engram table is replicated on each participating host; cache_bytes is per participating NPU.
- TP attention and FFN reductions are serialized at layer end; this is a coarse operator schedule.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。
- 配置 SHA256：`224e03259a9a7f29525e8e072cfd6381ec72e94f136d73a15464ae97d42b8284`。

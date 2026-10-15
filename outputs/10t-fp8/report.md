# 超节点资源建模报告

状态：**estimated**；模式：`estimate`。

模型：`synthetic-10t-moe-fp8`；场景：`training`；权重：`fp8`。

## 性能与约束

| 指标 | 数值 |
|---|---:|
| 理想执行时间必要下界 / s | 716.912 |

解析结果提供必要下界，尚未证明资源争用下的 SLO 可行性。

## 全局逻辑容量

单位为十进制 TB；实际分片、复制和生命周期见设备明细。

| 组成 | TB |
|---|---:|
| weights | 10 |
| gradients | 20 |
| master_weights | 40 |
| optimizer_states | 80 |
| total | 150 |

## 存储与通信

| 内存池 | 峰值或固定驻留 / GB | 配置容量 / GB |
|---|---:|---:|
| npu0.hbm | 1,896.6 | 4,000 |
| cpu0.dram | 30,000 | 64,000 |
| npu1.hbm | 1,896.6 | 4,000 |
| npu2.hbm | 1,896.6 | 4,000 |
| npu3.hbm | 1,896.6 | 4,000 |
| npu4.hbm | 1,896.6 | 4,000 |
| cpu1.dram | 30,000 | 64,000 |
| npu5.hbm | 1,896.6 | 4,000 |
| npu6.hbm | 1,896.6 | 4,000 |
| npu7.hbm | 1,896.6 | 4,000 |
| npu8.hbm | 1,896.6 | 4,000 |
| cpu2.dram | 30,000 | 64,000 |
| npu9.hbm | 1,896.6 | 4,000 |
| npu10.hbm | 1,896.6 | 4,000 |
| npu11.hbm | 1,896.6 | 4,000 |
| npu12.hbm | 1,896.6 | 4,000 |
| cpu3.dram | 30,000 | 64,000 |
| npu13.hbm | 1,896.6 | 4,000 |
| npu14.hbm | 1,896.6 | 4,000 |
| npu15.hbm | 1,896.6 | 4,000 |

物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。

| 网络域 | 物理传输 / GB |
|---|---:|
| SU | 60,969.8 |
| SO | 243,750 |
| D2H | 20,000 |
| H2D | 10,312.5 |

| 应用语义 | 传输 payload / GB |
|---|---:|
| fsdp_weights | 60,000 |
| quantization_metadata | 2,187.5 |
| tensor_parallel | 32.2123 |
| pipeline_activation | 0.134218 |
| fsdp_gradients | 60,000 |
| optimizer_offload | 30,000 |

应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。

按吞吐或训练步 SLO 得到的整任务时间窗口：1,000 s。各资源平均速率必要下界见 resources.csv；还需满足局部依赖与争用约束。

## 主要资源下界

| 资源 | 独占服务时间下界 / s |
|---|---:|
| cpu1.nic.tx | 716.912 |
| cpu2.nic.rx | 716.912 |
| cpu3.nic.tx | 716.912 |
| cpu0.nic.rx | 716.912 |
| link:su0.gateway>so.spine:SO | 358.456 |
| link:so.spine>su1.gateway:SO | 358.456 |
| link:su1.gateway>so.spine:SO | 358.456 |
| link:so.spine>su0.gateway:SO | 358.456 |
| cpu0.dram | 212.695 |
| cpu1.dram | 212.695 |

缓存容量和 MTE 通道数需要 kernel/硬件校准；当前输出工作集与搬运量下界。

## 参数来源

- `*`：assumed；Synthetic 10T capacity anchor. 4 TB HBM per device and 64 TB DRAM per host are hypothetical, not product specifications.

## 假设与复现

- Declared parameter-budget model: parameters and active work are uniform over layers and experts.
- EP dense weights are replicated; expert weights and token work are evenly distributed across EP ranks.
- MoE communication uses uniform routing and a continuous MoE-layer fraction; no expert imbalance is inferred.
- Activation ledger assumes streaming causal attention without a full attention-score matrix.
- Hidden boundaries are replicated over TP; saved activation elements/token/layer = 2H + intermediate/TP*(1 + MoE_fraction*top_k); checkpoint retains H.
- Activation/wire dtype differences add explicit packing/unpacking and retained communication buffers; FSDP tensors retain their own weight/gradient dtype.
- Forward FLOPs = 2*active_parameters*tokens plus causal attention; backward = 2*forward.
- Adam update uses 14 FLOPs/parameter; precision conversion uses 2 FLOPs/parameter; calibrate to measured kernels.
- NPU scalar updates/conversions use the FP32 profile and share a stream with matrix compute; if unavailable, use selected compute throughput.
- Optimizer starts after all backward phases; optimizer/backward overlap is not modeled.
- FP8 scale metadata uses continuous block counts; tensor alignment/padding requires profile calibration.
- Collective staging and precision-conversion buffers are additional to logical tensor capacity.
- Total and active parameters are declared budgets; generated layer kernels are architectural approximations.
- Active parameters derive as dense 200B plus 306.25B routed experts per token.
- This small-rank giant-memory profile keeps the event graph practical for capacity validation.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。
- 配置 SHA256：`c4a7c02588167762d1b1291d54df4ffba759587d88d81fd12497220ab84da4ae`。

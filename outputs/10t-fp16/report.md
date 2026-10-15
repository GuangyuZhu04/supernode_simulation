# 超节点资源建模报告

状态：**estimated**；模式：`estimate`。

模型：`synthetic-10t-moe-fp16`；场景：`training`；权重：`fp16`。

## 性能与约束

| 指标 | 数值 |
|---|---:|
| 理想执行时间必要下界 / s | 1,058.82 |

解析结果提供必要下界，尚未证明资源争用下的 SLO 可行性。

## 全局逻辑容量

单位为十进制 TB；实际分片、复制和生命周期见设备明细。

| 组成 | TB |
|---|---:|
| weights | 20 |
| gradients | 20 |
| master_weights | 40 |
| optimizer_states | 80 |
| total | 160 |

## 存储与通信

| 内存池 | 峰值或固定驻留 / GB | 配置容量 / GB |
|---|---:|---:|
| npu0.hbm | 2,502.06 | 4,000 |
| cpu0.dram | 30,000 | 64,000 |
| npu1.hbm | 2,502.06 | 4,000 |
| npu2.hbm | 2,502.06 | 4,000 |
| npu3.hbm | 2,502.06 | 4,000 |
| npu4.hbm | 2,502.06 | 4,000 |
| cpu1.dram | 30,000 | 64,000 |
| npu5.hbm | 2,502.06 | 4,000 |
| npu6.hbm | 2,502.06 | 4,000 |
| npu7.hbm | 2,502.06 | 4,000 |
| npu8.hbm | 2,502.06 | 4,000 |
| cpu2.dram | 30,000 | 64,000 |
| npu9.hbm | 2,502.06 | 4,000 |
| npu10.hbm | 2,502.06 | 4,000 |
| npu11.hbm | 2,502.06 | 4,000 |
| npu12.hbm | 2,502.06 | 4,000 |
| cpu3.dram | 30,000 | 64,000 |
| npu13.hbm | 2,502.06 | 4,000 |
| npu14.hbm | 2,502.06 | 4,000 |
| npu15.hbm | 2,502.06 | 4,000 |

物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。

| 网络域 | 物理传输 / GB |
|---|---:|
| SU | 90,032.3 |
| SO | 360,000 |
| D2H | 20,000 |
| H2D | 20,000 |

| 应用语义 | 传输 payload / GB |
|---|---:|
| fsdp_weights | 120,000 |
| tensor_parallel | 32.2123 |
| pipeline_activation | 0.134218 |
| fsdp_gradients | 60,000 |
| optimizer_offload | 40,000 |

应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。

按吞吐或训练步 SLO 得到的整任务时间窗口：1,000 s。各资源平均速率必要下界见 resources.csv；还需满足局部依赖与争用约束。

## 主要资源下界

| 资源 | 独占服务时间下界 / s |
|---|---:|
| cpu1.nic.tx | 1,058.82 |
| cpu2.nic.rx | 1,058.82 |
| cpu3.nic.tx | 1,058.82 |
| cpu0.nic.rx | 1,058.82 |
| link:su0.gateway>so.spine:SO | 529.412 |
| link:so.spine>su1.gateway:SO | 529.412 |
| link:su1.gateway>so.spine:SO | 529.412 |
| link:so.spine>su0.gateway:SO | 529.412 |
| link:npu4>su0.gateway:SO | 264.706 |
| link:su1.gateway>npu8:SO | 264.706 |

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
- 配置 SHA256：`9b71fb441f222f722ff44c7ca065a49529b241b8a9e0fc7dbf52290d6b98b175`。

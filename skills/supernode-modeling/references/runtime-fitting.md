# 让执行模型拟合真实推理

用于“让仿真更接近真实运行”“根据公开 benchmark 拟合参数”。重点是改变执行图、字节量、资源限制与服务时间，生成可重跑的拟合配置，并检查未参与拟合的点。来源登记和比较检查只辅助这一过程，不代替模型改进。

硬件接线、PCIe、NIC、分层互联和多路径差异另见 [硬件拓扑拟合](hardware-topology-fitting.md)。先使流量经过正确的共享资源，再生成与该拓扑匹配的 profile。

要适配不同 Ascend 环境和未采样消息量，使用 [多环境响应曲线](ascend-runtime-adaptation.md)，而不是扩展本页固定 H800 DeepEP 实验的系数。`collective_models` 支持同环境、同通信条件组的区间内插值；HCCL 与 msprof 的原始数据口径见 [采集指南](ascend-measurement-sources.md)。

## 已可使用的执行机制

### MoE 两个方向分别建模

在 canonical 场景中显式配置：

```yaml
execution:
  all_to_all_algorithm: direct
  precision:
    activation: bf16
    communication: bf16
    moe_dispatch: fp8
    moe_combine: bf16
    moe_dispatch_metadata_bytes_per_token: 240
    moe_combine_metadata_bytes_per_token: 0
```

`moe_dispatch` / `moe_combine` 省略时各自回退 `communication`，其他 TP/PP 通信仍使用原精度。metadata 是**每份 routed token 副本**的额外 wire byte 数，默认 0；不是每个原始请求一次，也不是 FLOPs。按路由数量重建两个方向的矩阵，保留 ragged batch 的返程目的地。量化转换、元数据复制、staging buffer 和实际传输使用各自字节量。

上例 240 来自 [DeepEP 旧版测试代码][test] 的 hidden=7168 情况：FP8 dispatch 的每份副本为 `7168 + (7168/128)*4 + 16 = 7408 byte`；BF16 combine 为 `7168*2 = 14336 byte`。所以两个方向并非等字节。维度、scale 格式或实现变化时重新计算，不能把 240 作为所有模型的默认值。代码允许的 FP8 scale 变体、随机 masked routing 和真实调度并未由这一个参数自动复现。

`direct` 让 rank 间传输并发竞争网络资源；`pairwise` 保留轮次序列。选择应符合目标 collective 实现，不能仅因为 direct 更快就通用于所有框架。这里仍是均匀专家路由期望模型，没有动态专家热点或分块计算通信重叠。

### 区分每卡和每主机网络能力

多 NIC 主机同时设置 `hardware.scale_out.injection_Bps` 与 `host_injection_Bps`，各自限制单卡与主机聚合带宽；共享上联、链路和 HBM 仍会产生争用。已有单 NIC 假设保持兼容。配置例子与默认上联规则见 [网络修正](public-benchmark-validation.md#已修正的网络表达缺口)。

### 用 collective profile 约束真实服务时间

`hardware.collective_profiles` 给指定调用提供 wall-time 服务约束，匹配通信种类、dispatch/combine 语义、执行阶段、world size、wire dtype、算法、总远端 payload、每 rank 发送/接收向量和硬件名称。未匹配时使用原模型；这不是跨消息大小的自动插值器。

保留所有传输、路径、缓冲区和共享资源，并把 profile timer 与原执行段并行，返回二者都完成后的时间：

```text
实际 collective 完成时间 = max(解析网络/转换执行完成时间, profile 服务时间完成点)
```

`duration_s` 包含该调用的启动，排除 warmup 和外部排队；与 kernel profile 的“不含启动服务时间”契约不同。`timing_scope: transport` 从传输前计时；`activation_collective` 从 pack 前至 unpack 后计时。DeepEP dispatch 内含量化，使用后者，不能在测得的整段时间之外再加一遍量化。遇到资源争用、慢链路或更慢的算法，真实传输会超过 profile timer，不会被强行裁短。

仅发送/接收向量相同仍不保证 peer 矩阵、张量形状和 runtime 相同。新生成 profile 通过 `applicability_signature` 锁定硬件抽象、peer 矩阵和物理 rank；旧无签名 profile 仍会报告 unguarded 匹配。`hardware_name` 与 hash 都不能认证真实平台。拟合配置的 `source` 应记录版本、测量/估算类型及限制；覆盖率表示发生匹配的调用比例，不表示整网精度。不要向旧模型直接复制整个硬件 profile；合并需要的字段并检查现有链路、rank 放置和精度。

## 公开数据驱动的可重跑拟合

仓库脚本使用 [DeepEP V1 公开低延迟微基准][legacy]，按 dispatch/combine 分别拟合：

```text
python skills/supernode-modeling/scripts/fit_deepep_collectives.py --output outputs/deepep-fit
```

这是通信阶段的数值拟合，不会加载模型权重或执行真实 GPU 测试。脚本构造 8 GPU/主机、每 GPU 400 Gb/s、每主机 8 路 NIC 的抽象 H800 网络，按每 rank 128 token、hidden7168、top8 建立矩阵；单向 SU 160 GB/s、HBM、均匀未 mask 路由、零链路启动等条件均在产物中声明。公开文档未绑定完整原始运行日志，这些是拟合时的工程假设。

拟合函数为 `T = startup + slowdown × physical_floor`，约束 `startup ≥ 0`、`slowdown ≥ 1`。`physical_floor` 根据构造图每项共享资源的累计工作量/能力求最大值；startup 与 slowdown 分别承载该微基准下的固定开销和有效传输损失。它们不改变 FLOPs、字节量或硬件标称带宽，也不是跨算子通用修正倍数。

仅用 **EP16/32** 的公开时延拟合，在 **EP64/128** 做留出检查。输出拟合系数、每点原始预测/拟合后执行/公开参考、误差、使用条件和可复用 profile。逐点查看误差；某个较大规模仍偏离，说明同一服务模型尚未覆盖全部协议/调度变化，不能只报告平均误差。

2026-10-22 执行结果（相同字节/网络假设，原始解析传输与拟合后执行的对照）：

| 留出点 | 公开 μs | 原始预测 μs | 拟合后 μs | 拟合后绝对相对误差 |
| --- | ---: | ---: | ---: | ---: |
| EP64 dispatch | 173 | 132.751 | 174.429 | 0.83% |
| EP64 combine | 314 | 256.901 | 312.000 | 0.64% |
| EP128 dispatch | 192 | 142.234 | 183.911 | 4.21% |
| EP128 combine | 369 | 275.251 | 331.500 | 10.16% |

这四个留出点的平均绝对相对误差从 **23.19% 降至 3.96%**。EP128 combine 仍偏快 10.16%，保留这一残差，没有用留出结果重新调整系数。拟合参数是 dispatch 的 startup=41.6776 μs、slowdown=1，以及 combine 的 startup=39 μs、slowdown≈1.062666；完整产物由上述命令重新生成。

留出点可验证这个固定形状、这组规模上的改善，不能证明任意 batch/上下文或模型 TPOT 的精度。迁移到目标推理场景时，先配置对应 wire 格式和网络，再使用匹配调用的 profile；未匹配部分继续解析执行。长期应从目标平台采集多 batch、多 payload、多 EP 的曲线，分别拟合 prefill 高吞吐 collective 与 decode 低延迟 collective。

生成的 `profiles.ep*.json` 是包含假设硬件与 collective profile 的片段，不是完整推理场景。完整 MoE 模型使用 `timing_scope: activation_collective`；仅 Topology 微基准调用可使用 transport scope。将片段合并到真实场景前保持其适用硬件、消息字节和 rank 条件，不能只改 `hardware_name` 让不相关平台强行匹配。检查 `model_metadata.collective_calibration` 的匹配、`profile_limited_collectives` / `physical_limited_collectives` 计数与阶段覆盖率；后者包括 pack、传输、unpack 及争用。包含转换的整段服务时间在时延拆解中归为其他通信，不伪分配成精确 kernel 子项。

## 下一步应拟合哪些运行行为

按主导误差选择下一项，不把所有缺失成本塞到 efficiency：

1. **MoE 热点和离散路由**：用逐层 token→expert/rank 分布驱动最忙 rank 的计算、权重访问和通信，计入冗余专家及迁移；当前均匀期望仍会低估小 batch 的不均衡。
2. **分块执行与重叠**：将 dispatch、专家 GEMM、combine 分块形成真实依赖，表达 DeepEP/Mega MoE 的通信进度与计算重叠；当前 direct 只改传输并发，不等价完整融合内核。
3. **计算与 KV 服务曲线**：按 batch、上下文、dtype、MLA/KDA layout 和形状采样，用同条件 kernel profile 替换 roofline 的粗效率；不得把网络耗时再次拟入计算 profile。
4. **服务调度**：加入到达队列、continuous batching、chunk budget、KV 抢占、非对称 P/D 池与动态投机接受数，使请求延迟和吞吐反映线上负载。现有固定 batch 与同步投机路径仍有这一边界。

继续使用 [运行时校准](runtime-calibration.md) 中的 trace 和阶段口径；仅需检查来源是否可比时再用 [公开 benchmark 比较器](public-benchmark-validation.md)。

[legacy]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/docs/legacy.md
[test]: https://github.com/deepseek-ai/DeepEP/blob/a56d6156febcd9976e55adc85b5155bfac9f28f8/tests/legacy/test_low_latency.py

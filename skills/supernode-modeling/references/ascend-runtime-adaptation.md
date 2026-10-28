# 适配不同 Ascend 环境的推理仿真

用于“不只拟合 DeepEP/DeepEP-Ascend 的几个点”“适配其他 Ascend 硬件、版本、消息量与模型负载”。目标是给目标环境建立独立响应模型，并在覆盖范围内预测未采样负载；不是把已有加速倍数搬到另一台设备。

## 当前可迁移的内容

| 条件变化 | 当前执行方式 |
| --- | --- |
| 同平台、软件、rank 布局和通信模式，消息大小或 batch 改变 | `hardware.collective_models` 在采样范围内分段线性插值，保留物理流量与争用 |
| 换 A2/A3/950 SKU、CANN/HCCL、驱动、框架版本、transport 或图模式 | 使用该环境的独立校准包；没有对应数据时使用解析模型并报告未覆盖 |
| 改 TP/EP、rank 放置、路由比例、I/O 共享、网络路径 | 单独的通信条件组；不跨 rank 数/拓扑插值，也不按型号比例缩放 |
| Dense/Attention/MoE 等计算算子 | 已有 exact-work kernel profile 可加平台、环境和结构元数据约束；不以通信曲线替代算子耗时 |
| 到达过程、continuous batching、动态 KV 抢占或在线服务队列 | 当前仍不模拟完整在线调度；用服务 trace 做端到端验证，不能吸收到硬件效率 |

因此，对不同 Ascend 环境的支持不依赖 DeepEP。HCCL 的 AllReduce、AllGather、ReduceScatter、AlltoAllV，以及其他后端的同范围观测均可作为输入。`algorithm: ring/pairwise/direct` 是本仿真器的传输模板，不能仅靠同名字段宣称复现某个 HCCL 实现。

## 为每个环境保存硬件和软件身份

场景/校准包的 `hardware.calibration_context` 包含以下必需的已知字符串；只有使用新的响应曲线才要求完整，旧场景保持兼容：

| 键 | 含义 |
| --- | --- |
| `device_sku` | 完整目标 SKU/产品形态，不能只有“Ascend” |
| `cann_version`, `hccl_version`, `driver_version` | 实际运行版本，不是参考文档版本 |
| `runtime`, `runtime_version` | 推理/测量运行栈及版本 |
| `transport` | 实际链路/传输路径，例如 HCCS、RDMA/RoCE、UB 的具体方式 |
| `mode` | 实际 eager/graph/加速模式等执行方式 |
| `timer_boundary` | `device_only` 或 `whole_call`；不能与客户端排队时延混用 |

额外键也参与匹配。应按需要记录 firmware、功耗/频率、实际 HCCL 算法配置、FP8 wire 格式、量化布局、HCCL_BUFFSIZE、对称内存等。当前 dtype 枚举 `fp8` 只描述仿真类型，不能自动识别 e4m3/e5m2/HiFloat8；同字节格式不同实现时用额外 context 隔离。

这些标签本身不会改变公式，也不是实机认证。硬件容量、FLOPs、HBM/链路能力、I/O 分组与实际 rank 放置仍需独立输入，见 [硬件拓扑拟合](hardware-topology-fitting.md)。身份未知时保持原始日志待补，不用虚构版本让校准包通过。

## 从观测生成响应模型

从项目根目录运行随 skill 提供的通用工具：

```text
python skills/supernode-modeling/scripts/build_runtime_calibration.py skills/supernode-modeling/assets/runtime-calibration.synthetic.json --output outputs/ascend-runtime-adaptation
```

示例有两个独立的 **synthetic A2/A3-like 环境**，数值与版本全部是虚构测试条件，不是厂商性能数据。它演示重复采样、未参加训练的区间内消息量以及超范围回退。输入结构可直接作为真实数据归一化的格式参考。

输入 JSON 为 `schema_version: 1`，包含：

- `environments`：每项 `id/hardware/calibration_context`。每个环境使用自己的硬件和软件条件，不混合拟合。
- `observations`：每项 `id/environment_id/kind/split/status/evidence_kind/source/duration_s`；本工具的 `kind` 为 `collective`，`split` 为 `train` 或 `validation`。
- 通信声明：`collective/semantic/phase/dtype/algorithm/timing_scope/ranks/size_bytes/precision`；AlltoAllV 必须提供完整 `payload_matrix`。`ranks` 是有序物理设备名，例如 `npu0`，不是未映射的逻辑序号。
- `metadata_match`：声明会影响实现的算子条件，并与事件元数据匹配。MoE 的 activation_collective dispatch/combine 要记录 hidden 宽度、tensor/额外 metadata 每 token 字节、top-k 和专家数量；不得仅凭总字节相等跨结构复用。

使用 `status: measured/public/synthetic` 与 `evidence_kind: measurement/estimate/synthetic` 区分来源公开、作者估算、真实测量和合成测试；公开的估算不等于实测。工具不自动下载数据或运行设备，也不认证来源声明。

数据量采用 **byte**，时间采用 **second**。`size_bytes` 对 AllGather/ReduceScatter/AllReduce 是当前引擎定义的每 rank 完整逻辑张量字节；AlltoAllV 的矩阵包含本地 diagonal，网络的采样横坐标只统计非对角物理发送 payload。不要把 HCCL 的 data_size、算法带宽或双向总量未经核对直接代入。实际采集、逐算子输入/输出口径与单位陷阱见 [Ascend 实测指南](ascend-measurement-sources.md)。

工具按环境、阶段、精度、算法、scope、物理 rank、完整 peer 比例和算子条件分组。同一点的重复训练观测取中位数，并保存样本数、最小/最大值与 observation ID；每组至少两个不同 payload 才能建曲线。验证点不参与中位数、训练范围或模型选择。输入硬件若已含 collective_profiles/models 会被拒绝，避免旧校准覆盖验证结果。

输出 `report.json`、可读报告和各自独立的 `environment.<id>.json`。每项保留原始观测、解析执行时间、应用模型后的真实仿真时间、训练范围、匹配方法与验证覆盖率。样本不够或没有被覆盖的验证点也会保留，不只展示成功点。

## 曲线怎样改变实际执行

对同一通信条件组，横轴为总远端发送 payload，纵轴为观测服务时间。新消息量落在两个采样点之间时线性插值；命中采样点时使用该点；范围外不外推。较大模型/新 batch 可以命中已有范围，但改变 rank 数、路由分布、软件或 shape 条件需要独立数据组。

归一化 peer 比例按 1e-12 精度记录并保留非零连接掩码，避免浮点缩放噪声把同一种路由拆散。这个工程容差不是任意流量矩阵的等价证明。

执行图仍包含传输、启动、HBM、NIC/PCIe/上联和争用，完成时间为物理执行与响应时间约束都满足后的时间。profile/curve 不会把过慢的物理路径强行裁短。现有 exact profile 优先，之后才尝试响应曲线；需要采用新校准包时检查是否仍残留优先级更高的旧 profile。

MoE 优先选择可用的 `activation_collective` 全调用模型；它覆盖 pack→网络→unpack。没有可用全调用模型而有 `transport` 模型时，只对网络段应用，转换成本继续单独执行。一次调用只统计所选边界，避免套两层计时。HCCL 网络数据不必伪装成 DeepEP 全调用数据才能进入真实推理图。

来源的 `device_only/whole_call` 和仿真中 dependency-ready→完成是两个概念。Device-only 样本排除的 Host 成本不会由一个字段自动补齐；不得把服务时间约束解释为含客户端排队的 TTFT/TPOT。

检查 `model_metadata.collective_calibration` 的 `exact_profile_collectives`、`exact_model_collectives`、`interpolated_collectives`、`out_of_range_collectives`、`matched_model_ids` 和 `unmatched_reasons`。`physical_limited_collectives` 表示解析执行/争用比采样服务约束更慢；覆盖率不代表准确率。

## 计算算子的环境适用性

`hardware.kernel_profiles` 仍按 kernel、FLOPs、HBM bytes 和 dtype/phase 精确匹配，但条目现在可以附加：

- `hardware_name`、完整 `calibration_context`：限制硬件和运行栈。
- `metadata_match`：核对真实事件已具有的元数据，例如 component、tokens、结构签名和 TP/EP；缺字段或不一致时回退。
- `source`：保留原始测量和转换方法。

结构化组件 kernel 暴露 `model_shape_signature`、`tensor_parallel_degree`、`expert_parallel_degree`；结构签名来自模型配置，不认证 checkpoint 内容、tile、张量布局或 fused kernel。先查看未校准事件再建立 selector，不为未暴露的 shape 字段造值。相同 FLOPs/bytes 不足以证明两种 kernel 同速。

kernel `duration_s` 仍为**不含 startup_s**的独占服务时间。msprof 的 Task Duration 可能包含派发与响应，不能原样重复加启动。没有可靠拆分时保留独立证据，不硬填纯 kernel 参数。kernel summary 分别报告 guarded/unguarded 匹配与 scope mismatch；有同工作量 scoped 记录却环境不符时，不通过 legacy 条目绕过限制。

## 在目标环境中采用和验收

选择对应环境产物，核对其 normalized 硬件值（包括显式列出的默认值），再合入目标配置。输出是独立片段，不会修改已有场景；完整场景仍需满足卡数、并行度和物理映射约束。可沿用 `calibration: <相对于场景的校准文件路径>`，然后 validate、运行并检查阶段覆盖率。

真实精度改进需要目标环境的留出数据。建议按 payload、batch、上下文、prefill/decode、rank 布局和竞争模式分别留出，不把同一次重复采样随机拆成独立泛化证据。先报告分阶段误差与覆盖范围，再检查完整推理的 TTFT/TPOT/吞吐；没有生产队列模型时不报告生产 P99 保证。

此前 **23.19%→3.96% 是 H800 DeepEP 固定形状的公开参考实验**。本通用示例仅验证多环境适配及插值行为，不能据其合成数据宣布 A2/A3/950 的误差降低。

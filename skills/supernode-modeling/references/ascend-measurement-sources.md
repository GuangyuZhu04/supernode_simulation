# Ascend 实测来源、口径与最小采集路径

核读日期：**2026-10-22**。本文是实际采集指南，不是已完成的 Ascend 真机验证报告。本轮未连接或运行 Ascend 设备；下面的工具能力与字段来自已核读的一手文档/源码，采样设计是本项目建议。此前约 **3.96%** 的留出误差来自 **NVIDIA H800 上的 DeepEP 公布数据**，不能写成 Ascend、HCCL 或任意新拓扑的仿真精度。

先保留原始数据及版本，再按计时边界映射到 [runtime-fitting.md](runtime-fitting.md) 的拟合流程。不同型号、后端、协议或设备布局建立不同范围的数据集；不能将一个 DeepEP/DeepEP-Ascend 点的系数推广到所有 Ascend 条件。

## 哪种原始数据能约束哪一层

| 数据源 | 应保存的原始交付件 | 适合约束 | 不能直接推导 |
| --- | --- | --- | --- |
| HCCL Test | 完整 stdout/stderr、实际命令/环境、Hostfile、所选物理设备、数据量逐点结果 | 特定通信算子/实现的 call 或 device 时间随数据量变化 | NIC 物理线速、MoE 路由矩阵、端到端 TTFT |
| Ascend DMI P2P | 单向/双向原表、src/dst device、链路类型、size、迭代数 | 指定路径的数据搬运与带宽边界 | AllReduce/AlltoAll 实现开销、未测路径的通用时延 |
| msprof | 原始 PROF 目录、timeline JSON/DB、op_summary、通信明细 | 同形状/精度/格式/实现的 task 时间；实际通信路径与重叠 | 将各 stream 时长相加得到 walltime；把 estimated 字段当实测 |
| AISBench / vLLM bench serve | 客户端请求日志、结构化结果、服务端配置、输入/输出 token、窗口定义 | 给定服务负载下的 TTFT/TPOT/ITL/吞吐验收 | 独立 kernel 或 collective 的纯服务耗时 |

表中的映射是建模建议。公开文档里的输出样例只证明字段格式；没有完整硬件、版本和计时条件时，不作为拟合标签。

## 最小真实数据路径

建议先完成一套**单一、可复现的 Ascend 部署**，而非先混合多个机型的数据：

1. 固定型号/产品形态、驱动固件、CANN/HCCL、推理栈及模型 checkpoint。记录物理 NPU、芯片/die、host、HiAM、逻辑超节点、NIC/IO group 到 rank 的映射，保留实际拓扑导出。A2、A3、950 的层次不能仅靠设备数互换；结构依据见 [hardware-topology-fitting.md](hardware-topology-fitting.md)。
2. 在目标物理 rank 集合上运行其已安装版本支持的 HCCL Test，分别保留 AllReduce、AllGather/ReduceScatter 及需要的 AlltoAll 类数据。先测单组、不并发的基础曲线，再另建竞争/重叠实验。项目建议至少覆盖小消息、中间区间和大消息，多次独立重复，并预留未参与拟合的数据量；具体点数和负载由实际目标决定。
3. 用 DMI 对本机实际存在的路径做补充测量：同 HiAM、跨 HiAM、跨 host 或跨超节点分别记录。DMI 的流量与 collective 不同，作为路径约束而不是替代 collective 样本。
4. 对目标模型的一小段稳定推理采集 msprof，按 prefill/decode、层/算子、shape、dtype、format 分组，核对模型执行确实调用了待校准实现。性能计时与 tracing 分开保存，避免把 tracing 扰动直接混入无 profiler 的基准。
5. 使用同一 checkpoint 和部署配置进行流式服务测量。先核对 token 计数、缓存、投机接受口径和到达过程，再比较服务指标。只有这一层匹配，才讨论 TTFT/TPOT/SLO 的准确度。

这五步不要求使用 DeepEP；HCCL、其他 MoE 通信后端以及普通稠密模型都能采用相同分层方法。没有对应样本的范围继续使用解析模型，并报告未覆盖。

## 每次采集都附带 manifest

以下是**归档字段建议**，不是可以直接粘贴进 Config 的新 schema。未知项用 null；不得用文档版本代替实际运行版本。

| 字段组 | 最少内容 |
| --- | --- |
| 平台 | 完整 SKU/板卡或服务器型号、物理设备与逻辑 rank 数、chip/die 口径、host/HiAM/SU 布局、链路/IO group、频率/功耗状态、驱动/固件 |
| 软件 | CANN、HCCL、工具版本或构建 commit/hash、torch/torch_npu、推理 runtime/backend、容器 digest |
| 运行策略 | 通信后端与算法、展开/加速模式、HCCL_BUFFSIZE、零拷贝/对称内存、UB/SDMA/RDMA/RoCE/UBoE、图捕获、并发和重叠、实际生效参数 |
| 工作量 | collective 与语义、rank 成员/顺序、dtype、shape、layout、实际 bytes、AlltoAllV counts 矩阵或逐 rank 收发量、pack/scale/index 字节 |
| 计时 | 单次开始/结束边界、Host/Device 时钟、是否包含启动/pack/unpack/同步、warmup、测量迭代数、聚合方法、采样窗口、profiler 开关 |
| 证据 | 原始文件相对路径及 SHA-256、实际命令与完整配置、采集日期、正确性结果、异常/丢弃样本、训练点和留出点标记 |

rank 数、总 bytes 相同仍可能有不同热点、路径和负载不均。保留矩阵比只保存总量更有用；发送和接收向量也不能证明两个完整矩阵相同。

## HCCL Test：优先导入实际 bytes 和时间

已核读 CANN 9.1 的[参数说明][hccl-params]、[执行说明][hccl-run]、[结果说明][hccl-results]与[机型约束][hccl-limits]。

- MPI 的进程数与工具的每节点设备数是不同输入。记录实际设备选择、Hostfile 顺序和通信域；不能把默认连续设备选择当成目标部署的任意映射。
- 数据量扫描使用起止 bytes 和步长/倍率；工具可能为对齐调整数据量，因此保存**回显实际 data_size**。
- `-n` 是测量迭代数，`-w` 是不计入性能统计的预热。参数页和旧样例中的默认值可能不同，保存实际回显，不依赖默认。
- `-t 0` 包含通信调用的全部耗时；`-t 1` 排除 Host 软件与 kernel 加载时间。后者有 AIV/CCU、buffer 和迭代数适用约束，不能假设加上开关就生效。两种计时不得放进同一曲线。
- profiling 会影响通信性能。分别标注带 profiler 与不带 profiler 的采集。
- `data_size(Bytes)` 是单 NPU 的数据量，`avg_time(us)`（部分版本拼作 `aveg_time`）是耗时，`alg_bandwidth` 是算法带宽，`check_result` 区分 success/failed/NULL。NULL 不等于通过验证。

固定源码 [oam-tools 5a6bfd0][hccl-code] 的 AllReduce 实现将总测量时间按迭代数求平均，并使用算子分配的数据量计算算法带宽；输出函数在 root rank 打印汇总。因此单张结果表不是逐 rank 原始时延分布，不能直接伪造 p99、最慢 rank 或方差。[源码计时计算][hccl-allreduce]

该版本文档示例中的 `8192 Bytes / 764.55 us` 约为 **0.010715 十进制 GB/s**，但打印为 **0.00998**，数值约对应 GiB/s。这里仅作单位一致性检查，不把样例视为实测参数。导入优先用 bytes 与 us 明确换算；保留原单位标签及原打印值，不把标为 GB/s 的列直接写入 `bandwidth_Bps`。算法流量也不是每条物理链路字节。

AllReduce、AllGather、ReduceScatter 的 input/output 分配语义不同。AlltoAllV 还需要每个 peer 的 counts、dtype 和本地/远端份额；仅知道 data_size 不能构造真实 MoE dispatch/combine 矩阵。普通 HCCL AlltoAll 的时间也不自动包含 MoE 路由、量化或合并计算。

## DMI P2P：保留方向和路径选择

MindCluster 26.1 的[跨机 P2P 文档][dmi-p2p]明确区分单向与双向结果，保存 `Size(Bytes)`、`Execute Times`、`Bandwidth(GB/s)`、`Elapsed Time(us)` 和源/目的设备。该页面分别列出 Atlas 850 的 CLOS8、Atlas 950 的 CLOS2/CLOS6 选择；不能将这些选项套到任意 A2/A3。

页面示例的单向 bytes/time 与显示带宽存在约 2 倍差异，且不能仅凭“总执行时长”推断如何计入迭代。正式数据需同时记录 chip/die 计数和工具计量规则；未解释前保留原数值、标记不可归一化，不用其反推全局带宽。具体可用参数以现场同版本帮助与对应产品文档为准；本文没有在设备上执行该命令。

对于模型需要的多个并发路径，单对 P2P 的结果只是边界；还要独立采集共享 uplink、PCIe/IO group 和双向并发情况。

## msprof：task、等待与重叠分开

已核读 [Ascend/msprof 固定版本的数据文件定义][msprof]。保留 `Device_id`、Task/Stream/Infer ID、Op Name/Type、Task Type、输入输出 shapes/dtypes/formats、Task Start Time、Task Duration、Task Wait Time；l0 采集可能没有 shape/type 等字段，不能用 N/A 自动合并不同实现。

`Task Duration(us)` 包含调度到加速器、执行及结束响应。`Task Wait Time` 是上一个 task 结束到当前 task 开始的间隔。导入 kernel 计时前先对齐启动边界：不要把等待算入 solo service，也不要把整 task 包络再次叠加相同 startup。shape/dtype/layout 对齐后，仍需核对算子实现和图执行方式。

通信明细的 src/dst rank、transport/link type、size 可帮助验证路径及流量；通信统计 CSV 的 Count/Total/Min/Avg/Max 是聚合结果。使用 timeline 核对计算通信重叠，不能把并行 stream 时长直接相加。

机型字段有实质区别：A3 的 SIO 数据分 die0/die1 通道，A2 对应字段为0时没有参考意义；950 DPU 表的 `Duration Estimated(us)` 明确为预估，不能标记 measured。对融合计算通信任务应保留融合边界，避免再分别加上已被包含的通信或计算。

## AISBench / vLLM：用于服务验收

[AISBench 固定版本指标定义][ais-metrics]区分请求 E2EL、TTFT、TPOT、ITL，以及整个 benchmark 的吞吐。TPOT/ITL不包含首 token；配置最大并发与实际平均并发不是同一个数。保留流式响应时间和实际生成 token，均值、分位值及全程/稳定阶段窗口分开记录。

其[服务化指南][ais-run]提供流式后端、请求速率/并发、timestamp 调度、输出长度等配置入口；[vLLM-Ascend 固定版本指南][vllm-bench]提供 online serve 与 offline throughput 两条路径。采集时保存真正使用的服务端启动配置和客户端配置，不能只记录 bench 命令。本文只核对公开入口，未运行这些服务 benchmark。

端到端指标同时受队列、客户端/网络、tokenizer、缓存、投机和调度影响。服务 TTFT 不能直接替换 device prefill 时间；客户端 ITL 也可能受分块返回影响。投机使用实际提交输出 token；满接受记账或草稿数量不作为有效生成量。未披露数据分布和计时边界的公开吞吐只作参考，不自动进入 kernel/collective 曲线。

## 型号与模式必须进入适用范围

CANN 9.1 文档的每节点 `-p` 范围分别为 A2 1–8、A3 1–16、950 1–8；这不等于它们拥有相同卡/die或网络结构。[机型约束][hccl-limits] 参数支持也不同：950 的特定低精度类型和加速模式不能默认用于 A2/A3；950PR 不支持文档中列出的 ccu_ms 模式。[参数说明][hccl-params]

950 的[官方架构白皮书][ascend950]第29–31页进一步限制 transport 适用范围：

- UB 2016 GB/s、PCIe 128 GB/s 是双向规格；不能直接放进单向带宽字段。
- UBoE 与 UB 静态复用 SerDes；启用相应 Ethernet 端口后，UB 可用端口相应减少。
- URMA RTP/CTP 可用端口能力分别受4/9个 Port 限制，不能为任意消息分配全部18个 x4 Port。
- 这些是结构/模式约束，不是已测有效带宽，也不是本轮硬编码的效率系数。

本地核读原件为40页、6,356,641字节，SHA-256：
`ece3405e6a17fabdd462338fb94266558649a6407a2f28008403211387b3a927`。
部署使用何种 transport、端口分配、CCU/AI CPU/AIV 模式，仍须记录现场配置及实际执行路径。

## 数据进入拟合前的检查

下列规则为本项目建议：

- 先确认正确性、单位、rank/shape、时间边界及物理拓扑；不满足的记录保留原始证据，但不输出“已校准”。
- 同一硬件/栈/模式范围内拟合；不同精度、路由策略、buffer 模式、加速模式、图模式分别分组。单纯模型同名或设备数相同不足以合并。
- 留出未参与拟合的点进行误差评估。不得用参与拟合点的重放误差宣称泛化精度；不要把范围内插值当跨型号或跨拓扑预测。
- 修改链路、IO groups、rank placement、协议或运行栈后重新核对 scope。物理容量/争用下界仍生效；测量不能用于“绕过”瓶颈。
- 报告至少同时给出：样本范围、原始来源、匹配/未匹配数、拟合/留出误差、物理下界主导状态和端到端验收状态。

[hccl-params]: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0005.html
[hccl-run]: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0004.html
[hccl-results]: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0006.html
[hccl-limits]: https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0007.html
[hccl-code]: https://gitcode.com/cann/oam-tools/blob/5a6bfd0f7c149968ed7517a4a7c17a3e10d07540/src/hccl_test/common/src/hccl_opbase_rootinfo_base.cc
[hccl-allreduce]: https://gitcode.com/cann/oam-tools/blob/5a6bfd0f7c149968ed7517a4a7c17a3e10d07540/src/hccl_test/opbase_test/hccl_allreduce_rootinfo_test.cc
[dmi-p2p]: https://www.hiascend.com/doc_center/source/zh/mindcluster/2610/toolbox/toolboxug/toolboxug_0242.html
[msprof]: https://github.com/Ascend/msprof/blob/8f0bed4db4f86c5258587a536e112ccfdbf86ceb/docs/zh/user_guide/profile_data_file_references.md
[ais-metrics]: https://github.com/AISBench/benchmark/blob/b5b88322d65b790c887b15a03e8d11cb30146be2/docs/source_zh_cn/base_tutorials/results_intro/performance_metric.md
[ais-run]: https://github.com/AISBench/benchmark/blob/b5b88322d65b790c887b15a03e8d11cb30146be2/docs/source_zh_cn/base_tutorials/scenes_intro/performance_benchmark.md
[vllm-bench]: https://github.com/vllm-project/vllm-ascend/blob/0b8cd0aeb01c25a4896ff78caddc834361a9af20/docs/source/developer_guide/performance_and_debug/performance_benchmark.md
[ascend950]: https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf

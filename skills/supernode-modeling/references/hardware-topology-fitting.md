# 根据硬件架构与拓扑拟合运行

用于已有 kernel/collective 拟合仍不能解释不同卡位、组网或 offload 下性能差异的任务。先构造真实物理路径和共享资源，再按 [运行拟合](runtime-fitting.md) 引入服务时间。卡数、型号和端口峰值相同，不保证可用带宽相同。

## 公开结构依据

核验日期：2026-10-22。

| 原始资料 | 进入模型的信息 | 仍须实际部署提供 |
| --- | --- | --- |
| [NVIDIA DGX H100/H200 硬件指南][dgx]，表 6 | 八个集群 ConnectX-7 端口，每卡最高 400 Gb/s；有明确 GPU↔NIC 对应和两组网络模块/CPU 归属 | 端口启用状态、PCIe bridge/NUMA 共享上联、GPUDirect 路径、交换机接线、实际单向有效速率 |
| [Ascend 官方亲和性说明，固定版本][ascend]，A3 | 两个处理器经 SIO 组成 HiAM；每节点八个 HiAM；节点内 HCCS-L1、节点间 HCCS-L2，跨逻辑超节点走 RoCE | 各级带宽/启动、端口聚合和实际编号映射。公开层级不自动确定某套系统的链路数值 |
| 同上，Atlas 950 | 节点内 X 轴、机架内 Y 轴、机架间 Z 轴具有不同局部性与逐级收敛带宽 | 每轴链路和共享出口的数值、实际 LRS/UB 端口关系；不能将全部 NPU 当同速 full mesh，也不能套 A3 的 HiAM 生成器 |

保留 `nvidia-smi topo -m`、NIC/PCIe 枚举、厂商拓扑导出或用户提供的连接表中的**物理编号**，再设置 `execution.rank_order`。NUMA 分组本身不是带宽瓶颈证据；只有确实经过同一 PCIe/UPI/NIC 资源的流量才共享容量。当前 `io_groups` 表达端点容量和静态亲和关系，不自动发现硬件、选择远端 NIC 或复现 PXN relay。

950 还需区分接口总能力与协议可用端口。[官方架构白皮书][950-whitepaper] 第 29–31 页给出 UB 2016 GB/s **双向**、PCIe 128 GB/s **双向**，并说明 UBoE 与 UB 静态复用端口、URMA RTP/CTP 的端口范围不同。不能把总双向带宽直接填为每条单向链路，也不能把复用接口容量相加。当前模型尚未自动按 URMA 协议分配端口预算；需要针对实际 transport 构建显式链路/共享容量，不能把新闻稿峰值当成任意 collective 的可用带宽。

## 端点 I/O：共享 NIC、PCIe 与 offload

可选 `hardware.io_groups` 将指定物理设备绑定到共享资源。例子中的 32 GB/s 是工程假设：

```yaml
hardware:
  io_groups:
    - id: pcie_root0
      members: [0, 1]
      domains: [SO, H2D, D2H]
      bandwidth_Bps: 3.2e10
      efficiency: 1
      duplex: full
      latency_s: 0
    - id: nic0
      members: [0]
      domains: [SO]
      bandwidth_Bps: 5.0e10
```

- `members` 是物理 NPU/GPU 索引，不随 `rank_order` 再映射；多个组可以重叠，表达 NIC 和 PCIe 两级限制。
- 只有实际路径含对应 domain 才收费。SO 在源端 tx、目的端 rx；D2H 只在源 NPU tx，H2D 只在目的 NPU rx。纯 SU/P2P 不会自动使用 NIC/PCIe 组。
- `duplex: full` 建立独立 tx/rx 资源；`shared` 则共用一个双向预算。同一 shared 组参与了两个端点时按两个端点分别计费。不要把此定义误当一条双端只计一次的物理 link；后者用 `explicit_links.shared_group`。
- `latency_s` 是每参与端点的附加启动延迟，默认 0。与已有 link latency 表达同一成本时不要两边重复填入。
- 组只增加容量约束，不增加逐跳物理流量。`io_group_charges` 列出每条传输使用的组；`resources.csv` 的 `io:<id>.tx/rx/shared` 可定位瓶颈。
- 每卡 `injection_Bps`、主机 `host_injection_Bps` 和 `uplink_Bps` 仍各自生效。新增八个独立 NIC 组不会自动消除旧配置的单 NIC 主机聚合上限。

未配置组时保持原有行为。先比较相同负载在真实端点映射和误当独立端口两种情况下的结果，再使用目标路径的微基准校准共享速率。

## 多路径：逐流 ECMP

在已存在多条等跳数路径的显式图中设置：

```yaml
hardware:
  topology: explicit
  routing_policy: su_so
  routing_algorithm: ecmp_hash
  routing_seed: 0
  # explicit_links 继续提供真实有向边及共享容量。
```

`routing_policy` 决定允许走哪些 SU/SO 域；`routing_algorithm` 决定如何从合法路径中选路。默认仍为 `shortest_hop`。`ecmp_hash` 对源、目的、当前节点和 seed 做稳定哈希，只沿剩余跳数递减的边前进。显式 `routes` 优先；查询顺序、Python 进程随机 seed 或缓存淘汰不会改变流的路径。

每个源/目的对选择一条路径，保留哈希碰撞和共享链路竞争。它不逐包喷洒、不动态避拥塞、不自动创建 spine，也不把单条流均分成理想带宽。不同链路仍按实际速率/时延计费；“等价”在此是单位路由代价下的等跳数，不是加权最短时间。需要固定 rail、加权路由或指定 GPU-NIC 通道时使用明确的图和 `routes`。

## A3 分层拓扑生成

```text
python skills/supernode-modeling/scripts/build_architecture_topology.py --spec skills/supernode-modeling/assets/a3-topology.example.json --output outputs/hardware-topology-fit/a3-hardware-fragment.json
```

输入 `hosts` 和 `sio/hccs_l1/hccs_l2` 三层的 `bandwidth_Bps`、`latency_s`、`source`，可附 efficiency 与 host_link。没有隐藏的厂商带宽默认值。示例中的所有速率/时延均为 assumed。

输出是 `hardware/sources/assumptions` 片段：每节点 16 个处理器、相邻偶奇编号组成 SIO 直连对；跨 HiAM 经过抽象节点 fabric；跨节点经过共享 HCCS-L2 上联。不是实际交换机内部端口复刻。每卡统一注入上限取 SIO/L1 最大有效速率，这是显式工程假设；不同层级同时工作的实际总注入能力仍须校准。

片段须与模型、HBM/计算、负载及并行配置合并；总卡数匹配后再 validate。未提供 host_link 时没有 H2D/D2H 边，不能用于需要 host offload 的场景。所有处理器位于一个逻辑 SU 域，跨逻辑超节点 RoCE 需额外图；不把 A3 结构复用到 950。

## 拟合 profile 随拓扑锁定

新生成的 collective profile 包含可选 `applicability_signature`。签名覆盖生效的硬件参数、I/O 组、路由/ECMP、物理 communicator rank 顺序、完整 peer payload 矩阵以及相关精度/计算条件。

即使总字节量和收发向量相同，换 peer 目的地、rank 放置、PCIe 共享、NIC 带宽或路由 seed，也可能改变通信时间。签名不同报告 `applicability_mismatch`，不继续套旧时延；重新确认目标硬件和原始测量后再生成适用 profile。它锁定的是声明的硬件抽象，不是芯片身份认证。

旧 profile 无签名时继续兼容，并在 `model_metadata.collective_calibration` 中计入 `unguarded_matched_collectives`；新匹配计入 `guarded_matched_collectives`。同一字节条件存在 guarded profile 时，不通过旧 unguarded 条目绕开检查。没有 guarded profile 时不对大矩阵计算签名，保留默认仿真开销。

生成工具使用 `collective_applicability_signature(hardware, physical_ranks, kind, size_bytes, payload_matrix, precision=...)`；让工具根据真实输入生成，不能手改 hash 让不相符条件通过。DeepEP 拟合脚本已自动带签名；网络声明改变后重新检查适用条件，不把上一轮 3.96% 误差移作新拓扑的保证。

## 可重跑的架构检查与实际效果

```text
python skills/supernode-modeling/scripts/evaluate_hardware_topology.py --output outputs/hardware-topology-fit
python skills/supernode-modeling/scripts/fit_deepep_collectives.py --output outputs/deepep-fit-topology
```

第一个命令输出每流路径、资源收费、瓶颈和输入片段。2026-10-22 的显式假设检查：

| 相同负载下的区别 | 仿真结果 | 解释 |
| --- | --- | --- |
| 两条 1 GB 流：独立 / 共享 50 GB/s NIC | 20 / 40 ms | 不再仅靠主机总 NIC 数判断是否独立 |
| 1 GB SO + 1 GB D2H：忽略 / 计入共享 32 GB/s PCIe | 20 / 62.5 ms | offload 和通信竞争同一实际出口 |
| 16 条 1 GB 流、两个 50 GB/s spine：单路径 / ECMP seed0 | 320 / 180 ms | 哈希为两路 9/7 条流，不虚构理想 160 ms |
| A3 抽象中八条 1 GB 流：同 HiAM / 跨 HiAM / 跨节点 | 5.0004 / 10.0016 / 50.0046 ms | 同样卡数和字节量，由局部性和聚合上联决定耗时 |

以上数值是容量和路径模型检查，**不是实机测量或新增准确率**。第二个命令保留上一轮 EP16/32 拟合、EP64/128 留出方式，检查新机制不会破坏已有校准。

进一步量化目标部署的精度，应分别采集同 HiAM/节点、跨节点、跨 SU，以及 SO 与 offload 混合负载的通信数据。先拟合端点/链路服务曲线，再用未参加拟合的 placement 和并发模式检验；保留 peer 矩阵与物理 rank 表。线上请求排队、NUMA CPU 调度、动态 rail 选择和包级拥塞控制仍不在这套模型中。

[dgx]: https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html
[ascend]: https://github.com/Ascend/mind-cluster/blob/c38f378dbf2070399b4ade868abfdbfcaa7c8736/docs/zh/scheduling/04_usage/03_basic_scheduling/01_affinity_scheduling/03_ascend_ai_processor_based_affinity.md
[950-whitepaper]: https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf

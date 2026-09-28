# 拓扑、通信与最小 SU

## 硬件配置与实际路径

`hardware.topology` 支持 `full_mesh/mesh/clos/fat_tree/explicit`，`su_size` 必须整除 `npu_count`。`rank_order` 是物理 NPU 索引的完整置换；它影响 communicator 是否跨域。`npus_per_host` 决定 `npu{i}` 的主机为 `cpu{i // npus_per_host}`。

`scale_up`, `scale_out`, `host_link` 分别声明 `bandwidth_Bps`, `latency_s`, `efficiency`, `injection_Bps`；支持 `uplink_Bps` 控制共享上联。有效带宽按效率折算，方向独立；host DDR、端点注入、链路和 SO 主机 NIC 都可能成为瓶颈。

- Full-mesh 同域走直接 SU 边；边按需生成。
- Mesh 使用二维邻接，`mesh_width` 可指定宽度，确定性最短跳路由。
- Clos/Fat-tree 使用 leaf 与单 spine 的流体容量抽象；`leaf_size` 和 SU `uplink_Bps` 控制上联汇聚。它不是完整交换芯片、ECMP 或包级模型。
- 跨 SU 路径使用独立 SO attachment、每主机 NIC、每域共享 uplink 和 SO spine。
- 远程内存/SSD 路径经过主机 bridge；硬件 profile 必须声明容量、带宽、延迟和随机 IOPS。

`explicit` 通过有向 `explicit_links` 建图：

```yaml
explicit_links:
  - id: link-a
    src: npu0
    dst: npu1
    bandwidth_Bps: 4e11
    latency_s: 1e-6
    domain: SU
    efficiency: 0.85
routes:
  npu0->npu1: [link-a]
```

反方向需要另一条有向边。`shared_group` / `shared_Bps` 可显式指定共享物理瓶颈，例如双向共用的半双工链路。显式拓扑需要提供任务所用的所有 host/remote/SU/SO 连通路径；引擎不会额外补上默认 host 链路。路由的 link IDs 必须连续并到达目标；未指定的路径使用确定性最短跳路由。

## Collective 与字节记账

ring all-gather、reduce-scatter、all-reduce 的大小口径分别是每 rank gather 后完整张量、reduce 前完整输入、all-reduce 完整输入。all-to-all-v 使用 rank-to-rank payload 矩阵，忽略不需网络通信的对角项。

collective 按实际轮次生成通信事件，并申请/释放 send 和 receive staging buffer。dispatch/combine 标签说明 all-to-all-v 的用途，属于同一笔字节的另一种分类。

`payload_bytes` 是传输事件实际携带的 payload；ring 中多轮转发会体现在 transport payload 总量。`physical_bytes_by_domain` 按路径每条边累加；多跳 mesh/Clos 可能大于 transport payload。DDR/HBM 访问字节、注入字节、物理链路字节是不同资源账本，不能全部相加当作通信量。

事件资源采用 roofline 独占速率，再对共享资源做归一化 max-min 分配；存在 startup、依赖与执行槽位。它可以评估拥塞与重叠，不模拟拥塞控制协议、报文、交换机 buffer 或真实 tail latency。

ring 会展开约 `ranks * (ranks-1)` 个事件/阶段。大 FSDP 群和长序列应先做容量/解析检查、缩小搜索集合；不要将小型样例的执行成本外推到数千 rank 的完整事件图。

## 最小 SU 搜索

`size-su --mode single` 将整个任务放入一个 SU；`--mode multi` 在集群约束下搜索每域卡数；`--mode both` 分别输出两种结论。

用 `targets.candidate_npu_counts` 指定总卡数候选，`candidate_su_sizes` 指定多域候选；多域还必须提供 `total_npus` 或 `max_total_npus`。`parallel_candidates` 可指定 `tp/pp/ep/dp/fsdp/rank_order` 组合，`max_candidates` 限制候选数量。未提供布局候选时保留 TP/PP/EP/FSDP、从总卡数推导 DP；改变 DP 会改变全局工作量，不能隐瞒。

```text
python -m supernode_sim size-su path/to/scenario.yaml --mode single --output outputs/su-single
python -m supernode_sim size-su path/to/scenario.yaml --mode multi --output outputs/su-multi
```

候选集合不是连续空间证明；只有显式检查过的更小配置都失败，才能表述为“该候选集合与当前策略下最小”。同时报告最佳配置、失败配置、SLO、总卡数/域数、DP/global batch、SO 能力，以及容量还是性能限制。没有 SLO 时，结果只是配置容量/执行可行性搜索，不应解释为业务 SLO 已满足。

实测校准应覆盖链路有效带宽、启动延迟、共享 uplink、host/remote 内存访问和实际 communicator。缓存层次容量和 MTE 通道数还需要 kernel tile、访存局部性及通道模型；架构级字节/带宽输出只能提供设计约束。

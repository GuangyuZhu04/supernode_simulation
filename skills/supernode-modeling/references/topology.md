# 拓扑、通信与最小 SU

## 一等输入：SU 域、SO 域与拓扑结构

| 输入 | 语义与映射 |
|---|---|
| 总 NPU 数 / SU 大小 | `npu_count` / `su_size`；SU 域 `d` 的物理成员是 `npu[d * su_size : (d + 1) * su_size]`，SU 域数为两者之商 |
| 主机与 rank 放置 | `npus_per_host` 与完整场景的 `execution.rank_order`；rank 的通信伙伴必须落到物理 NPU 后再判断跨域 |
| SO 域成员 | 高层规范 `scale_out.domains: [{id, su_domains}]`；每个 SO 域包含一个或多个 SU 域，全体 SU 域必须恰好分配一次 |
| 两层拓扑 | 高层规范 `scale_up.topology` 是 SU 域内部结构，`scale_out.topology` 是每个 SO 域内部结构；它们可不同 |
| SO 域间结构 | 高层规范 `inter_domain_links` 给出有向域间边、速率、延迟和效率；一个方向的声明不代表反方向 |
| 物理链路能力 | `bandwidth_Bps`、`latency_s`、`efficiency`，可选 `injection_Bps` / `uplink_Bps`；共享资源按实际约束声明 |

这里将一个 SU 域作为一个超节点单元。SO 域是互联这些单元的 fabric 分组，不等于 NPU 总数、DP 副本数，也不等于链路的 `domain: SO` 标签。已有原生配置只有 `hardware.su_size` 与 `hardware.scale_out` 速率时，默认模板代表所有 SU 域共用一个 SO fabric，不能推断成多个隔离 SO 域。

接收输入的两条路径：

- 已有可运行场景：保留其原生 `hardware`、`explicit_links`、`routes` 与来源，检查实际路径即可。任意非均匀图可用原生 explicit 表达；均匀的 `su_size` 标签不自动表示这些自定义节点的物理分组。
- 自然语言、仅有域规划或需要新的两层结构：调用 [supernode-topology 子 skill](../subskills/supernode-topology/SKILL.md)，形成独立规范并编译。完整字段见子 skill 的 [输入规范](../subskills/supernode-topology/references/specification.md)，机器契约由 `build_topology.py --schema` 导出。

例如“4 个超节点，每个 2 张卡、每主机 2 张卡，域内 SU full-mesh；前两个属于 SO 域 A，后两个属于 B，各 SO 域用 Clos，两域以双向 400 Gbit/s 互联”映射为 8 NPU、4 个 SU 域、2 个 SO 域。`400 Gbit/s = 5e10 byte/s` 是每方向的标称速率；效率另计。未提供的 SU/SO/host 速率及延迟需要补充或明确记为假设，不借用案例数值作为硬件规格。

编译器输出 `hardware.topology: explicit`、`routing_policy: su_so`、完整 SU/SO/host 有向边，以及 `sources` / `assumptions`；多个 SO 域及域间边落到显式交换节点与链路。完整场景使用 `--base` 合并，硬件片段仅用于构建场景。原始 topology spec 需与生成结果一起保留，不能把高层字段直接交给引擎。

当前子 skill 编译器使用等大、连续 SU 域和完全位于单一 SU 域内的主机，要求 `su_size % npus_per_host == 0`；不支持非均匀/重叠域成员或任意物理卡映射。多个 SO 域的有向域间图必须强连通。对于刻意隔离的 fabric，分别生成独立场景，或在原生 explicit 图中按任务验证连通性；不能补画不存在的互联。

编译后先检查每类实际路径：同 SU 的 NPU 对、跨 SU 同 SO 域的 NPU 对、跨 SO 域的 NPU 对、H2D/D2H，以及已启用的 remote/SSD。拓扑图标明 SU/SO 边界、网关、链路方向及速率；大图可以折叠重复结构，说明折叠口径。通信统计继续使用 payload 与逐跳物理字节的既有账本，域级归因需用生成图的成员/链路对应关系。

SU/SO 域规划是固定图输入。修改总卡数、SU 大小或域成员的候选必须重新编译，原生 `size-su` 不会重建显式图；仅扫描不改变物理结构的负载/精度参数时可复用该图。


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

反方向需要另一条有向边。`shared_group` / `shared_Bps` 可显式指定共享物理瓶颈，例如双向共用的半双工链路。显式拓扑需要提供任务所用的所有 host/remote/SU/SO 连通路径；引擎不会额外补上默认 host 链路。路由的 link IDs 必须连续并到达目标；未指定的路径使用确定性最短跳路由。`routing_policy` 默认 `shortest_hop`，保持显式图允许混合链路的原有行为；`su_so` 对 NPU 通信按 `su_size` 强制同 SU 仅用 SU 边、跨 SU 仅用 SO 边，显式 routes 也需遵守，避免经 host 或其他 fabric 绕行形成不真实的捷径。

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

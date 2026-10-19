# 拓扑规格与编译契约

本规格是自然语言与引擎原生配置之间的中间输入，和完整仿真场景区分。严格 schema 由同目录下的 `../scripts/build_topology.py --schema` 输出；未知字段会报错。JSON 与 YAML 使用同一契约。中间输入采用严格类型：卡数/宽度/索引写整数，速率与延迟写数值，不写带单位的字符串或布尔值；YAML 科学计数推荐 `5.0e+10`，避免被解析器读成字符串。

## 字段与域定义

| 字段 | 必需 | 含义 |
|---|---|---|
| `schema_version` | 是 | 字符串 `'1.0'` |
| `name` | 否 | 此拓扑的名称 |
| `npu_count` | 是 | 总卡数，正整数 |
| `su_size` | 是 | 每个 SU 域的卡数，须整除总卡数 |
| `npus_per_host` | 是 | 每台主机的卡数，须整除 `su_size` 且不大于 `su_size` |
| `scale_up` | 是 | 所有 SU 域共用的 SU 拓扑及链路参数 |
| `scale_out` | 是 | 所有 SO 域共用的域内拓扑及链路参数，附带 `domains` |
| `host_link` | 是 | H2D/D2H 主机链路参数 |
| `inter_domain_links` | 多 SO 域时需要 | SO 域网关之间的有向连接列表 |
| `sources` | 否 | 字段/对象名称到来源记录的映射 |
| `assumptions` | 否 | 假设说明字符串列表 |

`npu{i}` 属于 SU 域 `i // su_size` 和主机 `cpu{i // npus_per_host}`。SU 域索引从 0 开始，域 `j` 的成员为 `npu{j * su_size}` 到 `npu{(j + 1) * su_size - 1}`。此版本要求连续、统一大小的 SU 域和主机；没有任意 NPU 成员列表或不等大小 SU 域字段。

SO 域是 SU 域的分区。`scale_out.domains` 中每项为：

```yaml
id: so-a
su_domains: [0, 1]
```

`id` 使用字母、数字、`_` 或 `-`，且全局唯一；`su_domains` 是非空、无重复的 SU 索引列表。所有 SO 域的成员集合必须恰好覆盖全部 SU 域，每个 SU 域只出现一次。不能把 `su_size` 写成 SO 域中的 SU 个数。

## 拓扑与链路参数

`scale_up` 和 `scale_out` 均必需 `topology`、`bandwidth_Bps`、`latency_s`、`efficiency`。`host_link` 必需后三项，不包含 `topology`。

| 字段 | 约束与作用 |
|---|---|
| `topology` | `full_mesh/ring/mesh/clos/fat_tree` |
| `bandwidth_Bps` | 每条有向链路的原始带宽，byte/s，正数 |
| `latency_s` | 每条边的启动延迟，second，非负 |
| `efficiency` | 有效带宽系数，范围 `(0, 1]` |
| `injection_Bps` | 可选端点注入带宽，正数；省略时取 `bandwidth_Bps` 并记录假设 |
| `uplink_Bps` | 可选聚合上联容量，正数，见下述共享口径 |
| `mesh_width` | mesh 可选二维布局宽度，正整数；省略时由终端数确定 |
| `leaf_size` | Clos/Fat-tree 可选每 leaf 的最大终端数，正整数，最后一个 leaf 可部分填充 |
| `spine_count` | Clos/Fat-tree 可选 spine 数，正整数，省略时采用单 spine |

`host_link` 仅接受 `bandwidth_Bps/latency_s/efficiency/injection_Bps`。`mesh_width` 仅用于 mesh；`leaf_size/spine_count` 仅用于 Clos/Fat-tree；`scale_up.uplink_Bps` 仅用于 Clos/Fat-tree。错误组合和未支持字段会被拒绝，不会默默忽略。

SU 图的终端是同一 SU 的 NPU。SO 域内图的终端按 SU 索引排序为 `su{index}.gateway`，再加入本域 `so.{id}.gateway`；域间链路连接后者。每个 NPU 另有 SO attachment 到自己的 SU 网关，attachment 与 SU 链路分开计账。更改域成员会改变 SO 域内图，不只是改变标签。

- `full_mesh` 为全部终端的双向直连；`ring` 为相邻终端的双向环；`mesh` 为二维相邻终端的双向网格。反方向是独立有向边。
- `clos/fat_tree` 均为终端 → leaf → spine 的两层流体容量抽象，支持多个 spine；确定性最短跳选择一条路径，不进行 ECMP。`leaf_size` 是每个 leaf 的最大终端数：例如 SO 域有 2 个 SU 网关和 1 个域网关共 3 个终端，`leaf_size: 2` 产生终端数为 2、1 的两个 leaf。自然语言若要求每个 leaf 恰好满配，应指出终端数不整除的冲突或澄清，不能补造终端。
- Clos leaf 上联总容量取 `uplink_Bps`，省略时取 `bandwidth_Bps * leaf_size`。leaf 到每个 spine 的边使用此速率，并由按方向的 shared group 限制所有 spine 边的总容量；增添 spine 不增加 leaf 聚合容量。
- 每个 SU 的 SO attachment 按发送/接收方向共享总容量 `scale_out.uplink_Bps`；省略时取 `scale_out.bandwidth_Bps * (su_size / npus_per_host)`。SO 主机 NIC 使用 `scale_out.injection_Bps * efficiency`。`scale_out.uplink_Bps` 也约束 SO Clos 的 leaf 聚合上联；不能把它解释为两项可相加的额外带宽。
- 链路有效容量为原始速率乘 `efficiency`。拓扑使用各边延迟累加；交换机没有额外的包级处理、队列或 buffer 模型。

不要仅凭“Clos”或“Fat-tree”名称补出交换芯片端口数、完整生产布线、oversubscription、协议、实测延迟或 SLO。

## SO 域间连接

根字段 `inter_domain_links` 使用有向定义：

```yaml
- id: so-a-to-so-b
  src_domain: so-a
  dst_domain: so-b
  bandwidth_Bps: 5.0e+10
  latency_s: 5.0e-6
  efficiency: 0.8
```

每条链路必需 `id/src_domain/dst_domain/bandwidth_Bps/latency_s/efficiency`。`id/shared_group` 同样仅使用字母、数字、`_` 或 `-`；`shared_group/shared_Bps` 须同时提供或同时省略。源和目的必须指向已声明且不同的 SO 域，link ID 必须唯一。多个 SO 域时，其有向域间图必须强连通。双向链路需要明确写两条，不自动由一条推导反向。可选 `shared_group/shared_Bps` 用于共用物理瓶颈；同组有效共享容量必须一致。共享容量本身也乘相应 `efficiency`。

实际边连接 `so.{src_domain}.gateway` 和 `so.{dst_domain}.gateway`，使用原生 `domain: SO`，原生 link ID 为 `inter:{id}`。自然语言中“全互联”“环”“双向”可转成这些有向定义；只给“SO 域 A 和 B”不能推断两域间的速率和布线。

## 单位、来源与缺参

- `400 Gb/s = 400e9 bit/s / 8 = 5e10 byte/s = 50 GB/s`；`400 GB/s = 4e11 byte/s`，两者相差 8 倍。裸写“400G”时保留解释口径，无法确定时追问或列为未定参数。
- `1 us = 1e-6 s`；容量/速率使用十进制 GB/TB，GiB/TiB 则分别用 `2**30/2**40`。区分单向速率、双向合计、每端口和聚合注入。
- 中间输入的 `sources` 仅接受当前拓扑负责的 `hardware.npu_count/su_size/npus_per_host/topology/routing_policy/scale_up/scale_out/host_link/explicit_links/routes` 及这些路径的子字段，避免改写基场景的模型或计算 profile 来源。链路数值使用 `hardware.scale_up.*`、`hardware.scale_out.*`、`hardware.host_link.*` 等字段路径；SO 分区可用 `hardware.scale_out.domains` 作为来源元数据名称；原生硬件对象本身不新增此字段。重新编译应替换旧拓扑来源，不能保留旧 profile 的 measured 速率标签。
- 来源记录为 `{status: user/public/measured/assumed, description: ..., uri: ..., revision: ...}`；`uri/revision` 可省略。保留用户提供成员关系及资料依据，合成链路参数用 `assumed`。
- 未给硬件型号、实际端口或测量时，图可用于设计讨论；不能把演示数字写成某型号公开规格。结构已知但关键链路参数缺失时，可先交付草案和缺参清单；不要宣称它已成功编译或给出时间结论。

## 输出、合并与校验

指定 `--output` 时，扩展名 `.json` 输出 JSON，否则输出 YAML；省略 `--output` 则把 JSON 写到 stdout。没有 `--base` 时输出只有 `schema_version/hardware/sources/assumptions` 的硬件片段；编译器检查原生硬件定义与拓扑连通性。它缺少 model/workload，不能作为完整场景运行 `simulate` 或原生场景 `validate`。

`--base` 必须指向真实完整场景。编译器先解析原场景及 calibration，再替换卡数、拓扑、主机映射、SU/SO/host 链路字段，清除旧 `explicit_links/routes`，保留 model/workload/execution、NPU/CPU profile、remote/SSD 和 kernel profile。已启用 remote/SSD 时生成主机 bridge 路径；输出解除 calibration 引用，避免重载时以旧校准覆盖新拓扑。

`--base` 不自动更改并行策略。普通完整配置须满足 `tp * pp * ep * dp == npu_count`、`dp % fsdp == 0` 及 rank 放置要求；启用父 skill 的 PD 分离时按引擎双池规则 `npu_count == 2 * tp * ep`、DP=PP=1，并验证 P/D 物理映射。卡数冲突应明确调整用户接受的执行策略再编译，不能为了通过校验偷偷改变全局 batch 或 DP。

`--max-links` 默认 200000，可显式设置正整数上限；编译器在展开前估计链接数并限制大小。大 full-mesh 会产生平方级有向边；不要将提高上限当作保证仿真可执行。编译器错误应修正输入后再运行，不手工删边绕过校验。

Mermaid 图从编译后的真实节点/边派生，节点用 Mermaid 安全别名，显示标签保留原生 ID。图中标明原始速率、效率、有效速率的口径及 shared group；双向箭头只可合并两条确实存在且参数一致的反向边，半双工/共享容量另加标签。编译输出和结构化输入始终保留完整有向定义。

## 可重跑的两 SO 域示例

[../examples/two_domains.yaml](../examples/two_domains.yaml) 对应以下自然语言请求：

> 生成一个 8 卡超节点拓扑，每 2 卡为一个 SU 域、每台主机 2 卡，共 4 个 SU 域。SU 使用 full-mesh。SO 域 so-a 含 SU 0、1，so-b 含 SU 2、3；各 SO 域使用 leaf_size 为 2、2 个 spine 的 Clos 抽象。两域网关用 400 Gb/s 双向连接。先用合成演示参数：SU 每方向 900 GB/s、1 us、效率 0.85；SO 每方向 400 Gb/s、5 us、效率 0.8，聚合上联每方向 50 GB/s；host 每方向 64 GB/s、2 us、效率 0.85。只生成拓扑和图，不补充模型或负载。

```text
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py skills/supernode-modeling/subskills/supernode-topology/examples/two_domains.yaml --output outputs/two-domains-topology.yaml
```

该样例没有设备计算或存储 profile。若与 `examples/demo_training.yaml` 结合，先检查基场景的 4 卡配置与执行布局：8 卡拓扑通常需要在独立副本中明确把 `execution.dp` 改为 2，并说明全局 batch 的变化后再用 `--base`。不能直接把原 4 卡场景当作匹配的 8 卡负载。

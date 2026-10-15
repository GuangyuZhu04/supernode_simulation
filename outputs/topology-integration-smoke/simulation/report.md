# 超节点资源建模报告

状态：**feasible**；模式：`simulate`。

模型：`synthetic-moe-demo`；场景：`inference`；权重：`fp8`。

硬件：`synthetic-8npu-two-so-domains`；NPU 数量：8；每颗 HBM：80 GB。

## 性能与约束

| 指标 | 数值 |
|---|---:|
| 总执行时间 / s | 0.00368183 |
| 执行 TTFT / s | 0.00134596 |
| 批内最大平均 TPOT / s | 0.000861949 |
| 聚合吞吐 / tokens/s | 3,259.25 |
| 输入吞吐 / tokens/s | 104,296 |
| 每副本吞吐 / tokens/s | 1,629.62 |
| 理想执行时间必要下界 / s | 0.00360115 |
| SLO throughput_tokens_s | 满足；目标 1 |
| SLO ttft_s | 满足；目标 1 |
| SLO tpot_s | 满足；目标 1 |

## 全局逻辑容量

单位为十进制 TB；实际分片、复制和生命周期见设备明细。

| 组成 | TB |
|---|---:|
| weight | 0.002 |
| kv_data | 4.58752e-07 |
| kv_scales | 1.4336e-08 |
| kv_page_tables | 4.48e-10 |
| engram_table | 0.001 |

## 存储分解：组件 × 状态

口径：**仿真实际并发峰值**。表格使用十进制 GB（1 GB = 10⁹ byte）；CSV 和 JSON 使用 byte。

### 全模型逻辑参数（分片和复制之前）

结构清单按唯一参数所有权计数；共享权重别名不重复计入。精确形状、参数量、层号、分片标记和别名见 model_parameters.csv。

| 组件 | 总参数 / B | 每 token 活跃参数 / B | 独立参数张量数 | 层数 |
|---|---:|---:|---:|---:|
| Unassigned | 2 | 0.8 | 1 | 1 |

### 物理内存池容量与峰值

容量和分解均为每个内存池的数值；相同容量与分解的内存池合并展示，完整设备明细保留在 CSV。

| 池组 | 等价内存池 | 数量 | 单池容量 / GB | 单池固定驻留 / GB | 单池峰值或驻留 / GB |
|---|---|---:|---:|---:|---:|
| G1 | cpu0.dram, cpu1.dram | 2 | 2,000 | 1 | 1.0003 |
| G2 | cpu2.dram, cpu3.dram | 2 | 2,000 | 1 | 1.00017 |
| G3 | npu0.hbm, npu1.hbm, npu2.hbm, npu3.hbm | 4 | 80 | 2.68275 | 3.18488 |
| G4 | npu4.hbm, npu5.hbm, npu6.hbm, npu7.hbm | 4 | 80 | 2.68275 | 3.1846 |

### 每池同一峰值时刻的构成

每组中各组件和状态可以相加得到上表的单池总值。不同池的峰值时刻可能不同。

**G1 · cpu0.dram, cpu1.dram · 单池 GB**

| 组件 | engram_table | KV | 合计 |
|---|---:|---:|---:|
| Unassigned | 1 | 0.000304704 | 1.0003 |
| **总计** | 1 | 0.000304704 | **1.0003** |

**G2 · cpu2.dram, cpu3.dram · 单池 GB**

| 组件 | engram_table | KV | 合计 |
|---|---:|---:|---:|
| Unassigned | 1 | 0.00016928 | 1.00017 |
| **总计** | 1 | 0.00016928 | **1.00017** |

**G3 · npu0.hbm, npu1.hbm, npu2.hbm, npu3.hbm · 单池 GB**

| 组件 | 激活 | 缓冲 | expert_buffer | kv_write_buffer | 量化元数据 | 运行时预留 | 权重 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| System | 0 | 0.564393 | 0 | 0 | 0 | 2 | 0 | 2.56439 |
| Unassigned | 0.00131072 | 0 | 0.000393216 | 3.3792e-05 | 0.01875 | 0 | 0.6 | 0.620488 |
| **总计** | 0.00131072 | 0.564393 | 0.000393216 | 3.3792e-05 | 0.01875 | 2 | 0.6 | **3.18488** |

**G4 · npu4.hbm, npu5.hbm, npu6.hbm, npu7.hbm · 单池 GB**

| 组件 | 激活 | 缓冲 | expert_buffer | kv_write_buffer | 量化元数据 | 运行时预留 | 权重 | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| System | 0 | 0.564393 | 0 | 0 | 0 | 2 | 0 | 2.56439 |
| Unassigned | 0.00104858 | 0 | 0.000393216 | 1.6896e-05 | 0.01875 | 0 | 0.6 | 0.620209 |
| **总计** | 0.00104858 | 0.564393 | 0.000393216 | 1.6896e-05 | 0.01875 | 2 | 0.6 | **3.1846** |

### 各组件独立峰值（不可相加）

下表每项取该组件在整段已观测执行中的最大值，各项可能出现在不同时间；相加不能得到设备峰值。

| 池组 | 组件 | 单池组件独立峰值 / GB |
|---|---|---:|
| G1 | Unassigned | 1.0003 |
| G2 | Unassigned | 1.00017 |
| G3 | System | 2.56439 |
| G3 | Unassigned | 0.620488 |
| G4 | System | 2.56439 |
| G4 | Unassigned | 0.620209 |

所有内存池同一时刻的合计峰值为 29.4787 GB，发生于 0.00104551 s。该数值按全局时间线重放得到，未相加各池独立峰值；不同内存类型的容量不能互相替代。

张量、层号、形状、精度、组件来源与申请/释放时刻见 tensors.csv；组件状态交叉表见 component_memory.csv，状态汇总见 state_memory.csv，逐池峰值见 memory_peaks.csv。

缺失组件归属保留为 `Unassigned`；通用缓冲和运行时归为 `System`。已知优化器按其状态语义拆分；无法确定语义的合并状态才保持未拆分，不推定 Adam/Muon 的具体张量。

## 通信

物理链路字节量按路径上的每一跳累计；与应用 payload 分开统计。

| 网络域 | 物理传输 / GB |
|---|---:|
| H2D | 0.00283034 |
| D2H | 0.000827904 |
| SU | 0.0128451 |
| SO | 0.0706478 |

| 应用语义 | 传输 payload / GB |
|---|---:|
| lookup | 0.000802816 |
| kv_write | 0.000827904 |
| tp | 0.0128451 |
| dispatch | 0.00321126 |
| combine | 0.00321126 |
| kv_read | 0.00202752 |

应用语义与 collective 分类是同一批 payload 的不同视图，不能相加。

按吞吐或训练步 SLO 得到的整任务时间窗口：12 s。各资源平均速率必要下界见 resources.csv；还需满足局部依赖与争用约束。

## 主要资源下界

| 资源 | 独占服务时间下界 / s |
|---|---:|
| npu0.hbm | 0.00262972 |
| npu1.hbm | 0.00262972 |
| npu2.hbm | 0.00262972 |
| npu3.hbm | 0.00262972 |
| npu4.hbm | 0.00196085 |
| npu5.hbm | 0.00196085 |
| npu6.hbm | 0.00196085 |
| npu7.hbm | 0.00196085 |
| npu0.compute | 0.000155411 |
| npu1.compute | 0.000155411 |

缓存容量和 MTE 通道数需要 kernel/硬件校准；当前输出工作集与搬运量下界。

## 参数来源

- `*`：assumed；Synthetic MoE and hardware; fixed ragged batch, uniform expert routing.
- `hardware.npu_count`：user；User requested eight NPUs.
- `hardware.su_size`：user；User requested two NPUs in each contiguous SU domain.
- `hardware.npus_per_host`：user；User requested two NPUs per host.
- `hardware.topology`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.routing_policy`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.scale_up`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.scale_out`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.host_link`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.explicit_links`：assumed；Structure and SO domain membership are user-specified; all link rates, latency and efficiency are synthetic, including both 400 Gb/s domain links.
- `hardware.routes`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.name`：user；Compiled from the supplied structured topology specification.；C:\D_disk\Github\supernode_simulation\skills\supernode-modeling\subskills\supernode-topology\examples\two_domains.yaml
- `hardware.scale_out.domains`：user；so-a contains SU 0 and 1; so-b contains SU 2 and 3.
- `hardware.scale_up.bandwidth_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_up.latency_s`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_up.efficiency`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_up.injection_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_out.bandwidth_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_out.latency_s`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_out.efficiency`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_out.injection_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.scale_out.uplink_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.host_link.bandwidth_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.host_link.latency_s`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.host_link.efficiency`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.
- `hardware.host_link.injection_Bps`：assumed；Synthetic demonstration parameter; not a measured production hardware specification.

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
- Integration fixture explicitly doubles DP and maps EP communicators across the two SO domains.
- All SU domains have equal size and contiguous NPU indices.
- 400 Gb/s is 50 GB/s per direction before efficiency; 900 GB/s is byte/s.
- Reverse SO domain links are independent full-duplex edges.
- Device memory, compute, model, workload and execution strategy are intentionally unspecified.
- Topology bandwidths are bytes/s per direction; generated reverse links have independent capacity unless an explicit shared_group couples them.
- Uniform SU domains use contiguous NPU indices; each SO fabric includes its SU gateways followed by one domain gateway terminal.
- NPU routes use SU only within one SU domain and SO only across SU domains; deterministic shortest-hop routing does not model ECMP or packet switching.
- scale_out.clos is a two-tier leaf/spine fluid abstraction; each leaf's spine links share one aggregate capacity per direction.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。
- 配置 SHA256：`9b0d93f36ff405c9655948a2711954c65e675e50e4bee47f737374e2420b177b`。

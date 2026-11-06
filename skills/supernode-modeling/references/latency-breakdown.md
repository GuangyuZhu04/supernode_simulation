# 推理 / 训练时延与算子占比

每次推理结果在原有容量、通信量、吞吐和 SLO 之外，报告 TTFT 和 TPOT（用户称 TPOP）的总时延与组成。`tpot_s` 是规范字段，`tpop_s` 是相同数值的兼容别名，不能把两者相加或视为两个不同指标。

TTFT 取所有 DP 副本、所有请求中最大的首 token 完成时间；TPOT 先求每请求首 token 之后的平均 token 间隔，再取请求间最大值。两项最大值可能来自不同请求。`metrics.latency_breakdown.ttft` 和 `.tpot` 各自保留 `replica`、`sequence`、`total_s` 和 `components_s`；分项属于决定该指标的同一个请求，不能分别取每类分项最大值再组合。

| `components_s` 字段 | 含义 |
|---|---|
| `compute_s` | kernel 的计算 / 访存服务时间及启动，不等同于仅 FLOPs / 峰值算力；新算子账本将其细分 |
| `communication_node_s` | 同主机 SU 或 host 搬运路径的通信服务时间 |
| `communication_scaleup_s` | 跨主机 SU 资源的通信服务时间 |
| `communication_scaleout_s` | SO 资源的通信服务时间 |
| `communication_other_s` | 内存带宽限制或未分类通信路径的服务时间 |
| `synchronization_wait_s` | 显式纯同步 / barrier 的时间 |
| `scheduling_wait_s` | 依赖已就绪后等待独占调度槽位的时间 |
| `memory_wait_s` | 依赖已就绪后等待内存准入的时间 |
| `resource_contention_s` | 共享资源争用带来的额外执行时间 |
| `other_s` | 其余已归因时间 |

拆解依据仿真实际完成顺序和关键依赖路径。分项可以相加得到对应总时延；所有 rank 的服务时间总和、独立组件最大耗时、并发通信流量比例都不是这个口径。依赖同步所等的上游工作已经归到该上游的计算 / 通信 / 等待分项，不能再附加一份“同步等待”。通信服务时间按实际路径的限制资源归因，启动延迟按各跳链路域归因；跨多个域的传输不会把各域完整服务时间重复相加，也不按字节占比摊分时延。共享资源造成的减速独立计入争用项。

TPOT 对相邻 token 完成之间的时间窗分别归因，再取该请求的平均值。跨越 token 边界的事件按时间窗覆盖比例分配服务、争用或准入等待；引擎未保留更细的事件内进度，因此分项是这个分辨率下的模型归因。调度槽位和内存准入同时阻塞时优先计入槽位等待，不重复累计。完成时间相同的前驱、独占服务时间相同的限制资源按标识稳定择一，不把同一段时间重复放到多个分项。

JSON 和 CSV 使用秒；中文报告与网页拆解表使用毫秒。普通 `simulate` 的 `result.json` 保留完整拆解；`report.md` 展示聚合两项；`latency_breakdown.csv` 每行表示一个候选的一项聚合指标并保留请求标识。`sweep`、`size-su` 和 `feedback` 的 CSV 包含所有已评估候选，feedback 报告另展示最佳可行候选的拆解。流式推理 API 的 `metrics` 同样保留拆解，局部阶段按实际执行范围解释。

`estimate` 没有实际调度轨迹，不能把解析下界写成已执行 TTFT / TPOT；训练不定义这两项推理指标。执行未完成不提供完整执行指标；独立 Decode 不提供 TTFT；只输出一个 token 不提供 TPOT。相应记录为 `null`，CSV 留空，报告显示“—”，并保留不可用原因，不补零。固定 batch 仿真不包含请求到达队列、冷启动加载或生产 P99 延迟。

## 最终报告的算子表

在原有总时延、网络域拆解和资源统计之外，展示两层视图：窗口总开销，以及按耗时降序的完整算子表。推理窗口仍取上面同一个请求的 TTFT / TPOT；训练使用 `metrics.training_latency_breakdown.step`（所有训练步平均）、`first_step`、`steady_step`（第 2 步起平均，至少两步）。训练窗口按实际 step completion 划分，覆盖前向、反向、重算、同步、optimizer 和 offload 等实际路径。阶段名称使用事件的 `phase`，没有证据的阶段保留 `unspecified`。

每个窗口记录的 `operator_breakdown` 包含 `total_s`、`components_s`、`component_shares_pct` 和 `operators`。每个算子行保留 `phase`、`component`、`operator`、`total_s`、`share_pct`、`components_s`、`component_shares_pct`。

| 展示列 | 计算口径 |
|---|---|
| 算子耗时 / ms | `operators[].total_s × 1000`；同阶段、组件、算子名合并，不累加并发 rank |
| 占窗口 / % | `share_pct = 算子归因耗时 / 窗口总时延 × 100` |
| 算子内计算 / 通信 / 访存 / 启动等 % | `component_shares_pct = 该项归因耗时 / 该算子归因耗时 × 100` |

`components_s` 的新细分口径与旧 `latency_breakdown.components_s` 分开保存：

- `compute_s`：限制资源为计算或 IOPS 的基线服务。
- `memory_access_s`：限制资源为 HBM / DRAM / memory 的基线服务；包括通信端点访存限制，不等于 `memory_wait_s`。
- `communication_s`：网络/搬运限制的基线服务、链路启动及整 collective 计时边界。整 collective profile 可能含打包，不再臆造内部拆分。
- `launch_s`：kernel 的显式启动开销。融合 kernel 保留融合后的启动成本，不再添加被融合成员的启动。
- `synchronization_wait_s`、`scheduling_wait_s`、`memory_wait_s`、`resource_contention_s`：保留原有等待和争用含义，不能为让计算/通信/访存三项加和为 100% 而隐藏它们。
- `unattributed_service_s`：整 kernel profile 限制的服务时间，不能由该总时间反推出计算/访存实测比例。
- `other_s`：其余有时间贡献但无法分类的开销。

引擎以 `max(work / capacity)` 建模同时需求：计算和访存有重叠，细分表将基线服务归给限制资源，同值按资源 ID 稳定择一。因此一个算子的某项可能是 0%，表示没有归入该项的关键路径时间，不能解释为没有计算或访存需求。启动与争用等另列，各行内部比例与算子占窗口比例各自加和为 100%；舍入可产生尾差。TPOT 和训练平均步必须先聚合分项秒值再计算比例，不能直接平均各 token / 各步的百分比。跨窗口裁剪继续遵守上面的时间覆盖口径。

通信使用实际通信算子名称单列，不凭前后相邻关系将通信时间摊给某个 GEMM。`kernel_traces` 展开后使用真实子算子名；未展开的组件 kernel 保留现有粒度，融合算子按融合整体展示。没有组件标签时保留 `Unassigned`。表中只列对该关键路径有时间贡献的算子；不在该路径上不表示算子未执行。此表不是全卡累计工作时间或硬件 profiler 的利用率。

`report.md` 和网页结果页同时展示窗口总开销与完整算子排行，网页可以切换窗口。`operator_breakdown.csv` 每行是一项候选、一个窗口、一个算子，秒值列以 `_s` 结尾，比例列以 `_pct` 结尾；`window_total_s` 是外层分母，`total_s` 是算子内分母。候选搜索保留所有候选的表，feedback 正文展示推荐候选。算子占比在原始 trace 精简前生成，紧凑导出和流式缓存仍保留同样的数值。

零耗时窗口的比例为 `null` / 空值 / “—”，不生成 NaN 或无穷值；旧结果缺少算子账本、仅估算、执行失败、窗口未定义时明确标注不可用，不从精简事件记录重建百分比。若只有一个训练步，稳态步不可用；TTFT/TPOT 在训练中仍不可用。

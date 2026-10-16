# TTFT / TPOT（TPOP）时延拆解

每次推理结果在原有容量、通信量、吞吐和 SLO 之外，报告 TTFT 和 TPOT（用户称 TPOP）的总时延与组成。`tpot_s` 是规范字段，`tpop_s` 是相同数值的兼容别名，不能把两者相加或视为两个不同指标。

TTFT 取所有 DP 副本、所有请求中最大的首 token 完成时间；TPOT 先求每请求首 token 之后的平均 token 间隔，再取请求间最大值。两项最大值可能来自不同请求。`metrics.latency_breakdown.ttft` 和 `.tpot` 各自保留 `replica`、`sequence`、`total_s` 和 `components_s`；分项属于决定该指标的同一个请求，不能分别取每类分项最大值再组合。

| `components_s` 字段 | 含义 |
|---|---|
| `compute_s` | kernel 的计算 / 访存服务时间，不等同于仅 FLOPs / 峰值算力 |
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

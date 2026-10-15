# 超节点资源建模报告

状态：**completed**；模式：`size-su`。

结果限定于显式候选集合和给定调度策略，完整配置与失败原因见 JSON。

| 候选 | 状态 | SU 卡数 | 总卡数 | 执行时间 / s |
|---|---|---:|---:|---:|
| single 搜索 | feasible | 4 | — | — |
| multi 搜索 | feasible | 1 | — | — |
| single/0 | feasible | 4 | 4 | 0.00271194 |
| single/1 | feasible | 8 | 8 | 0.00271194 |
| multi/0 | feasible | 1 | 4 | 0.00437921 |
| multi/1 | feasible | 1 | 8 | 0.00437921 |
| multi/2 | feasible | 2 | 4 | 0.00306502 |
| multi/3 | feasible | 2 | 8 | 0.00306502 |
| multi/4 | feasible | 4 | 4 | 0.00271194 |
| multi/5 | feasible | 4 | 8 | 0.00271194 |
| multi/6 | feasible | 8 | 8 | 0.00271194 |

## 假设与复现

- Minimum is over the complete declared candidate set and fixed scheduler only.
- Without parallel_candidates, TP/PP/EP/FSDP are retained and DP is derived from candidate device count.
- batch_size is per DP replica; changing DP changes aggregate workload.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。

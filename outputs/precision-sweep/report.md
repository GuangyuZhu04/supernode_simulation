# 超节点资源建模报告

状态：**completed**；模式：`sweep`。

结果限定于显式候选集合和给定调度策略，完整配置与失败原因见 JSON。

| 候选 | 状态 | SU 卡数 | 总卡数 | 执行时间 / s |
|---|---|---:|---:|---:|
| 0 | estimated | 8 | 16 | — |
| 1 | estimated | 8 | 16 | — |
| 2 | estimated | 8 | 16 | — |
| 3 | estimated | 8 | 16 | — |

## 假设与复现

- Each combination is independently validated. Invalid combinations are retained with diagnostics.
- 固定负载执行模型，不含请求到达排队或生产尾延迟。
- FP8 scale、转换缓冲与额外副本按配置单独计入。
- 未经实测校准的合成配置用于架构比较。

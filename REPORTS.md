# 仿真结果报告索引

本分支归档本地已经生成的仿真报告及配套汇总数据，保留原 outputs/ 路径。快照时间：2026-10-15T19:31:52+08:00（Asia/Shanghai）。

共 17 份报告，含 21 个 Markdown/HTML 报告文件。原始报告字节保持不变；容量不可行、SLO 失败、外推条件及未校准假设均以原报告为准。

> **Ascend 960/970 当前为阶段快照：完成 152/288 组，complete=false，来源/审计问题 1 项。原报告生成时间：2026-10-15T19:27:50+08:00。** 此快照不能视为全部扫描的最终结论。

| 报告 | 网页版本 | 归档状态 |
|---|---|---|
| [10t-fp16：超节点资源建模报告](outputs/10t-fp16/report.md) | — | 已有归档结果；边界见原报告 |
| [10t-fp8：超节点资源建模报告](outputs/10t-fp8/report.md) | — | 已有归档结果；边界见原报告 |
| [10t-inference：超节点资源建模报告](outputs/10t-inference/report.md) | — | 已有归档结果；边界见原报告 |
| [active-sweep：超节点资源建模报告](outputs/active-sweep/report.md) | — | 已有归档结果；边界见原报告 |
| [Kimi K3 约10T：Ascend 960DT / 970 推理仿真](outputs/ascend960-970-kimi-10t/report.md) | [HTML](outputs/ascend960-970-kimi-10t/report.html) | 阶段快照，扫描未完成 |
| [Ascend 950DT × Kimi K3：1024 卡长上下文流量仿真](outputs/ascend_a5_1024_long_context/report.md) | [HTML](outputs/ascend_a5_1024_long_context/results.html) | 已有归档结果；边界见原报告 |
| [Kimi K3 专家扩容至约10T：1024卡推理流量与延迟](outputs/kimi-k3-10t-1024/report.md) | [HTML](outputs/kimi-k3-10t-1024/report.html) | 已有归档结果；边界见原报告 |
| [Kimi K3 约10T：Megatron SP16 / SP32 推理仿真](outputs/kimi-k3-10t-sp/report.md) | [HTML](outputs/kimi-k3-10t-sp/report.html) | 已有归档结果；边界见原报告 |
| [Kimi K3 × Ascend 950DT：参考场景对照](outputs/kimi-k3-ascend950-comparison.md) | — | 已有归档结果；边界见原报告 |
| [kimi-k3-ascend950-fp16：超节点资源建模报告](outputs/kimi-k3-ascend950-fp16/report.md) | — | 已有归档结果；边界见原报告 |
| [kimi-k3-ascend950-fp8：超节点资源建模报告](outputs/kimi-k3-ascend950-fp8/report.md) | — | 已有归档结果；边界见原报告 |
| [kimi-k3-ascend950-mxfp4：超节点资源建模报告](outputs/kimi-k3-ascend950-mxfp4/report.md) | — | 已有归档结果；边界见原报告 |
| [kimi-k3-ascend950-training-muon-128：超节点资源建模报告](outputs/kimi-k3-ascend950-training-muon-128/report.md) | — | 已有归档结果；边界见原报告 |
| [kimi-k3-ascend950-training-muon：超节点资源建模报告](outputs/kimi-k3-ascend950-training-muon/report.md) | — | 已有归档结果；边界见原报告 |
| [precision-sweep：超节点资源建模报告](outputs/precision-sweep/report.md) | — | 已有归档结果；边界见原报告 |
| [su-search：超节点资源建模报告](outputs/su-search/report.md) | — | 已有归档结果；边界见原报告 |
| [simulation：超节点资源建模报告](outputs/topology-integration-smoke/simulation/report.md) | — | 已有归档结果；边界见原报告 |

包含报告正文、HTML 所需脚本/图表、汇总 CSV、导出结果 JSON、配置及直接引用的审计附件。未归档逐事件 events.csv、timeline.json、tensors.csv、检查点、全部 case/raw 结果、缓存、临时日志和 PPT 构建产物；本分支是报告快照，并非完整复现实验包。

完整文件清单和逐文件 SHA-256 / Git blob SHA-1 见 [report-manifest.json](outputs/report-manifest.json)。GitHub 将 HTML 显示为源码，下载仓库后可在本地浏览器打开。

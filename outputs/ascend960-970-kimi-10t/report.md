# Kimi K3 约10T：Ascend 960DT / 970 推理仿真

生成时间：2026-10-15T20:19:10+08:00（Asia/Shanghai）。完成 288/288 组，prefill 可行 240 组；审计问题 0 个。

| 芯片 | 并行 | 完成 | prefill 可行 | TTFT 范围(s) | decode 步范围(ms) |
|---|---|---:|---:|---:|---:|
| ascend960dt | tp8 | 48 | 24 | 10.76–41.35 | 28.46–35.59 |
| ascend960dt | sp16 | 48 | 48 | 8.57–182.64 | 19.43–26.57 |
| ascend960dt | sp32 | 48 | 48 | 7.80–160.56 | 19.45–26.59 |
| ascend970 | tp8 | 48 | 24 | 5.70–21.71 | 27.62–32.37 |
| ascend970 | sp16 | 48 | 48 | 4.39–95.69 | 18.67–23.41 |
| ascend970 | sp32 | 48 | 48 | 3.96–83.47 | 18.70–23.44 |

本实验使用未来硬件公开路线图参数，属于未实测校准的预测。A6/A7 是用户代称；报告不声称官方已确认其与 960/970 的命名对应。

模型保持 10.171385035872T、3328 个专家、93 层（69 KDA/24 MLA）、top-32/top-64；FP8 权重/计算，FP16 通信/激活/KV，FP32 KDA 状态。

并行分别为 TP8/SP1/EP128、TP16/SP16/EP64、TP32/SP32/EP32。Megatron SP 共享 TP rank：两次 AG+RS/层，embedding RS、output AG；单 token decode 的每个非空 EP 源按 TP 整倍数 padding。MoE 在各 TP lane 独立发送完整 latent。

固定 1024 卡；960DT 使用官方 960E 最大 4096 卡统一互联域的 1024 卡子集。970 采用用户确认的 960E 拓扑继承假设，官方尚未披露 970 独立拓扑。8 卡节点、64 卡分组用于和前次比较；64 卡分组不是已核验的 960/970 实际机柜 BOM。

全部 1024 卡映射为一个非阻塞 SU 域，域内 SO 流量为 0；逻辑路径记一个 SU edge，不能解释为实际仅经过一根线或一层交换机。报告单独保留跨 64 卡分组的 SU 载荷。外部二层四平面 Clos、多轨拓扑不参与本次调度。

互联公开数值 2.2/4.4TB/s 未注明方向。本次按 Tx+Rx 总和保守折半为单向 1.1/2.2TB/s；2μs D2D RTT 折为单向 1μs。带宽方向折半、非阻塞性及传播时延拆分都是建模假设。

FP8 计算峰值来自 2026 路线图；计算效率 50%、通信效率 80% 等沿用原仿真假设，尚无实机校准。非 FP8 的 FP16/BF16/FP32 峰值占位沿用原配置，不是本次查证的 960/970 官方规格。

和归档 950DT 比较时，HBM 容量/带宽、计算峰值及网络拓扑同时改变：原跨 64 卡 400Gb/s SO 改为全域 SU，因此结果不是纯芯片代际加速比。

TTFT 仅在 prefill 可行时给出；TPOT 是首个 decode 间隔。prefill OOM 后的独立 decode 假定压缩 KV 已驻留，列在 standalone_decode_step_s，不能当成该请求已完成 prefill。

planned 是完整阶段需求；actual 在失败阶段仅为执行到失败前已完成组件的通信。physical_*_hopbytes 是模型拓扑路径累计，不是实测端口流量。SI：GB=10^9B，TB=10^12B。

最终扫描使用冻结的同步通信轮执行表示：对通过严格资格检查、各流进度相同的通信轮合并调度事件；这不改变网络算法、每条流载荷或物理资源工作量。适用性、两例原始执行对照与误差检查保存在 audit/synchronous-rounds。generated_coalesced_rounds / generated_original_transfers 仅统计生成图，组件缓存重放不会重新生成事件，不能解释为整个阶段的实际传输次数。

最初的原始逐流执行器 code/runtime 与其 sha256:447b…3645 归档仍保留，原始原生网络对照在 audit/native-reference-cases；最终执行器由实验 manifest.runtime_manifest 指定。旧版本尚未重跑的原始文件标为 stale_provenance，时延留空且不计入完成数量。

## 数据与复现

- [交互报告](report.html)；[288 组结果](summary.csv)；[归档对比](comparison.csv)。
- [执行阶段流量](phase_traffic.csv)；[计划阶段流量](planned_phase_traffic.csv)；[8 卡节点流量](node_traffic.csv)；[64 卡比较分组流量](rack_traffic.csv)。
- 冻结运行版本 `sha256:78919546d7eb336972226e9f71cc396718eafeafc5999327c398eeb025c40620`；运行清单 `code/runtime-synchronous-rounds-manifest.json`；[审计状态](report_status.json)。
- [同步轮资格及等价性审计](audit/synchronous-rounds/README.md)；[288 组资格检查](audit/synchronous-rounds/all-cases-eligibility.json)。

## 来源

- [2026 华为官方路线图与 960E 系统](https://www.huawei.com/cn/news/2026/9/hc-wang-keynote)：960DT 2027Q1，970 2028；960E 最大 4096 卡，统一内存编址。
- [2026 华为官方 960 超节点拓扑说明](https://www.huawei.com/cn/news/2026/9/hc-ascend960-supernode)：正交架构、全液冷、NPO、UnifiedBus；跨超节点二层四平面 Clos 不在本次 1024 卡子集内。
- [华为 2026 会场芯片参数原始幻灯片照片（媒体托管）](https://cdn.wccftech.com/wp-content/uploads/2026/09/Huawei-Ascend-960DT-960PR-970-980-Chips-and-Atlas-SuperPods-_2-1.jpg)：960DT：2 PF8，288GB，9.6TB/s HBM，2.2TB/s 互联；970：3.6 PF8，288GB，14.4TB/s HBM，4.4TB/s 互联。
- [华为 2026 会场 960E 原始幻灯片照片（媒体托管）](https://cdn.wccftech.com/wp-content/uploads/2026/09/Huawei-Ascend-960DT-960PR-970-980-Chips-and-Atlas-SuperPods-_1.jpg)：1024→4096 卡；全域 D2D RTT 2μs；2027Q3。

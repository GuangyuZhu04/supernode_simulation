# 超节点资源与流量仿真 PPT 内容文档

面向模型、系统与网络架构团队，说明 supernode-modeling Skill 如何将模型、负载、并行策略和 SU/SO 拓扑转为可复跑配置，并由 supernode_sim 分析容量、流量、时延与候选方案。

内容基线：2026 年 10 月 19 日仓库内的主 Skill、references 与拓扑子 Skill。主讲 19 页，约 20–25 分钟；附录 3 页，共 22 页。第 14–17 页加入 2026-10-15 的 A5 / Ascend 950DT 上 10T 级 Kimi K3 历史仿真案例；数值复用归档报告与 CSV，不表示已由当前引擎重新执行。附录 A01 仍为合成逻辑容量锚点。

## 一 生成约定

- 主讲严格采用 01–19，附录 A01–A03；不将目录、图源或代码另生成页面。
- 每页 2–3 条短文案和一个主要表格或流程。解释放 speaker notes，来源放页脚；保留限定条件。
- 16:9、白底、深蓝标题与蓝色强调；正文建议 22–26 pt，图中标签不小于 18 pt。精简文字，不以缩小字号压缩内容。
- 概念图不使用暗示实测占比的柱长、面积或百分比。正式数值图必须来自同一场景的真实 JSON/CSV，并写明单位、配置与范围。
- 日期采用 2026-10-19；不填写未提供的作者、机构、联系方式。第 19 页结束主讲，附录按需跳转。

## 二 主讲顺序

| 页码 | 标题 | 建议分钟 |
|---|---|---:|
| 01 | 超节点资源与流量仿真 | 0.5 |
| 02 | 从可编辑网页到可复跑结果 | 1 |
| 03 | 用统一契约固定比较条件 | 1 |
| 04 | 拓扑先编译，再验证实际路径 | 1 |
| 05 | 事件依赖与资源准入决定执行 | 1.5 |
| 06 | 训练同时计算状态、搬运与更新 | 1 |
| 07 | 推理状态随结构与阶段变化 | 1.5 |
| 08 | 用组件账本解释同一时刻的峰值 | 1 |
| 09 | 流量按语义、路径与资源分别计量 | 1 |
| 10 | TTFT 与 TPOT 沿关键路径拆解 | 1.5 |
| 11 | Feedback 在延迟约束下提高 Decode 吞吐 | 1.5 |
| 12 | 按分析目标选择普通或流式运行 | 1 |
| 13 | 配置、结果与失败原因一起交付 | 1 |
| 14 | A5 上的 10T Kimi K3 合成扩容场景 | 1 |
| 15 | 柜间流量主导，Top-k 与 batch 放大通信 | 1.5 |
| 16 | TP8 基线完成 128k，长 Prefill 触及容量边界 | 1.5 |
| 17 | TP/SP 联合调整：容量与时延改善，流量增加 | 1.5 |
| 18 | 结论受负载、抽象与校准范围约束 | 1 |
| 19 | 用同一场景闭合分析与选型 | 0.5 |

建议讲述合计约 22 分钟；讨论与转场后控制在 20–25 分钟。

## 三 主讲内容

### 01 超节点资源与流量仿真

**核心结论** 用同一场景连接容量、通信、时延与方案选择。

**上屏文案**

- supernode-modeling Skill × supernode_sim 引擎
- 训练与固定 batch 推理的资源和流量分析
- 可编辑输入、可解释结果、可复跑方案

**图表与布局** 一条四步流程：场景 → 资源与事件 → 容量 / 流量 / 时延 → 方案判断。副标题“能力、口径与工作流”，日期 2026-10-19。

**讲解备注** 系统回答能否装下、能否满足目标、瓶颈在哪里。Skill 负责组织场景与解释结果，数值计算由仓库或已安装的 supernode_sim 完成，不在 Skill 中维护另一套引擎。

**来源** [主 Skill](../skills/supernode-modeling/SKILL.md)，开头与“解释结果时的关键口径”。

### 02 从可编辑网页到可复跑结果

**核心结论** 本地网页与 CLI/API 使用同一配置和数值引擎。

**上屏文案**

- 默认准备并打开可编辑本地网页
- 输入校验 → 执行任务 → 查看历史快照
- 网页支持 estimate / simulate

**图表与布局** 五步流程：启动或复用服务 → 编辑完整配置 → 校验 → 提交任务 → 图表与产物。流程下方一行文字：“sweep / size-su / feedback / 流式推理继续使用 CLI/API”。

**讲解备注** 网页由 supernode_sim.web 提供本地 HTTP 服务，无需 npm 构建；展示模型、拓扑、硬件、并行和 Workload，支持 JSON、日志、取消、历史快照及结果图表。启动网页本身不提交任务。只有启动器返回 started/reused 且资源检查通过，才可报告服务就绪；实际 URL 以返回值为准。历史输入先复制再编辑。本页说明现有能力，不声称本次已启动服务或运行实验。

**来源** [frontend.md](../skills/supernode-modeling/references/frontend.md)，“页面来源与运行条件”“与仿真任务衔接”。

### 03 用统一契约固定比较条件

**核心结论** 模型、负载、精度、并行与拓扑共同定义一个可比较场景。

**上屏文案**

- 总参数、活跃参数与 activation 分开
- TP × PP × EP × DP = NPU 数
- SP 共用 TP rank；FSDP 属于 DP 子组

**图表与布局** 使用输入契约表，表下标注“byte / second / FLOP；来源：public / user / measured / assumed”。

| 输入 | 必须固定或显式变更 |
|---|---|
| model / workload | 张量结构、训练或推理负载、每 DP 副本 batch |
| execution | 并行布局、各状态精度、offload、rank 放置 |
| hardware / targets | 容量、SU/SO 与链路、时延和吞吐目标 |
| sources / assumptions | 公开资料、用户输入、实测与假设 |

**讲解备注** architecture 按真实张量形状派生参数；只有预算时，未知结构列 Unassigned，不缩放账本凑厂商约数。权重驻留、计算、梯度、优化器、KV、通信和权重回传精度独立设置。SP 是 sequence_parallel 开关，开启时 degree=TP，要求结构化推理、TP>1、PP=1；不能重复乘入卡数。FSDP 满足 DP 整除关系。改变 DP 会改变聚合负载；HF 导入只生成模型片段与 missing_fields，仍需完整场景校验。

**来源** [configuration.md](../skills/supernode-modeling/references/configuration.md)，“配置来源”；[inference.md](../skills/supernode-modeling/references/inference.md)，“Megatron 式 sequence parallel”。

### 04 拓扑先编译，再验证实际路径

**核心结论** SU 域、SO 域及有向连接必须形成明确的物理路径。

**上屏文案**

- 自然语言 → topology spec → 原生 hardware
- SU 内卡数与 SO 域成员分别声明
- 改变卡数或域成员后重新编译

**图表与布局** 一个两层拓扑示意：4 个 SU 分属 2 个 SO 域，域间画方向明确的双向边；旁注“示意不含性能假设”。图下流程：“成员 / 连通校验 → explicit + su_so → communicator 路径核对”。

**讲解备注** 子 Skill 仅生成拓扑时无需模型或负载；有完整场景则用 --base 保留模型、精度、负载、SLO 和 rank 放置。高层 scale_out.domains 不能直接写入引擎 schema。当前编译器采用等大连续 SU 域、单个 SU 内的完整主机，多 SO 域有向图须强连通。编译后核对同 SU、跨 SU 同 SO、跨 SO，以及 H2D/D2H 和已启用的 remote/SSD 路径。原生 size-su 不自动重建显式图，改变物理结构的每个候选须重新编译；单向声明不代表反方向。

**来源** [topology.md](../skills/supernode-modeling/references/topology.md)，“一等输入”；[拓扑子 Skill](../skills/supernode-modeling/subskills/supernode-topology/SKILL.md)，“生成与交付”“保持模型语义”。

### 05 事件依赖与资源准入决定执行

**核心结论** 下界用于筛查必要条件，实际仿真再检查争用、容量与目标。

**上屏文案**

- Graph 连接依赖、工作量与张量生命周期
- 就绪后仍须通过内存和 slot 准入
- estimate 通过不代表 SLO 可行

**图表与布局** 一个主流程：配置校验 → 构图与下界 → 执行 → 指标 / SLO。将执行框展开为四步环：“完成释放 → ready 准入 → 共享速率 → 时间推进”，不展开完整函数图。

**讲解备注** 普通引擎以 Event.deps、work、startup、allocs/releases、slots 表达执行；资源分别约束速率、容量和互斥。roofline 独占速率与归一化 max-min 描述共享争用，计算与访存不直接相加。构图循环不代表运行时串行。配置合法、图与路径合法、执行可行、SLO 达标是不同关口；初始容量失败和 memory_admission_deadlock 均保留诊断。流式执行范围见第 12 页。

**来源** [topology.md](../skills/supernode-modeling/references/topology.md)，“Collective 与字节记账”；[配置口径](../skills/supernode-modeling/references/configuration.md)；[事件执行器图](diagrams/08-event-engine.mmd)。

### 06 训练同时计算状态、搬运与更新

**核心结论** offload 与重计算改变容量，也带来明确的传输和执行开销。

**上屏文案**

- 非交错 1F1B、FSDP、TP/EP 与梯度同步
- optimizer offload：梯度 D2H → 更新 → 权重 H2D
- Adam / Muon 按参数组分别建账

**图表与布局** 一条训练步流程：前向 / 反向 → 梯度同步 → optimizer → step 完成。optimizer 框内画 NPU→Host→NPU 的搬运闭环，标明 gradient 与 weight_transfer 精度。

**讲解备注** checkpoint 用重计算换 activation 留存，但不消除前向工作集。Dense/shared 与专家参数的复制/分片不同，不能将全部状态统一除以卡数。Muon 需矩阵形状，矩阵 momentum、正交化暂存和辅助 Adam m/v 分列；其分组是建模策略。FP8 权重驻留不意味着 H2D 按 FP8。大型结构化 offload 可显式设置 inflight_components_per_rank: 2；该策略影响峰值与耗时，须随结果披露。10T Adam 逻辑容量放附录 A01，不当作设备峰值或 Muon 配方。

**来源** [training.md](../skills/supernode-modeling/references/training.md)，“参数和执行策略”；[memory-breakdown.md](../skills/supernode-modeling/references/memory-breakdown.md)，“Adam 与 Muon”。

### 07 推理状态随结构与阶段变化

**核心结论** 固定 batch 的 Prefill、Decode 与状态布局必须联合建模。

**上屏文案**

- Prefill 产生首 token，活跃序列继续 Decode
- MLA 缓存随长度增长；KDA 保存固定状态
- SP、KV offload 与 Engram 改变工作集和通信

**图表与布局** 以一张结构/策略对照表为主图，顶部标“预加载权重后的固定 batch”。

| 结构或策略 | 状态与通信含义 |
|---|---|
| MLA / KDA | latent 或展开 KV / 每序列 recurrent + conv state |
| SP | Norm、残差等按序列分片；组件前 AG、后 RS |
| KV offload / Engram | 取回与写入 / 查询、缓存和 IOPS；共享实际路径 |

**讲解备注** 完成序列退出 Decode，不补入新请求。MLA latent KV 在 TP 间复制；KDA recurrent heads 按 TP 分片，不能套用 context parallel。SP 仅支持结构化推理、TP>1、PP=1，等长 collective 存在 padding；attention/KV 仅计有效 token。K3 样例的 compressed MLA 为 512+64=576 元素/token/MLA 层，是明确布局假设，官方 HF 参考实现为展开 K/V。分页、scale、转换与取回 buffer 分列。KV offload 与 Engram 要联合检查 host/remote 资源；Engram 只预测资源成本，不预测模型质量。

**来源** [inference.md](../skills/supernode-modeling/references/inference.md)，“负载与性能指标”“KV 结构”“Engram”“sequence parallel”；[K3 案例](../skills/supernode-modeling/references/ascend950-kimi-k3.md)，“模型必须保留的细节”。

### 08 用组件账本解释同一时刻的峰值

**核心结论** 逻辑容量、复制分片后的驻留与执行峰值分别报告。

**上屏文案**

- 组件 × 状态矩阵追溯容量来源
- 峰值构成来自同一时刻的存活张量
- 独立组件峰值不能直接相加

**图表与布局** 一张概念矩阵，不填虚构容量；右下标“HBM / DDR：容量、驻留、峰值、余量、失败资源”。

| 组件组 | 权重与训练状态 | activation / 工作区 | 推理状态 |
|---|---|---|---|
| Embedding / Norm / Output / AttnRes | 按张量归属 | 留存、重算、残差 | 按实际结构 |
| MLA / KDA | 参数与优化器状态 | 计算暂存 | KV / recurrent + conv |
| Dense / MoE | 复制与分片 | 路由及通信 buffer | 按实际结构 |

**讲解备注** K3 明细保留 WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output layer 及实际存在的 AttnRes。weight、gradient、master、Adam m/v、Muon momentum、activation、KV/KDA、量化元数据、通信与优化器 workspace 分列。MoE top-k 不缩小全部常驻专家权重。DRAM 与 DDR 不重复计容。resident_only、simulated 与 observed_partial 分别表示驻留、完整执行和已观察前缀；OOM 后观察峰值不能作为完整任务安全容量。完整数值从组件、状态和峰值 CSV 读取。

**来源** [memory-breakdown.md](../skills/supernode-modeling/references/memory-breakdown.md)，“建议输出”“数值口径”；[主 Skill](../skills/supernode-modeling/SKILL.md)，“解释结果时的关键口径”。

### 09 流量按语义、路径与资源分别计量

**核心结论** 网络瓶颈取决于 collective 轮次、rank 放置和共享路径资源。

**上屏文案**

- AG / RS / AR 用 ring；A2AV 用 payload 矩阵
- payload 与逐跳 physical bytes 分开
- 链路、注入、NIC 和共享上联共同约束

**图表与布局** 一个三层流程：“通信语义 → 算法轮次 → 实际链路与资源”。配一个点对点小例子：B 字节经过 h 跳，payload=B，physical=h×B；明确“单次点对点示意”。

**讲解备注** ring 的多轮转发计入 transport payload 总量；dispatch/combine 是 A2AV 的语义视图，不能再加一次。HBM/DDR、注入与物理链路属于不同资源账本，不合并成一个网络字节数。SU/SO 链路类别不是具体 SO 域 ID；域级细分需成员表及逐事件路径归因。速率须区分 bps/Bps、方向、效率和共享容量；400 Gbit/s=50 GB/s/方向。路径静态确定，当前抽象不模拟 ECMP、报文 buffer 或拥塞控制协议。

**来源** [topology.md](../skills/supernode-modeling/references/topology.md)，“硬件配置与实际路径”“Collective 与字节记账”。

### 10 TTFT 与 TPOT 沿关键路径拆解

**核心结论** 同一请求的分项可加和为总时延，不能累加并发 rank 耗时。

**上屏文案**

- TTFT：所有请求首 token 完成时间的最大值
- TPOT：每请求平均间隔，再取请求间最大值
- 计算、分域通信与等待沿关键路径归因

**图表与布局** 一张归因表；不以流量比例画耗时占比。表下注“TPOT = TPOP；JSON / CSV 为秒，中文报告 / 网页为毫秒”。

| 组成 | 内容 |
|---|---|
| 计算 | kernel 计算 / 访存服务时间 |
| 通信 | node 内、跨主机 SU、SO、其他路径 |
| 等待与争用 | 显式同步、slot、内存准入、共享资源争用 |
| 其他 | 其余已归因时间 |

**讲解备注** TTFT 与 TPOT 的最坏请求可能不同，结果各自保留 replica、sequence、total_s、components_s。各类分项必须来自决定该指标的同一请求，不逐项取最大再拼接。上游依赖等待已归入上游工作，不重复加同步时间；跨域通信也不把各域完整服务时长重复相加。TPOT 按相邻 token 时间窗归因后求平均，跨窗事件按覆盖比例分配。estimate、训练、未完成执行、独立 Decode 的 TTFT，以及单 token 的 TPOT 不可用时保留 null/空白，不补零。查看 metrics.latency_breakdown 和 latency_breakdown.csv。

**来源** [latency-breakdown.md](../skills/supernode-modeling/references/latency-breakdown.md)，指标选择、components_s、可加和规则与不可用状态。

### 11 Feedback 在延迟约束下提高 Decode 吞吐

**核心结论** 固定硬件与模型，完整执行每个候选后再给出推荐。

**上屏文案**

- 搜索 TP / SP / EP、Prefill 长度和 batch
- 同一次执行同时满足 TTFT、TPOT 与资源约束
- 在声明候选集内最大化 Decode 吞吐

**图表与布局** 一个闭环流程：固定模型与硬件 → 枚举候选 → 完整 Prefill+Decode → 筛选目标 → 推荐与 Pareto。底部列推荐字段：“TP/SP/EP、长度、batch、TTFT/TPOT、Decode 吞吐、峰值”。

**讲解备注** 保持硬件、拓扑、PP/DP/FSDP、rank_order、精度和 offload 不变；SP 共用 TP rank。一个候选的 Decode batch 也用于其 Prefill，不能拼接不同候选的最优长度与最大 batch。Decode 吞吐=Σ(O_i−1)/(max(last_i)−min(first_i))；首 token 属于 Prefill，不用 batch/TPOT 替代。普通输出吞吐仍含 Prefill，另行报告；有原吞吐目标时继续检查。output horizon 保持一致且至少 2，max_candidates 超限会拒绝，不截断搜索。auto 对结构化 DP=PP=1 使用流式完整执行，其他用普通仿真；独立 Decode 不用于联合可行性。推荐、各候选失败原因与时延拆解均保留；无可行项不自动放宽目标或增加硬件。

**来源** [feedback.md](../skills/supernode-modeling/references/feedback.md)，“固定项与联合可行性”“吞吐、延迟与排序口径”“结果与复核”。

### 12 按分析目标选择普通或流式运行

**核心结论** 控制仿真器开销时保持原场景、设备约束与失败案例完整。

**上屏文案**

- 完整生命周期审计：普通 estimate / simulate
- 大型结构化推理、DP=PP=1：优先流式
- 有界缓存、串行批量与按需明细控制开销

**图表与布局** 使用运行方式对照表。

| 方式 | 适用范围与交付 |
|---|---|
| 普通引擎 | 训练、其他并行布局、完整组件 / 张量生命周期 |
| 流式完整执行 | 结构化推理、DP=PP=1；性能、流量、设备峰值与诊断 |
| 检查点实验 | 匹配配置 / 源码指纹后，恢复完成态分析与导出 |

**讲解备注** run_streaming 逐组件执行，保留依赖、争用与内存准入，支持 SP；不提供普通报告的完整 trace，不能直接交给普通 write_report。phase=all 才覆盖联合 Prefill/Decode；独立 Decode 跳过 Prefill，不证明端到端可行。时序缓存默认 64 MiB / 256 项，属于对象预算而非进程 RSS 上限；每个并发进程另占图和缓存，批量默认串行。大型默认导出精简事件与临时 buffer，保留完整统计；保留张量表超过 50,000 行时写 tensors.csv.gz，以 artifact_export 为准。--full-detail 保留普通 CSV 和全部明细。检查点是已完成数值执行的恢复，不是 active/ready 中途续算；精简 JSON 不能补造丢失生命周期。

**来源** [resource-usage.md](../skills/supernode-modeling/references/resource-usage.md)，“执行方式”“缓存和批量实验”“文件大小与恢复”；[inference.md](../skills/supernode-modeling/references/inference.md)，“大规模结构化推理”。

### 13 配置、结果与失败原因一起交付

**核心结论** 每项结论都应回到实际配置、命令、版本与引擎产物。

**上屏文案**

- 保留配置、来源、运行版本与命令
- 报告结果、候选范围和失败原因
- 校准覆盖与未校准假设分别说明

**图表与布局** 使用“问题 → 产物”映射表。

| 核查目的 | 主要产物 |
|---|---|
| 复跑与结论 | config.resolved.json、result.json、report.md |
| 容量与瓶颈 | component_memory / state_memory / memory_peaks / resources CSV |
| 时延与搜索 | latency_breakdown.csv、candidates.csv、best.config.json |
| 事件与导出范围 | events.csv、timeline.json、artifact_export |

**讲解备注** 表中为按命令适用的产物，流式 API 交付紧凑 JSON 与配置，不冒充普通完整报告。Feedback 的 config.resolved.json 是基础场景，best.config.json 才是推荐场景。sweep 枚举受控变量；size-su 的最小值仅限定已检查候选与策略，改变 DP 需披露负载变化，显式图需逐候选重编译。搜索 completed 不代表有可行项；区分执行错误、资源不可行、SLO 未达标和指标不可用。实测 profile 应覆盖实际 kernel、collective 与平台路径；旧结果可读，但代码或配置指纹不匹配时不能当作当前已验证缓存。先看汇总与导出清单，再按行读取大型明细。

**来源** [configuration.md](../skills/supernode-modeling/references/configuration.md)，“CLI”“外部 profile 与 HF 导入”；[feedback.md](../skills/supernode-modeling/references/feedback.md)，“结果与复核”；[topology.md](../skills/supernode-modeling/references/topology.md)，“最小 SU 搜索”。

### 14 A5 上的 10T Kimi K3 合成扩容场景

**核心结论** 以 1024 卡的历史实验贯通模型扩容、流量解释、容量失败与并行方案比较。

**上屏文案**

- 3328 routed experts，精确总参数 10.171385T
- A5 / Ascend 950DT 假设平台，1024 卡
- 2026-10-15 历史结果；合成模型、未实机校准

**图表与布局** 以四行场景契约表为主图，顶部小标签“案例 1 / 4”。表下标注“基线 48 组；SP 扩展 96 组。本次复用归档结果，未重跑当前引擎”。

| 固定项 | 案例配置 |
|---|---|
| 模型 | 93 层：69 KDA + 24 MLA；92 MoE 层；Top32 / Top64 |
| 基线并行 | TP8 × EP128 × PP1 × DP1 = 1024；SP 关闭 |
| 拓扑与容量 | 8 卡/节点、64 卡/SU 柜、16 柜/SO 域；144 GB HBM/卡 |
| 精度与负载 | FP8 权重/计算；FP16 通信/activation/KV；128k–1M，B1–B32 |

**讲解备注** 本案例对应用户所称 A5，归档配置的卡型名称为 Ascend950DT；平台参数与拓扑为显式建模假设，不能将其等同实机测量。仅增加 routed experts 和 router 输出维度，保留 hidden=7168、latent=3584、专家宽度=3072、2 个 shared experts。EP128 要求专家数可整除，因此选择 3328 专家，即每 EP 26 专家；精确总参数为 10.171385035872T，比 10T 多 1.71385%。Top32 / Top64 活跃参数分别为 154.414012512B / 251.654756448B。这是合成扩容设计，不是官方发布的 10T 模型，也不是早期约 2.77948T backbone 实验。每卡 HBM 标称 4 TB/s；SU 单向 1 TB/s/卡、效率 80%；SO 为 400 Gbit/s/节点、效率 80%，即有效 40 GB/s/节点，柜间共享上联 400 GB/s/方向，显式跨柜路径 4 hops。FP32 KDA 状态单列；GB/TB 采用十进制。输入 128k/256k/512k/1M 对应 131072/262144/524288/1048576 token/序列；batch 为 1/2/4/8/16/32。序列分配至 EP，B≤32 时每个活跃 EP 持有一条完整序列；TP8 留在一个节点内，不将上下文均匀切到 1024 卡。基线报告完成于 2026-10-15 16:43，SP 报告完成于同日 17:58（Asia/Shanghai）；这些是历史归档证据，不代表 2026-10-19 当前引擎重跑。

**来源** [1024 卡基线报告：场景与指标](../outputs/kimi-k3-10t-1024/report.md#场景与指标)、[基线 summary.csv](../outputs/kimi-k3-10t-1024/summary.csv)、[SP 报告](../outputs/kimi-k3-10t-sp/report.md)、[SP summary.csv](../outputs/kimi-k3-10t-sp/summary.csv)。

### 15 柜间流量主导，Top-k 与 batch 放大通信

**核心结论** TP8 基线中，柜间 payload 占 85.01%–89.50%；Top64 使 MoE / SO 流量翻倍。

**上屏文案**

- 128k 完整 Prefill 与单步 Decode，均为全 1024 卡聚合
- Top64 总 payload 为 Top32 的 1.900 倍
- 同一 Top-k 下随 batch 线性增加

**图表与布局** 一张四行代表结果表，顶部小标签“案例 2 / 4 · TP8 / EP128”。每行只列成功完成的 128k 案例，流量采用十进制单位。表下标注“有向发送计一次；SO 物理 hopbytes = SO payload × 4”。

| Top-k / Batch | Prefill 总 TB | Decode 总 GB/步 | Decode 柜间 GB/步 | 柜间占比 |
|---|---:|---:|---:|---:|
| 32 / B1 | 48.803 | 0.372 | 0.317 | 85.01% |
| 32 / B32 | 1,561.684 | 11.915 | 10.129 | 85.01% |
| 64 / B1 | 92.712 | 0.707 | 0.633 | 89.50% |
| 64 / B32 | 2,966.792 | 22.635 | 20.258 | 89.50% |

**讲解备注** 数值逐项取自基线 summary.csv 和 phase_traffic.csv 中 context=128k、batch_size=1/32 的完整阶段；不是从柱长或带宽反推。payload 为逻辑 transport payload，每个有向发送计一次；dispatch/combine 是 all_to_all_v 的语义视图，不能再次叠加。Top64 的 MoE / SO 为 Top32 的 2 倍，而 TP all-reduce 不变，因此总 payload 实际为 1.899738 倍。当前布局下单步 Decode 网络 payload 不随上下文长度变化，序列长度仍影响 KV 读与 attention 计算；完整 Prefill 流量随总 token 数线性增加。256k–1M 基线 Prefill 失败，对应 planned_phase_traffic.csv 只能说明完整计划需求，不能当实际完成流量；本页避免将其混入成功结果。柜内 payload 与 SU hopbytes 相等；柜间 4-hop 物理流量为柜间 payload 的 4 倍，不能把两种口径相加。HBM 工作量是本地读写流量，不能当 HBM 容量。该历史 pairwise EP128 每次 dispatch/combine 有 127 轮，92 层各执行两次，TP 使用 ring；报告估算 collective 启动延迟下界约 450–473 ms/步，尚无真实 collective 校准，因此总字节/总带宽不足以解释 Decode 时延。

**来源** [基线 phase_traffic.csv](../outputs/kimi-k3-10t-1024/phase_traffic.csv)，筛选 context=128k、batch_size=1/32，字段 transport_payload_bytes / so_payload_bytes；[基线 summary.csv](../outputs/kimi-k3-10t-1024/summary.csv)；[网络流量口径](../outputs/kimi-k3-10t-1024/report.md#网络流量)；[计划流量](../outputs/kimi-k3-10t-1024/planned_phase_traffic.csv)。

### 16 TP8 基线完成 128k，长 Prefill 触及容量边界

**核心结论** 48 组基线中仅 12 组 128k 完成端到端执行，长上下文首先受 Prefill 工作集限制。

**上屏文案**

- 128k：所有 Top-k / batch 组合完成
- 256k / 512k / 1M：36 组 Prefill 均失败
- 失败后的独立 Decode 不能证明端到端可行

**图表与布局** 一张四行上下文对照表，统一选 Top32 / B1，顶部小标签“案例 3 / 4 · TP8 / EP128”。表中 TTFT / TPOT 仅展示端到端指标，失败项用“—”。表下标注“已准入峰值不含被拒绝申请；失败所需为当时容量下界，非最终峰值”。

| 输入 | TTFT s | TPOT ms | 已准入峰值 GB/卡 | 失败所需 GB/卡 |
|---|---:|---:|---:|---:|
| 128k | 1,046.782 | 463.531 | 118.971 | — |
| 256k | — | — | 126.462 | 145.257 |
| 512k | — | — | 141.483 | 179.071 |
| 1M | — | — | 62.663 | 167.914 |

**讲解备注** TTFT 是预加载权重后、固定 batch 开始到批内最晚首 token 的时间；TPOT 仅为首 token 后的一个 Decode 间隔，未覆盖连续服务、排队、continuous batching 或 P99。表中 128k 行 decode_source=end_to_end；后三行 Prefill 未完成，端到端时延留空。128k 的 Top32 / B32 为 TTFT 1109.548 s、TPOT 487.698 ms，Top64 / B1 为 2084.445 s / 471.472 ms，Top64 / B32 为 2208.881 s / 496.575 ms，均完成；不能与 standalone Decode 混读。256k/512k/1M 的 Prefill 均在 AttnRes 申请触发 memory_admission_deadlock；失败层分别为 49/25/1（1-based），已准入峰值分别为 126.462/141.483/62.663 GB，失败申请所需分别为 145.257/179.071/167.914 GB。这里的“所需”是 144 GB + shortage_bytes，表示当时加入被拒绝申请的容量下界，不是完整任务最终峰值；低于 144 GB 的已准入峰值不代表可行。Top-k 与 B1–B32 各组共享这一失败边界，因为当前映射让每个活跃 EP 持有一条完整序列。失败配置的 TTFT 与端到端 TPOT 留空；预置完整 KV 后所得独立 Decode 只说明条件性单步成本。128k 峰值来自同一张卡同一时刻，主要构成为 activation 67.660 GB、AttnRes residual 15.032 GB、weight 23.603 GB、runtime 8.000 GB、KV 3.624 GB，另有量化元数据与其他状态；各值经舍入，不用独立组件峰值相加。权重与复制关系决定静态 HBM 约 32.597 GB/卡，不能将逻辑 10.171T 直接除以 1024 作为驻留量。

**来源** [基线 summary.csv](../outputs/kimi-k3-10t-1024/summary.csv)，字段 ttft_s / tpot_s / decode_source / prefill_peak_hbm_bytes / failure_required_hbm_bytes；[失败诊断](../outputs/kimi-k3-10t-1024/failure_diagnostics.json)；[同峰值组件](../outputs/kimi-k3-10t-1024/peak_components.csv)；[内存峰值与失败定位](../outputs/kimi-k3-10t-1024/report.md#内存峰值与失败定位)。

### 17 TP/SP 联合调整：容量与时延改善，流量增加

**核心结论** 两档 TP/SP/EP 组合完成了长上下文实验，并展示了更低峰值、较短 Decode 与更高通信量之间的取舍。

**上屏文案**

- SP 扩展 96/96 组完成，覆盖至 1M
- TP / EP 同时调整，算法轮次与 payload 一起变化
- 旧基线缺少边界通信；不能归因为 SP 单变量收益

**图表与布局** 一张三行对照表，统一选 Top32 / 128k / B1，顶部小标签“案例 4 / 4 · 归档方案比较”。表下标注“2026-10-15 历史结果；SP 使用冻结 runtime，未实机校准”。

| 方案 | TTFT s | TPOT ms | Prefill 峰值 GB/卡 | Decode 总 GB/步 |
|---|---:|---:|---:|---:|
| TP8 / EP128，SP 关 | 1,046.782 | 463.531 | 118.971 | 0.372 |
| TP16 / SP16 / EP64 | 1,041.874 | 246.775 | 38.132 | 1.952 |
| TP32 / SP32 / EP32 | 1,040.291 | 147.866 | 34.889 | 6.627 |

**讲解备注** SP 共用 TP 通信组，不额外乘入卡数。SP16/32 分别跨 2/4 个 8 卡节点，仍完全在一个 64 卡 SU 柜内；norm 与 AttnRes 按序列切分，组件前 AG、后 RS。对应 Top32 / 128k / B1 的 TTFT 分别为旧基线 1046.782 s、SP16 1041.874 s、SP32 1040.291 s，Prefill 时延变化远小于 Decode。Top32 / 1M / B1 在 SP16/32 下的 Prefill 峰值为 99.688/85.346 GB、TTFT 为 8422.024/8365.714 s、TPOT 为 262.629/163.721 ms；旧基线 1M Prefill 失败，不能将其独立 Decode 与新方案的端到端 TPOT 当作同一种服务能力比较。全体 96 组含 Top32/64、4 档上下文与 6 档 batch，均完成；可行性仅针对这些已检查配置。Compressed MLA KV 仍在 TP 间复制，不能再除以 SP；KDA heads 按 TP 分片。当前 MoE 实现复制各 TP slice 的完整 latent dispatch，使 SP16/32 的柜间 payload 为 TP8 的 2/4 倍；单 query 的 Top32 柜间 payload 分别为 0.633078/1.266156 GB，基线为 0.316539 GB。Decode 等长 collective 将每个非空 EP lane 的 1 个 query padding 到 16/32 行，进一步增加 AG/RS，但 attention、MoE 计算与 KV 只计有效 token。比较复用真实旧基线，旧引擎未计 embedding/output 边界通信，SP 版本新增 RS/AG；同时 TP/EP、算法轮次和工作集发生变化，因此这不是控制所有其他变量的 SP 因果实验。实验期间源码曾并发修改，归档以恢复的冻结源码、配置 SHA 与 producer 证据核验 96 份结果；54 份磁盘观测指纹不同于运行中已加载 runtime。冻结 runtime ID 为 sha256:6759029adf34d1c8da4893051b2163ffd376243fcc61ad470fa70ae86d6907d2，两组隔离完整重放通过；不能称与当前工作区引擎匹配。结果仍基于均匀路由、粗粒度 roofline 与所选 collective，没有实机 kernel / collective 校准。

**来源** [SP summary.csv](../outputs/kimi-k3-10t-sp/summary.csv)，筛选 top_k=32、context=128k、batch_size=1；[旧基线 summary.csv](../outputs/kimi-k3-10t-1024/summary.csv)；[逐组 comparison.csv](../outputs/kimi-k3-10t-sp/comparison.csv)；[SP phase_traffic.csv](../outputs/kimi-k3-10t-sp/phase_traffic.csv)；[与归档 TP8 基线比较](../outputs/kimi-k3-10t-sp/report.md#与归档tp8基线比较)；[runtime 来源审计](../outputs/kimi-k3-10t-sp/audit/runtime-provenance.json)；[冻结源码清单](../outputs/kimi-k3-10t-sp/runtime-frozen/manifest.json)。

### 18 结论受负载、抽象与校准范围约束

**核心结论** 现有输出支持架构比较和敏感性分析，实机准确性需要对应证据。

**上屏文案**

- 固定 batch 不含到达队列、冷启动或 P99
- 流体资源与均匀 MoE 不覆盖所有生产行为
- 公开结构、硬件规格和运行假设分开

**图表与布局** 一张“可解释 / 需补证据”边界表。

| 可解释的范围 | 不能据此直接推出 |
|---|---|
| 声明负载下的容量、流量、关键路径 | continuous batching、PD 分离、生产 tail latency |
| 确定性路径与共享资源争用 | 包级 ECMP、拥塞控制、周期级行为 |
| 结构账本与显式 profile | 实机误差百分比、缓存最优容量、MTE 通道数 |

**讲解备注** K3 文本 backbone 为 93 层、69 KDA+24 MLA，形状派生约 2.77948T，排除视觉和训练 MTP；官方 2.8T/104B 为约数。950DT 144 GB、4 TB/s 为公开资料，样例以 SI 容量建模，链路方向、效率、Host/SO 和 compressed KV 另列假设。64/128 NPU 样例的 SU 划分不是 Atlas 950 真实域上限。通用非交错 1F1B/FSDP2/CPU offload 不等于 K3 生产训练；未覆盖动态冗余专家、分布式 Muon、activation offload 等完整策略。没有实测不报告误差百分比或“符合实机”。仅有字节、工作集与搬运需求不能确定缓存层级容量或 MTE 通道数。

**来源** [inference.md](../skills/supernode-modeling/references/inference.md)，“MoE 与模型近似”；[topology.md](../skills/supernode-modeling/references/topology.md)，“Collective 与字节记账”；[K3 / 950 案例边界](../skills/supernode-modeling/references/ascend950-kimi-k3.md)。

### 19 用同一场景闭合分析与选型

**核心结论** 通过明确输入、统一计算与真实产物形成可审查的方案依据。

**上屏文案**

- 固定场景：结构、负载、精度、布局与拓扑
- 解释瓶颈：容量、流量、时延与等待
- 受控搜索：保留推荐、失败原因和校准范围

**图表与布局** 一个五步闭环：编辑场景 → 校验与运行 → 账本与时延拆解 → 受控调整 / Feedback → 复跑验证。中心写“同一配置 · 同一口径 · 可追溯”。

**讲解备注** 主讲到此结束。容量、网络和性能必须来自一致的场景假设。先核对输入与可行性，再比较候选，最终用实测校准收敛绝对预测。附录提供容量交叉检查、可复跑命令和来源，不将未运行的结果作为本次交付结论。

**来源** [主 Skill](../skills/supernode-modeling/SKILL.md)；[项目说明](../README.md)。

## 四 备查附录

### A01 10T 逻辑容量锚点

**核心结论** 降低权重精度只改变对应状态，全部训练状态不会随之减半。

**上屏文案**

- 10T 全参数 Adam；十进制 TB
- FP8 / FP16 权重为 10 / 20 TB
- 基础状态合计为 150 / 160 TB

**图表与布局** 使用原始数值表或两根堆叠柱。柱中只包含四种状态，合计作为顶部标签，不画成第五段。

| 状态 | FP8 权重方案 TB | FP16 权重方案 TB |
|---|---:|---:|
| weight | 10 | 20 |
| gradient FP16 | 20 | 20 |
| master FP32 | 40 | 40 |
| Adam m/v FP32 | 80 | 80 |
| 基础状态合计 | 150 | 160 |

**讲解备注** 这是合成逻辑容量交叉检查，假设 1e13 参数全部更新；不含 scale、额外副本、activation、通信 buffer、workspace 或复制。不是集群物理总容量、设备峰值或 Muon 状态。master+m/v offload 到 CPU 为 120 TB。每步 FP16 梯度 D2H payload=20 TB；H2D 按 weight_transfer 取 FP8 10 TB 或 FP16 20 TB，另计 FP8 元数据和其他 collective。驻留 FP8、回传 FP16 时仍按 20 TB 回传。

**来源** [training.md](../skills/supernode-modeling/references/training.md)，“10T FP8/FP16 交叉检查”。

### A02 命令与产物速查

**核心结论** 先校验完整场景，再按问题选择入口并检查真实输出。

**上屏文案**

- 配置合法 → 路径合法 → 执行可行 → SLO 达标
- 网页任务与搜索 / 流式入口分别使用
- 命令是复跑示例，不代表本次已执行

**图表与布局** 上屏使用入口表；完整命令放 speaker notes，不把长路径挤入正文。

| 问题 | 入口 | 重点检查 |
|---|---|---|
| 编辑与单次分析 | 网页、validate / estimate / simulate | 配置、汇总、容量和时延拆解 |
| 参数与规模选择 | sweep / size-su | 候选、失败原因、负载与拓扑 |
| 延迟约束下吞吐选择 | feedback | candidates、最佳配置、联合 SLO |
| 大型推理 / 恢复 | run_streaming / run_reference_experiment.py | 阶段范围、导出清单、匹配指纹 |

**讲解备注** 从项目根目录使用项目 Python。输出目录按实验区分，保留已有产物。示例命令如下；不同命令按各自实际导出清单交付。

| 用途 | 命令 |
|---|---|
| 网页 | python skills/supernode-modeling/scripts/launch_frontend.py --project-root . |
| 校验 | python -m supernode_sim validate examples/demo_inference.yaml |
| 估算 | python -m supernode_sim estimate examples/demo_inference.yaml -o outputs/demo-estimate |
| 仿真 | python -m supernode_sim simulate examples/demo_inference.yaml -o outputs/demo-inference |
| 扫描 | python -m supernode_sim sweep examples/training_10t_fp8.yaml --axes examples/sweep_precision.yaml --mode estimate -o outputs/precision-sweep |
| SU 搜索 | python -m supernode_sim size-su examples/demo_inference.yaml --mode both -o outputs/su-search |
| Feedback | python -m supernode_sim feedback examples/demo_inference.yaml --search examples/feedback_inference.yaml -o outputs/demo-feedback |
| 推荐复核 | python -m supernode_sim simulate outputs/demo-feedback/best.config.json -o outputs/demo-feedback-best |
| 完成态恢复 | python -u scripts/run_reference_experiment.py examples/kimi_k3_ascend950_training_muon_128.yaml -o outputs/kimi-k3-ascend950-training-muon-128 |

需要逐事件 / 张量生命周期时，普通 estimate/simulate 添加 --full-detail。结构化推理且 DP=PP=1 可用 Python API run_streaming(config, phase='all')；紧凑结果与配置按 API 文档保存，不直接交给普通 write_report。Feedback 推荐不存在时不执行 best.config.json 复核命令。

**来源** [configuration.md](../skills/supernode-modeling/references/configuration.md)；[frontend.md](../skills/supernode-modeling/references/frontend.md)；[resource-usage.md](../skills/supernode-modeling/references/resource-usage.md)。

### A03 来源与详细图索引

**核心结论** 当前能力以仓库 Skill 和真实产物为准，公开事实与运行假设可分别追溯。

**上屏文案**

- 能力与口径：主 Skill、references、拓扑子 Skill
- 模型与平台：官方来源快照及字段级来源
- 详细函数图留在仓库，按需查阅

**图表与布局** 使用四组来源卡片。每组显示简称与路径；完整链接保留在备注或文档中。

| 来源组 | 本稿依据 |
|---|---|
| 工作流与配置 | [主 Skill](../skills/supernode-modeling/SKILL.md)、[frontend](../skills/supernode-modeling/references/frontend.md)、[configuration](../skills/supernode-modeling/references/configuration.md) |
| 模型与资源 | [training](../skills/supernode-modeling/references/training.md)、[inference](../skills/supernode-modeling/references/inference.md)、[memory-breakdown](../skills/supernode-modeling/references/memory-breakdown.md) |
| 拓扑、时延与搜索 | [topology](../skills/supernode-modeling/references/topology.md)、[子 Skill](../skills/supernode-modeling/subskills/supernode-topology/SKILL.md)、[latency-breakdown](../skills/supernode-modeling/references/latency-breakdown.md)、[feedback](../skills/supernode-modeling/references/feedback.md) |
| 规模、案例与公开事实 | [resource-usage](../skills/supernode-modeling/references/resource-usage.md)、[K3 / 950](../skills/supernode-modeling/references/ascend950-kimi-k3.md)、[官方来源快照](sources/README.md) |
| A5 / 10T 历史仿真 | [TP8 基线报告](../outputs/kimi-k3-10t-1024/report.md)、[TP/SP 对照报告](../outputs/kimi-k3-10t-sp/report.md)、[基线 CSV](../outputs/kimi-k3-10t-1024/summary.csv)、[TP/SP CSV](../outputs/kimi-k3-10t-sp/summary.csv) |

**讲解备注** 内容日期 2026-10-19 表示本稿读取仓库文档的基线，不把旧架构说明日期当作当前引擎修订。正式数值另保留运行命令、配置与代码版本。K3 结构与 950 规格的公开来源、核验日期和不确定性见 docs/sources/README.md；不以型号相同替代性能验证。

**来源** 上表所列仓库文件；[项目说明](../README.md)。

## 五 详细图源链接（不生成额外幻灯片）

主讲按本稿重绘简图。以下既有函数图用于深入核查，不将旧总览图中的入口列表当作当前能力全集；不在本文重复 Mermaid 源码。

| 内容 | 原始图文件 |
|---|---|
| 架构与普通 CLI 主链 | [00-architecture.mmd](diagrams/00-architecture.mmd)、[01-entry-and-run.mmd](diagrams/01-entry-and-run.mmd) |
| 参数预算 / 结构化训练 | [02-training-budget.mmd](diagrams/02-training-budget.mmd)、[03-training-structured.mmd](diagrams/03-training-structured.mmd) |
| optimizer offload | [04-optimizer-offload.mmd](diagrams/04-optimizer-offload.mmd) |
| 结构化 / 参数预算推理 | [05-inference-structured.mmd](diagrams/05-inference-structured.mmd)、[06-inference-budget.mmd](diagrams/06-inference-budget.mmd) |
| 通信路径与执行器 | [07-collective-topology.mmd](diagrams/07-collective-topology.mmd)、[08-event-engine.mmd](diagrams/08-event-engine.mmd) |
| 账本、传统搜索与恢复 | [09-memory-and-report.mmd](diagrams/09-memory-and-report.mmd)、[10-sweep-and-su-search.mmd](diagrams/10-sweep-and-su-search.mmd)、[11-reference-checkpoint.mmd](diagrams/11-reference-checkpoint.mmd) |

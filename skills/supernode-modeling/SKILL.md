---
name: supernode-modeling
description: Model supernode training and inference memory, compute, and communication with supernode_sim, automatically preparing and opening its editable local web dashboard. Use for component-level memory and Adam/Muon state breakdowns; Kimi K3 and Ascend 950 scenarios; offload, KV/KDA state, SU/SO topology generation and minimum-SU searches; or feedback searches for TP/SP/EP, prefill length and decode batch that maximize decode throughput under TTFT/TPOT targets on fixed hardware and topology.
---

# 超节点资源与通信建模

默认先准备并打开可编辑的流量仿真网页，再把问题转为可复现配置，调用 Python 引擎计算，解释资源需求、瓶颈和适用条件。前端与引擎由 `supernode_sim` 统一提供；此 skill 包含调用说明和自动启动脚本，**不包含独立的 Python 引擎**，需要 `supernode_simulation` 仓库或已安装的 `supernode_sim` 包。

## 找到可运行环境

优先使用用户已给定的项目路径，其次检查 `SUPERNODE_SIM_ROOT` 环境变量，再检查当前工作目录及其父目录中的 `supernode_sim/` 与 `pyproject.toml`。如果本文件位于项目的 `skills/supernode-modeling/`，其上两级目录就是项目根目录。被复制到全局 skills 目录后，不能把该目录视为引擎所在项目。

在项目根目录、项目 Python 解释器下执行：

```text
python -m supernode_sim --help
python -m supernode_sim schema
```

包已安装时也可使用 `python -m supernode_sim`；示例文件仍位于原仓库。若找不到仓库或模块，报告具体缺少的路径/依赖，不伪造执行结果。引用配置文件路径、输出目录时遵循所在 shell 的引号规则。

## 调用时自动准备前端

每次启用本 skill，默认先执行 [前端启动流程](references/frontend.md)，无需等用户另行请求网页。用户明确指定只用 CLI、不要网页或运行环境不能启动服务时，按其要求或实际能力继续数值工作，并说明网页状态。

1. 运行本 skill 自带的 `scripts/launch_frontend.py`，已定位仓库时传入 `--project-root <项目目录>`；仅使用安装包时省略该参数。脚本检查现有页面资源、选择 Python 环境并启动或复用本地服务，输出机器可读 JSON。当前前端无需 npm / Node 构建，复用项目 `supernode_sim/web_static/`，保持输入编辑、实验运行与结果展示使用同一引擎配置。
2. 只有返回 `started` 或 `reused` 且资源检查成功，才报告网页就绪。用可用浏览器工具打开返回的 `inputs_url`；没有浏览器工具时提供实际 URL 和能力限制，不声称已打开。保护已有标签页中的未提交输入，优先使用新任务标签页或本任务已创建的标签页。
3. 将用户给定或本次整理的完整配置载入网页，按请求开展仿真。页面启动本身不自动提交数值任务。`estimate` / `simulate` 可用网页现有 API，保存输入快照并在历史运行中展示结果；其他分析命令继续使用其原入口，明确网页支持范围。
4. 重复调用复用匹配服务；保留交付的服务供用户继续编辑和运行。失败时读取脚本返回的原因和日志，按具体原因修复后再试，不能循环重启、停止其他服务或用未验证 URL 代替真实页面。

网页包含模型、拓扑、硬件、并行度和 Workload 的输入展示与编辑、完整字段 JSON 编辑、schema 校验、运行日志、取消、历史快照以及流量/内存/利用率/时间线图表。编辑当前配置会同步下一次仿真；历史快照先复制再编辑。操作、输入同步及回退细节见 [frontend.md](references/frontend.md)。

## 按问题选择资料和命令

- 配置、来源、单位、CLI：读 [configuration.md](references/configuration.md)。
- 大型仿真、批量实验或本机内存/磁盘紧张：读 [resource-usage.md](references/resource-usage.md)，选择流式运行、缓存预算与导出粒度。
- 按组件分析存储、activation、梯度、Adam/Muon：读 [memory-breakdown.md](references/memory-breakdown.md)。
- Kimi K3 / Ascend 950 样例和真实性边界：读 [ascend950-kimi-k3.md](references/ascend950-kimi-k3.md)。
- 训练/FSDP2/optimizer offload/10T 容量：读 [training.md](references/training.md)。
- 推理/KV 压缩与 offload/Engram：读 [inference.md](references/inference.md)。
- 固定模型、拓扑和硬件，输入 TTFT、TPOT（用户也可能写 TPOP），反求 TP/SP/EP、Prefill 长度及 Decode batch：读 [feedback.md](references/feedback.md)，运行 `feedback`，在满足目标的有限候选内优先最大化 Decode 吞吐量。
- 接收 SU/SO 拓扑、scale-out 域、分析链路瓶颈/最小域搜索：读 [topology.md](references/topology.md)。
- 用户用自然语言生成或修改 scale-up / scale-out 拓扑：按子 skill [supernode-topology](subskills/supernode-topology/SKILL.md) 生成拓扑规范、原生配置和拓扑图，再回到本 skill 做流量仿真。仅生成拓扑时在子 skill 内完成，不要求模型或负载。

先检查用户提供的配置与可核实资料，再补齐问题所需的参数。只询问会改变结论且无法从现有资料获得的缺失项，使用明确字段名。模型名称不是结构证据；对未公开结构采用用户认可或明确标注的 synthetic 假设，不把假设写成某模型的公开规格。

保留用户场景，在工作目录创建可重跑配置。先 `validate`，再按任务运行 `estimate`、`simulate`、`sweep`、`size-su` 或 `feedback`；`estimate` 提供必要下界，不能证明 SLO 可行。硬件显然无法容纳的配置也有分析价值，保留失败原因而不自动放大硬件。

## 拓扑作为输入

把拓扑与模型、负载、并行布局一起作为场景输入，先读用户提供的原生场景、拓扑文件或自然语言描述。收集会影响实际路径的以下信息；已有配置足够时直接使用：

- 总 NPU 数、每个 scale-up 域的 NPU 数、每主机 NPU 数，以及物理 NPU 与并行 rank 的映射。
- scale-out 域的数量、标识和成员 SU 域；区分“SU 域内卡数”和“SO 域内超节点数”。多个 SO 域之间是否互联、连接方向、链路能力也要显式输入。
- SU 域内拓扑、SO 域内拓扑与域间连接；带宽、延迟、效率、端点注入、共享上联及双向共享容量的口径。

输入可以是已有 `hardware` 原生配置，也可以是子 skill 的独立 topology spec。后者先经 `build_topology.py` 编译；不能把 `scale_out.domains` 等高层字段直接放入严格的引擎 schema。详见 [拓扑输入契约](references/topology.md)。保持已有的模型、精度、负载、SLO 和 rank 放置；改变卡数后若并行乘积不匹配，报告冲突，不自动改 DP/global batch。

先校验拓扑成员、连通性和实际 communicator 路径，再做数值分析。记录 SU/SO 域划分、链路来源、拓扑规范与编译后配置；报告同 SU、跨 SU 同 SO 域、跨 SO 域通信的路径及共享瓶颈。引擎原生字节统计按 SU/SO 链路类别汇总，SO 域级细分需要结合成员表和逐事件路径归因，不能把链路类别当成具体的 SO 域 ID。

显式图的 `explicit_links` 随卡数与域成员固定。扫描或搜索改变 `npu_count` / `su_size` / 域成员时，每个候选必须重新生成拓扑并校验；不能直接复用原图跑 `size-su`，以免图与候选域划分不一致。普通内置模板继续使用原有搜索流程。

## 解释结果时的关键口径

- `total_params`、每 token 的 `active_params` 和中间 activation 张量是三个不同量。
- 独立配置权重驻留、计算、梯度、优化器、KV、通信与权重回传精度。FP8 驻留不意味着 H2D 按 FP8 传输。
- 区分逻辑容量、实际复制/分片容量、生命周期峰值；DRAM 与 DDR 不重复计容。
- 存储报告应给出“组件 × 状态”矩阵及逐层/逐设备明细。Kimi K3 至少展示 WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output layer，并保留实际存在的 AttnRes；未知结构列为 Unassigned，不按百分比伪分配。
- 将 weight、gradient、master、Adam m/v、Muon momentum、activation、KV、KDA recurrent/conv state、量化元数据、通信/计算 workspace 分列；Muon 的正交化暂存和辅助 Adam 参数组单列。
- 表明每张表的加和规则：逻辑静态容量可加；不同组件在不同时刻取得的峰值不能相加当作同一时刻设备峰值。应同时查看设备池峰值时刻的组件构成。
- 区分 transport payload 与逐跳物理字节量；dispatch/combine 是 all-to-all-v 的语义视图，不能再相加。
- 给出配置、版本/来源、命令、结果文件，以及结论成立的约束。执行失败、不可行和缺参分别处理。
- 缓存/MTE 的工作集、访问量和搬运需求不能直接换算成 L1/L2/L3 容量或通道数；需要硬件与 kernel 校准。
- 固定 batch 仿真不提供生产请求排队或 P99 保证；SU “最小”限定为已检查候选集合与调度策略。
- Feedback 的“最优”限定为所声明候选集内满足 TTFT/TPOT 与资源约束的最高 Decode 吞吐。Prefill 与 Decode 用同一候选完整执行；SP 共用 TP rank。报告所选长度和 batch，不把不同候选的长度/batch 极值拼成一个推荐。

正式数值引用引擎生成的 JSON/CSV/报告；容量锚点可用于交叉检查。报告没有的数据应标注缺失或扩展需求，不补写成仿真输出。

继续中断的大型实验时，先检查输出文件与 `checkpoint/`，区分数值执行已完成和明细导出已完成。可用 `python -u scripts/run_reference_experiment.py <配置> -o <原输出目录>` 恢复已提交的匹配检查点。它会校验配置和数值模块指纹，跳过已完成的数值计算。精简结果 JSON 不能补造缺失的逐张量生命周期；若没有原始记录，须重放并保存检查点。

大型场景默认按报告中的导出清单精简原始事件和临时通信缓冲记录，容量统计与真实并发峰值保持完整；超过 50,000 行的保留张量明细无损压缩为 `tensors.csv.gz`。优先阅读 `report.md`、`component_memory.csv`、`state_memory.csv` 和 `model_parameters.csv`；需要检查全部逐事件生命周期时，在 `estimate` 或 `simulate` 命令添加 `--full-detail`。读取明细前检查 `artifact_export` 的实际文件名，不整份加载大型 CSV/JSON。不要把导出行数减少解释为仿真忽略了这些资源。

只需要性能、流量、容量峰值和失败诊断的大型结构化推理，DP=PP=1 时优先使用 [流式推理](references/inference.md#大规模结构化推理)。需要完整组件/张量生命周期、训练或其他并行布局时继续使用普通引擎。批量实验默认串行并复用匹配结果；每个并发进程会分别占用图和缓存内存。

大型 optimizer offload 场景可显式配置 `execution.optimizer.inflight_components_per_rank: 2`，限制同时在途的组件更新并保留搬运/计算重叠；这是执行策略，会影响峰值和耗时，应随结果说明。未设置时保留原有全部提交策略。

用户指出仿真与现实存在差距时，先核对结构、精度、复制/分片与模型范围，再核对平台 SKU、单位、带宽方向、拓扑和实测 kernel/collective。用可核实 profile 替换合成硬件，并说明尚未校准的影响。型号相同不等于已验证性能准确；没有实测结果时不报告误差百分比或“符合真实情况”。

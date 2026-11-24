---
name: supernode-modeling
description: Model supernode training and inference resources, optional acceleration features, and TTFT/TPOT (TPOP) latency breakdowns with supernode_sim, automatically opening its editable local web dashboard. Use for pretraining, SFT, DPO, PPO/GRPO post-training with operator-level rollout/scoring/update costs, megakernel fusion, prefill/decode disaggregation, DSpark draft/verify, CSA/HCA/KDA attention, quantization, checkpoint/offload, Adam/Muon, Kimi K3 and Ascend 950, SU/SO topology and minimum-SU searches, or feedback searches for TP/SP/EP, prefill length and decode batch under latency targets.
---

# 超节点资源与通信建模

实际开展建模时，默认先准备并打开可编辑的流量仿真网页，再把问题转为可复现配置，调用 Python 引擎计算，解释资源需求、瓶颈和适用条件。仅维护 skill、参考资料或引擎代码时无需启动前端。前端与引擎由 `supernode_sim` 统一提供；此 skill 包含调用说明和自动启动脚本，**不包含独立的 Python 引擎**，需要 `supernode_simulation` 仓库或已安装的 `supernode_sim` 包。

展示训练仿真结果时，交付输出目录中的 `training_summary.json` 和 `training_breakdown.svg`。用平均步关键路径解释时延，用同卡时间并集解释通信暴露和空泡，用峰值卡的同步快照解释 HBM。报告计算与暴露通信的主要算子时注明各自分母。估算或未完成的仿真请报告数据不可用。

用公开训练日志校准 MFU 时，先对齐每步 token 数、激活卡数、精度峰值和日志的 FLOPs 分子。对比固定容量与 ragged EP 时，同时报告发送行、有效专家计算和填充开销。Marin 535B-A23B 的分阶段对照、可运行命令与适用范围见 [训练校准说明](references/training.md#marin-训练校准)。

训练从单个超节点扩展到多个超节点时，用同一份已解析的模型、硬件速率和软件配置运行 `training-scaling`。先声明强扩展或弱扩展，再分列每卡发送字节、每卡暴露通信、步关键路径通信、峰值 HBM 和实测误差。实测留出点只用于评分。步骤、实测字段口径和 64/128/256 卡示例见 [跨规模训练验证](references/training.md#跨规模训练验证)。

## 找到可运行环境

优先使用用户已给定的项目路径，其次检查 `SUPERNODE_SIM_ROOT` 环境变量，再检查当前工作目录及其父目录中的 `supernode_sim/` 与 `pyproject.toml`。如果本文件位于项目的 `skills/supernode-modeling/`，其上两级目录就是项目根目录。被复制到全局 skills 目录后，不能把该目录视为引擎所在项目。

在项目根目录、项目 Python 解释器下执行：

```text
python -m supernode_sim --help
python -m supernode_sim schema
```

包已安装时也可使用 `python -m supernode_sim`；示例文件仍位于原仓库。若找不到仓库或模块，报告具体缺少的路径/依赖，不伪造执行结果。引用配置文件路径、输出目录时遵循所在 shell 的引号规则。

## 调用时自动准备前端

每次使用本 skill 实际开展建模，默认先执行 [前端启动流程](references/frontend.md)，无需等用户另行请求网页；维护 skill 或引擎代码不触发此流程。用户明确指定只用 CLI、不要网页或运行环境不能启动服务时，按其要求或实际能力继续数值工作，并说明网页状态。

1. 运行本 skill 自带的 `scripts/launch_frontend.py`，已定位仓库时传入 `--project-root <项目目录>`；仅使用安装包时省略该参数。脚本检查现有页面资源、选择 Python 环境并启动或复用本地服务，输出机器可读 JSON。当前前端无需 npm / Node 构建，复用项目 `supernode_sim/web_static/`，保持输入编辑、实验运行与结果展示使用同一引擎配置。
2. 只有返回 `started` 或 `reused` 且资源检查成功，才报告网页就绪。用可用浏览器工具打开返回的 `inputs_url`；没有浏览器工具时提供实际 URL 和能力限制，不声称已打开。保护已有标签页中的未提交输入，优先使用新任务标签页或本任务已创建的标签页。
3. 将用户给定或本次整理的完整配置载入网页，按请求开展仿真。页面启动本身不自动提交数值任务。`estimate` / `simulate` / `serve` 可用网页 API，保存输入快照并在历史运行中展示结果；服务任务单独保存模型与请求两份快照。其他分析命令继续使用其原入口，明确网页支持范围。
4. 重复调用复用匹配服务；保留交付的服务供用户继续编辑和运行。失败时读取脚本返回的原因和日志，按具体原因修复后再试，不能循环重启、停止其他服务或用未验证 URL 代替真实页面。

网页包含模型、拓扑、硬件、并行度和 Workload 的输入展示与编辑、完整字段 JSON 编辑、schema 校验、运行日志、取消、历史快照以及流量/内存/利用率/时间线图表。编辑当前配置会同步下一次仿真；历史快照先复制再编辑。集群拓扑与执行计划按所选配置动态生成，共用当前草稿或运行快照来源；实际阶段仅使用引擎记录。操作、输入同步及回退细节见 [frontend.md](references/frontend.md)。

## 按问题选择资料和命令

- 配置、来源、单位、CLI：读 [configuration.md](references/configuration.md)。
- 借鉴其他仿真器、提高训练/推理执行粒度：读 [参考程序映射](references/reference-simulators.md) 和 [子算子执行描述](references/operator-traces.md)。使用 `hardware.kernel_traces` 将有依据的组件展开为串行子算子，再按实际子算子 profile 校准；同时核对阶段、矩阵维度和逐请求长度分布，不将更细粒度或 synthetic 检查称为实测精度提升。
- 根据 DavidV100 MicroBenchmark 改进核内计算、搬运或指令供给建模：读 [DavidV100 微架构服务需求](references/david-v100-microarchitecture.md)，在有范围约束的子算子上配置 `resource_demands` 与 `dependency_floor_s`。同物理资源先合并需求，独立资源才取最大值；保留 HBM 总流量与争用。报告参数仅适用于其 DavidV100 测试条件，不传播为其他 Ascend 型号的默认规格，也不把资源下界称为整网实测校准。
- 用户提供 NPUSim/cannsim 报告、引用 CANN 算子仿真或要求改进算子级时序：读 [NPU 算子仿真接入](references/npu-simulator.md)。用 `build_npu_sim_calibration.py` 检查并将单 launch CA summary 绑定到有 shape、tiling、layout 和 binary 身份的已展开子算子；保留 HBM 争用与内核流水重叠，区分 simulator、synthetic 和硬件实测，不把 PV 功能结果当作性能校准。
- 用户指定 msModeling/ServingCast 或请求到达、排队、多实例服务指标：读 [ServingCast 源码映射](references/msmodeling-servingcast.md) 与 [服务仿真](references/serving.md)，使用独立 `serve`。当前是隔离实例的 FCFS 固定成批；continuous batching、token budget、抢占和异构 PD 队列尚未实现，不能由固定 batch 结果乘并发数替代。
- 用户指定 SimuMax-Zhanlu、TorchTitan 仿真或训练 MFU：读 [湛卢训练参考](references/zhanlu-training.md)，检查 `metrics.training_accounting` 的阶段、dtype 与分子。额外重算、optimizer、通信打包不可无说明混进模型 MFU。
- 选择训练/推理加速特性、megakernel、PD 分离、DSpark、CSA/HCA/KDA 或对比 feature 开关：读 [features.md](references/features.md)，使用顶层 `features` 或 `--features` 文件；先运行 `list-features` 查看当前支持范围。DeepInfer 式小 P / 大 D、分阶段 batch 和相关优化另读 [deepinfer.md](references/deepinfer.md)。
- 提高仿真器自身运行速度、大型仿真、批量实验或本机内存/磁盘紧张：读 [resource-usage.md](references/resource-usage.md)，选择流式运行、多核候选搜索、缓存预算与导出粒度；用相同配置的基准验证主机耗时与数值一致性。
- 按组件分析存储、activation、梯度、Adam/Muon：读 [memory-breakdown.md](references/memory-breakdown.md)。
- Kimi K3 / Ascend 950 样例和真实性边界：读 [ascend950-kimi-k3.md](references/ascend950-kimi-k3.md)。
- 预训练、继续预训练、SFT、DPO、后训练 RL（PPO/GRPO）：先读 [训练任务分类与参数](references/training-tasks.md)，用 `workload.training` 或 `features.training_task` 建立任务流程，再按 [training.md](references/training.md) 选择 FSDP2、checkpoint、Adam/Muon、optimizer offload 等执行策略。
- 推理/KV 压缩与 offload/Engram：读 [inference.md](references/inference.md)。
- 推理 TTFT、TPOT（TPOP）与训练步的时延、算子耗时占比及算子内计算 / 通信 / 访存 / 启动占比：读 [latency-breakdown.md](references/latency-breakdown.md)。
- 固定模型、拓扑和硬件，输入 TTFT、TPOT（用户也可能写 TPOP），反求 TP/SP/EP、Prefill 长度及 Decode batch：读 [feedback.md](references/feedback.md)，运行 `feedback`，在满足目标的有限候选内优先最大化 Decode 吞吐量。
- TTFT、TPOT（用户可能写作 TPOP）、实测对齐或性能优化：先读 [runtime-calibration.md](references/runtime-calibration.md)；引用 K3/950DT 的 10 ms 数据时再读 [kimi-k3-runtime-evidence.md](references/kimi-k3-runtime-evidence.md)。
- 希望仿真拟合真实运行、根据 DeepInfer/NVIDIA/Ascend 数据改进预测：先读 [执行模型拟合](references/runtime-fitting.md)，实际调整 wire 精度/元数据、算法、NIC 资源和 collective/kernel 服务时间，运行拟合与留出实验并交付前后结果。不能仅输出真实性检查或缺失条件清单。
- 按硬件架构/拓扑提高拟合，或分析 GPU-NIC 亲和性、PCIe 与 offload 争用、多 spine、A3 HiAM/HCCS 层级：读 [硬件拓扑拟合](references/hardware-topology-fitting.md)，使用 `io_groups`、显式分层图及可选逐流 ECMP，检查 profile 的拓扑签名。不要只调整全局效率或给所有卡对套同一带宽。
- 适配其他 Ascend 机型/运行环境、超出单一 DeepEP 数据点的 batch/消息量，或接入 HCCL/msprof 数据：先读 [Ascend 多环境适配](references/ascend-runtime-adaptation.md)，用 `build_runtime_calibration.py` 从各环境独立观测生成 `collective_models`，按覆盖范围插值并运行留出与完整执行检查。数据采集按 [Ascend 实测来源](references/ascend-measurement-sources.md)；此前 H800 DeepEP 的误差不能标为 Ascend 精度。
- 与公开指标做严格比较：读 [公开数据对比](references/public-benchmark-validation.md)，使用证据目录与 `scripts/compare_inference_benchmarks.py`。逐条件可比后才计算误差；`public` 只表示来源公开，作者估算与硬件实测须用 `evidence_kind` 区分。
- 接收 SU/SO 拓扑、scale-out 域、分析链路瓶颈/最小域搜索：读 [topology.md](references/topology.md)。
- 用户用自然语言生成或修改 scale-up / scale-out 拓扑：按子 skill [supernode-topology](subskills/supernode-topology/SKILL.md) 生成拓扑规范、原生配置和拓扑图，再回到本 skill 做流量仿真。仅生成拓扑时在子 skill 内完成，不要求模型或负载。

先检查用户提供的配置与可核实资料，再补齐问题所需的参数。只询问会改变结论且无法从现有资料获得的缺失项，使用明确字段名。模型名称不是结构证据；对未公开结构采用用户认可或明确标注的 synthetic 假设，不把假设写成某模型的公开规格。

保留用户场景，在工作目录创建可重跑配置。先 `validate`，再按任务运行 `estimate`、`simulate`、`sweep`、`size-su` 或 `feedback`；`estimate` 提供必要下界，不能证明 SLO 可行。硬件显然无法容纳的配置也有分析价值，保留失败原因而不自动放大硬件。

Feature 是模型与执行策略输入，不是固定加速倍率。对同一路径的重复设置保留冲突检查；resolved config 使用 canonical 字段重跑。Attention 类型变化需要实际模型逐层结构和维度，不能默认质量等价；DSpark 类路径必须有显式 draft 成本、驻留及接受数。核对报告中的 `model_metadata.feature_plan`，并区分已选择、解析建模、profile 驱动和仍不支持的功能。

Megakernel 要指定有序 kernel 组及融合启动成本，检查 `model_metadata.megakernel` 的实际匹配、跳过原因和未继承的成员 profile；零匹配不能声称提速。PD 使用两个独立物理池，当前相同 TP/EP、DP=PP=1，总卡数为 `2 × TP × EP`；双份权重、状态和首 token ID 传输按实际拓扑计费。TTFT 是 P 池首 token 完成，交接影响首个 decode 间隔，不是线上客户端延迟或生产 P99。

DeepInfer 类部署选型优先比较 P 的小 EP、可选小 TP/SP 与请求 bs=1，以及 D 的 Wide-EP 与请求 bs=128/256；先声明 bs 是每 Attention replica 还是整个池。保留官方小 EP/Attention DP 基线，小 SP 是本项目候选，不能归为官方当前部署。不同 P/D 布局或 batch 使用独立阶段实验，不能套用现有同布局 PD handoff；具体 batch 换算、候选和能力边界见上述参考。

## 训练任务的执行与核对

先确定预训练、SFT、DPO 或 RL 算法，再配置任务专用参数。新任务使用 `workload.training`；原有未分类训练仍保持兼容。SFT 分开上下文、有效数据和监督标签；DPO 的 batch 计偏好对；RL 分开生成 batch、更新 microbatch、采样组、更新轮数及 actor/reference/reward/critic，不能统一套用普通训练 token 公式。

按 [training-tasks.md](references/training-tasks.md) 的参数表和 feature 矩阵填写实际执行模型，使用相应 `examples/training_task_*.yaml` 起步。训练任务先构图，再统一展开算子与应用 profile；绑定时保留任务、角色、阶段和形状身份。对 checkpoint、packing、loss mask、reference、update epochs 等选项，核对它改变的是工作量、状态、依赖还是仅统计分母，不使用固定提速倍率。

报告增加 `metrics.training_task` 和阶段 CSV，分列处理/有效/监督/生成 token、actor/critic 优化步与阶段时间；RL 的 step 表示完整迭代。MFU 只计可训练角色的模型前反向，冻结评分、rollout、loss 和优势计算单列。当前 RL 为共池串行、PP=FSDP=1；LoRA、异步 rollout、独立资源池等能力边界见任务参考，不把无法执行的参数写成已经支持。用 `scripts/evaluate_training_tasks.py` 可复跑五类任务及参数对照。

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
- 保留原有输出，并给出 TTFT、TPOT（TPOP）总时延及计算、node 内 / scale-up / scale-out / 其他通信、显式同步、调度槽位、内存准入等待和资源争用拆解。引用 `metrics.latency_breakdown` 与 `latency_breakdown.csv`；分项沿关键依赖路径可加和，不累加并发 rank 或重复计入上游同步依赖。搜索结果保留逐候选 CSV，并展示推荐候选拆解。仅估算、训练、执行失败或指标未定义时标注不可用，不补零；口径和请求选择见 [latency-breakdown.md](references/latency-breakdown.md)。
- 最终报告同时展示算子耗时排行：推理分别按 TTFT、TPOT，训练按平均步、首步与可用的稳态步统计。使用对应记录的 `operator_breakdown` 和 `operator_breakdown.csv`，列出阶段、组件、实际算子名、耗时、占窗口比例，以及算子内计算、通信、访存、启动、等待、争用、未细分/其他比例。明确两个分母；保留完整表及来源，不仅列 Top N。计算/访存按限制资源归因，不能相加重叠服务；整 kernel 校准无内部证据时保留未细分。训练记录在 `metrics.training_latency_breakdown`，不将训练步命名为 TTFT/TPOT。细化与融合边界、零耗时和缺失处理见上述参考。
- 给出配置、版本/来源、命令、结果文件，以及结论成立的约束。执行失败、不可行和缺参分别处理。
- 缓存/MTE 的工作集、访问量和搬运需求不能直接换算成 L1/L2/L3 容量或通道数；需要硬件与 kernel 校准。
- 固定 batch 仿真不提供生产请求排队或 P99 保证；SU “最小”限定为已检查候选集合与调度策略。
- Feedback 的“最优”限定为所声明候选集内满足 TTFT/TPOT 与资源约束的最高 Decode 吞吐。Prefill 与 Decode 用同一候选完整执行；SP 共用 TP rank。报告所选长度和 batch，不把不同候选的长度/batch 极值拼成一个推荐。

正式数值引用引擎生成的 JSON/CSV/报告；容量锚点可用于交叉检查。报告没有的数据应标注缺失或扩展需求，不补写成仿真输出。

继续中断的大型实验时，先检查输出文件与 `checkpoint/`，区分数值执行已完成和明细导出已完成。可用 `python -u scripts/run_reference_experiment.py <配置> -o <原输出目录>` 恢复已提交的匹配检查点。它会校验配置和数值模块指纹，跳过已完成的数值计算。精简结果 JSON 不能补造缺失的逐张量生命周期；若没有原始记录，须重放并保存检查点。

大型场景默认按报告中的导出清单精简原始事件和临时通信缓冲记录，容量统计与真实并发峰值保持完整；超过 50,000 行的保留张量明细无损压缩为 `tensors.csv.gz`。优先阅读 `report.md`、`component_memory.csv`、`state_memory.csv` 和 `model_parameters.csv`；需要检查全部逐事件生命周期时，在 `estimate` 或 `simulate` 命令添加 `--full-detail`。读取明细前检查 `artifact_export` 的实际文件名，不整份加载大型 CSV/JSON。不要把导出行数减少解释为仿真忽略了这些资源。

只需要性能、流量、容量峰值和失败诊断的大型结构化推理，DP=PP=1 时优先使用 [流式推理](references/inference.md#大规模结构化推理)。需要完整组件/张量生命周期、训练或其他并行布局时继续使用普通引擎。`sweep`、`size-su`、`feedback` 可通过 `--workers` 并行评估独立候选；按主机内存选择进程数，每个进程分别占用图和缓存。单次仿真的依赖、争用、内存准入和校准条件仍完整执行，不以减少模型层数或输出长度代替仿真器优化。

`speculative_decode` 支持普通 `simulate` 和 `run_streaming` API；流式投机复用相同 draft/verify/commit hooks，按实际提交 token 数归因时延，并保留状态快照/回滚。小 batch 按实际 EP 请求归属计状态；标量 draft profile 必须覆盖最大 local batch，并在所有 rank 计费。`megakernel` 与 `prefill_decode_disaggregation` 仍须普通 `simulate`，流式路径不处理跨组件融合或跨池交接。PD 暂不与投机、KV offload、Engram 组合；未知收益不能用一个效率系数替代缺失语义。

大型 optimizer offload 场景可显式配置 `execution.optimizer.inflight_components_per_rank: 2`，限制同时在途的组件更新并保留搬运/计算重叠；这是执行策略，会影响峰值和耗时，应随结果说明。未设置时保留原有全部提交策略。

用户指出仿真与现实存在差距时，先对齐计时边界、负载与软件执行路径，再核对结构、精度、复制/分片、平台 SKU、拓扑及实测 kernel/collective。使用 [校准流程](references/runtime-calibration.md) 分开校准 prefill/decode、服务时间和启动开销；检查实际匹配覆盖率。公开的整网 TPOT 不能填入单算子 profile，投机 decode 的吞吐不能充当普通 decode 的耗时，缺失 TTFT 保持未知。没有同条件留出实测时，交付可复跑的预测和明确限制，不报告误差百分比或“符合真实情况”。

真实硬件的 SO 建模须分别检查单卡注入、主机聚合和共享上联。多 NIC 主机可显式设置 `hardware.scale_out.host_injection_Bps`；这是每主机每方向的带宽，未提供时保留原共享单注入口径。MoE 使用 `precision.moe_dispatch/moe_combine` 和各自 metadata 字节表达非对称流量；`hardware.collective_profiles` 在保留字节与争用的基础上提供同条件服务时间约束。不能把双向带宽、每卡端口能力或 NCCL busbw 当作同一个数。配置、拟合命令和仍未实现的热点/动态调度见 [执行模型拟合](references/runtime-fitting.md)。

`io_groups.members` 使用物理设备编号；相邻 rank 不代表相同 NIC/PCIe/HiAM。高层 topology DSL 暂未提供 I/O/ECMP 字段时，在生成的 canonical 硬件中配置并重新校验。A3 分层生成器需要各级速率/时延及来源，不能套用到 950；未公开的数值明确作为假设。新 collective profile 的 `applicability_signature` 锁定硬件、路由、rank 和 peer 矩阵；改变这些条件后检查匹配，旧无签名匹配须标 unguarded。容量/路径检查与实机留出误差分开报告。

Ascend 响应曲线通过 `hardware.calibration_context` 锁定 SKU、CANN/HCCL/驱动、runtime、transport、mode 和来源计时边界，按完整 peer 比例分组；只对同条件已测 payload 区间插值，不跨设备/软件/rank 数外推。MoE whole-call 模型必须有五项 shape/format 条件；HCCL transport 模型只约束网络，pack/unpack 成本保留。报告 exact/interpolated/out_of_range 和未覆盖部分，synthetic 例子不能变成精度结论。kernel profile 可附硬件/环境/结构元数据匹配条件，未知形状不能凭同 FLOPs/bytes 复用。

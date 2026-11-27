# 昇腾 970DT 训练集群四场景分析｜PPT 制作交接稿

版本日期为 **2026 年 11 月 27 日**。本文面向接手制作演示文稿的新 Codex 会话。工作区根目录为 `C:\D_disk\Github\supernode_simulation`。所有本地链接均相对本文所在的 `docs/` 目录，命令从仓库根目录运行。

## 1. 新会话的任务

> 请阅读本文，使用 **ppt-master-huawei-v2** 华为 PPT 技能，在本仓库另存一份中文、16:9、可编辑的 PowerPoint。采用观点式红标题，表格保留原生单元格，每页备注给出案例、字段、比较条件和来源。开场先讲训练目标，再用一页表概括四个场景。正文依次讲长序列切分与通算掩盖、显存 offload、故障与故障恢复、集群规模扩展的线性度。按本文数字与原始记录核对，并在交付前逐页渲染。

建议主标题为 **《昇腾 970DT 训练集群效率与吞吐分析》**，建议另存为 `outputs/presentations/昇腾970DT训练集群四场景分析-20261127.pptx`。采用 **15 页主报告＋6 页附录与文献，共 21 页** 的结构。11 月 25 日的 18 页版本可用于参考版式、中文术语和表格样式。新版的算子、切分、offload 与故障数字使用下面列出的 11 月 27 日数据。[上一版 PPT](../outputs/umdk-issue7-training-20261125-eff100/昇腾970DT训练仿真分析-20261125.pptx)

**开工最短路径。** 先读第 2 节场景总表和第 3 节数据索引，再按第 5 节逐页制作。已备好 16 页的正文矢量元素、演讲备注素材，以及五张附录的数值和加粗掩码。即使从头排版，本文也附有全部五张数据表。[可编辑的单页素材包](../outputs/training-20261127-four-scenario-evidence/editable-slide-sources.zip) · [附录矩阵与加粗掩码](../outputs/training-20261127-four-scenario-evidence/appendix_matrices.json)

## 2. 开场的四场景总表

第 2 页先定义训练目标，第 3 页展示下表。**场景 01–04 是汇报顺序，T1/T2/T3 是数据实验编号**。保留这个区别，以便听众按报告 ID 查数。

| 正文章节 | 本地仿真的对象与范围 | 对照设计 | 开场带走的一句话 |
|---|---|---|---|
| **01 长序列切分与通算掩盖** | T2 的 50T 模型、50 万购置卡、六档超节点。另取 T3 的 100T 做同布局分块实验 | Attention 的 TP×CP 与专家层 EP 共用一组物理卡。固定并行度和预取，比较 1/2/4/8 块 | 在选定的 100T 布局上，八块使 4M 步时缩短 **30.84%**，同布局的 128K 步时增加 **11.59%** |
| **02 显存 offload** | T3 固定 50 万卡、每个超节点 4,096 卡，先比较 1.6T/10T/50T/100T，再深入 **10T、1M** | 真实显存上限下对照常驻和迁移。固定迁移 52 层，再扫主机预取深度 0/1/2/4/8 | 一套布局由 **260.935 降至 258.912 GB/卡**而可运行。另一套原本可常驻的布局同样迁移时，步时增加 **6.28%** |
| **03 故障与故障恢复** | 故障归档中的 **50T、1M、SU4096**，覆盖 10 万、20 万、50 万购置卡。各故障策略使用配对的 32 条 48 小时日历 | 比较全局重启、加快建链、全流程提速、同域热备、完整副本隔离和保存间隔。展示隔离事件和 48 小时时间账本 | 50 万卡案例中，仅把建链由 60 秒降至 30 秒，净吞吐提高 **5.93%**。完整副本隔离将有效训练占比从 **5.98% 提到 77.02%** |
| **04 集群规模扩展的线性度** | **10T MoE**，SU4096，10 万/20 万/50 万购置卡，十档序列 | 同一长度下固定全局批量和有效工作量，用 10 万卡作基线。跨长度使用含因果注意力的有效计算吞吐 | 卡数增至五倍，4K 加速 **2.44 倍**、4M 加速 **4.39 倍**，强扩展效率为 **48.9%** 和 **87.8%** |

健康训练每优化步处理 **1,024 条有效序列**。十档长度为 **4K、8K、16K、32K、64K、128K、256K、512K、1M、4M**，K 和 M 按 1,024 和 1,048,576 换算。2M 未采样。新版健康矩阵有 130 个报告点、110 个不同负载。T1 与 T3 在 **10T、50 万卡、SU4096** 的十档上重合，T2 与 T3 在 **50T、50 万卡、SU4096** 的十档上重合。故障研究是在独立的故障日历上使用已有健康布局，另计实验覆盖范围。

## 3. 数据地图与取数方法

### A　健康训练、模型切分与权重迁移

主目录为 `outputs/training-20261127-tpcp-ep-fsdp2/`。它提供 T2、T3，以及故障研究的原始健康布局。[研究说明与复现入口](../outputs/training-20261127-tpcp-ep-fsdp2/README.md) · [模型与硬件输入](../outputs/training-20261127-tpcp-ep-fsdp2/inputs.json) · [归档核验](../outputs/training-20261127-tpcp-ep-fsdp2/verification.json)

| 要找的数据 | 文件、连接键与口径 |
|---|---|
| 每个场景的最短步时、两种 MFU 和吞吐 | 以 `id` 定位报告行，以 `source` 定位共享案例目录。`throughput_policy/throughput_layout` 对应 `step_s、max_tokens_s、max_purchased_mfu_pct`。`mfu_policy/mfu_layout` 对应独立选优的 `max_active_mfu_pct`。[optimal_grid.csv](../outputs/training-20261127-tpcp-ep-fsdp2/optimal_grid.csv) |
| 某一获选布局的参与卡数、MFU、显存、重叠与分块收益 | 在同一 `id` 下按 `mode` 或 `best_throughput=True` 选行。可读 `active_cards、active_mfu、purchased_mfu、peak_gb、chunk_gain_pct、wall_overlap_pct` 等。[metrics.csv](../outputs/training-20261127-tpcp-ep-fsdp2/metrics.csv) |
| 固定同一布局的 1/2/4/8 块步时 | 读 `cases/<source>/training_summary.json` 的 `policies.ON/OFF.trials`。匹配同一组 `args = [TP, CP, EP, DP, 稠密FSDP度, 专家FSDP度]`、同一重算策略，再按 `blocks` 比较。获选运行的 `details` 提供 `control、wall_s、overlap_s`。[100T、4M 单点](../outputs/training-20261127-tpcp-ep-fsdp2/cases/T3-M4-S4194304/training_summary.json) · [算子分解](../outputs/training-20261127-tpcp-ep-fsdp2/operators.csv) |
| 同一布局的主机迁移和深度扫描 | 在 `offload_sweep.csv` 固定 `id、checkpoint、layers_offloaded` 和布局后比较 `host_depth`。取 `peak_gb、step_s、tokens_s、active_mfu、host_exposed_s、h2d_gb、d2h_gb`。购置卡 MFU 乘上 `active_cards/cards` 折算。单点 JSON 还提供 `off.required_gb、best、trials、same_layers_jit_s、relaxed_resident_reference_s`。[offload_sweep.csv](../outputs/training-20261127-tpcp-ep-fsdp2/offload_sweep.csv) · [10T、1M 单点](../outputs/training-20261127-tpcp-ep-fsdp2/cases/T3-M2-S1048576/offload.json) |
| 如何解释“最优” | 布局先经容量和代价筛选，对入围者做事件调度。健康方格的最优仅覆盖实际跑过且满足容量的候选。常驻布局比较重算开/关与 2/4/8 块，单块是匹配对照。权重迁移从可救回的超限布局中选代表做深度和层数扫描。[布局搜索](../outputs/training-20261127-tpcp-ep-fsdp2/run_study.py) · [权重迁移实现](../outputs/training-20261127-tpcp-ep-fsdp2/offload.py) · [算量、组网和时间线](../outputs/training-20261127-tpcp-ep-fsdp2/model.py) |

### B　新版 T1 的 10T 集群扩展实验

使用 `outputs/training-20261127-10t-cluster-scaling/`。这里有 30 点，10 万、20 万卡共新跑 20 点，50 万卡复用并核对 A 的十个 `T3-M2` 点。A 中旧的 `family=T1` 是 **50T**，仅在故障归档按旧案例编号回溯时使用。新版健康扩卡图和附录的 T1 均取 B。[30 点明细](../outputs/training-20261127-10t-cluster-scaling/cluster_scaling.csv) · [验证与模型参数](../outputs/training-20261127-10t-cluster-scaling/validation.json) · [复跑脚本](../outputs/training-20261127-10t-cluster-scaling/run_t1.py) · [利用率与计算吞吐参考图](../outputs/training-20261127-10t-cluster-scaling/10T集群利用率与计算吞吐.png)

取数按 `purchased_cards、sequence_tokens`。正文使用同一吞吐获选布局的 `step_s、cluster_effective_eflop_per_s、purchased_card_mfu_pct、raw_tokens_per_s、active_cards`。另有 `equivalent_256k_tokens_per_s、same_length_speedup_vs_100k、strong_scaling_efficiency_vs_100k_pct`。独立优化的参与卡利用率在 `max_active_card_mfu_pct_separately_optimized`，相应布局与最快布局可能不同。`source_case` 指向可还原单块对照的目录。

### C　故障日历、恢复策略与进度

使用 `outputs/training-20261127-fault-recovery/`。该批沿用原健康实验的 130 个报告点和 110 个不同负载，共有 488 个保留的健康算子配置。主表固定各场景的健康吞吐最优布局，32 次试验在各故障策略间使用同一物理日历。50T、1M、50 万卡示例在该批的 ID 是 **`T1-N500000-S1048576`**。[故障研究说明](../outputs/training-20261127-fault-recovery/README.md) · [实参 campaign.json](../outputs/training-20261127-fault-recovery/campaign.json) · [布局 profiles.json](../outputs/training-20261127-fault-recovery/profiles.json) · [核验记录](../outputs/training-20261127-fault-recovery/verification.json)

| 用途 | 取数方式 |
|---|---|
| 主布局六种对照 | 以 `id、policy` 连接。策略为 `healthy、restart、su_spares、replica、restart_fast、restart_10min`。读 `net_tokens_s、goodput_pct、full_train_s、reduced_train_s、recovery_stop_s、write_s、rollback_equiv_s`。[training_reliability.csv](../outputs/training-20261127-fault-recovery/training_reliability.csv) |
| 故障后的再次选优 | `best_goodput.csv` 按恢复策略在既有健康布局中重新选择净吞吐最高项。第 9–12 页的方案对照固定健康选定布局，采用上一行的主表。[best_goodput.csv](../outputs/training-20261127-fault-recovery/best_goodput.csv) · [所有保留布局](../outputs/training-20261127-fault-recovery/all_layouts.csv) |
| 恢复单元、事件时间线与分位数 | `study.py` 给出连续排卡的单元划分和日历。`reliability.py` 与 `training_reliability.py` 定义故障、状态转换、保存及进度。事件时间线取首条代表轨迹，P50/P95 另取里程碑表，保留已观测端点及截尾数。[研究实现](../outputs/training-20261127-fault-recovery/study.py) · [故障引擎](../supernode_sim/reliability.py) · [训练恢复实现](../supernode_sim/training_reliability.py) · [里程碑 CSV](../outputs/training-20261127-fault-recovery/recovery_milestones.csv) · [隔离事件原始记录](../outputs/training-20261127-fault-recovery/incidents/T1-N500000-S1048576_replica.json.gz) |

### D　两项新增配对对照与制页素材

新增证据已保存到固定目录 `outputs/training-20261127-four-scenario-evidence/`。后续会话可直接使用下列文件。正式输入数据、补算和版式草稿的用途各自列明。[文件 SHA-256 清单](../outputs/training-20261127-four-scenario-evidence/checksums.json)

| 内容 | 依据 |
|---|---|
| 10T、1M、健康最快常驻布局的同布局卸载对照 | 归档的 `OFF-R3` 常驻步时已经重现。保持 52 层及其余布局参数固定，另算按需与深度 2。[原始补算 JSON](../outputs/training-20261127-four-scenario-evidence/offload_10t_1m_matched.json) · [复现脚本](../outputs/training-20261127-four-scenario-evidence/replay/offload_10t_1m.py) |
| 全局重启仅缩短通信组建链 | 固定 50T、1M、50 万卡、健康 `ON-R3` 布局，将 `rebuild_s` 从 60 调为 30。32 条基线逐轨迹字段与原归档吻合。[补算汇总](../outputs/training-20261127-four-scenario-evidence/fault_rebuild_summary.json) · [64 条试验记录](../outputs/training-20261127-four-scenario-evidence/fault_rebuild_trials.csv) · [复现脚本](../outputs/training-20261127-four-scenario-evidence/replay/fault_rebuild.py) |
| 第 16–20 页可编辑表数据 | JSON 的顶层键为页码。每页有 `headers、rows、values、bold、negative`。二维数组含表头和行标签，14 行×12 列。按 `rows` 写文字、`bold` 设粗体、`negative` 设负值颜色。[制作矩阵](../outputs/training-20261127-four-scenario-evidence/appendix_matrices.json) |
| 可复用的主图和备注草稿 | ZIP 中 `pages/pageNN.body.svgfrag` 是 1280×720 SVG 的中央内容，`pages/pageNN.json` 含观点标题、核心句、结论条、口播、术语和红字释义。覆盖第 4–14、16–20 页。第 1、2、3、15、21 页按本文制作，公共页眉、页脚和备注分别组装。[素材包](../outputs/training-20261127-four-scenario-evidence/editable-slide-sources.zip) |

## 4. 全稿统一的指标与假设

1. **健康集群吞吐。** 真实词元吞吐 = 每步有效 token 数 ÷ 健康步时。标准实验每步有效 token 数为 `1,024 × S`。跨长度另报有效模型计算吞吐 `F_step ÷ 步时 ÷ 10^18`，单位为 **EFLOP/s**，EFLOP 是算量单位。
2. **模型有效算量。** 令 `P` 为每 token 激活参数数，`L` 为层数，`H` 为隐藏宽度，`S` 为单条序列长度，`B` 为有效序列数。本轮完整因果注意力代理取 `F_step = B × [6PS + 6LH·S(S+1)]`。这计入参数矩阵与因果注意力的有效前反向 FLOPs。重算、优化器、启动、搬运等耗时计入分母。
3. **两种 MFU。** 购置卡 MFU = 有效 FLOP/s ÷（购置卡数 × 单卡标称峰值）。参与卡 MFU 使用实际参与卡数作分母。同一模型、长度和购置卡数下，最短步时同时给出最高 tokens/s、有效 EFLOP/s 和购置卡 MFU。单独最大化参与卡 MFU 可能选中更少卡、更长步时的布局。
4. **强扩展。** 固定模型、长度、全局批量后，`加速比 = 10 万卡步时 ÷ 目标步时`，`强扩展效率 = 加速比 ÷（目标购置卡数 / 10 万）`。每个规模可按同一规则重选布局。256K 等效 token/s 用有效 FLOP/s 除以 **256K 时**每 token 的有效 FLOPs，正文与原始 token/s 分列。
5. **重叠。** `wall_overlap_pct` 是代表卡上两类或以上服务同时活跃的时间并集占比。`chunk_gain_pct = 100 × (1 − 多块步时/同布局单块步时)` 衡量净收益，含分块启动、附加访存与争用。
6. **故障净吞吐与有效训练占比。** `net_tokens_s` 是窗末已提交且未回滚的 token ÷ 48 小时。`goodput_pct = 100 × net_tokens_s ÷ 纯健康训练吞吐`。最后一次写完的持久检查点另以 `durable_tokens` 记账。四类墙钟时间为满编训练、降容训练、全作业停训恢复和同步保存，和为 48 小时。回滚工作用满速等效小时另列。
7. **硬件和调度。** Ascend 970DT 每卡 FP8 4 PFLOP/s、FP32 0.2 PFLOP/s、HBM 14.4 TB/s，SU 内/间单向带宽 800/100 GB/s。计算和带宽效率输入均为 100%。每卡物理显存 288 GB，按 90% 准入，即 **259.2 GB**。主机链路八卡共享每方向 64 GB/s，按每卡 8 GB/s 建模。PP=1、专家 TP=1、均匀专家路由、规则排卡，使用均衡代表卡事件调度。报告卡数指购置规模，图上列出实际参与卡数时单独标识。
8. **分片和检查点。** `TP×CP` 与 `EP×专家副本数` 覆盖同一组物理卡，外层 DP 决定组的份数。稠密和专家 FSDP2 分片度分别定义参数归属。完整层权重预取深度固定为一层，主机权重预取深度是另一独立设置。激活重计算控制反向时重放前向，外部恢复检查点则用于训练进度的持久保存。

## 5. 正文逐页制作脚本

### 第 1–3 页　训练目标与场景地图

| 页 | 观点和画法 | 可直接写入页面的证据 |
|---|---|---|
| 1 封面 | 《昇腾 970DT 训练集群效率与吞吐分析》，使用华为模板。时间与报告范围可放在开场备注 | 背景为健康算子仿真、故障恢复实验及 10T 扩卡补跑 |
| 2 训练目标 | 以“同一负载下，参与卡效率最高的布局为何更慢”开场。左边放同点双布局，右边定义集群吞吐、购置卡 MFU、参与卡 MFU。页尾引出故障后按保留 token 评价进度 | **50T、50 万卡、SU4096、1M。** 参与卡 MFU 最优布局用 262,144 卡，步时 15.336 s，参与卡 MFU **61.81%**、购置卡 MFU **32.41%**、吞吐 **70.01 百万 token/s**。最快布局用 491,520 卡，14.207 s，三项分别为 **35.58%、34.98%、75.58 百万 token/s**。A 的 `T3-M3-S1048576` 与 `T1-N500000-S1048576` 为同一输入 |
| 3 四场景总览 | 使用第 2 节的四行表，按“场景、负载、对照、核心发现”四列排。可让场景名称跳到第 4、6、9、13 页 | 说明健康矩阵 130 点、110 个不同负载。可靠性结果附带 48 小时、32 条配对轨迹的窗口条件 |

### 第 4–5 页　场景 01　长序列切分与通算掩盖

**第 4 页：4M 序列上，SU4096 的吞吐约为 SU128 的两倍。** 左图固定 50T、50 万卡、4M，每一档超节点分别选最快可行布局。六档 SU128/256/512/1024/4096/12288 的步时依次为 **211.682 / 161.928 / 161.920 / 124.163 / 105.673 / 105.673 s**，吞吐依次为 **20.29 / 26.52 / 26.53 / 34.59 / 40.64 / 40.64 百万 token/s**。右图画 SU4096 上的获选布局：Attention 的 TP8×CP256 与专家层 EP2048×1 专家副本映射到同一组 2,048 卡，外层 DP240，共 491,520 张参与卡。稠密 FSDP512 份、物理步长 8；专家 FSDP2 份、物理步长 2,048，两类组跨度同为 4,096 卡。SU4096 与 SU12288 的十个长度均出现相同吞吐获选布局和步时。这组结果包含布局重新选优。配 GTP 论文解释切分与聚合时机。[A 场景表](../outputs/training-20261127-tpcp-ep-fsdp2/optimal_grid.csv) · [GTP](https://arxiv.org/abs/2611.32057)

**第 5 页：同一布局切成八块，128K 变慢，4M 加快。** 固定 100T、50 万卡、SU4096、TP8、CP256、EP2048、DP240、稠密/专家 FSDP512/16，开启重算、权重常驻。用两行时序图并列 KV 搬运→注意力计算，以及专家分发→GEMM→合并。下表中的数字为秒/步。

| 序列 | 单块 | 两块 | 四块 | 八块 | 八块相对单块 |
|---|---:|---:|---:|---:|---:|
| 128K | 11.583 | 11.667 | 11.862 | 12.926 | 步时增加 11.59% |
| 4M | 254.763 | 209.838 | 187.388 | 176.189 | 步时缩短 30.84% |

4M 八块运行的多类服务重叠 **83.138 s，占 47.19%**；其中计算和通信同时活跃约 83.014 s。与单块相比净省 **78.574 s**。可见图和备注都应分别标明“时间区间占比”和“步时净降幅”。同一布局的 4K 因整除约束仅有单块、两块数据。论文中的 1.88 倍是 MegaScale-MoE 整套系统相对 Megatron-LM、352B MoE 和 1,440 张 Hopper 的结果。[128K 对照](../outputs/training-20261127-tpcp-ep-fsdp2/cases/T3-M4-S131072/training_summary.json) · [4M 对照](../outputs/training-20261127-tpcp-ep-fsdp2/cases/T3-M4-S4194304/training_summary.json) · [MegaScale-MoE](https://arxiv.org/abs/2505.11432)

### 第 6–8 页　场景 02　显存 offload，主案例为 10T

**第 6 页：先交代模型负载和需要驻留的状态。** 四种模型均在 50 万购置卡、SU4096 上各自选择十档长度中的最高吞吐。1.6T 是 DSV4 基准，后三个是同系列外推配置。下表的工期均为 100 万亿 token ÷ 对应峰值吞吐，假设按该长度持续健康运行。

| 总参数 | 每 token 激活参数 | 峰值长度 | 峰值吞吐（百万 token/s） | 预算天数 | 4M 吞吐获选的激活重算 |
|---|---:|---:|---:|---:|---|
| 1.6T | 490 亿 | 256K | 542.32 | 2.13 | 关 |
| 10T | 1,590 亿 | 512K | 328.39 | 3.52 | 关 |
| 50T | 5,120 亿 | 512K | 89.28 | 12.96 | 开 |
| 100T | 1 万亿 | 1M | 40.11 | 28.85 | 开 |

40 个逐点最高吞吐候选均为权重常驻。**10T 有 61 层、1,792 个路由专家，每 token 选择 24 个，另有一个共享专家。** 本实验迁移指定层的 FP8 训练权重分片及量化系数。FP32 主权重、Adam 一二阶矩、梯度和激活仍留在设备。前向、反向各按需要 H2D，优化器更新后 D2H 回写。将这张驻留示意图作为下一页的入口。[A 模型输入](../outputs/training-20261127-tpcp-ep-fsdp2/inputs.json) · [A 逐点最优](../outputs/training-20261127-tpcp-ep-fsdp2/optimal_grid.csv)

**第 7 页：权重迁移可以救回超限布局，也会增加已可行布局的步时。** 固定 **10T、1M、50 万卡、SU4096、关闭激活重算**。下面两套布局各自进行同布局开关对照。迁移点都取 **52/61 层、提前 2 层预取**。

| 布局与权重策略 | 每卡 HBM（GB） | 步时（s） | 百万 token/s | 购置卡 MFU |
|---|---:|---:|---:|---:|
| A 常驻，TP8/CP64/EP64/DP976，稠密/专家 FSDP512/64，2 块 | 260.935 | 超限 | — | — |
| A 迁移，499,712 卡参与 | 258.912 | 6.608 | 162.48 | 30.10% |
| B 常驻，TP8/CP128/EP256/DP480，稠密/专家 FSDP512/16，8 块 | 148.047 | 4.718 | 227.57 | 42.16% |
| B 迁移，491,520 卡参与 | 146.025 | 5.014 | 214.13 | 39.67% |

A 的 259.2 GB 准入线决定开和关的可行性。归档还给了**放宽容量**的 6.323 s 常驻参考，适合在备注中解释搬运代价。B 是该输入的健康最快常驻布局，补跑复现了它的原始步时。B 迁移同样的层数省 **2.022 GB/卡**，步时增加 **6.28%**。SuperOffload 论文使用 GH200、Grace CPU 和 NVLink-C2C，并包括 CPU 优化器等设计。图中昇腾对照数字取 A、D。[A 单点](../outputs/training-20261127-tpcp-ep-fsdp2/cases/T3-M2-S1048576/offload.json) · [B 布局补跑](../outputs/training-20261127-four-scenario-evidence/offload_10t_1m_matched.json) · [SuperOffload](https://arxiv.org/abs/2509.21271)

**第 8 页：固定迁移 52 层，提前两层预取比按需搬运缩短 19.61% 步时。** 只比较上页布局 A。每卡每步 H2D **8.585 GB**、D2H **2.146 GB**，迁移字节数相同。

| 主机预取深度 | 每卡 HBM（GB） | 步时（s） | 百万 token/s | 主机传输暴露时间（s） |
|---:|---:|---:|---:|---:|
| 0，按需 | 258.830 | 8.221 | 130.61 | 1.342 |
| 1 | 258.871 | 6.631 | 161.92 | 0.294 |
| 2 | 258.912 | 6.608 | 162.48 | 0.293 |
| 4 | 258.995 | 6.608 | 162.48 | 0.292 |
| 8 | 259.160 | 6.608 | 162.48 | 0.293 |

深度 2 到 8 的步时在毫秒精度下持平，显存余量从 **0.288 GB** 缩至 **0.040 GB**。若允许层数随深度改变，此布局另有 46 层、深度 2 的候选；固定层数图统一取 52 层。[实际五档扫描](../outputs/training-20261127-tpcp-ep-fsdp2/offload_sweep.csv)

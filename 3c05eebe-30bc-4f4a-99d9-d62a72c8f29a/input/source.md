<!-- Slide number: 1 -->

Kimi K3 扩容专家池至 10.17T，保留混合注意力骨架
01 模型演进 · 以公开 Kimi K3 文本骨干为起点，构造 10T 推理设计

核心思想
参数容量扩大约 3.66 倍，top-32 每 token 激活 154.4B，仅占总量 1.52%。

保留计算骨架

混合注意力
93 层：69 KDA + 24 MLA

稀疏前馈
92 MoE + 1 Dense

跨层残差聚合
AttnRes
隐藏维度 7168 · 潜空间维度 3584
专家中间维 3072 · 共享专家 2 个
扩大路由专家池，同步增加路由器输出维度

扩大参数容量，独立设定激活规模

比较维度
当前 Kimi K3
10T 扩容设计

2.779T
10.171T
结构推导总参数

路由专家 / MoE 层
896
3328

每 token 选择专家
top-16
top-32

结构推导活跃参数
104.19B
154.41B
3328 个专家：沿用早期 EP128 整除约束，向近 10T 规模取整
最新实验采用 EP32，每个专家并行分片对应 104 个专家
边界：10T 为合成扩容，模型质量未验证；仅文本骨干，不含视觉、投影器和训练 MTP。来源：官方配置快照及 2026-10-19 仿真清单。

本页亮点
★ 总参数扩大约 3.66 倍，活跃参数约增加 48.2%。
★ 参数总量决定容量压力，激活规模决定主要计算负载，两者分开设计。

![](Image75.jpg)
1
Huawei Proprietary - Restricted Distribution

### Notes:
【第1页 · Kimi K3 扩容专家池至 10.17T，保留混合注意力骨架】

这一页说明我们怎样从公开 Kimi K3 文本骨干构造接近十万亿参数的推理研究对象。首先保留原有的计算骨架：总计九十三层，其中六十九层为 KDA，二十四层为 MLA；第一层使用稠密前馈，其余九十二层为 MoE，并保留 AttnRes 跨层残差聚合。隐藏维度、潜空间维度、专家中间维和两个共享专家均不改变。扩容动作集中在每个 MoE 层的路由专家池，从八百九十六增加到三千三百二十八，同时扩大路由器输出维度。这个数值来自早期 EP128 的整除约束，是近十万亿规模的工程取整，不代表最新实验使用 EP128；最新实验使用 EP32，每个专家并行分片对应一百零四个专家。按同一结构计数口径，总参数从二点七七九万亿增加到十点一七一万亿，约为原来的三点六六倍。主实验把每个 token 选择的专家从十六个增加到三十二个，活跃参数由约一千零四十二亿增加到一千五百四十四亿，增加约百分之四十八点二，只占新模型总量的百分之一点五二。因此，扩容后的参数容量压力和每次推理的计算负载需要分别分析。这里的十万亿模型是合成扩容设计，尚未训练或验证质量；统计范围只有文本骨干，不含视觉编码器、投影器和训练时的多 token 预测层。

过渡：模型总量和活跃量被拆开之后，下一页进一步说明这些参数如何转化为推理集群的容量、计算和通信要求。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• Kimi K3：Kimi K3（模型产品名）——公开模型是本次架构推导的起点，10T 版本是本项目合成扩容研究对象。
• T：Trillion，万亿——参数计数单位，1T 等于 10 的 12 次方个参数。
• B：Billion，十亿——参数计数单位，1B 等于 10 的 9 次方个参数。
• KDA：Kimi Delta Attention，Kimi 德尔塔注意力——本文本骨干中有六十九层，使用递归状态的注意力结构。
• MLA：Multi-head Latent Attention，多头潜在注意力——文本骨干保留二十四层门控 MLA，潜在表示用于降低注意力缓存需求。
• MoE：Mixture of Experts，混合专家——每个 token 仅选择部分路由专家参与计算，扩容增加专家池容量。
• Dense：Dense Feed-forward Layer，稠密前馈层——Kimi K3 文本骨干第一层为稠密层，本次扩容保持不变。
• AttnRes：Attention Residuals，注意力残差——按注意力方式聚合跨层残差表示，本次沿用原结构。
• token：Token，词元——模型处理文本的基本离散单位，页面活跃参数按每个词元计数。
• top-16 / top-32：Top-k Expert Routing，前 k 个专家路由——分别表示每个词元选择十六个或三十二个路由专家，本次主实验取三十二个。
• EP128 / EP32：Expert Parallelism，专家并行——数值为专家并行分片数；3328 沿用 EP128 整除取整，而最新实验使用 EP32。
• MTP：Multi-Token Prediction，多词元预测——训练时的预测扩展，本页的文本骨干参数计数不包含该部分。
• Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有信息，限制分发——依所选华为版式规范保留的模板页脚文字。

【标红内容详解 · 逐条一句话解释】
• 「3.66 倍」：同口径结构计数中，10,171,385,035,872 ÷ 2,779,484,476,000 = 3.65945，四舍五入为 3.66 倍。
• 「1.52%」：最新 manifest 的主实验活跃参数 154,414,012,512 ÷ 总参数 10,171,385,035,872 = 1.51812%，四舍五入为 1.52%。
• 「10.171T」：最新仿真 manifest 的结构推导总参数为 10,171,385,035,872；该数值对应 3328 路由专家的文本骨干合成扩容设计。
• 「48.2%」：主实验活跃参数相对当前基线的增幅为 (154,414,012,512 ÷ 104,189,619,808 - 1) × 100% = 48.2048%，取一位小数为 48.2%。

【来源与数据口径】
• docs/sources/README.md（官方来源快照与建模边界，核验日期 2026-09-23）
• outputs/kimi-k3-ascend950-fp8/config.resolved.json：model.total_params、model.active_params、model.architecture
• 工作区 input/manifest.json：created_at=2026-10-19T17:50:42.617776+08:00，主实验 top_k=32，total_params=10171385035872，active_params=154414012512
• 工作区 content/content1.md：3328 专家沿用早期 EP128 整除约束；最新 EP32 分片为 104 个专家

<!-- Slide number: 2 -->

10T 推理需同时满足显存峰值、HBM 带宽与低时延互联
02 集群要求 · 已仿真的部署条件；以下资源指标采用 1M 输入、batch=1、top-32、output=2

核心思想
1024 卡 TP32×EP32，SP32 共用 TP ranks；PP=DP=1（未搜索最小卡数）

显存：按物理副本与阶段峰值预算
61.44 GB/卡
F 完整请求的每卡显存峰值

10.17 TB
逻辑 FP8 权重

19.88 TB
集群物理权重 + scale

每卡 runtime + workspace
8 GB + 256 MB
实际峰值 ≠ 逻辑容量均分
峰值含副本、缓存、临时缓冲及运行时开销

计算与访存：分阶段配置资源
每卡规格
950DT
960DT

HBM 容量
144 GB
288 GB

HBM 带宽
4 TB/s
9.6 TB/s

FP8 标称峰值
1 PFLOP/s
2 PFLOP/s
两平台计算效率均假设 50%
8.34 TB/步
decode 集群累计 HBM 读写
prefill 计算量：1.611 EFLOP
HBM 读写量为累计流量，不是存储容量

互联：专家分发需要低启动时延
1024 卡单逻辑 SU · 非阻塞 · SO=0
950DT
960DT

单向带宽
1 TB/s
1.1 TB/s

启动/传播
2.1 μs
1 μs
两平台互联有效效率均假设 80%
86% / 87%
decode 关键路径的通信启动占比
prefill 载荷：772.87 TB
decode 载荷：3.31 GB/步
F：SP32 + MLA compressed + FP8 权重/计算/激活/通信/KV；KDA 状态 FP32。1M=1,048,576 token；batch=1、top-32、output=2。
来源：2026-10-19 最新 report.md、summary.csv、phase_metrics.csv、latency_breakdown.csv 与配置。硬件、拓扑及效率均为未实测校准的仿真条件。

本页亮点
★ SP 与压缩、量化共同控制显存峰值
★ decode 优先关注 collective 启动时延与 HBM 服务时间

![](Image96.jpg)
2
Huawei Proprietary - Restricted Distribution

### Notes:
【第2页 · 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联】

这一页把集群条件分成显存、计算与访存、互联三个维度。已仿真的部署是1024卡，张量并行32乘专家并行32，序列并行与张量并行共用设备，流水线并行和数据并行均为1。这是固定实验配置，并不是已经搜索出的最小卡数。以下工作量都对应1M输入、单请求、top-32和输出2个token，采用完整优化F。首先，FP8逻辑权重约10.17TB，但计入复制和量化scale后，集群物理权重约19.88TB；再叠加缓存、运行时与工作空间，完整运行的每卡峰值为61.44GB，因此不能把逻辑模型大小简单除以1024。其次，两平台每卡HBM容量分别为144和288GB，带宽为4和9.6TB每秒，FP8标称峰值为1和2PFLOP每秒，计算效率统一假设50%。F的prefill计算约1.611EFLOP，而单步decode的集群HBM累计读写约8.34TB，这个数是流量，不是容量。最后，网络采用1024卡单逻辑非阻塞SU，单向带宽1和1.1TB每秒，效率80%，启动与传播延时2.1和1微秒。在这个模型中，decode关键路径里通信启动约占86%和87%，说明优化集体通信的启动次数与时延很重要。所有硬件、拓扑和效率参数都是未实测校准的仿真条件，尤其路线图平台不代表已验证部署；F采用FP8的量化质量也未在这项性能仿真中验证。

过渡：资源条件明确后，下一页解释仿真如何把模型、负载与集群配置转换成可核对的时延和资源账本。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• 10T：Ten Trillion Parameters，十万亿参数规模——本方案模型参数量为约10.171万亿，标题按规模简写为10T。
• HBM：High Bandwidth Memory，高带宽存储器——加速卡上的存储资源，页面区分其容量、带宽和累计读写量。
• F：Full Optimization Configuration，完整优化配置（实验标签）——SP32、MLA压缩缓存和FP8权重/计算/激活/通信/KV的组合，KDA状态保持FP32。
• TP32：Tensor Parallelism 32，32路张量并行——把单层张量运算分到32个设备rank。
• EP32：Expert Parallelism 32，32路专家并行——将MoE专家分布在32个专家并行rank，和TP32共同构成1024卡布局。
• SP32：Sequence Parallelism 32，32路序列并行——把序列相关激活分片，复用TP的32个rank，不额外相乘增加卡数。
• TP：Tensor Parallelism，张量并行——本页的SP与TP共用同一组rank。
• PP：Pipeline Parallelism，流水线并行——沿模型层划分流水段，本次为1。
• DP：Data Parallelism，数据并行——不同副本处理不同输入，本次为1。
• ranks：Process Ranks，并行进程编号——标识参与并行操作的设备或进程；此处SP与TP共享rank。
• FP8：8-bit Floating Point，8位浮点格式——F用于权重、计算、激活、通信及KV缓存的精度假设。
• FP32：32-bit Floating Point，32位浮点格式——F中的KDA递归状态继续采用此格式。
• scale：Quantization Scale，量化缩放因子——物理权重总量含低精度量化所需的缩放元数据。
• runtime：Runtime Reservation，运行时预留——本次每卡额外预留8GB运行时空间。
• workspace：Workspace，算子工作空间——本次每卡工作空间预算为256MB。
• 950DT / 960DT：950DT / 960DT（仿真平台名称）——采用报告冻结的硬件路线图规格和效率假设，960DT相关参数尚未实机校准。
• GB / TB / MB：Gigabyte / Terabyte / Megabyte，吉字节/太字节/兆字节——本报告按十进制字节量显示，分别为10^9、10^12和10^6字节。
• TB/s：Terabytes per Second，每秒太字节——页面的HBM与互联带宽单位，互联为单方向带宽。
• PFLOP/s：Peta Floating-point Operations per Second，每秒千万亿次浮点运算——用于每卡FP8标称算力峰值，模拟有效计算效率为50%。
• EFLOP：Exa Floating-point Operations，百亿亿次浮点运算——总计算工作量单位，1 EFLOP为10^18次浮点运算。
• decode：Autoregressive Decoding，自回归解码——在已有上下文状态上生成后续token，页面给出其单步资源需求。
• prefill：Prompt Prefill，提示词预填充——处理全部输入token并建立后续解码所需状态的阶段。
• SU：Scale-Up，域内扩展互联——此处假设1024卡处在一个逻辑非阻塞互联域，逻辑边不代表单根物理线。
• SO：Scale-Out，域间扩展互联——本次单逻辑SU模型中不产生SO流量，故SO=0。
• μs：Microsecond，微秒——启动与传播延时的单位，1微秒等于10^-6秒。
• MLA compressed：Multi-head Latent Attention with Compressed Cache，多头潜在注意力压缩缓存——F用压缩的潜在向量缓存降低MLA缓存容量和访存。
• KV：Key and Value，键与值——注意力机制的缓存数据，F中的相关缓存采用FP8假设。
• KDA：Kimi Delta Attention，Kimi Delta注意力——模型的递归注意力组件，其状态保持FP32。
• 1M：One Megatoken，一个百万级token输入档位——本报告严格取1,048,576个输入token。
• token：Token，词元——模型读写文本的基本离散单元。
• batch：Batch Size，批大小——此页的batch=1表示一个请求同时参与推理。
• top-32：Top-32 Expert Routing，前32专家路由——每个token选用32个路由专家，不代表改变top-k后质量等价。
• output：Output Token Count，输出词元数量——此页采用output=2，对应完整prefill与后续首个decode间隔。
• collective：Collective Communication，集体通信——并行rank间的广播、规约和全互换等通信操作，启动开销影响decode关键路径。
• report.md / summary.csv / phase_metrics.csv / latency_breakdown.csv：Report / Summary / Phase Metrics / Latency Breakdown，报告/汇总/阶段指标/时延拆解文件——对应2026年10月19日最新仿真结果与阶段资源、关键路径归因数据。
• Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有—限制传播——所用华为风格模板的固定页脚声明。

【标红内容详解 · 逐条一句话解释】
• 「1024 卡 TP32×EP32」：来自冻结并行配置；32路张量并行乘32路专家并行组成1024卡，SP32复用TP ranks，未搜索更小卡数。
• 「61.44」：最新summary.csv中1M、batch=1、top-32、output=2、F两平台完整运行峰值均为61.436GB/卡，页面取两位小数。
• 「8.34」：最新phase_metrics.csv对应F单步decode的集群累计HBM读写8.338TB，页面取两位小数；该值是流量而非容量。
• 「86% / 87%」：由latency_breakdown.csv中950DT的36.3258ms/42.4228ms及960DT的17.2980ms/19.9022ms得到，分别约为86%与87%的decode通信启动占比。
• 「SP 与压缩、量化共同控制显存峰值」：1M、batch1、top32下B→C→D→E→F峰值122.304→102.977→85.346→75.478→61.436GB/卡，来自最新summary.csv。
• 「decode 优先关注 collective 启动时延与 HBM 服务时间」：F、1M、batch1的关键路径中，950DT通信启动36.3258ms和HBM服务5.9676ms，960DT分别17.2980ms和2.4865ms，构成主要decode耗时。

【来源与数据口径】
• input/report.md（更新时间2026-10-19T19:56:52+08:00）
• summary.csv：F、1M、batch1、top32、output2，单卡峰值61.436GB
• phase_metrics.csv：逻辑权重10.171TB、物理权重含scale19.877TB，decode HBM读写8.338TB/步
• latency_breakdown.csv：通信启动与阶段时延
• 冻结硬件与并行配置：TP32×EP32×PP1×DP1，1024卡逻辑SU，SP复用TP ranks

<!-- Slide number: 3 -->

仿真 Skill 将配置编译为资源事件图，评估容量与时延
03 仿真方法 · supernode-modeling 编排，supernode_sim 执行

核心思想
Skill 负责配置与解释；Python 引擎执行可复现的数值计算。

01 输入与严格校验
结构 · 精度 · 负载
模型张量、输入/输出长度与固定 batch
并行 · 拓扑 · 目标
并行策略、SU/SO 链路图与 SLO

严格校验
并行乘积 · 专家整除 · 路径连通
特性兼容 · 参数来源与假设标记
自然语言拓扑 → 原生链路图

02 编译与资源竞争
① 逐组件张量账本
权重、KV/KDA 状态、激活与工作区
② 资源事件图与执行

计算
HBM
网络

依赖约束 + 多资源共同推进
先估下界，再做重叠与公平共享调度
内存准入 / 释放；保留阶段与并发峰值
训练/推理共用资源内核

03 输出与解释复现
时延与吞吐
TTFT · TPOT · 吞吐量 · 通信分解
容量与瓶颈
单卡峰值 · 容量失败 · 资源占用

解析配置与报告
CSV / JSON · 代码指纹 · 可重复运行
结果带来源、假设与适用条件
本次：结构化流式推理

校准闭环
调硬件 / 并行 / 特性
候选扫描
实机校准
更新配置

边界：固定 batch，无线上排队/P99；未做实机 kernel 校准；解析下界不能证明可行。来源：项目仿真技能说明与系统架构文档。

本页亮点
★ 同一资源语义统一计算、存储与网络
★ 预测带来源和约束，可复现、可校准

![](Image106.jpg)
3
Huawei Proprietary - Restricted Distribution

### Notes:
【第3页 · 仿真 Skill 将配置编译为资源事件图，评估容量与时延】

这一页说明仿真结果是怎样产生的。入口是 supernode-modeling 技能工作流，它把用户问题整理为配置并解释结果，真正的数值计算由 supernode_sim 的 Python 引擎完成。第一步先固定模型结构、精度、负载、并行策略、拓扑和目标，给关键字段记录公开、用户、假设或实测来源，并分别检查配置、图连通性和执行约束；自然语言拓扑要先编译成原生链路图。第二步按组件建立权重、注意力状态、激活和工作区账本，再把依赖关系与计算、访存、通信工作量映射到资源事件。一个事件可以同时使用计算与 HBM 资源，采用屋顶线最大项和最大最小公平共享语义，不能把计算与访存耗时简单相加；内存容量则通过准入与生命周期释放单独约束。训练与推理共享这一资源内核，训练还需要梯度与优化器状态。本次使用结构化流式推理，保留性能、流量、阶段资源和并发峰值，而不是据导出明细多少判断仿真精度。第三步输出首词时延、逐词时延、吞吐量、容量失败和瓶颈，再保存解析后的配置、结果与代码指纹以供复跑。反馈环节可以扫描硬件、并行和特性候选，并用实机算子与集合通信数据校准；这里的实机校准是后续工作，本次尚未完成。结果仅对应固定批次，不包含在线请求排队或生产环境第九十九百分位保证，解析下界也不等于可行性证明。

过渡：方法和边界明确后，下一页直接看最新 kimi-k3-10t 场景的仿真结果及主要瓶颈。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• Skill：Skill，技能工作流——将用户问题整理为严格配置、调用引擎并解释数值结果的说明与编排层。
• supernode-modeling：supernode-modeling（技能名），超节点建模技能——本项目训练和推理资源分析的交互入口，不是独立数值引擎。
• supernode_sim：supernode_sim（Python 包名），超节点仿真引擎——完成配置解析、构图、资源执行、指标与报告生成的代码包。
• Python：Python（编程语言名），Python 语言——本项目数值仿真内核采用的程序语言。
• batch：Batch，批次——在固定并行布局下共同执行的一组请求，本次没有线上动态排队模型。
• SU/SO：Scale-Up / Scale-Out，域内扩展 / 域间扩展——描述超节点域内与跨域连接，带宽、延迟、共享与注入限制均需配置。
• SLO：Service Level Objective，服务水平目标——对吞吐或时延的目标约束，配置通过不代表执行一定满足目标。
• KV：Key-Value Cache，键值缓存——自回归推理保留的注意力键和值状态，其容量与精度、长度及分片策略有关。
• KDA：Kimi Delta Attention，Kimi Delta 注意力——页面张量账本覆盖该注意力结构的循环和卷积等状态。
• HBM：High Bandwidth Memory，高带宽内存——计算卡上的高带宽内存，其容量和服务带宽分别建模。
• TTFT：Time To First Token，首词时延——仿真中完成第一个输出 token 的时间，不包含线上请求队列。
• TPOT：Time Per Output Token，每输出词元时延——反映 decode 阶段的输出间隔，不能直接视为生产 P99 指标。
• CSV：Comma-Separated Values，逗号分隔值——用于导出结构化指标和组件资源明细的表格数据格式。
• JSON：JavaScript Object Notation，JavaScript 对象表示法——用于保存结构化配置、仿真输出和可复现元数据的格式。
• P99：99th Percentile，第九十九百分位——生产延迟分布的高分位指标，固定 batch 仿真没有提供这类保证。
• kernel：Compute Kernel，计算内核——需要同平台同负载实测来校准的底层算子实现，本次尚未进行实机校准。
• Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有信息与受限分发声明——来自用户指定华为风格模板的固定页脚。

【标红内容详解 · 逐条一句话解释】
• 「可复现的数值计算」：依据项目技能与架构说明，正式结果来自 Python 引擎，并通过解析配置、数值输出和代码指纹支持复跑。
• 「01 输入与严格校验」：配置、拓扑连通、特性兼容与执行可行性是分开的校验关口，输入合法不代表资源和时延一定满足。
• 「02 编译与资源竞争」：将张量、依赖及各类工作量映射到共同资源语义，同时模拟重叠、争用和内存生命周期。
• 「03 输出与解释复现」：技能负责读取引擎生成的指标、失败诊断及来源约束，保留配置和代码指纹以便复跑。
• 「训练/推理共用资源内核」：两类工作负载共享图与执行资源语义，训练另有梯度和优化器状态，不能直接沿用推理容量。
• 「本次：结构化流式推理」：本次 10T 场景采用结构化流式路径保留性能、通信、容量峰值和失败诊断；精简记录不表示忽略了相应资源。
• 「同一资源语义统一计算、存储与网络」：结构化组件图、拓扑资源映射与共享执行器统一记录计算、访存、通信及内存准入，来源docs/simulation-architecture.md。
• 「预测带来源和约束，可复现、可校准」：严格配置标注参数来源、输出resolved配置与代码指纹，实测kernel profile按工作量匹配，来源仿真Skill与系统架构文档。

【来源与数据口径】
• skills/supernode-modeling/SKILL.md
• docs/simulation-architecture.md
• skills/supernode-modeling/references/inference.md

<!-- Slide number: 4 -->

完整优化使 1M 上下文可完成，960DT 首步解码约 19.90 ms
04 最新结果 · 2026-10-19 19:56（北京时间）· 1024卡，batch=1，top-32，输出2 token；模型侧预测

核心思想
104组全部完成仿真：94组可完成请求，10组容量失败；完整优化覆盖128k至1M。

首 token 时间 TTFT：上下文越长，计算与传输负担越高

950DT
960DT
单位：秒（越低越好）

5.66
128k

4.37

12.87
256k

9.51

32.04
512k

22.17

89.37
1M

56.99

0
20
40
60
80
100

解码与显存：完整优化 F 的同条件结果
输入
950DT
960DT
峰值
首步 TPOT / ms
GB / 卡

128k
39.25
18.58
31.90

256k
39.70
18.77
36.12

512k
40.61
19.15
44.56

1M
42.42
19.90
61.44
峰值在两平台相同；1M = 1,048,576 token。
TPOT 为首步间隔，未代表长生成稳态或 P99。
F：SP32、MLA压缩、FP8权重/计算/激活/通信/KV；KDA状态FP32。所有结果使用相同 TP32×EP32、单逻辑SU。
来源：2026-10-19 最新 report / summary / verification。TTFT 不含排队及冷加载；10T设计、硬件路线图和低精度质量未实机验证。

本页亮点
1M、960DT：相较仅启用SP的基线B，完整F的 TTFT 降低49.9%，显存峰值降低49.8%。
无SP的A基线在两平台512k与1M均容量失败；失败场景不计算加速比。

![](Image100.jpg)
4
Huawei Proprietary - Restricted Distribution

### Notes:
【第4页 · 完整优化使1M上下文可完成，960DT首步解码约19.90 ms】

本页采用2026年10月19日19点56分更新的104组实验。94组可完成请求，10组因容量不足中止。图和表选取相同的1024卡、TP32乘EP32、batch为1、top-32、输出两个token的完整优化F，避免把不同负载混在一起。F依次包含SP32、MLA缓存压缩、FP8权重与计算、FP8激活与通信以及FP8 KV，KDA状态保持FP32。随着上下文从128k增加到1M，950DT首token从5.66秒增至89.37秒，960DT从4.37秒增至56.99秒；1M首步解码间隔分别是42.42和19.90毫秒。上下文增加同时影响传输及MLA二次计算项，因此首token时间不能线性外推。两平台的布局和状态容量相同，峰值均从31.90GB增至61.44GB每卡。无SP的FP16展开缓存基线A在两平台512k和1M均无法完成；与仅启用SP的可行基线B比较，1M、960DT的完整优化F将首token时间和峰值约减半。这里全部是权重加载后的模型侧预测，输出两个token只测到一个解码间隔；无排队、冷加载与生产P99保证，也没有实机kernel校准或量化质量验证。

过渡：后续可依据相同负载采集真实kernel与collective数据，校准互联启动时间及HBM服务时间，再评估生产SLO。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• Kimi K3：Kimi K3（模型产品名）——扩容设计采用其公开文本骨干结构。
• 950DT / 960DT：Huawei Ascend 950DT / 960DT（硬件型号）——分别采用资料中的硬件路线图参数及显式效率、拓扑假设。
• TTFT：Time To First Token，首token时间——从模型执行开始到首token完成，不含请求排队、冷加载。
• TPOT：Time Per Output Token，每输出token耗时——本页输出2个token，TPOT为首token后的一个解码间隔。
• TP / EP / SP：Tensor / Expert / Sequence Parallelism，张量/专家/序列并行——TP32与EP32形成1024卡；SP32与TP共享ranks，不额外乘卡数。
• MLA：Multi-head Latent Attention，多头潜在注意力——F使用压缩缓存布局，保存潜在状态而非参考实现展开K/V。
• KDA：Kimi Delta Attention，Kimi增量注意力——递归与卷积状态采用该结构的独立账本。
• FP8 / FP16 / FP32：8 / 16 / 32-bit Floating Point，8/16/32位浮点——F主路径使用FP8，KDA递归状态保持FP32；基线A是FP16设计。
• KV：Key-Value，键值缓存——F将MLA缓存指定为FP8并计入scale。
• SU：Scale-Up，域内高速互联——1024卡映射为单逻辑非阻塞域。
• batch / top-32 / token：Batch / Top-k / Token，批量/前k个专家/词元——本页为单请求、每token选择32个routed专家，固定输出2个token。
• k / M / GB / ms：kilo / million / Gigabyte / millisecond，长度标签/吉字节/毫秒——128k是131072token，1M是1048576token；GB=10^9B，ms=10^-3s。
• P99：99th Percentile，第99百分位——本次固定batch仿真不提供生产尾时延保证。
• A / B / F：Experiment Stages A / B / F，实验方案代号——A为FP16展开缓存无SP；B仅加SP32；F包含全部五项叠加优化。
• report / summary / verification：Report / Summary / Verification，报告/汇总/验证文件——最新实验目录中的追溯证据。
• Huawei Proprietary - Restricted Distribution：华为专有信息，限制分发——按用户指定华为风格模板保留的页脚。

【标红内容详解 · 逐条一句话解释】
• 「94组可完成请求」：verification.json记载104组已完成仿真，其中94组feasible、10组infeasible；完成仿真不代表请求可行。
• 「4.37 / 9.51 / 22.17 / 56.99」：红色条形为同条件F方案960DT的TTFT，来源summary.json，按128k、256k、512k、1M排序；分别为4.368742、9.511830、22.172952、56.994974秒，图中保留两位小数。
• 「19.90」：1M、960DT、F、batch1、top32、output2的TPOT=0.019902162394657807秒，转毫秒并四舍五入。
• 「49.9%」：同条件1M、960DT，B方案TTFT=113.77648384870794秒，F=56.99497440539057秒，按1-F/B计算降低49.9062%。
• 「49.8%」：同条件1M从B峰值122304165278B降至F峰值61435794595B，按1-F/B计算降低49.7680%。

【来源与数据口径】
• C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/report.md
• C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/summary.csv
• C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/verification.json
• C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/configs/960dt-F_kv-b1-k32-o2.json
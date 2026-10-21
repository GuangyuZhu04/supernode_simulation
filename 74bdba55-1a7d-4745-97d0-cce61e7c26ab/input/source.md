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
公开 Kimi K3 快照
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
并行扫描以 EP32 为起点，并向 EP16 / 8 / 4 / 2 扩展
边界：10T 为合成扩容，模型质量未验证；仅文本骨干，不含视觉、投影器和训练 MTP。来源：官方配置快照与2026-10-20并行扫描。

本页亮点
★ 总参数扩大约 3.66 倍，活跃参数约增加 48.2%。
★ 参数总量决定容量压力，激活规模决定主要计算负载，两者分开设计。

![](Image75.jpg)
1
Huawei Proprietary - Restricted Distribution

### Notes:
【第1页 · Kimi K3 扩容专家池至 10.17T，保留混合注意力骨架】

这一页说明我们怎样从公开 Kimi K3 文本骨干构造接近十万亿参数的推理研究对象。首先保留原有的计算骨架：总计九十三层，其中六十九层为 KDA，二十四层为 MLA；第一层使用稠密前馈，其余九十二层为 MoE，并保留 AttnRes 跨层残差聚合。隐藏维度、潜空间维度、专家中间维和两个共享专家均不改变。扩容动作集中在每个 MoE 层的路由专家池，从八百九十六增加到三千三百二十八，同时扩大路由器输出维度。这个数值来自早期 EP128 的整除约束，是近十万亿规模的工程取整，不代表最新实验使用 EP128；最新实验使用 EP32，每个专家并行分片对应一百零四个专家。按同一结构计数口径，总参数从二点七七九万亿增加到十点一七一万亿，约为原来的三点六六倍。主实验把每个 token 选择的专家从十六个增加到三十二个，活跃参数由约一千零四十二亿增加到一千五百四十四亿，增加约百分之四十八点二，只占新模型总量的百分之一点五二。因此，扩容后的参数容量压力和每次推理的计算负载需要分别分析。这里的十万亿模型是合成扩容设计，尚未训练或验证质量；统计范围只有文本骨干，不含视觉编码器、投影器和训练时的多 token 预测层。本次新增并行度扫描仍使用同一个10.171T合成模型与top-32，以EP32为起点，向EP16、EP8、EP4、EP2等候选扩展，并同步调整TP和DP；专家规模及质量假设没有随布局改变。结果在第4至5页比较。

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
• 2026-10-20补充：outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/verification.json及audit/verification.json

<!-- Slide number: 2 -->

10T 推理需同时满足显存峰值、HBM 带宽与低时延互联
02 基线集群要求 · 原并行配置的资源锚点；1M、batch=1、top-32、output=2

核心思想
1024 卡 TP32×EP32，SP32共用TP；PP=DP=1为基线，最新并行扫描见后两页

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
来源：2026-10-19基线报告及配置；2026-10-20并行扫描见第4–5页。硬件、拓扑与效率未实测校准。

本页亮点
★ SP 与压缩、量化共同控制显存峰值
★ decode 优先关注 collective 启动时延与 HBM 服务时间

![](Image96.jpg)
2
Huawei Proprietary - Restricted Distribution

### Notes:
【第2页 · 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联】

这一页把集群条件分成显存、计算与访存、互联三个维度。本页沿用的原并行基线是1024卡，张量并行32乘专家并行32，序列并行与张量并行共用设备，流水线并行和数据并行均为1。这是固定实验配置，并不是已经搜索出的最小卡数。以下工作量都对应1M输入、单请求、top-32和输出2个token，采用完整优化F。首先，FP8逻辑权重约10.17TB，但计入复制和量化scale后，集群物理权重约19.88TB；再叠加缓存、运行时与工作空间，完整运行的每卡峰值为61.44GB，因此不能把逻辑模型大小简单除以1024。其次，两平台每卡HBM容量分别为144和288GB，带宽为4和9.6TB每秒，FP8标称峰值为1和2PFLOP每秒，计算效率统一假设50%。F的prefill计算约1.611EFLOP，而单步decode的集群HBM累计读写约8.34TB，这个数是流量，不是容量。最后，网络采用1024卡单逻辑非阻塞SU，单向带宽1和1.1TB每秒，效率80%，启动与传播延时2.1和1微秒。在这个模型中，decode关键路径里通信启动约占86%和87%，说明优化集体通信的启动次数与时延很重要。所有硬件、拓扑和效率参数都是未实测校准的仿真条件，尤其路线图平台不代表已验证部署；F采用FP8的量化质量也未在这项性能仿真中验证。这些数字专门作为batch=1、TP32/EP32/DP1的资源锚点；并不表示所有最新候选仍使用DP1。后两页在固定1024卡和全局请求数下比较新布局，峰值和最终驻留随副本复制及上下文变化，不能套用本页61.44GB到全部布局。

过渡：资源条件明确后，下一页解释仿真如何把模型、负载与集群配置转换成可核对的时延和资源账本。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• 10T：Ten Trillion Parameters，十万亿参数规模——本方案模型参数量为约10.171万亿，标题按规模简写为10T。
• HBM：High Bandwidth Memory，高带宽存储器——加速卡上的存储资源，页面区分其容量、带宽和累计读写量。
• F：Full Optimization Configuration，完整优化配置（实验标签）——SP32、MLA压缩缓存和FP8权重/计算/激活/通信/KV的组合，KDA状态保持FP32。
• TP32：Tensor Parallelism 32，32路张量并行——把单层张量运算分到32个设备rank。
• EP32：Expert Parallelism 32，32路专家并行——将MoE专家分布在32个专家并行rank，和TP32共同构成1024卡布局。
• SP32：Sequence Parallelism 32，32路序列并行——把序列相关激活分片，复用TP的32个rank，不额外相乘增加卡数。
• TP：Tensor Parallelism，张量并行——本页的SP与TP共用同一组rank。
• PP：Pipeline Parallelism，流水线并行——沿模型层划分流水段；本次基线及全部并行扫描均为1。
• DP：Data Parallelism，数据并行——不同副本处理不同输入；本页基线DP1，后两页扫描更大的DP，SP不额外相乘。
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
• 2026-10-20补充：outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/verification.json及audit/verification.json

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
本次：副本级流式 + DP聚合

校准闭环
调硬件 / 并行 / 特性
候选扫描
实机校准
更新配置

边界：固定batch、无排队/P99；DP按单逻辑非阻塞SU独立副本聚合，并经原生整图核对；未做真机校准。

本页亮点
★ 同一资源语义统一计算、存储与网络
★ 预测带来源和约束，可复现、可校准

![](Image106.jpg)
3
Huawei Proprietary - Restricted Distribution

### Notes:
【第3页 · 仿真 Skill 将配置编译为资源事件图，评估容量与时延】

这一页说明仿真结果是怎样产生的。入口是 supernode-modeling 技能工作流，它把用户问题整理为配置并解释结果，真正的数值计算由 supernode_sim 的 Python 引擎完成。第一步先固定模型结构、精度、负载、并行策略、拓扑和目标，给关键字段记录公开、用户、假设或实测来源，并分别检查配置、图连通性和执行约束；自然语言拓扑要先编译成原生链路图。第二步按组件建立权重、注意力状态、激活和工作区账本，再把依赖关系与计算、访存、通信工作量映射到资源事件。一个事件可以同时使用计算与 HBM 资源，采用屋顶线最大项和最大最小公平共享语义，不能把计算与访存耗时简单相加；内存容量则通过准入与生命周期释放单独约束。训练与推理共享这一资源内核，训练还需要梯度与优化器状态。本次每个DP副本使用原生DP1结构化流式推理。在单逻辑非阻塞SU、无跨副本共享瓶颈的明确条件下，各副本资源独立：时延取副本最大值，流量与计算量累加，空闲副本仍计静态驻留。副本拆分已用1024卡原生整图和完整93层样例核对；没有把流式引擎改为通用DP执行器。第三步输出首词时延、逐词时延、吞吐量、容量失败和瓶颈，再保存解析后的配置、结果与代码指纹以供复跑。反馈环节可以扫描硬件、并行和特性候选，并用实机算子与集合通信数据校准；这里的实机校准是后续工作，本次尚未完成。结果仅对应固定批次，不包含在线请求排队或生产环境第九十九百分位保证，解析下界也不等于可行性证明。

过渡：接下来两页先比较不同并行组合，再展示上下文、batch和资源账本的变化。

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
• DP：Data Parallelism，数据并行——不同副本接收不同请求；本次以经验证的独立副本方式聚合，而非放开流式引擎的DP1限制。

【标红内容详解 · 逐条一句话解释】
• 「可复现的数值计算」：依据项目技能与架构说明，正式结果来自 Python 引擎，并通过解析配置、数值输出和代码指纹支持复跑。
• 「01 输入与严格校验」：配置、拓扑连通、特性兼容与执行可行性是分开的校验关口，输入合法不代表资源和时延一定满足。
• 「02 编译与资源竞争」：将张量、依赖及各类工作量映射到共同资源语义，同时模拟重叠、争用和内存生命周期。
• 「03 输出与解释复现」：技能负责读取引擎生成的指标、失败诊断及来源约束，保留配置和代码指纹以便复跑。
• 「训练/推理共用资源内核」：两类工作负载共享图与执行资源语义，训练另有梯度和优化器状态，不能直接沿用推理容量。
• 「本次：副本级流式 + DP聚合」：当前单逻辑非阻塞SU下，独立DP副本按最大时延和累计工作量聚合，已与原生整图核对；只适用于该资源独立条件。
• 「同一资源语义统一计算、存储与网络」：结构化组件图、拓扑资源映射与共享执行器统一记录计算、访存、通信及内存准入，来源docs/simulation-architecture.md。
• 「预测带来源和约束，可复现、可校准」：严格配置标注参数来源、输出resolved配置与代码指纹，实测kernel profile按工作量匹配，来源仿真Skill与系统架构文档。

【来源与数据口径】
• skills/supernode-modeling/SKILL.md
• docs/simulation-architecture.md
• skills/supernode-modeling/references/inference.md
• 2026-10-20补充：outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/verification.json及audit/verification.json

<!-- Slide number: 4 -->

保留TP32、降低EP，可同时降低时延并提升吞吐
04 仿真结果（1/2） · 2026-10-20 · F加速 · 1024卡/平台 · 1M · 全局bs32 · top32 · output2

核心思想
兼顾布局decode吞吐+27.1% / +34.0%，TTFT同步下降；解码优先有首词时延代价。

同负载比较：六组平台/并行组合

TP/SP,EP,DP
TTFT
TPOT
decode
峰值HBM / GB
平台 / 策略
本地bs
s
ms
tokens/s
容量占比

32/32/1
131.39
43.08
742.9
62.67 / 43.5%
950DT · 原并行
本地 bs 32

32/8/4
128.79
33.88
944.5
112.66 / 78.2%
950DT · 兼顾时延
本地 bs 8

16/16/4
171.88
24.76
1,292.3
112.44 / 78.1%
950DT · 解码优先
本地 bs 8

32/32/1
78.00
20.17
1,586.2
62.67 / 21.8%
960DT · 原并行
本地 bs 32

32/4/8
72.50
15.06
2,125.3
183.46 / 63.7%
960DT · 兼顾时延
本地 bs 4

8/8/16
148.32
7.26
4,407.6
275.56 / 95.7%
960DT · 解码优先
本地 bs 2

6组均整程可行；SP与TP共享ranks，不作独立乘数。

decode吞吐
tokens/s
950DT
原并行

742.9
兼顾时延

944.5
解码优先

1,292.3

0
1500
3000
4500
960DT
原并行

1,586.2
兼顾时延

2,125.3
解码优先

4,407.6

0
1500
3000
4500
两平台横轴统一为0–4500 tokens/s
口径：HBM占比=峰值/144或288 GB；F配置、同全局负载；TPOT为首个decode间隔，非长生成稳态/P99。
来源：2026-10-20并行度扫描 summary.json / recommendations.json；全部为未校准模型预测，非实测。

本页亮点
兼顾方案：decode吞吐+27.1% / +34.0%，TTFT同步下降。
960DT解码优先：TPOT 7.26 ms，代价是首词148.32 s与95.7%的显存占比。

![](Image149.jpg)
4
Huawei Proprietary - Restricted Distribution

### Notes:
【第4页 · 保留TP32、降低EP，可同时降低时延并提升吞吐】

这一页把两个平台都固定为1024卡、1M输入和全局32个请求，用相同F精度与加速特性比较六种并行布局。公平比较看的是全局负载，本地批大小会随数据并行副本数变化；SP复用TP的设备，不额外增加卡数。原并行都采用TP/SP32、EP32、DP1。保留TP32、降低EP并增加DP后，950DT的解码吞吐由742.9升到944.5，增幅27.1%；960DT从1586.2升到2125.3，增幅34.0%。两者首词时延还分别从131.39和78.00秒下降到128.79和72.50秒，因此这组方案兼顾首词和解码。它也有资源代价：每卡峰值显存分别提高到112.66和183.46GB，对应容量占比78.2%和63.7%。继续降低TP能把解码推得更快，960DT的解码优先布局达到4407.6个token每秒、间隔7.26毫秒，但首词等待增至148.32秒，显存占比达95.7%。六组均通过完整请求的容量与执行可行性检查。这里的TPOT和吞吐只取输出2个token时的首个decode间隔，不代表长生成稳态或线上P99；没有排队和冷加载，所有数字仍是未经实机校准的模型预测。

过渡：下一页固定这两组兼顾布局，继续看上下文与批大小变化，并核对扩展数据并行后的集群资源工作量。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• TP：Tensor Parallelism，张量并行——将单层张量计算分配到多个并行设备；标题TP32指32路张量并行。
• SP：Sequence Parallelism，序列并行——对序列相关数据分片，本页与TP共享同一组ranks。
• EP：Expert Parallelism，专家并行——将专家模型分配到多个并行设备；本页通过降低EP增加数据并行副本数。
• DP：Data Parallelism，数据并行——多个模型副本处理不同请求；全局bs等于活跃副本数乘本地bs。
• TP32：Tensor Parallelism 32，32路张量并行——标题强调兼顾方案保留32路张量并行，表中TP/SP一项同时给出TP与SP。
• TTFT：Time to First Token，首词时延——从请求开始到首个输出token的模拟时间，不含排队和冷加载。
• TPOT：Time per Output Token，单输出token时间——本页取首个decode间隔，不是长生成稳态平均间隔。
• decode：Autoregressive Decoding，自回归解码——在上下文状态上生成后续token，本页吞吐来自首个decode间隔。
• HBM：High Bandwidth Memory，高带宽存储器——页面给出完整请求每卡实际峰值，以及该峰值对单卡容量的占比。
• GB：Gigabyte，吉字节——十进制容量单位，1 GB为10^9字节；两平台容量分别144与288GB。
• s：Second，秒——TTFT使用的时间单位。
• ms：Millisecond，毫秒——TPOT使用的时间单位，1 ms为0.001秒。
• tokens/s：Tokens per Second，每秒token数——decode吞吐单位，按全局完成的token数除以首个decode间隔计算。
• tokens：Tokens，模型标记单元——语言模型处理和生成的基本序列单元。
• F：Full Optimization Configuration，完整优化配置（实验标签）——本次沿用SP开启、压缩MLA缓存和FP8权重/计算/激活/通信/KV，KDA递归状态保留FP32；SP规模随表中TP变化。
• 1M：One Mebi-token Context，约百万token上下文——本次1M输入精确为1,048,576个token。
• bs：Batch Size，批大小——全局bs32指全平台同时处理32个请求；本地bs指一个活跃DP副本处理的请求数。
• bs32：Batch Size 32，批大小32——页眉指定同一全局负载，为全部六组比较共同使用的32个请求。
• top32：Top-32 Expert Selection，前32专家选择——沿用本次实验每个token选择32个专家的路由配置。
• output2：Output Length 2，输出长度2——每个请求生成2个token，因此只观察到首个decode间隔。
• 950DT：950DT（仿真平台名称）——使用报告冻结的950DT硬件规格与效率假设，未做实机校准。
• 960DT：960DT（仿真平台名称）——使用报告冻结的960DT硬件规格与效率假设，未做实机校准。
• ranks：Process Ranks，并行进程编号——标识参与并行操作的设备或进程；SP与TP共享rank，不作为独立乘数。
• P99：99th Percentile，99百分位数——线上尾时延统计；本页固定batch、无排队，未估算此项。
• summary.json：Summary in JavaScript Object Notation，JSON格式仿真汇总文件——保存并行度扫描的结构化结果，是本页六组数据的源文件。
• recommendations.json：Recommendations in JavaScript Object Notation，JSON格式建议文件——记录并行度扫描中的推荐布局，供方案选择与数据核对。
• Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有信息，限制传播——沿用原稿保密页脚声明。

【标红内容详解 · 逐条一句话解释】
• 「+27.1% / +34.0%」：2026-10-20 summary.json中950dt-1M-tp32-ep8-dp4-sp32-g32与960dt-1M-tp32-ep4-dp8-sp32-g32相对于各平台原并行950dt-1M-tp32-ep32-dp1-sp32-g32与960dt-1M-tp32-ep32-dp1-sp32-g32，首个decode间隔吞吐分别提高27.1%和34.0%，比较均固定1024卡及全局bs32。
• 「TTFT同步下降」：同一summary.json中兼顾方案的TTFT由950DT原并行131.39秒降至128.79秒、960DT原并行78.00秒降至72.50秒，说明本次吞吐改善没有牺牲首词时延。
• 「同负载比较：六组平台/并行组合」：六行均取2026-10-20 summary.json所列1M、全局bs32、1024卡的完整请求可行点，本地bs随DP变化而不是扩大总请求数。
• 「decode吞吐」：表格与横条图使用summary.json的decode_tokens_s字段，表示首个decode间隔的全平台输出速率；两平台图共用0到4500 tokens/s范围。
• 「128.79」：2026-10-20 summary.json的950dt-1M-tp32-ep8-dp4-sp32-g32预测为128.79秒，对比950dt-1M-tp32-ep32-dp1-sp32-g32的131.39秒，表示首词时延降低；两点均为1M、全局bs32。
• 「33.88」：2026-10-20 summary.json的950dt-1M-tp32-ep8-dp4-sp32-g32预测为33.88毫秒，对比950dt-1M-tp32-ep32-dp1-sp32-g32的43.08毫秒，表示首个decode间隔缩短；两点均为1M、全局bs32。
• 「944.5」：2026-10-20 summary.json的950dt-1M-tp32-ep8-dp4-sp32-g32预测为944.5tokens/s，对比950dt-1M-tp32-ep32-dp1-sp32-g32的742.9tokens/s，表示全平台解码吞吐增加；两点均为1M、全局bs32。
• 「72.50」：2026-10-20 summary.json的960dt-1M-tp32-ep4-dp8-sp32-g32预测为72.50秒，对比960dt-1M-tp32-ep32-dp1-sp32-g32的78.00秒，表示首词时延降低；两点均为1M、全局bs32。
• 「15.06」：2026-10-20 summary.json的960dt-1M-tp32-ep4-dp8-sp32-g32预测为15.06毫秒，对比960dt-1M-tp32-ep32-dp1-sp32-g32的20.17毫秒，表示首个decode间隔缩短；两点均为1M、全局bs32。
• 「2,125.3」：2026-10-20 summary.json的960dt-1M-tp32-ep4-dp8-sp32-g32预测为2,125.3tokens/s，对比960dt-1M-tp32-ep32-dp1-sp32-g32的1,586.2tokens/s，表示全平台解码吞吐增加；两点均为1M、全局bs32。
• 「深红色兼顾时延系列」：深红条对应summary.json的950dt-1M-tp32-ep8-dp4-sp32-g32与960dt-1M-tp32-ep4-dp8-sp32-g32，吞吐944.5和2,125.3 tokens/s；各自灰色原并行是742.9和1,586.2 tokens/s。
• 「浅红色解码优先系列」：浅红条对应summary.json的950dt-1M-tp16-ep16-dp4-sp16-g32与960dt-1M-tp8-ep8-dp16-sp8-g32，吞吐1,292.3和4,407.6 tokens/s，均高于各自原并行，但TTFT增为171.88和148.32秒。
• 「7.26 ms」：2026-10-20 summary.json中960dt-1M-tp8-ep8-dp16-sp8-g32的首个decode间隔为7.26毫秒，是本页六组中最低值，相比960DT原并行20.17毫秒明显缩短。
• 「148.32 s与95.7%」：同一960dt-1M-tp8-ep8-dp16-sp8-g32的TTFT为148.32秒、每卡峰值275.56GB占288GB的95.7%，相对960DT原并行78.00秒和21.8%体现追求最低TPOT的首词及容量代价。

【来源与数据口径】
• 2026-10-20 · C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/summary.json
• 2026-10-20 · C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/recommendations.json
• input/selected-data.json：selected六行与boundaries；所有数字为未经实机校准的F配置模型预测。
• 2026-10-20 selected case：950dt-1M-tp32-ep32-dp1-sp32-g32；replica case：950dt-1M-tp32-ep32-dp1-sp32-lb32
• 2026-10-20 selected case：950dt-1M-tp32-ep8-dp4-sp32-g32；replica case：950dt-1M-tp32-ep8-dp4-sp32-lb8
• 2026-10-20 selected case：950dt-1M-tp16-ep16-dp4-sp16-g32；replica case：950dt-1M-tp16-ep16-dp4-sp16-lb8
• 2026-10-20 selected case：960dt-1M-tp32-ep32-dp1-sp32-g32；replica case：960dt-1M-tp32-ep32-dp1-sp32-lb32
• 2026-10-20 selected case：960dt-1M-tp32-ep4-dp8-sp32-g32；replica case：960dt-1M-tp32-ep4-dp8-sp32-lb4
• 2026-10-20 selected case：960dt-1M-tp8-ep8-dp16-sp8-g32；replica case：960dt-1M-tp8-ep8-dp16-sp8-lb2

<!-- Slide number: 5 -->

增大batch可近线性提升吞吐，扩容DP仍需核对资源代价
05 仿真结果（2/2） · 2026-10-20 · 固定第4页兼顾布局，比较四种上下文与六档全局bs；F配置、每平台1024卡

核心思想
全局bs1→32：固定布局吞吐31.4–31.5倍，TPOT仅增1.6%–2.0%；通信流量与驻留容量需一起核算。

固定兼顾布局 · 四种上下文（全局bs32）
平台
上下文
TTFT(s)
TPOT(ms)
吞吐(tok/s)
峰值HBM(GB)
950DT
128k
10.58
30.71
1,042.0
65.04

950DT
256k
22.72
31.16
1,026.9
71.84
950DT
512k
51.74
32.07
997.9
85.45

950DT
1M
128.79
33.88
944.5
112.66

960DT
128k
6.30
13.74
2,329.7
109.53
960DT
256k
13.38
13.92
2,298.2
120.09
960DT
512k
29.92
14.30
2,237.5
141.22

960DT
1M
72.50
15.06
2,125.3
183.46

固定1M · 六档batch吞吐
950DT
960DT

tokens/s
2,125.3

2,000

1,074.0

1,000

539.9

944.5
270.0

135.0

67.5
477.5

240.1

0

120.4

60.2
30.1
1
2
4
8
16
32
全局batch（log2档位）
TPOT（bs1→32）：950DT 33.22→33.88 ms；960DT 14.82→15.06 ms

资源账本
prefill计算
decode计算
prefill SU
decode SU
decode HBM
全系统最终驻留
1M、全局bs32
EFLOP
TFLOP/步
PB
GB/步
TB/步
TB

两平台 · 原并行
51.54
74.01
24.73
106.03
27.33
43.67
950DT · 兼顾时延
51.54
74.01
22.61
104.01
27.66
74.95

950DT · 解码优先
43.65
66.46
11.97
30.72
20.23
68.98
960DT · 兼顾时延
51.54
74.01
19.78
101.31
27.71
116.67

960DT · 解码优先
39.71
62.69
5.58
9.53
17.24
193.66
工作量≠利用率；驻留=decode完成态，非峰值之和；GB/TB/PB十进制。来源：2026-10-20 summary.json；未校准预测，TPOT为一次decode间隔。

本页亮点
bs1→32：固定兼顾布局吞吐31.4–31.5倍，TPOT增幅1.6%–2.0%。
全量1008组合：534整程可行；66仅decode可行；408静态容量不足。

![](Image206.jpg)
5
Huawei Proprietary - Restricted Distribution

### Notes:
【第5页 · 增大batch可近线性提升吞吐，扩容DP仍需核对资源代价】

这一页比较上下文长度、全局batch和资源开销，所有结果都是2026年10月20日的F配置模型预测，每个平台固定1024卡。左表和右图沿用上一页在1M、全局bs32下选出的兼顾布局：950DT采用TP/SP32、EP8、DP4，960DT采用TP/SP32、EP4、DP8；没有对每个上下文或batch重新寻找最优。左表显示，上下文从128k增长到1M后，首词时延和峰值HBM上升，解码吞吐略降。右图固定1M，把全局bs从1提高到32，950DT吞吐提升31.3766倍，TPOT增长1.9869%；960DT对应31.4893倍和1.6218%，因此概括为吞吐31.4至31.5倍、TPOT增长1.6%至2.0%。横轴按log2档位等距，纵轴吞吐是线性刻度。负载口径是active_DP取配置DP与全局bs的较小值，本地bs为全局bs除以active_DP，吞吐按全局bs除以TPOT计算，不再额外乘DP。下表把计算工作量、通信流量、HBM读写和最终驻留容量分开核算；原并行的两平台资源值一致，所以合并一行。降低通信负载不等于降低驻留，扩容DP会复制更多权重。最终驻留是解码完成后所有1024卡的容量加和，包含空闲DP的静态权重，不是各卡峰值的加和，也不是利用率。全量1008个部署组合中，534组整程可行、66组仅decode可行、408组静态容量不足；仅解码可行不能当作端到端成功。TPOT只代表一次首个decode间隔，未做真机校准，也没有P99结论。

过渡：以上两页给出了固定1024卡、相同全局负载下的可行选择及资源代价；后续应按实际上下文、并发和首词时延要求筛选布局，再用真机数据校准，而不能将这组预测直接当作交付性能。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
• batch：Batch，批次——同一轮服务的请求批次；本页横轴改变的是全局请求批大小。
• bs：Batch Size，批大小——全局bs是系统请求数；本地bs随活跃DP数分摊，吞吐不重复乘DP。
• DP：Data Parallelism，数据并行——独立副本并行数；活跃副本数取配置DP与全局bs中的较小值。
• 950DT：950DT（平台配置名，源数据未提供英文展开）——本次仿真中的950dt硬件配置，固定兼顾布局为TP/SP32、EP8、DP4。
• 960DT：960DT（平台配置名，源数据未提供英文展开）——本次仿真中的960dt硬件配置，固定兼顾布局为TP/SP32、EP4、DP8。
• F：F precision configuration，F精度配置（实验定义）——实验采用SP开启、压缩MLA、权重/计算/激活/通信/KV为FP8、KDA递归状态为FP32。
• TTFT：Time To First Token，首个输出词元时延——从请求计算开始到首个输出词元的预测时延，排除排队及加载时间。
• TPOT：Time Per Output Token，每个输出词元耗时——本实验output2条件下的一次首个decode间隔；不是尾时延或P99。
• HBM：High Bandwidth Memory，高带宽存储器——左表给出单卡整程真实峰值容量；资源表给出全系统解码阶段累计读写量。
• tok/s：Tokens per Second，每秒词元数——本页为全系统decode吞吐，等于全局bs除以TPOT秒数。
• tokens/s：Tokens per Second，每秒词元数——折线图纵轴单位，与左表tok/s同义。
• s：Second，秒——左表TTFT时延单位。
• ms：Millisecond，毫秒——TPOT时延单位，1000毫秒等于1秒。
• k：Kilo-context notation，千级上下文标记——本实验128k、256k、512k分别表示131072、262144、524288输入词元。
• M：Mega-context notation，百万级上下文标记——本实验1M表示1048576输入词元；该上下文记法不改变容量单位的十进制口径。
• log2：Base-two Logarithm，以2为底的对数——batch轴按1、2、4、8、16、32六个等距档位显示，不是batch线性距离。
• prefill：Prefill，预填充阶段——处理完整输入上下文，资源表给出全系统该阶段累计计算量与SU物理通信量。
• decode：Decode，解码阶段——自回归生成阶段，本页流量和计算量按一次decode步累计。
• EFLOP：Exa Floating-point Operations，百亿亿次浮点运算——10的18次方次浮点运算；这里是累计计算工作量，不是每秒计算性能。
• TFLOP：Tera Floating-point Operations，万亿次浮点运算——10的12次方次浮点运算；表内按每次decode步计量。
• SU：Scale Up，节点内扩展互联——本实验只有一个逻辑无阻塞SU，表内为全系统SU物理通信流量，SO流量为0。
• GB：Gigabyte，吉字节——十进制10的9次方字节；左表为峰值容量，资源表decode SU为每步通信量。
• TB：Terabyte，太字节——十进制10的12次方字节；HBM列是每步累计读写量，驻留列是最终存量。
• PB：Petabyte，拍字节——十进制10的15次方字节；资源表为prefill阶段累计SU通信量。
• summary.json：Summary JSON file，仿真汇总文件——本页数据引用2026-10-20并行度扫描的原生summary.json，经selected-data.json整理。
• Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有信息、限制分发——沿用原稿的保密分发声明。
• Huawei：Huawei（公司名），华为——页面页脚中沿用的品牌与专有信息声明主体。
• TP：Tensor Parallelism，张量并行——在多个计算卡之间切分模型张量。
• SP：Sequence Parallelism，序列并行——本实验SP与TP共享同一组卡，兼顾布局固定SP32。
• EP：Expert Parallelism，专家并行——将专家分配到不同卡组；本页固定沿用第4页兼顾布局。

【标红内容详解 · 逐条一句话解释】
• 「31.4–31.5倍」：来源为2026-10-20 summary.json及selected-data.json.batch_growth；固定1M和第4页兼顾布局，以全局bs1为基准到bs32，950DT为31.376582881823552倍、960DT为31.489308336783225倍，分别显示为31.4倍和31.5倍，不是逐负载重选最优。
• 「1.6%–2.0%」：同一batch_growth、同一固定布局及bs1基准：950DT的TPOT由33.220237439249836增长到33.88028588262293毫秒，增幅1.9868865915848133%；960DT由14.81636597332383增长到15.056656884156775毫秒，增幅1.6217938411185129%；范围取一位小数，不能把两者都写成1.6%。
• 「534整程可行」：2026-10-20并行度扫描summary.json的验证汇总：1008部署组合中534组prefill与decode都可行，才属于整程成功。
• 「66仅decode可行」：同一1008组合验证汇总中有600组decode可行而仅534组prefill可行，差额66组仅解码可行，不能计入端到端成功；另408组静态容量不足。
• 「960DT（红色系列）」：红线与红点对应固定TP/SP32、EP4、DP8的960DT兼顾布局；横轴为全局bs1、2、4、8、16、32的log2档位，纵轴为原生模型预测decode tokens/s；灰线为950DT固定TP/SP32、EP8、DP4。
• 「固定兼顾布局 · 四种上下文（全局bs32）」：左表为selected-data.json.contexts的8个可行case；全局bs固定32，沿用1M/global_bs32选定的两平台兼顾布局，不对各上下文重新优化。
• 「固定1M · 六档batch吞吐」：右图为selected-data.json.batches的12个可行case；上下文固定1M、每平台1024卡，以相同全局batch口径比较两平台。
• 「资源账本」：下表为selected-data.json.resources的5行，全系统工作量和最终驻留分列；两平台原并行各项资源值一致而合并，时延和吞吐未合并。
• 「67.5」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g1，固定兼顾布局下全局bs=1的全系统decode吞吐为67.492933273952 tokens/s；红色数值仅标识960DT系列，不代表实测。
• 「135.0」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g2，固定兼顾布局下全局bs=2的全系统decode吞吐为134.985866547904 tokens/s；红色数值仅标识960DT系列，不代表实测。
• 「270.0」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g4，固定兼顾布局下全局bs=4的全系统decode吞吐为269.971733095809 tokens/s；红色数值仅标识960DT系列，不代表实测。
• 「539.9」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g8，固定兼顾布局下全局bs=8的全系统decode吞吐为539.943466191617 tokens/s；红色数值仅标识960DT系列，不代表实测。
• 「1,074.0」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g16，固定兼顾布局下全局bs=16的全系统decode吞吐为1074.024618926326 tokens/s；红色数值仅标识960DT系列，不代表实测。
• 「2,125.3」：来源为2026-10-20 summary.json案例 960dt-1M-tp32-ep4-dp8-sp32-g32，固定兼顾布局下全局bs=32的全系统decode吞吐为2125.305786417415 tokens/s；红色数值仅标识960DT系列，不代表实测。

【来源与数据口径】
• 2026-10-20；原始仿真汇总：C:/D_disk/Github/supernode_simulation/outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/summary.json；页面选数：input/selected-data.json。
• 上下文表（contexts）：950dt-128k-tp32-ep8-dp4-sp32-g32；950dt-256k-tp32-ep8-dp4-sp32-g32；950dt-512k-tp32-ep8-dp4-sp32-g32；950dt-1M-tp32-ep8-dp4-sp32-g32；960dt-128k-tp32-ep4-dp8-sp32-g32；960dt-256k-tp32-ep4-dp8-sp32-g32；960dt-512k-tp32-ep4-dp8-sp32-g32；960dt-1M-tp32-ep4-dp8-sp32-g32。
• batch折线图（batches）：950dt-1M-tp32-ep8-dp4-sp32-g1；950dt-1M-tp32-ep8-dp4-sp32-g2；950dt-1M-tp32-ep8-dp4-sp32-g4；950dt-1M-tp32-ep8-dp4-sp32-g8；950dt-1M-tp32-ep8-dp4-sp32-g16；950dt-1M-tp32-ep8-dp4-sp32-g32；960dt-1M-tp32-ep4-dp8-sp32-g1；960dt-1M-tp32-ep4-dp8-sp32-g2；960dt-1M-tp32-ep4-dp8-sp32-g4；960dt-1M-tp32-ep4-dp8-sp32-g8；960dt-1M-tp32-ep4-dp8-sp32-g16；960dt-1M-tp32-ep4-dp8-sp32-g32。
• 资源账本（resources）：950dt-1M-tp32-ep32-dp1-sp32-g32；950dt-1M-tp32-ep8-dp4-sp32-g32；950dt-1M-tp16-ep16-dp4-sp16-g32；960dt-1M-tp32-ep4-dp8-sp32-g32；960dt-1M-tp8-ep8-dp16-sp8-g32；共用原并行另核对960dt-1M-tp32-ep32-dp1-sp32-g32，各列全系统资源数值相等。
• 增长率：selected-data.json.batch_growth；950DT吞吐31.376582881823552倍、TPOT增幅1.9868865915848133%；960DT吞吐31.489308336783225倍、TPOT增幅1.6217938411185129%。
• 全量可行性：selected-data.json.verification，deployment_points=1008，feasible=534，infeasible=66（decode可行、prefill不可行），static_capacity_rejected=408。
• 边界：F配置、1024卡/平台、单逻辑SU、PP1、SP共享TP；TP×EP×DP=1024；全局bs=active_DP×本地bs，active_DP=min(DP,全局bs)；output2仅有一次首个decode间隔；未校准模型预测，无P99。
• 资源定义：计算与流量为全系统累计工作量，非算力/带宽利用率；GB/TB/PB按十进制；最终驻留为decode完成后所有1024卡加和，包含空闲DP的静态权重，不是单卡峰值的加和。
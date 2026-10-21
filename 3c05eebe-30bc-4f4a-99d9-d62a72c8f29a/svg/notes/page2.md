【第2页 · 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联】

这一页把集群条件分成显存、计算与访存、互联三个维度。本页沿用的原并行基线是1024卡，张量并行32乘专家并行32，序列并行与张量并行共用设备，流水线并行和数据并行均为1。这是固定实验配置，并不是已经搜索出的最小卡数。以下工作量都对应1M输入、单请求、top-32和输出2个token，采用完整优化F。首先，FP8逻辑权重约10.17TB，但计入复制和量化scale后，集群物理权重约19.88TB；再叠加缓存、运行时与工作空间，完整运行的每卡峰值为61.44GB，因此不能把逻辑模型大小简单除以1024。其次，两平台每卡HBM容量分别为144和288GB，带宽为4和9.6TB每秒，FP8标称峰值为1和2PFLOP每秒，计算效率统一假设50%。F的prefill计算约1.611EFLOP，而单步decode的集群HBM累计读写约8.34TB，这个数是流量，不是容量。最后，网络采用1024卡单逻辑非阻塞SU，单向带宽1和1.1TB每秒，效率80%，启动与传播延时2.1和1微秒。在这个模型中，decode关键路径里通信启动约占86%和87%，说明优化集体通信的启动次数与时延很重要。所有硬件、拓扑和效率参数都是未实测校准的仿真条件，尤其路线图平台不代表已验证部署；F采用FP8的量化质量也未在这项性能仿真中验证。这些数字专门作为batch=1、TP32/EP32/DP1的资源锚点；并不表示所有最新候选仍使用DP1。后两页在固定1024卡和全局请求数下比较新布局，峰值和最终驻留随副本复制及上下文变化，不能套用本页61.44GB到全部布局。

过渡：资源条件明确后，下一页解释仿真如何把模型、负载与集群配置转换成可核对的时延和资源账本。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- 10T：Ten Trillion Parameters，十万亿参数规模——本方案模型参数量为约10.171万亿，标题按规模简写为10T。
- HBM：High Bandwidth Memory，高带宽存储器——加速卡上的存储资源，页面区分其容量、带宽和累计读写量。
- F：Full Optimization Configuration，完整优化配置（实验标签）——SP32、MLA压缩缓存和FP8权重/计算/激活/通信/KV的组合，KDA状态保持FP32。
- TP32：Tensor Parallelism 32，32路张量并行——把单层张量运算分到32个设备rank。
- EP32：Expert Parallelism 32，32路专家并行——将MoE专家分布在32个专家并行rank，和TP32共同构成1024卡布局。
- SP32：Sequence Parallelism 32，32路序列并行——把序列相关激活分片，复用TP的32个rank，不额外相乘增加卡数。
- TP：Tensor Parallelism，张量并行——本页的SP与TP共用同一组rank。
- PP：Pipeline Parallelism，流水线并行——沿模型层划分流水段；本次基线及全部并行扫描均为1。
- DP：Data Parallelism，数据并行——不同副本处理不同输入；本页基线DP1，后两页扫描更大的DP，SP不额外相乘。
- ranks：Process Ranks，并行进程编号——标识参与并行操作的设备或进程；此处SP与TP共享rank。
- FP8：8-bit Floating Point，8位浮点格式——F用于权重、计算、激活、通信及KV缓存的精度假设。
- FP32：32-bit Floating Point，32位浮点格式——F中的KDA递归状态继续采用此格式。
- scale：Quantization Scale，量化缩放因子——物理权重总量含低精度量化所需的缩放元数据。
- runtime：Runtime Reservation，运行时预留——本次每卡额外预留8GB运行时空间。
- workspace：Workspace，算子工作空间——本次每卡工作空间预算为256MB。
- 950DT / 960DT：950DT / 960DT（仿真平台名称）——采用报告冻结的硬件路线图规格和效率假设，960DT相关参数尚未实机校准。
- GB / TB / MB：Gigabyte / Terabyte / Megabyte，吉字节/太字节/兆字节——本报告按十进制字节量显示，分别为10^9、10^12和10^6字节。
- TB/s：Terabytes per Second，每秒太字节——页面的HBM与互联带宽单位，互联为单方向带宽。
- PFLOP/s：Peta Floating-point Operations per Second，每秒千万亿次浮点运算——用于每卡FP8标称算力峰值，模拟有效计算效率为50%。
- EFLOP：Exa Floating-point Operations，百亿亿次浮点运算——总计算工作量单位，1 EFLOP为10^18次浮点运算。
- decode：Autoregressive Decoding，自回归解码——在已有上下文状态上生成后续token，页面给出其单步资源需求。
- prefill：Prompt Prefill，提示词预填充——处理全部输入token并建立后续解码所需状态的阶段。
- SU：Scale-Up，域内扩展互联——此处假设1024卡处在一个逻辑非阻塞互联域，逻辑边不代表单根物理线。
- SO：Scale-Out，域间扩展互联——本次单逻辑SU模型中不产生SO流量，故SO=0。
- μs：Microsecond，微秒——启动与传播延时的单位，1微秒等于10^-6秒。
- MLA compressed：Multi-head Latent Attention with Compressed Cache，多头潜在注意力压缩缓存——F用压缩的潜在向量缓存降低MLA缓存容量和访存。
- KV：Key and Value，键与值——注意力机制的缓存数据，F中的相关缓存采用FP8假设。
- KDA：Kimi Delta Attention，Kimi Delta注意力——模型的递归注意力组件，其状态保持FP32。
- 1M：One Megatoken，一个百万级token输入档位——本报告严格取1,048,576个输入token。
- token：Token，词元——模型读写文本的基本离散单元。
- batch：Batch Size，批大小——此页的batch=1表示一个请求同时参与推理。
- top-32：Top-32 Expert Routing，前32专家路由——每个token选用32个路由专家，不代表改变top-k后质量等价。
- output：Output Token Count，输出词元数量——此页采用output=2，对应完整prefill与后续首个decode间隔。
- collective：Collective Communication，集体通信——并行rank间的广播、规约和全互换等通信操作，启动开销影响decode关键路径。
- report.md / summary.csv / phase_metrics.csv / latency_breakdown.csv：Report / Summary / Phase Metrics / Latency Breakdown，报告/汇总/阶段指标/时延拆解文件——对应2026年10月19日最新仿真结果与阶段资源、关键路径归因数据。
- Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有—限制传播——所用华为风格模板的固定页脚声明。

【标红内容详解 · 逐条一句话解释】
- 「1024 卡 TP32×EP32」：来自冻结并行配置；32路张量并行乘32路专家并行组成1024卡，SP32复用TP ranks，未搜索更小卡数。
- 「61.44」：最新summary.csv中1M、batch=1、top-32、output=2、F两平台完整运行峰值均为61.436GB/卡，页面取两位小数。
- 「8.34」：最新phase_metrics.csv对应F单步decode的集群累计HBM读写8.338TB，页面取两位小数；该值是流量而非容量。
- 「86% / 87%」：由latency_breakdown.csv中950DT的36.3258ms/42.4228ms及960DT的17.2980ms/19.9022ms得到，分别约为86%与87%的decode通信启动占比。
- 「SP 与压缩、量化共同控制显存峰值」：1M、batch1、top32下B→C→D→E→F峰值122.304→102.977→85.346→75.478→61.436GB/卡，来自最新summary.csv。
- 「decode 优先关注 collective 启动时延与 HBM 服务时间」：F、1M、batch1的关键路径中，950DT通信启动36.3258ms和HBM服务5.9676ms，960DT分别17.2980ms和2.4865ms，构成主要decode耗时。

【来源与数据口径】
- input/report.md（更新时间2026-10-19T19:56:52+08:00）
- summary.csv：F、1M、batch1、top32、output2，单卡峰值61.436GB
- phase_metrics.csv：逻辑权重10.171TB、物理权重含scale19.877TB，decode HBM读写8.338TB/步
- latency_breakdown.csv：通信启动与阶段时延
- 冻结硬件与并行配置：TP32×EP32×PP1×DP1，1024卡逻辑SU，SP复用TP ranks
- 2026-10-20补充：outputs/kimi-k3-10t-1m-acceleration-20261019/parallelism-20261020/verification.json及audit/verification.json

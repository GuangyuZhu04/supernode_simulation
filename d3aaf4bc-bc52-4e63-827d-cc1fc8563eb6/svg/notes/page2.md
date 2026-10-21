【第2页 · 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联】

这一页保留显存、计算与访存、互联三个资源维度。统一部署为1024卡，TP32乘EP32，SP32复用TP ranks，PP和DP均为1；固定1M上下文、top-32、output=2、完整组合F。本次显存表采用2026年10月20日最新并行扫描中的原布局切片，并与同日独立decode batch扫描逐点交叉核对；prefill和decode各展示batch为1、2、4、8、16、32的结果。batch表示整个DP副本并发处理的序列数，不是每卡batch。950DT与960DT的单卡阶段峰值完全相同：prefill在batch 1至16为61.435794595GB，batch32为62.673769635GB，增加约2.015%；decode六档均为42.642575567GB。表中均按两位小数展示。所有这12个完整请求点均完成prefill与decode，未使用失败前的峰值推断完整请求容量。两个阶段的batch各自变化，本表不表示测试了prefill batch与decode batch不同的分离部署。单卡峰值近似稳定是EP32请求分布的结果：batch不超过32时，每个活跃EP lane最多持有一条序列，新增请求主要启用更多lane。prefill在batch32仍存在额外批量工作集峰值，因此不能把所有显存简单按batch线性外推，也不能泛化到batch大于32或其他并行布局。集群输出末端常驻量随batch逐档为28.810、29.289、30.248、32.164、35.998、43.665TB，从batch1到32增长约51.563%。这是一致时刻的常驻量加和，不是单卡峰值乘1024，也不是各卡独立峰值之和。原页静态预算仍适用：FP8逻辑权重10.17TB，物理权重含复制与scale为19.88TB，每卡runtime为8GB，workspace为256MB，均已计入相应容量账本。计算与互联两栏继续使用batch1基线：每卡HBM容量144/288GB、带宽4/9.6TB每秒，FP8标称峰值1/2PFLOP每秒，计算效率50%；prefill计算1.611EFLOP，decode集群累计HBM读写8.34TB每步。互联为单逻辑非阻塞SU，SO为0，单向带宽1/1.1TB每秒，效率80%，启动及传播时延2.1/1微秒，decode关键路径通信启动占比约86%/87%。这些流量和时延占比不能套用到更大batch。GB和TB均为十进制单位。所有数据是未实测校准的模型预测，FP8质量也未验证。第4至5页继续讨论其他并行布局，不能将本页峰值直接套用到所有候选。

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
- batch / batch size：Batch Size，批大小——本页为整个DP副本的并发序列数；DP1时等于1024卡模型实例并发，显存扫描1/2/4/8/16/32，其他资源用batch1。
- top-32：Top-32 Expert Routing，前32专家路由——每个token选用32个路由专家，不代表改变top-k后质量等价。
- output：Output Token Count，输出词元数量——此页采用output=2，对应完整prefill与后续首个decode间隔。
- collective：Collective Communication，集体通信——并行rank间的广播、规约和全互换等通信操作，启动开销影响decode关键路径。
- report.md / summary.csv / phase_metrics.csv / latency_breakdown.csv：Report / Summary / Phase Metrics / Latency Breakdown，报告/汇总/阶段指标/时延拆解文件——对应2026年10月19日最新仿真结果与阶段资源、关键路径归因数据。
- Huawei Proprietary - Restricted Distribution：Huawei Proprietary - Restricted Distribution，华为专有—限制传播——所用华为风格模板的固定页脚声明。
- EP lane：Expert Parallel lane，专家并行请求通道——请求按本地序号模EP分布；EP32且batch≤32时，每个活跃lane最多持有一条序列，lane内由TP32共同处理。

【标红内容详解 · 逐条一句话解释】
- 「1024 卡 TP32×EP32」：来自冻结并行配置；32路张量并行乘32路专家并行组成1024卡，SP32复用TP ranks，未搜索更小卡数。
- 「8.34」：最新phase_metrics.csv对应F单步decode的集群累计HBM读写8.338TB，页面取两位小数；该值是流量而非容量。
- 「86% / 87%」：由latency_breakdown.csv中950DT的36.3258ms/42.4228ms及960DT的17.2980ms/19.9022ms得到，分别约为86%与87%的decode通信启动占比。
- 「decode 优先关注 collective 启动时延与 HBM 服务时间」：F、1M、batch1的关键路径中，950DT通信启动36.3258ms和HBM服务5.9676ms，960DT分别17.2980ms和2.4865ms，构成主要decode耗时。
- 「62.67」：2026-10-20并行扫描，1M、F、TP32/EP32/SP32/DP1、batch32，两平台prefill单卡峰值62.673769635GB，较batch1的61.435794595GB增长约2.015%。
- 「28.81 → 43.67」：同一配置下batch1到32的输出末端集群常驻量由28.810017063936TB增至43.665410067456TB，约增51.563%；不是各卡峰值之和。
- 「batch≤EP32：每个活跃 EP lane 最多 1 条序列，单卡峰值不按 batch 倍增」：本轮请求按序号模EP映射至lane；batch从1到32只增加活跃lane，单lane最大序列数保持1，不能据此推断其他布局或更大batch。

【来源与数据口径】
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\parallelism-20261020\summary.csv；筛选1M、TP32/EP32/DP1/SP32，batch1/2/4/8/16/32，两平台12点；prefill_peak_hbm_GB、decode_peak_hbm_GB、resident_cluster_TB。
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\decode-batch-20261020\summary.csv；筛选1M、F_kv及相同batch；decode_peak_hbm_bytes_max_per_card / 1e9、resident_hbm_bytes_all_cards_at_end / 1e12。
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\parallelism-20261020\README.txt；batch、EP请求分布与阶段计时定义。
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\report.md；模型与硬件条件，物理权重含scale19.877TB，runtime8GB、workspace256MB。
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\phase_metrics.csv；F、1M、batch1的prefill计算1.611EFLOP、decode HBM累计读写8.338TB/步。
- C:\D_disk\Github\supernode_simulation\outputs\kimi-k3-10t-1m-acceleration-20261019\latency_breakdown.csv；F、1M、batch1的decode通信启动占比。

【六档显存数据】
| batch size | Prefill 单卡峰值 GB | Decode 单卡峰值 GB | 输出末端集群常驻 TB |
|---|---|---|---|
| 1 | 61.44 | 42.64 | 28.81 |
| 2 | 61.44 | 42.64 | 29.29 |
| 4 | 61.44 | 42.64 | 30.25 |
| 8 | 61.44 | 42.64 | 32.16 |
| 16 | 61.44 | 42.64 | 36.00 |
| 32 | 62.67 | 42.64 | 43.67 |

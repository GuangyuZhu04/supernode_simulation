【第2页 · CED、CSA2 与 Engram 改写资源路径，阶段成本逐项计入】

这一页解释三个机制如何进入仿真，而不是用总参数或活跃参数乘上一个经验系数。左侧CED首先让全量提示词只经过32层编码器，然后将编码器最终状态投影成解码器的全局键值缓存。它仍要让末尾128个token完整重放32层解码器，以建立各层滑动窗口状态并产生首token。因此首token时间包含这段有界重放、索引和输出头，不能把旧模型的首token时延直接除以二。启用DSpark时，初始预填充还包含草稿网络末窗口的种子状态，具体成本在后页分开解释。

中间CSA2只有4个全局KV持有层和8个索引持有层，按Full、Reindex、Reuse三种模式生成或复用状态。这些位置由公开Flash层位映射到合成结构，缓存共享质量没有实测。初级索引面对全历史打分，每个分块都先跨张量并行分片进行FP32全归约，再执行top-512选择，不能让各分片各自选top-k后假装等价。后续解码器索引限定在16,384个候选位置。这里提前聚集候选后再打分是本地仿真的优化执行假设；锁定的公开参考实现采用全量打分后屏蔽。因此PPT把这种策略直接标为假设，不将其包装成已验证实现性能。

右侧Engram的ngram表采用FP8值和每32元素一字节scale，每个数据并行副本共3.5473TB。表按行分片放到该副本所属主机，而不是每主机重复一整份。主场景是冷查表，没有缓存命中或免费预取重叠。请求从原主机哈希出地址，经过主机到设备、域内互联与设备到主机到达持有行的主机，查到的行再通过主机链路返回并在张量并行组内分发。本地直达不会重复计域内传输。CPU、主机DRAM带宽与随机访问、主机链路、互联及设备字节均单独入账，图中压缩了路由细节但没有省略成本。这些机制共同说明，参数规模决定一部分容量，最终时延还取决于阶段顺序、状态驻留和每条数据路径。

过渡：明确资源路径后，再看同一1024卡硬件输入下的容量准入，区分预填充峰值与生成峰值。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- Flash：（产品名）DeepSeek-V4.1-Flash——公开架构参考；与本材料合成Pro-10T规模分开。
- Pro-10T：（研究配置名）合成Pro万亿级结构——由Flash机制推演的研究配置，不是公开Pro规格。
- 10T：10 Trillion，十万亿参数——本实验名义骨干参数规模，实际审计为10.1225T。
- CED：Causal Encoder-Decoder，因果编码器—解码器——批量prompt先经过编码器，再投影解码器全局KV。
- CSA2：Compressed Sparse Attention 2，第二代压缩稀疏注意力——跨层共享KV和索引，并复用稀疏位置。
- Engram：（组件名）条件记忆模块——从主机侧ngram表查找并传回选中行。
- SWA：Sliding Window Attention，滑动窗口注意力——本实验窗口长度为128 token。
- KV：Key/Value cache，键值缓存——推理中驻留的注意力历史状态。
- Full：Full Mode，完整模式——生成共享KV/index，并计算稀疏索引的层模式。
- Reindex：Reindex Mode，重索引模式——复用共享KV并重新计算索引。
- Reuse：Reuse Mode，复用模式——复用已有KV及稀疏位置。
- ngram：N-gram，N元词组——本实验条件记忆表的索引组织方式。
- TP：Tensor Parallelism，张量并行——对注意力头、专家中间维等张量分片。
- DP：Data Parallelism，数据并行——在独立权重副本上分配已存在的全局请求。
- DRAM：Dynamic Random Access Memory，动态随机存取存储器——主机内存，用于分片存放Engram表。
- CPU：Central Processing Unit，中央处理器——执行hash与主机侧处理成本。
- FP8：8-bit Floating Point，8位浮点——主对照的权重、计算、激活、通信和KV精度。
- FP4：4-bit Floating Point，4位浮点——公开参考采用的部分存储格式，本实验未假设其加速倍率。
- FP32：32-bit Floating Point，32位浮点——router、mHC、norm等显式标量运算精度。
- SU：Scale Up，域内互联——单逻辑1024卡无阻塞互联及端点注入模型。
- H2D：Host to Device，主机到设备传输——主机内存与加速卡间传输方向。
- D2H：Device to Host，设备到主机传输——加速卡回传主机内存的方向。
- IOPS：Input/Output Operations Per Second，每秒输入输出操作数——主机查表随机访问能力的约束。
- TTFT：Time To First Token，首token时间——本稿统一单独标示prefill全局bs=1的执行时间。
- token：Token，词元——模型输入输出的基本离散单位。
- owner：Cache Owner，缓存持有层——仅指定层生成并驻留共享全局KV/index。
- top-512：Top-512 Sparse Index，512项稀疏索引——每层稀疏注意力索引选中位置数量。
- TB：Terabyte，十进制太字节——本稿TB=10^12字节。
- T：Trillion，万亿——在参数计数中为10^12个参数。
- JSON：JavaScript Object Notation，结构化对象格式——模型、实验与验证账本格式。
- Huawei Proprietary - Restricted Distribution：华为专有资料—限制分发——沿用用户要求的华为模板固定页脚。
- L：Sequence Length，序列长度——图中全量输入提示词的token数量。
- tile：Tiled Computation Block，分块计算单元——索引打分以有限工作区内的分块执行，每块归约及启动均计费。
- scale：Quantization Scale，量化缩放因子——lookup表按每32个FP8元素保存一个scale字节，计入主机容量。
- host link：Host–Device Link，主机设备链路——模型中主机与设备之间的双向传输资源，具有独立带宽和启动开销。
- host：Host，主机——持有部分ngram表并处理哈希、查表与设备传输的CPU主机。
- methodology.md：Methodology Document，仿真方法文档——给出架构来源、执行阶段、精度、资源路径和不确定性边界。
- model.json：Model Configuration JSON，模型配置文件——保存本地合成模型的结构、共享层位与候选索引参数。
- inference/model.py：Reference Inference Model，参考推理模型实现——来自锁定Flash仓库，仅作为公开机制证据，未执行下载的代码。

【标红内容详解 · 逐条一句话解释】
- 「CED：长提示与有界重放」：依据methodology.md的CED阶段定义，编码器主体、解码器KV投影、有界重放及输出头依次执行。
- 「CSA2：共享状态，逐级稀疏」：公开参考保留跨层共享和三种模式，合成结构映射共享层位；候选提前聚集单列为本地执行假设。
- 「Engram：主机分片，冷查表计费」：本实验每DP副本持有一份ngram表并按行分片到所属主机，冷查表路径包含CPU、DRAM、host与SU实际成本。
- 「阶段成本逐项计费」：methodology.md按计算、缓存、索引、通信、主机查表和传输显式建模，没有用参数倍率代替新增机制成本。
- 「128 token」：model.json的window_size为128，CED用该末窗口完整重放32层解码器以建立SWA和首token状态。
- 「16,384」：model.json设置candidate_topk_blocks=2048、candidate_block_size=8，乘积为16384；提前聚集候选是仿真优化假设。
- 「3.5473」：由ngram参数3,439,800,000,000乘以FP8每元素1字节及1/32 scale开销得到每DP副本3.54729375TB，不是每主机一份。
- 「成本全部保留」：bulk活跃参数口径没有包括有限重放与末位置输出头，但执行图包含这些阶段，TTFT并非仅编码器主体耗时。
- 「仿真假设」：公开PyTorch参考先全量打分再屏蔽，本地采用候选提前聚集；此假设与层位映射、FP8统一精度一起限定结果适用范围。

【来源与数据口径】
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020\methodology.md
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020\model.json
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020\sources\inference\model.py
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020\sources\inference\engram.py
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020\sources\config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py

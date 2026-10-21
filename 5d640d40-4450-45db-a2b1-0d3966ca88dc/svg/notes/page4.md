【第4页 · 新增架构用独立解析模块接入原生资源内核，结果可追溯】

这一页承接原版材料的仿真Skill流程，但需要把本次适配范围讲清楚。DeepSeek的CED、CSA2、mHC、Engram和三层DSpark通过独立的解析模块编译为阶段图，调用冻结的supernode_sim原生Graph、Event、Allocation、Resource以及engine.simulate，没有把这些尚未进入通用streaming接口的特性写成通用API已原生支持。输入包括逐矩阵结构、精度、1024卡拓扑和固定全局batch；DP仅分摊已有请求，活跃副本数量不超过全局batch。扫描包含45个并行候选、四种二进制上下文和六档decode batch，TP16与TP32因为保留output groups=8而明确标为不兼容。各组件按barrier串行；计算与HBM在单个kernel内共同约束，只把时间归给主导资源，通信每轮同时受端点和启动约束。仿真使用资源封装缩约相同速率的rank组，仍保留全物理工作量；封装中代表性工作量不能被误认成全卡流量。17组校验核对小规模原生通信图、异构计算、DP主机独立、整数路由守恒、缓存owner、投机状态和容量失败等。全量共有15120行decode，7560行可行、3528行容量不可行、4032行结构不兼容，另有1440行prefill。这些行包括AR和六档条件提交数，不是15120次不同模型训练或独立硬件实测。失败项保留空时延，不能在图表中当成零延迟。prefill按全局bs1和bs8另列，decode批次扫描不改写首token的含义。输入、原始阶段结果、代码指纹、矩阵审计及六类资源时延账本均可追溯。验证证明了方程、加和与缩约一致性，并未证明实际kernel精度、学习路由、接收率、排队、动态批处理或P99；下一步仍需要实机校准。

过渡：在统一的硬件和负载口径下，下一页比较参考布局与解码优先布局的收益及代价。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- Flash：（产品名）DeepSeek-V4.1-Flash——公开架构参考；与本材料合成Pro-10T规模分开。
- CED：Causal Encoder-Decoder，因果编码器—解码器——批量prompt先经过编码器，再投影解码器全局KV。
- CSA2：Compressed Sparse Attention 2，第二代压缩稀疏注意力——跨层共享KV和索引，并复用稀疏位置。
- mHC：Manifold-constrained Hyper-Connections，流形约束超连接——以四路残差流及显式FP32操作建模的跨层连接。
- Engram：（组件名）条件记忆模块——从主机侧ngram表查找并传回选中行。
- DSpark：（机制名）DeepSeek DSpark——三层草稿网络并行提出五个位置，主模型验证六个query。
- KV：Key/Value cache，键值缓存——推理中驻留的注意力历史状态。
- TP：Tensor Parallelism，张量并行——对注意力头、专家中间维等张量分片。
- EP：Expert Parallelism，专家并行——在不同rank间分布专家权重。
- DP：Data Parallelism，数据并行——在独立权重副本上分配已存在的全局请求。
- SP：Sequence Parallelism，序列并行——与TP共用卡，分片pointwise和残差等token状态。
- PP：Pipeline Parallelism，流水线并行——本实验固定PP=1。
- HBM：High Bandwidth Memory，高带宽内存——NPU卡上存储，容量和带宽分别约束。
- FP8：8-bit Floating Point，8位浮点——主对照的权重、计算、激活、通信和KV精度。
- FP32：32-bit Floating Point，32位浮点——router、mHC、norm等显式标量运算精度。
- TTFT：Time To First Token，首token时间——本稿统一单独标示prefill全局bs=1的执行时间。
- P99：99th Percentile，99百分位——本模型未模拟请求排队与尾时延。
- bs：Batch Size，批大小——decode为全局同时参与序列数；TTFT的prefill批次另列。
- batch：Batch Size，批大小——在相应阶段同时处理的序列数量。
- prefill：Prompt Prefill，提示词预填充——含编码器bulk、解码器KV投影及有界replay。
- decode：Autoregressive Decode，自回归生成——生成阶段；可选择AR或DSpark轮调度。
- rank：Parallel Rank，并行进程编号——模型中独立资源端点编号。
- owner：Cache Owner，缓存持有层——仅指定层生成并驻留共享全局KV/index。
- FLOPs：Floating Point Operations，浮点运算次数——工作量而非利用率或时延。
- supernode_sim：（软件名）超节点流量仿真引擎——本次使用冻结的原生资源和调度语义。
- CSV：Comma-Separated Values，逗号分隔数据——全量结构化结果格式。
- JSON：JavaScript Object Notation，结构化对象格式——模型、实验与验证账本格式。
- SHA256：Secure Hash Algorithm 256-bit，256位安全散列——源文件固定版本及审计指纹。
- Huawei Proprietary - Restricted Distribution：华为专有资料—限制分发——沿用用户要求的华为模板固定页脚。
- roofline：Roofline Model，屋顶线模型——同一矩阵阶段以计算和HBM服务时间中的较慢者约束完成。
- collective：Collective Communication，集合通信——本实验ring/pairwise轮次、载荷和启动均显式计入。
- min：Minimum，最小值函数——活跃DP副本数是DP数量与全局batch的较小值。
- output groups：Output Projection Groups，输出投影分组——保留8组结构导致TP16/32不兼容。
- bytes：Bytes，字节数——物理内存和链路账本的流量单位。
- null：Null Value，空值——不可行结果未执行或没有有效时延，不能当作零。
- kernel：Compute Kernel，计算内核——本实验未用实测kernel启动或吞吐profile校准。
- profile：Measured Performance Profile，实测性能档案——用于把解析模型校准到真实算子的测量资料。
- batching：Batching，请求批处理——动态请求合批尚未纳入模型。

【标红内容详解 · 逐条一句话解释】
- 「17 组不变量验证」：verification.json记录17组结构和原生内核缩约核对，范围不是硬件性能准确度。
- 「17」：verification.json的count=17，覆盖张量、容量、通信及状态账本。
- 「性能精度仍待实机校准」：methodology.md明确未使用kernel实测profile，没有性能误差承诺。
- 「15,120」：summary.csv行数，包含AR和DSpark的六个条件C值；不是独立实测次数。
- 「1,440」：prefill.csv独立行数，prefill全局批次为1和8。
- 「固定全局负载，不以新增请求制造 DP 收益」：experiment.json规定active_DP=min(DP,global_batch)，吞吐按实际已给定batch计算。
- 「7,560」：result-verification.json和summary.csv状态计数中的可行decode行。
- 「3,528」：summary.csv中容量不满足当前预留策略的decode行；时延为空。
- 「4,032」：summary.csv中TP16/32与output groups=8不兼容的decode行。

【来源与数据口径】
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/methodology.md
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/verification.json
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/result-verification.json
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/provenance.json
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/experiment.json

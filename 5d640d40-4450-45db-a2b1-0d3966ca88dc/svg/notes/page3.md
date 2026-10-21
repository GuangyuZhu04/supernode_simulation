【第3页 · 10T 推理需分开预算 HBM 与主机内存，并按阶段校验峰值】

这一页保留原版材料的显存、算力和互联三条主线，并补上DeepSeek条件记忆的主机内存路径。所有数字以一百零四万八千五百七十六token、1024卡、TP8、EP128、DP1、SP8的AR参考布局为锚点。首先要区分阶段：prefill全局批大小为1时，单卡预留峰值是62.0232GB；decode全局批大小为32时，峰值只有27.3956GB。它们是两个独立阶段的负载口径，不能把后者作为整批32个请求同时prefill的容量证明。prefill批大小8另有78.7364GB记录。峰值包括权重物理副本、最终KV预留、每lane激活arena、通信暂存、8GB运行时和256MB工作区，不能用逻辑参数简单均分来替代。两平台的容量账本一致，但容量上限、HBM带宽、FP8计算速率和SU启动时间不同；效率分别固定为计算50%和互联80%，FP32峰值沿用50TFLOP/s假设。实验把主要矩阵、激活、通信和KV统一为FP8，显式计入router、mHC及norm的FP32工作量。公开参考中的FP4专家和KV只是来源记录，没有凭空换算为性能倍率。Engram的3.4398T逻辑参数另存主机，FP8与每32元素一个scale使每DP副本占约3.5473TB；该表按行分片到副本所属主机，而非每台主机都放整表。冷查表经过CPU、DRAM或IOPS、host链路、SU和TP通信，所有流量和启动均入账。扩大DP会复制权重和整张逻辑表，即使副本闲置仍保留容量。这些配置是未校准模型的输入，不是硬件规格实测，也不承诺实际物理布线。

过渡：明确资源边界后，下一页说明如何将新架构接入已有仿真内核并校验结果。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- 10T：10 Trillion，十万亿参数——本实验名义骨干参数规模，实际审计为10.1225T。
- mHC：Manifold-constrained Hyper-Connections，流形约束超连接——以四路残差流及显式FP32操作建模的跨层连接。
- Engram：（组件名）条件记忆模块——从主机侧ngram表查找并传回选中行。
- KV：Key/Value cache，键值缓存——推理中驻留的注意力历史状态。
- ngram：N-gram，N元词组——本实验条件记忆表的索引组织方式。
- AR：Autoregressive decoding，自回归解码——每轮提交一个输出token的基线。
- TP：Tensor Parallelism，张量并行——对注意力头、专家中间维等张量分片。
- EP：Expert Parallelism，专家并行——在不同rank间分布专家权重。
- DP：Data Parallelism，数据并行——在独立权重副本上分配已存在的全局请求。
- SP：Sequence Parallelism，序列并行——与TP共用卡，分片pointwise和残差等token状态。
- HBM：High Bandwidth Memory，高带宽内存——NPU卡上存储，容量和带宽分别约束。
- DRAM：Dynamic Random Access Memory，动态随机存取存储器——主机内存，用于分片存放Engram表。
- FP8：8-bit Floating Point，8位浮点——主对照的权重、计算、激活、通信和KV精度。
- FP4：4-bit Floating Point，4位浮点——公开参考采用的部分存储格式，本实验未假设其加速倍率。
- FP32：32-bit Floating Point，32位浮点——router、mHC、norm等显式标量运算精度。
- SU：Scale Up，域内互联——单逻辑1024卡无阻塞互联及端点注入模型。
- SO：Scale Out，域间互联——本实验普通通信不经过此路径。
- IOPS：Input/Output Operations Per Second，每秒输入输出操作数——主机查表随机访问能力的约束。
- TTFT：Time To First Token，首token时间——本稿统一单独标示prefill全局bs=1的执行时间。
- bs：Batch Size，批大小——decode为全局同时参与序列数；TTFT的prefill批次另列。
- prefill：Prompt Prefill，提示词预填充——含编码器bulk、解码器KV投影及有界replay。
- decode：Autoregressive Decode，自回归生成——生成阶段；可选择AR或DSpark轮调度。
- TFLOP：Tera Floating Point Operations，万亿次浮点运算——10的12次方次浮点运算。
- PFLOP/s：Peta FLOPs Per Second，每秒千万亿次浮点运算——硬件名义计算速率。
- GB：Gigabyte，十进制吉字节——本稿GB=10^9字节。
- TB：Terabyte，十进制太字节——本稿TB=10^12字节。
- MB：Megabyte，十进制兆字节——本稿MB=10^6字节。
- T：Trillion，万亿——在参数计数中为10^12个参数。
- μs：Microsecond，微秒——10^-6秒。
- s：Second，秒——时延单位。
- 1M：1,048,576 Tokens，一百万级词元上下文——本稿1M采用二进制长度。
- 950DT：（硬件配置名）Ascend 950DT——本实验采用的144GB、4TB/s、1PFLOP/s假设配置。
- 960DT：（硬件配置名）Ascend 960DT——本实验采用的288GB、9.6TB/s、2PFLOP/s假设配置。
- CSV：Comma-Separated Values，逗号分隔数据——全量结构化结果格式。
- JSON：JavaScript Object Notation，结构化对象格式——模型、实验与验证账本格式。
- Huawei Proprietary - Restricted Distribution：华为专有资料—限制分发——沿用用户要求的华为模板固定页脚。
- scale：Quantization Scale，量化缩放因子——随FP8权重、KV和Engram表存储的额外字节。
- router：Expert Router，专家路由器——以FP32计算显式路由工作量。
- norm：Normalization，归一化——本实验显式计入FP32归一化计算。
- host link：Host Link，主机到加速卡链路——64GB/s名义带宽、85%效率与2μs启动的继承假设。
- cache：Cache，缓存——主场景不假设Engram查表缓存命中。

【标红内容详解 · 逐条一句话解释】
- 「prefill 峰值高于 decode」：summary.csv参考布局记录prefill bs1峰值62.023152704GB、decode bs32峰值27.395639616GB；负载不同，分别展示。
- 「62.02」：prefill.csv中参考布局、1M、AR、全局bs1的每卡峰值；不是decode容量。
- 「27.40」：summary.csv中参考布局、1M、AR、全局bs32的decode阶段每卡峰值。
- 「主对照统一 FP8」：experiment.json规定矩阵存储/计算/激活/通信/KV统一FP8，FP32操作另计。
- 「Engram：每副本主机驻留」：methodology.md定义每DP副本完整ngram表，按行分散在副本所属主机。
- 「3.5473」：3.4398T查表参数按FP8和每32元素1字节scale计算为3.54729375TB，每DP副本一份。

【来源与数据口径】
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/experiment.json
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/summary.csv
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/prefill.csv
- C:\D_disk\Github\supernode_simulation\outputs\deepseek-v41-pro-10t-20261020/methodology.md

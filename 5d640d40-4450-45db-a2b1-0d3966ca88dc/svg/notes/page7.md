【第7页 · DSpark 在 C=4 条件下，同布局折算提速 2.95× / 2.58×】

这一页在固定的 1M 上下文和 decode 全局 batch32 下，单独看 DSpark 带来的条件收益。950DT 固定 TP/EP/DP/SP=8/32/4/8，960DT 固定 4/16/16/4，两者都使用前面已经选出的 AR 最小 TPOT 布局。因而这里的加速比只回答：在相同硬件、负载和布局上，加入草稿机制后，提交多个输出是否足以覆盖额外工作。布局优化的收益已经在前页展示，此处不再重复计入。

执行过程分成三个明确的阶段。第一阶段用 3 层草稿网络同时产生 5 个位置，并执行 5 次马尔可夫串行校正及置信度处理。第二阶段让完整目标网络验证 6 个输入位置，其中也包含当前待处理输入；因此不能把验证成本按五个草稿位置计数。第三阶段进行接受判断和状态提交。全局 KV 可以追加后截断，滑窗和池化状态要保留候选前缀的版本并按提交结果恢复，已验证的目标隐藏态还要投影并追赶草稿窗口。替换或奖励输出尚无对应隐藏态，它作为下一轮输入继续处理，不能免费补出这部分状态。

C=4 时，950DT 的草稿、目标验证、状态提交分别为 1.093315、25.434584、0.089863 ms，合计 26.617762 ms；960DT 对应为 0.283210、7.620857、0.031197 ms，合计 7.935263 ms。状态提交成本虽小却非零，目标验证是主要时间开销。每轮时间除以提交数后，TPOT 为 6.654 和 1.984 ms，同布局 AR 比值分别是 2.95× 和 2.58×。

右表继续把 C 从 1 扫到 6。C=1 时，两平台比值都小于一，新增的草稿和验证工作无法被摊薄；到 C=2 才开始超过同布局 AR。C 表示每轮最终提交的输出数，包含替换或奖励输出，它不是接受率，也不能直接换算成接受概率。本次循环路由是解析工作负载条件，真实文本中的路由相关性、接受情况和模型质量尚未测量。最后区分两个时间概念：round 是一轮成批完成的间隔，TPOT 是将其除以 C 得到的输出服务成本。客户端会如何逐个收到 token，还取决于输出交付和调度；不能把该折算值当成逐 token ITL。

过渡：下一页回到普通 AR，将 prefill 和 decode 的延迟拆成可加资源项，解释布局优化实际省在哪里、又付出了什么代价。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- DSpark：（机制名）DeepSeek DSpark——三层草稿网络并行提出五个位置，主模型验证六个query。
- KV：Key/Value cache，键值缓存——推理中驻留的注意力历史状态。
- AR：Autoregressive decoding，自回归解码——每轮提交一个输出token的基线。
- TP：Tensor Parallelism，张量并行——对注意力头、专家中间维等张量分片。
- EP：Expert Parallelism，专家并行——在不同rank间分布专家权重。
- DP：Data Parallelism，数据并行——在独立权重副本上分配已存在的全局请求。
- SP：Sequence Parallelism，序列并行——与TP共用卡，分片pointwise和残差等token状态。
- TPOT：Time Per Output Token，每输出token时间——AR为轮时延；DSpark为轮时延除以提交数C。
- ITL：Inter-Token Latency，客户端相邻token到达间隔——DSpark的折算TPOT不等同客户端ITL。
- bs：Batch Size，批大小——decode为全局同时参与序列数；TTFT的prefill批次另列。
- decode：Autoregressive Decode，自回归生成——生成阶段；可选择AR或DSpark轮调度。
- draft：Draft Proposal，草稿提出——DSpark三层草稿网络和Markov校正阶段。
- verify：Target Verification，主模型验证——主模型对六个输入位置执行完整验证。
- commit：State Commit，状态提交——接受并写回状态，包含非零恢复与追赶开销。
- ms：Millisecond，毫秒——10^-3秒。
- 1M：1,048,576 Tokens，一百万级词元上下文——本稿1M采用二进制长度。
- C：Committed Tokens Per Round，每轮提交词元数——条件输入C=1至6，包含replacement或bonus，不是接受率。
- 950DT：（硬件配置名）Ascend 950DT——本实验采用的144GB、4TB/s、1PFLOP/s假设配置。
- 960DT：（硬件配置名）Ascend 960DT——本实验采用的288GB、9.6TB/s、2PFLOP/s假设配置。
- CSV：Comma-Separated Values，逗号分隔数据——全量结构化结果格式。
- Huawei Proprietary - Restricted Distribution：华为专有资料—限制分发——沿用用户要求的华为模板固定页脚。
- round：Round，草稿验证提交的一轮——包含 draft、target verify 与 state commit，完成后可成批提交 C 个输出。
- serving：Inference Serving，推理在线服务——接收请求、调度推理并向客户端交付结果的系统；本实验没有进行真实在线服务校准。

【标红内容详解 · 逐条一句话解释】
- 「C≥2」：summary.csv 的同布局条件扫描中，两平台 C=1 的比值均小于一，C=2 才开始超过 AR。
- 「收益门槛 C≥2」：该门槛仅对当前 1M、decode bs32、所选布局与循环路由仿真条件成立。
- 「2.95×」：950DT：同布局 AR 19.627832888073 ms / DSpark C=4 6.654440422798 ms，未混入布局改善。
- 「2.58×」：960DT：同布局 AR 5.111085152725 ms / DSpark C=4 1.983815872743 ms，未混入布局改善。
- 「不等于客户端 ITL」：methodology.md 将 TPOT 定义为 round/C 的折算服务成本，round 是成批输出完成间隔，未建模客户端逐 token 交付。
- 「C=4」：这里每轮提交数固定为四，是仿真条件输入，包含替换或奖励输出，不代表实测接受率。
- 「6.654」：950DT 的 C=4 轮时间除以四，单位为毫秒；其状态提交成本已经计入。
- 「1.984」：960DT 的 C=4 轮时间除以四，单位为毫秒；其状态提交成本已经计入。
- 「一轮包含完整验证与非零状态提交」：三阶段成本来自 dspark_round_breakdown.csv；该面板说明本地状态维护没有作为免费操作。
- 「C=1…6 的条件敏感性：AR 基线 = 1.00×」：比值以每个平台相同布局的 AR TPOT 为分子，各 C 值是独立的条件输入。

【来源与数据口径】
- outputs/deepseek-v41-pro-10t-20261020/summary.csv
- outputs/deepseek-v41-pro-10t-20261020/dspark_round_breakdown.csv
- outputs/deepseek-v41-pro-10t-20261020/methodology.md
- outputs/deepseek-v41-pro-10t-20261020/model.json
- input/deck-data.json: dspark_cases / dspark_sensitivity / ar_cases

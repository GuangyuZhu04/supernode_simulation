【第8页 · decode 收益主要来自通信启动减少，prefill 需承担计算代价】

这一页用相同的四组 AR 结果回答布局优化实际改变了哪类延迟。左边是 1M 单请求 prefill，全局 batch 为一，单位是秒；右边是已经完成预填充后的全局 batch32 自回归 decode，单位是毫秒。上下文相同并不意味着两列代表同一批请求的端到端时间，也不能把单请求首 token 时间视为三十二条请求同时预填充的耗时。每个平台的两行分别是统一参考布局和该平台最小 TPOT 布局。

原始账本有六项可加归因：计算受限、HBM 受限、SU 传输、通信启动、主机查表与哈希、主机传输。图中为节约列宽，把最后两项合并为主机，原始数值完整保留在备注附表。每个矩阵 kernel 同时比较计算和 HBM 的 roofline 时间，只将较慢的一项归入该阶段，二者不会再次相加。不同算子阶段和通信阶段按仿真调度串行累加，因此六项总和等于阶段延迟。组件维度拆解是另一个可加视角，不能再与此处资源归因相加。

先看 decode，950DT 的通信启动由 43.213 ms降到 16.602 ms，净减 26.6112 ms；960DT 由 20.584 降到 3.594 ms，净减 16.990 ms。减少 EP 压低了 dispatch/combine 串行轮数，解释了本轮时延的大部分改善。这不是把全网带宽利用率直接换算成性能；每个端点的注入和每轮启动仍受显式资源约束。

再看 prefill，参考布局的 SU 服务分别占总时间 49.69% 和 60.74%，长上下文索引与通信仍是重要成本。切到 decode 最优布局后，950DT 计算归因从 12.812 增到 13.643 s；960DT 从 9.698 增到 20.047 s。这使总 TTFT 分别上升 1.7% 和 25.8%。单请求只激活一个 DP 副本，更多 DP 不会自动缩短这一个请求；较少并行资源与每目标 rank 的工作变化，会抵消传输和启动的节省。因此应同时约束首 token 时间、decode 服务成本和容量，不能用一个最小 TPOT 结果概括所有目标。本页是条件仿真的资源归因，未进行真机 kernel 校准；表格独立取整会出现尾差，原值的加和检查已经通过。

过渡：材料至此完成从架构、资源约束、布局扫描到阶段归因的闭环；下一步以真实 kernel、路由和接受情况校准这些条件预测。

【本页名词表 · 英文简称：英文全称，中文全称——一句话解释】
- AR：Autoregressive decoding，自回归解码——每轮提交一个输出token的基线。
- TP：Tensor Parallelism，张量并行——对注意力头、专家中间维等张量分片。
- EP：Expert Parallelism，专家并行——在不同rank间分布专家权重。
- DP：Data Parallelism，数据并行——在独立权重副本上分配已存在的全局请求。
- SP：Sequence Parallelism，序列并行——与TP共用卡，分片pointwise和残差等token状态。
- HBM：High Bandwidth Memory，高带宽内存——NPU卡上存储，容量和带宽分别约束。
- SU：Scale Up，域内互联——单逻辑1024卡无阻塞互联及端点注入模型。
- TTFT：Time To First Token，首token时间——本稿统一单独标示prefill全局bs=1的执行时间。
- TPOT：Time Per Output Token，每输出token时间——AR为轮时延；DSpark为轮时延除以提交数C。
- bs：Batch Size，批大小——decode为全局同时参与序列数；TTFT的prefill批次另列。
- prefill：Prompt Prefill，提示词预填充——含编码器bulk、解码器KV投影及有界replay。
- decode：Autoregressive Decode，自回归生成——生成阶段；可选择AR或DSpark轮调度。
- token：Token，词元——模型输入输出的基本离散单位。
- ms：Millisecond，毫秒——10^-3秒。
- s：Second，秒——时延单位。
- 1M：1,048,576 Tokens，一百万级词元上下文——本稿1M采用二进制长度。
- 950DT：（硬件配置名）Ascend 950DT——本实验采用的144GB、4TB/s、1PFLOP/s假设配置。
- 960DT：（硬件配置名）Ascend 960DT——本实验采用的288GB、9.6TB/s、2PFLOP/s假设配置。
- CSV：Comma-Separated Values，逗号分隔数据——全量结构化结果格式。
- Huawei Proprietary - Restricted Distribution：华为专有资料—限制分发——沿用用户要求的华为模板固定页脚。
- hash：Hash，哈希寻址——主机按 ngram 索引计算查表地址，算入主机查表与哈希延迟。
- kernel：Compute Kernel，计算内核——每个内核按计算与 HBM roofline 较慢者计费，二者不重复归因。

【标红内容详解 · 逐条一句话解释】
- 「prefill 计算归因增加」：prefill.csv 中最小 TPOT 布局的 compute-bound 归因相对参考布局上升，其他资源项不能完全抵消这部分增长。
- 「26.6112 ms」：950DT decode bs32：参考通信启动减去最优通信启动，等于 43.2132−16.602=26.6112 ms。
- 「26.611 ms」：950DT decode bs32 通信启动净减少值，图内保留三位小数，原值为 26.6112 ms。
- 「16.990 ms」：960DT decode bs32：通信启动由 20.584 ms 降至 3.594 ms，净减少 16.990 ms。
- 「+0.831 s」：950DT prefill bs1：计算受限归因差为 0.831230790599 s，带来 TTFT 取舍。
- 「+10.350 s」：960DT prefill bs1：计算受限归因差为 10.3497572112 s，带来 TTFT 取舍。
- 「1.7% / 25.8%」：950DT/960DT 的最小 TPOT 布局 TTFT，相对各自参考布局分别按最优/参考−1 计算。
- 「prefill bs1｜单请求首 token（s）」：左图使用独立 prefill 全局 batch1 账本，单位为秒；其六项之和等于 TTFT。
- 「decode bs32｜AR 单轮（ms）」：右图使用 1M、AR、decode 全局 batch32 账本，单位为毫秒；AR 每轮提交一个输出。
- 「46.580」：950DT 最小 TPOT 布局的 prefill bs1 总时间，单位为秒，六项原始归因加和验证通过。
- 「19.628」：950DT 最小 TPOT 布局的 AR decode bs32 单轮时间，单位为毫秒，六项原始归因加和验证通过。
- 「42.835」：960DT 最小 TPOT 布局的 prefill bs1 总时间，单位为秒，六项原始归因加和验证通过。
- 「5.111」：960DT 最小 TPOT 布局的 AR decode bs32 单轮时间，单位为毫秒，六项原始归因加和验证通过。

【来源与数据口径】
- outputs/deepseek-v41-pro-10t-20261020/prefill.csv
- outputs/deepseek-v41-pro-10t-20261020/latency_breakdown.csv
- outputs/deepseek-v41-pro-10t-20261020/summary.csv
- outputs/deepseek-v41-pro-10t-20261020/methodology.md
- input/deck-data.json: ar_cases[*].prefill / decode_latency
- 原始六项附表｜prefill bs1 950dt reference tp8-ep128-dp1-sp8（s）：compute_bound=12.8120506198；hbm_bound=8.53720271342；su_transfer=22.7527000397；communication_startup=1.6681806；host_lookup_hash=0.005542772736；host_transfer=0.00936779294118；total=45.7850445386
- 原始六项附表｜prefill bs1 950dt fastest_decode tp8-ep32-dp4-sp8（s）：compute_bound=13.6432814104；hbm_bound=8.55299620949；su_transfer=22.7098264474；communication_startup=1.6415694；host_lookup_hash=0.017339252736；host_transfer=0.0152660329412；total=46.5802787529
- 原始六项附表｜prefill bs1 960dt reference tp8-ep128-dp1-sp8（s）：compute_bound=9.69762450375；hbm_bound=2.86094388942；su_transfer=20.6842727633；communication_startup=0.794378；host_lookup_hash=0.005542772736；host_transfer=0.00936779294118；total=34.0521297222
- 原始六项附表｜prefill bs1 960dt fastest_decode tp4-ep16-dp16-sp4（s）：compute_bound=20.047381715；hbm_bound=2.93778616301；su_transfer=19.4107630499；communication_startup=0.335218；host_lookup_hash=0.064525172736；host_transfer=0.0388589929412；total=42.8345330935
- 原始六项附表｜decode bs32 950dt reference tp8-ep128-dp1-sp8（ms）：compute_bound=0.000260096；hbm_bound=2.9576343685；su_transfer=0.17797702875；communication_startup=43.2132；host_lookup_hash=0.000121536；host_transfer=6.70588235294e-05；total=46.3492600881
- 原始六项附表｜decode bs32 950dt fastest_decode tp8-ep32-dp4-sp8（ms）：compute_bound=0.000260096；hbm_bound=2.9576343685；su_transfer=0.06774982875；communication_startup=16.602；host_lookup_hash=0.000121536；host_transfer=6.70588235294e-05；total=19.6278328881
- 原始六项附表｜decode bs32 960dt reference tp8-ep128-dp1-sp8（ms）：compute_bound=0.002766767；hbm_bound=1.2300832275；su_transfer=0.161797298864；communication_startup=20.584；host_lookup_hash=0.000121536；host_transfer=6.70588235294e-05；total=21.9788358882
- 原始六项附表｜decode bs32 960dt fastest_decode tp4-ep16-dp16-sp4（ms）：compute_bound=0.00430964876；hbm_bound=1.4595747075；su_transfer=0.0530036068182；communication_startup=3.594；host_lookup_hash=0.000123072；host_transfer=7.41176470588e-05；total=5.11108515273

【原始资源时延六项明细】
[
  {
    "phase": "prefill",
    "unit": "s",
    "chip": "950dt",
    "objective": "reference",
    "layout": "tp8-ep128-dp1-sp8",
    "context_tokens": 1048576,
    "global_batch": 1,
    "components": {
      "compute_bound": 12.81205061978111,
      "hbm_bound": 8.537202713424001,
      "su_transfer": 22.75270003968008,
      "communication_startup": 1.6681805999999912,
      "host_lookup_hash": 0.005542772736,
      "host_transfer": 0.009367792941176472
    },
    "total": 45.78504453856282,
    "check_sum": 45.78504453856236
  },
  {
    "phase": "prefill",
    "unit": "s",
    "chip": "950dt",
    "objective": "fastest_decode",
    "layout": "tp8-ep32-dp4-sp8",
    "context_tokens": 1048576,
    "global_batch": 1,
    "components": {
      "compute_bound": 13.643281410379778,
      "hbm_bound": 8.55299620948797,
      "su_transfer": 22.709826447360097,
      "communication_startup": 1.6415693999999903,
      "host_lookup_hash": 0.017339252736,
      "host_transfer": 0.015266032941176469
    },
    "total": 46.58027875290514,
    "check_sum": 46.58027875290501
  },
  {
    "phase": "prefill",
    "unit": "s",
    "chip": "960dt",
    "objective": "reference",
    "layout": "tp8-ep128-dp1-sp8",
    "context_tokens": 1048576,
    "global_batch": 1,
    "components": {
      "compute_bound": 9.697624503746534,
      "hbm_bound": 2.8609438894199886,
      "su_transfer": 20.684272763345184,
      "communication_startup": 0.7943779999999899,
      "host_lookup_hash": 0.005542772736,
      "host_transfer": 0.009367792941176472
    },
    "total": 34.05212972218861,
    "check_sum": 34.05212972218887
  },
  {
    "phase": "prefill",
    "unit": "s",
    "chip": "960dt",
    "objective": "fastest_decode",
    "layout": "tp4-ep16-dp16-sp4",
    "context_tokens": 1048576,
    "global_batch": 1,
    "components": {
      "compute_bound": 20.04738171496024,
      "hbm_bound": 2.9377861630066637,
      "su_transfer": 19.410763049890715,
      "communication_startup": 0.33521799999999413,
      "host_lookup_hash": 0.06452517273599999,
      "host_transfer": 0.03885899294117647
    },
    "total": 42.83453309353432,
    "check_sum": 42.83453309353479
  },
  {
    "phase": "decode",
    "unit": "ms",
    "chip": "950dt",
    "objective": "reference",
    "layout": "tp8-ep128-dp1-sp8",
    "context_tokens": 1048576,
    "global_batch": 32,
    "components": {
      "compute_bound": 0.00026009600000000013,
      "hbm_bound": 2.957634368499984,
      "su_transfer": 0.1779770287500006,
      "communication_startup": 43.213199999999944,
      "host_lookup_hash": 0.00012153599999999998,
      "host_transfer": 6.705882352941176e-05
    },
    "total": 46.349260088073684,
    "check_sum": 46.34926008807346
  },
  {
    "phase": "decode",
    "unit": "ms",
    "chip": "950dt",
    "objective": "fastest_decode",
    "layout": "tp8-ep32-dp4-sp8",
    "context_tokens": 1048576,
    "global_batch": 32,
    "components": {
      "compute_bound": 0.00026009600000000013,
      "hbm_bound": 2.957634368499984,
      "su_transfer": 0.06774982875000025,
      "communication_startup": 16.60199999999994,
      "host_lookup_hash": 0.00012153599999999998,
      "host_transfer": 6.705882352941176e-05
    },
    "total": 19.627832888073346,
    "check_sum": 19.627832888073453
  },
  {
    "phase": "decode",
    "unit": "ms",
    "chip": "960dt",
    "objective": "reference",
    "layout": "tp8-ep128-dp1-sp8",
    "context_tokens": 1048576,
    "global_batch": 32,
    "components": {
      "compute_bound": 0.002766767000000001,
      "hbm_bound": 1.230083227499999,
      "su_transfer": 0.1617972988636362,
      "communication_startup": 20.58400000000001,
      "host_lookup_hash": 0.00012153599999999998,
      "host_transfer": 6.705882352941176e-05
    },
    "total": 21.978835888187064,
    "check_sum": 21.978835888187174
  },
  {
    "phase": "decode",
    "unit": "ms",
    "chip": "960dt",
    "objective": "fastest_decode",
    "layout": "tp4-ep16-dp16-sp4",
    "context_tokens": 1048576,
    "global_batch": 32,
    "components": {
      "compute_bound": 0.004309648759999999,
      "hbm_bound": 1.4595747075000005,
      "su_transfer": 0.05300360681818141,
      "communication_startup": 3.5939999999999777,
      "host_lookup_hash": 0.00012307199999999998,
      "host_transfer": 7.411764705882354e-05
    },
    "total": 5.111085152725247,
    "check_sum": 5.111085152725218
  }
]

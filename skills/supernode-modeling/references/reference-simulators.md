# 外部仿真器的方法与本项目映射

用于根据外部仿真程序细化训练、推理的执行过程、工作量参数与校准边界。
核验日期：2026-10-31。以下链接为实际读取的一手文档或源码；分支内容可能继续变化。

用户提供的 ChatGPT 共享链接在本次环境中只返回登录页面；后续已直接补充 msModeling、SimuMax、TorchTitan 参考名单。
指定参考分别见 [msModeling / ServingCast](msmodeling-servingcast.md) 与 [训练仿真参考](zhanlu-training.md)。
本文件的 ASTRA-sim、SimAI/AICB、Vidur 保留为独立核验的通用补充，不把它们写成已读取的原对话内容。

## 借鉴范围

本项目继续使用自己的解析模型、事件图、资源争用和校准接口。
下述映射表示可吸收的方法与参数，不表示完整复现外部程序，也不表示直接兼容其 trace 格式。
不要把外部论文的误差、示例默认值或 GPU profile 当作本项目的精度或 Ascend 参数。

| 参考 | 已核验方法 | 当前落点 | 边界 |
|---|---|---|---|
| ASTRA-sim | 执行 DAG、依赖与资源准入；计算和网络后端分层 | 现有事件图、拓扑、roofline 与 kernel profile；新增 serial operator trace | 不提供 Chakra 导入或周期/报文级复现 |
| SimAI / AICB | 框架负载生成，按训练步骤记录计算与通信；可使用 AIOB 测时 | `model.architecture`、训练执行策略、collective 校准、阶段限定 trace | 不直接读取 AICB workload，不自动复现 Megatron/DeepSpeed 调度 |
| Vidur | token、sequence、communication 分类 profiling；请求调度与执行预测分层 | `workload_shape`、分阶段 kernel profile、固定变长 batch | 新增 `serve` 仅独立实例的 FCFS 固定成批；没有 Vidur 式连续 batching |

## ASTRA-sim：把依赖、资源和服务时间分开

[Workload layer](https://astra-sim.github.io/astra-sim-docs/workload-layer/overview.html)
说明其以 Chakra 执行 DAG 表达工作量，节点在依赖满足并取得资源后执行。
[Collective scheduler](https://astra-sim.github.io/astra-sim-docs/system-layer/collective-scheduler.html)
进一步区分 collective 就绪和设备可同时发起的 stream/phase 数。

- 本项目的组件/通信事件依赖决定何时可以开始，资源争用决定实际完成时间。
- `hardware.kernel_traces` 可将一个组件 kernel 展开为串行子算子；原有前后依赖仍然有效。
- 子算子服务时间可以由 `hardware.kernel_profiles` 提供；不能用实测耗时取消真实计算或访存需求。
- 该接口不能表达任意分支 DAG、独立通信 stream、外部 collective 内部分块或新的跨算子重叠。
- 若问题依赖 collective 发起并发限制、包级拥塞控制或框架原生 trace，应明确当前覆盖范围并扩展执行模型。

[Roofline backend](https://astra-sim.github.io/astra-sim-docs/compute-backend/roofline-compute-backend.html)
区分 measured runtime 与 roofline，并使用算子 FLOPs、张量大小、内存带宽和峰值算力。
本项目保留同样的证据区分，但使用自己的字段、单位和资源共享算法。

## SimAI / AICB：训练流程和通信工作量

[AICB training tutorial](https://raw.githubusercontent.com/aliyun/aicb/master/training/tutorial.md)
按初始化、前向、反向、优化器更新组织训练负载；通信项记录类型、group、group size 和消息量。
启用 AIOB 或提供对应计算测量文件才有相应实测依据；未启用的固定计算时间不能用于证明性能。

| 外部关注参数 | 本项目输入或检查项 |
|---|---|
| micro/global batch、训练迭代、序列长度 | `workload.batch_size`、`gradient_accumulation`、`training_steps`、`sequence_length`；每步序列数为 microbatch × accumulation × DP |
| TP / PP / EP、层数及矩阵形状 | `execution.tp/pp/ep/dp`、`model.architecture`；校验设备乘积和物理 rank 放置 |
| ZeRO / distributed optimizer | 本项目的 `execution.fsdp`、`reshard_after_forward`、`sync_each_microbatch`；不能直接把 ZeRO stage 数值填入 FSDP |
| 重计算、参数预取、优化器 offload | `execution.features.activation_checkpointing`、`execution.prefetch`、`execution.features.optimizer_offload` |
| 算子时间、通信类型/group/消息量 | 阶段限定 `hardware.kernel_traces` / `kernel_profiles` 与既有 collective profile/model |
| reduce/allgather/prefetch bucket 与驻留阈值 | 无完整一一对应字段；现有有限下一层预取不等于 DeepSpeed 的 bucket 和 max-live-parameters 策略 |

参数表用于语义映射，不是外部 CLI 的自动转换器。
本项目训练为非交错 1F1B；virtual pipeline、任意 gradient bucket、训练 SP 和框架特定重叠不可由改名获得。
前向、反向、重计算 trace 应分别按实际 `phase` 匹配，不要把前向测量自动倍乘后标为反向实测。
拆分 backward 不能提前放行梯度同步，重计算不能产生额外 optimizer step。
本项目原有组件预算未表达的算子或流量，需要修改编译模型或显式记录新增访存，不能靠 trace 隐藏差额。

[SimAI tutorial](https://raw.githubusercontent.com/aliyun/SimAI/master/docs/Tutorial.md)
区分 analytical 带宽估算与 NS-3 网络仿真；不要将解析链路模型解释成包级网络仿真。
其当前文档还说明手动 `-busbw` 路径未接入当前开源 analytical binary；引用外部命令时须核对版本。

## Vidur：按算子输入性质选择校准维度

[Vidur profiling](https://github.com/microsoft/vidur/blob/main/docs/profiling.md)
将设备计算 profile 与网络拓扑 profile 分开采集，并单独考虑 CPU 调度、采样等实现开销。
[Runtime predictor](https://raw.githubusercontent.com/microsoft/vidur/main/vidur/execution_time_predictor/sklearn_execution_time_predictor.py)
按 token 算子、prefill attention、decode attention 和 CPU 开销使用不同特征。

| 计算性质 | 应保留的输入 | 本项目映射 |
|---|---|---|
| MLP、projection、norm 等 token 工作量 | 当前 query tokens、矩阵维度、dtype、TP | `operator_tokens`、`query_tokens`，以及可选子算子 `matmul` |
| Prefill attention | 每请求 query/context 长度、平方和与已有 KV | `query_length_squared_sum`、`context_length_squared_sum`、`kv_cache_tokens_before` |
| Decode attention | 活跃序列数、历史 KV 长度、head/TP 与布局 | `source_batch_size`、`context_tokens`、长度范围、结构化组件 metadata |
| 网络通信 | communicator、payload、拓扑、算法、软件环境 | 既有 `collective_profiles` / `collective_models`；不使用 token kernel trace 代替 |
| 请求到达与调度 | 到达时间/QPS、batch cap、token budget、chunk、KV 水位 | `serve` 接受显式到达与 batch cap；尚无 token budget/chunk/抢占；实测 trace 分析与服务仿真分开 |

以上 `workload_shape` 字段是结构化 kernel metadata 的观测描述，不是新的 workload 配置项。
其中 `sequence_signature` 保存有序 query/context 配对的身份；仅总 tokens 相同不能证明 shape 相同。
`dense_causal_pairs_reference` 为稠密因果 attention 参考对数，不是 KDA/CSA/HCA 的真实 FLOPs。
推理 `sequence_scope` 为 `ep_source_live_requests`；训练为 `dp_microbatch_before_ep`，并记录 `training_ep_divisor`。
MoE 均衡专家工作量、SP padding 可能使 `operator_tokens` 与 `query_tokens` 不同，不能再机械除一次 EP。

[Vidur 论文](https://arxiv.org/html/2405.05465v2)讨论少量 profiling 与预测，以及在线请求中单批误差改变后续 batching 的现象。
本项目本轮借鉴其特征分组；新增 operator trace 仍是精确适用范围匹配，不提供 Vidur 的回归器或 serving scheduler。
不要将其公开 GPU 数据自动迁移到不同 SKU、拓扑、CANN/HCCL 或模型实现。

## 新增 operator trace 的适用契约

- 外层键是原组件 kernel 名；每条 trace 带 `id`、`source`、`hardware_name`、`phase`、`dtype`。
- 原组件 `flops` 与 `memory_bytes` 必须匹配，非空 `metadata_match` 进一步限定实际结构和 `workload_shape`。
- 有软件栈证据时填写 `calibration_context`；没有声明的环境信息不会自动得到验证。
- `operations` 是有序串行子算子，逐个声明 `kernel`、`flops`、`memory_bytes`、`startup_s`、`workspace_bytes`。
- 可选 `matmul.m/n/k/batch` 描述 GEMM，必须满足 `FLOPs = 2 × batch × m × n × k`。
- 子算子 FLOPs 总和必须等于原组件；访存总和不得低于原组件，可以显式计入拆分产生的额外中间读写。
- 子算子 workspace 和启动成本不能遗漏；拆得更细可能增加实际时间和瞬时内存。
- 子算子继续按其名称和工作量匹配 `hardware.kernel_profiles`；profile 的服务时间与启动时间分别计量。
- trace 的 `source` 说明拆分依据；它本身不证明每个子算子耗时已经实测校准。
- 未命中保持原组件路径并检查报告；不能仅因配置里出现 trace 就声称该路径已采用。

## 如何验证精度改进

先验证语义：FLOPs 守恒、访存不漏计、依赖顺序、梯度/更新次数、workspace 生命周期以及结束序列退出。
再验证 shape：相同总 tokens 而长度分布不同的 prefill 不应被误认；长 KV decode 不应套短 KV 数据。
核对阶段、dtype、设备、结构或环境变化时不会误用原 trace/profile，并对比普通图与适用的 streaming 路径。
synthetic 案例只能验证机制；真实准确度需要同模型、硬件、软件、负载和计时边界下的独立观测。
保留未用于拟合的配置，分别报告 TTFT、TPOT、step time 误差和覆盖范围，不只报训练集拟合分数。
Vidur 源码中的全量训练后评分也不能替代本项目的留出验证；外部论文误差不作为本项目验收阈值。

具体输入示例、匹配行为、输出与执行命令见 [operator-traces.md](operator-traces.md)。

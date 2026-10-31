# msModeling / ServingCast：服务层过程与接入边界

核验日期：2026-10-31。用户已指定 msModeling；本资料按公开文档和真实源码核对，不以转述作为实现证据。
官方入口：[GitCode](https://gitcode.com/Ascend/msmodeling)、[服务化指南](https://mindstudio-docs-master.readthedocs.io/zh-cn/latest/msmodeling/docs/zh/user_guide/msmodeling_serving_cast_simulation_user_guide/)。
源码通过 Ascend 官方 GitHub 镜像读取，锁定 [01898f5fd30a53b21bbd24b798d038294205b85e](https://github.com/Ascend/msmodeling/commit/01898f5fd30a53b21bbd24b798d038294205b85e)，提交于 2026-10-30。
该版本的服务化指南明确标为实验性；后续版本可能改变接口与行为。未核实的组织贡献归属不作为结论。

## 已核验的服务过程

`LoadGen → 客户端 pending / 全局并发门 → Serving → Instance → DP Engine → BatchScheduler → ModelRunner`

- 每个实例组声明角色、实例数和设备/并行布局；每个 DP rank 建立一个独立 Engine。
- 实例与 Engine 都选择当前 workload 最小者，平局选择首项；该分数不是剩余耗时预测。
- running queue 优先，随后处理 waiting queue；每次调度受 query token budget 约束，prefill 可以分块。
- decode window 是原子单位；剩余预算不足时不越过 FIFO 队头。
- KV 以 block/slot 管理；不足时简单抢占 running 队尾，释放其 KV，随后重计算已有历史。
- PD 分离先在 P 完成首 token，再传输 KV 到 D；D 端需要有容量才能接收并继续 decode。

实现依据：[instance.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/instance.py)、[engine.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/engine.py)、[KV 管理](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/kv_cache_manager.py)。

## 上游参数与单位

下表是 [config.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/config.py) 的代码默认值，不是某 Ascend 型号的实测参数。

| 字段 | 默认值 / 要求 | 单位或口径 |
|---|---|---|
| `load_gen.load_gen_type` | 必填；入口仅实现 `fixed_length` | 请求生成类型 |
| `num_requests` | 必填 | 请求数 |
| `num_input_tokens` / `num_output_tokens` | 必填 | 每请求输入 / 输出 tokens |
| `request_rate` | 必填，非负 | req/s；0 表示无间隔尝试发送 |
| `serving_config.max_concurrency` | 100 | 全局在途请求数，不是 tokens |
| `block_size` | 128 | KV tokens/block |
| `max_tokens_budget` | 8192 | 每次调度 query token 预算 |
| `num_instances` / `num_devices_per_instance` | 必填 | 实例数 / 每实例设备数 |
| `pd_role` | 必填：`prefill/decode/both` | 实例角色 |
| `device_type` | `TEST_DEVICE` | 已注册设备画像 |
| `parallel_config.world_size/tp_size/dp_size/ep_size/moe_tp_size` | 均为 1 | 并行规模；不能照抄到本项目独立维度乘积 |
| `dcp_size` | 1 | 复用 TP 设备，须整除 TP |
| `host2device_bandwidth` / `host2device_rate` | `1e10` / `0.5` | B/s / 有效系数 |
| `device2device_bandwidth` / `device2device_rate` | `4e9` / `0.5` | PD KV 传输 B/s / 有效系数 |
| `enable_preprocessing_modeling` / `enable_kv_transfer_modeling` | 均为 `False` | 输入 H2D / PD KV 时延开关 |
| `enable_interpolate` / `enable_multi_process` | `True` / `False` | 耗时插值 / 预测并行 |
| `num_processes` / `predict_steps` / `interpolation_seed` | 10 / 20 / 1234 | worker 数 / 预测前瞻步数 / 初始化种子 |
| `num_mtp_tokens` / `mtp_acceptance_rate` | 0 / `[0.9,0.6,0.4,0.2]` | speculative token 数 / 接受贡献列表 |

以上开关位于 `model_config`；通信字段位于实例的 `communication_config`。
MLP/LM-head 独立 TP/DP 及 `moe_dp_size` 默认 `None`，具体继承与约束由下游处理。
[PD 示例](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/example/pd_disaggregation/README.md)的 1P×4 卡、2D×2 卡、32 请求、4 req/s 等是示例值，不是上述默认值。
PD 角色组合必须同时有 P 与 D 且不含 `both`；聚合部署只含 `both`。
通信按 `bytes / (bandwidth × rate)`，每 Engine 的 Channel 串行排队；不是显式 SU/SO 多链路共享模型。
来源：[communication.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/communication.py)。

## 文档与源码的差异、重要简化

1. PD README 写“泊松到达”，但 [load_gen.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/load_gen.py)实际为 `interval=1/request_rate`，即确定性等间隔，rate=0 时为零间隔。
2. 指南把 `--instance_config_path` 写成路径列表；该版本 [main.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/main.py)实际只接收单路径，一个 YAML 内可包含多实例组。
3. README 将 `predict_steps` 描述为插值采样点数；源码实际用于 multi-process 未来 batch 预测循环，不能据文档将它当作插值精度旋钮。
4. 插值采样调用 TensorCast 的 `execution_time_s['analytic']`，不是 NPU 实测。二维特征只有 `sum(seq_len)` 与 `sum(query_len)`，会丢失长度分布；范围外 NaN 会报错。
5. MTP 采用平均接受推进，decode query window 为 `num_mtp_tokens+1`，KV 只增长接受部分；禁用插值和多进程预测。MTP 混合 P/D batch 串行拆成两次同阶段调用，不表示真实 mixed-batch 时延。
6. 并发门满时请求停在客户端；获得槽后才记录 `LEAVES_CLIENT`，客户端等待不进入其请求 TTFT/E2E。准入使用 0.1 s 轮询，可能产生时间量化。

第 3–5 项依据 [model_runner.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/model_runner.py)；第 6 项依据 [utils.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/utils.py)。
ReadTheDocs 的 PD 示例相对链接在核验时返回 404；可用以上锁定的仓库文件。

## 指标不可直接混用

[request.py](https://github.com/Ascend/msmodeling/blob/01898f5fd30a53b21bbd24b798d038294205b85e/serving_cast/request.py)以实际发送时刻起计 `CLIENT_TTFT`，以服务到达起计 `SERVER_TTFT`。
二者差值 `ADMISSION_WAIT` 在该实现通常约为零，因为并发槽等待发生在实际发送之前。
TPOT 为首 token 后至末 token 的平均间隔；仅一个输出 token 时返回零，而本项目保持未定义/null。
上游 P99 是逐请求标量的分位数，不能与逐 token ITL P99 混用。
整体吞吐分母为首个实际发送至最后完成的时间窗；其单请求输出速率另有 1 ms 分母下限，不应照搬为本项目统一公式。

## 本项目 `serve` 的独立服务层

新增 `supernode_sim.serving` / CLI `serve` 借鉴请求到达、实例、准入与指标分层；每个批次仍调用原有仿真引擎。
具体命令参数先运行 `python -m supernode_sim serve --help`，服务配置与模型/硬件基线分开保存。

| 本项目服务配置 | 含义 |
|---|---|
| `requests` | 显式请求数组；每项 `id/arrival_s/input_tokens/output_tokens` |
| `arrival_s` | 从实验开始起的秒数；是本实验客户端计时边界 |
| `instance_count` | 相互独占、隔离的实例数；每实例完整使用基线硬件 |
| `max_batch_size` | 单批最大请求数，与每实例 DP=1 的基线对应 |
| `max_batch_wait_s` | 为组批允许的等待窗口，秒 |
| `scheduler_overhead_s` | 显式调度开销假设，秒 |
| `slo.ttft_s` / `slo.tpot_s` | 服务请求阈值，秒；取代此次服务运行的基线 targets |

策略是 nonpreemptive FCFS 固定成批：一个批次完成后实例才能接收下一批；批内序列完成不补入新请求。
因此它是有到达与排队的固定组批服务模型，不是 continuous batching，也不是 ServingCast 的完整复现。
当前不支持 token budget、chunked prefill、KV 抢占/重计算、异构 P/D 池或共享实例资源干扰。
不能用 `max_batch_size` 冒充 token budget，也不能把实例复制解释为原拓扑内增加 DP。
原始基线配置及 targets 仍保存；执行采用服务层 SLO，避免把固定 batch 阈值重复作为请求层判据。

本项目客户端 TTFT 为 `first_token-arrival`，服务端 TTFT 为 `first_token-admission`，准入等待为 `admission-arrival`。
该口径包含从输入 arrival 边界开始的排队，与上游排除客户端并发槽等待的口径不同，比较时必须对齐。
服务层按到达/完成事件推进，不复制上游固定 0.1 s 轮询；不将尚未实现的处理或网络延迟隐含加入指标。
统计请求分位数时同时保留样本数、失败/未完成状态、单 token TPOT 未定义情况及计时窗口。

## 参数迁移与验证

只迁移有证据的方法和单位；上游内建常量、`TEST_DEVICE`、示例有效带宽不写入本项目 A3/A5/950 硬件默认值。
子算子、矩阵维度和 profile 继续遵循 [operator trace](operator-traces.md) 的精确适用条件与来源要求。
先验证 FCFS、实例占用、到达前不执行、token 数和时间分解，再用独立实测比较服务延迟与吞吐。
synthetic 队列实验只能验证实现；模拟 P99 仅对给定到达、调度、服务时间与隔离假设成立，不是生产 P99 保证。

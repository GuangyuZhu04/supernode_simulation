# 湛卢公开训练仿真参考与本项目映射

核验日期：2026-10-31。根据用户补充的名单读取以下公开源码；它们不等于完整的内部 Zhanlu-Simulation / Zhanlu-Modeling 系统。专利或演讲的架构/性能结论不能补成未公开的运行参数。

| 项目 | 已核验分支与 revision | 可吸收内容 |
|---|---|---|
| SimuMax-Zhanlu-Enhance | `feat/zhanlu`，`739c5676028ae269180b69b692eb4bb28c6201be` | 分阶段计算、矩阵形状与服务时间、通信/重算及利用率分子 |
| torchtitan-npu-simulator | `feat/npu-simulator`，`2e988ab42d56c43c19c781eb8e9afc1f446693f4` | meta/autograd 训练图捕获、算子 FLOPs、通信量与存储生命周期 |
| torchtitan-simulator | `main`，`af83ddcfb61b3fe6dd3589ae58e6e0eb8ecc2cd3` | 图依赖、compute/comm 执行槽及 mock 成本模型 |

## SimuMax：分开记录模型工作与额外执行

[perf_llm.py](https://github.com/ZhanluModelSim/SimuMax-Zhanlu-Enhance/blob/739c5676028ae269180b69b692eb4bb28c6201be/simumax/core/perf_llm.py#L5211)保留 model/useful/recompute 工作量；该 revision 名为 MFU 的指标分子包含重计算。因此只看“MFU”名称不能判断与另一程序的指标可比。

本项目新增 `metrics.training_accounting`，从实际事件图汇总：

| 本项目字段 | 定义 |
|---|---|
| `model_forward_backward_flops` | 当前编译模型的 F+B 算子计算量代理；不含重算、optimizer、pack/转换 |
| `model_recompute_flops` | 重放前向产生的模型计算量 |
| `executed_npu_flops` | 所有 NPU compute 资源的实际工作量；不把计时 limiter 当 FLOPs |
| `npu_flops_by_phase` | forward/backward/recompute/optimizer/其他阶段分区 |
| `npu_flops_by_role` | 模型 F+B、模型重算、optimizer、auxiliary 的不相交分区 |
| `npu_by_dtype` | 各计算精度的 FLOPs、裸峰值和归一计算时间 |
| `model_mfu` | `Σ_dtype(F+B FLOPs / 该 dtype 单卡裸峰值) / (完整 NPU 数 × 总时间)` |
| `executed_compute_utilization` | 相同分母，分子使用全部 NPU 执行 FLOPs |
| `cpu` | CPU optimizer/其他工作单列，使用 CPU 峰值和主机数 |

两种利用率是峰值归一的工作量比例，不是 profiler 实测 busy fraction。不能将 FP32 与 BF16 FLOPs 直接相加后除以同一个峰值；也不能用已乘 efficiency 的有效峰值作 MFU 分母。未知资源身份会令比例不可用，失败/零时长执行不报告 MFU。

开启 checkpoint 时，F+B 分子应保持一致，重计算和总执行工作增加；将额外重算放进 MFU 分子，可能造成“做更多无用功，MFU 更高”的误读。新账本不替代对训练收敛、框架实际 FLOPs 与算子实现的验证。

## 矩阵参数与阶段

[config.py](https://github.com/ZhanluModelSim/SimuMax-Zhanlu-Enhance/blob/739c5676028ae269180b69b692eb4bb28c6201be/simumax/core/config.py)区分阶段对应的 shape、dtype/out_dtype、累加、tile、padding、operand bytes、算术强度和启动延迟。
[matrix_service.py](https://github.com/ZhanluModelSim/SimuMax-Zhanlu-Enhance/blob/739c5676028ae269180b69b692eb4bb28c6201be/simumax/core/matrix_service.py)提供计算/访存效率曲线、tensor quantum 和串行分段服务；当前特定 BF16/ND 适用限制必须保留。

映射到本项目时：

- GEMM 的 `M/N/K/batch`、FLOPs、HBM bytes 进入 [operator trace](operator-traces.md)；forward、dgrad、wgrad、recompute 分别描述，不把输入梯度和权重梯度的轴机械照抄。
- 计算、权重、梯度、activation、通信和 optimizer 的精度分别进入 `execution.precision`；不把 storage dtype 当作 compute dtype。
- 核实后才将独占服务时间/启动时间写入 `hardware.kernel_profiles`，同时限定 shape、phase、SKU 和软件环境。
- 本项目当前 trace 保持原逻辑 FLOPs，不能直接塞入含额外物理 padding 的执行 FLOPs；如需模拟 padding/tiling，须扩展编译工作量并保留逻辑与物理两本账。
- 不迁移未经同条件验证的效率曲线系数，也不把 SimuMax 的峰值字段默认值当作 Ascend A3/A5 实测。

## TorchTitan NPU：图、通信与存储边界

[trainer.py](https://github.com/ZhanluModelSim/torchtitan-npu-simulator/blob/2e988ab42d56c43c19c781eb8e9afc1f446693f4/torchtitan_npu/simulator/trainer.py#L496)按 execution kind 与重复次数归总节点工作，配置分别控制参数存储、保存 activation 的 offload、FSDP/EP 通信精度和 selective activation checkpoint。
[op_cost_model.py](https://github.com/ZhanluModelSim/torchtitan-npu-simulator/blob/2e988ab42d56c43c19c781eb8e9afc1f446693f4/torchtitan_npu/simulator/cost/op_cost_model.py)按输出元素与归约维度计算 GEMM；grouped wgrad 不能额外重复乘专家数。

其 collective 的逻辑 comm bytes 不能直接替代本项目拓扑路由后的 transport/逐跳 physical bytes。既有 TP/EP/FSDP 图和共享链路仍决定通信时间；不能把 logical allreduce bytes 与 ring busbw 混为一项。

[memory/estimator.py](https://github.com/ZhanluModelSim/torchtitan-npu-simulator/blob/2e988ab42d56c43c19c781eb8e9afc1f446693f4/torchtitan_npu/simulator/memory/estimator.py)按存储身份去重，并将入阶段前已经存活的内存计入阶段峰值。对应本项目初始 allocation、事件 allocation/release、tensor alias 与组件峰值；不能将“本阶段新增分配”当作阶段总峰值，也不能相加不同时间的独立峰值。

框架原生图捕获在本项目尚未实现；当前显式结构与 operator trace 是架构近似。文档中的真实设备训练 benchmark 也不证明 simulator 的成本预测已经校准。

## TorchTitan mock 成本不能当硬件依据

[cost_model.py](https://github.com/ZhanluModelSim/torchtitan-simulator/blob/af83ddcfb61b3fe6dd3589ae58e6e0eb8ecc2cd3/torchtitan/experiments/simulator/cost_model.py#L391)明确使用 mock 默认成本，包含固定计算/访存/通信能力、延迟及噪声种子。
[des_engine.py](https://github.com/ZhanluModelSim/torchtitan-simulator/blob/af83ddcfb61b3fe6dd3589ae58e6e0eb8ecc2cd3/torchtitan/experiments/simulator/des_engine.py)的计算/通信槽和依赖可作为执行组织参考，不能证明本项目已支持同样调度或真实 MTE 通道。

不将 mock 峰值、随机噪声或简化通信公式填入真实机型 profile；缺少同环境测量时继续使用明确假设与敏感性范围。

## 复现与检查

```powershell
python -m supernode_sim simulate examples/features/synthetic-training.yaml -o outputs/reference-training
```

在 `report.md` 查看训练工作量口径，在 `result.json → metrics.training_accounting` 查看分阶段/角色/dtype 的守恒关系。checkpoint 开关对比应保持模型、硬件、microbatch、累积步数、DP 和精度一致。trace 或 profile 改变执行拆解/时延后，不能重复计算父组件或计时 limiter 的工作。

要声称准确度改善，仍须从同条件框架运行获取独立 step-time、算子/通信与内存峰值观测；上述公开实现启发的是参数与过程分解，不是本项目的误差保证。

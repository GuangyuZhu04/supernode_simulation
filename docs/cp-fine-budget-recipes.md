# CP 细粒度：共享投影与路由专家预算配方

实现位于 `scripts/cp_fine_trace_specs.py`。模块只生成配方；图变换、严格元数据守卫、原生 `apply_kernel_traces`、中间缓冲生命周期和额外 FP32 事件由 CP fine adapter 处理。原 coarse CP 图和核心引擎保持不变。

## 参数与适用范围

- `P_expert`：用户总参数预算中路由专家部分；`E`：专家数；`L`：模型层数。
- `H`：当前预算模型的 MoE hidden width；`B`：全 POD batch；`k`：top-k。
- 逻辑中间宽度 `I = P_expert / (3 × H × E × L)`。
- 每个 EP owner 的期望路由行数 `M = B × k / EP`。

这些是三矩阵 gated FFN 的预算恒等式。`I` 和路由行数可以是小数，且各专家的整数 token 分配未知，因此不写入原生 `MatmulShape`，也不把它们称为官方 checkpoint 或可部署 GEMM 形状。当前 2.8T 案例的 Kimi/V4.1 逻辑宽度分别约为 3170.902 和 7372.347。

## 专家子算子

设父专家事件的矩阵 FLOPs 为 `F`、FP8 权重读取为 `W`。当前 CP 父事件只计专家权重 HBM 读取，生成器会严格核对这一前提。

| 子算子 | 参数矩阵 FLOPs | HBM byte |
|---|---:|---:|
| gate budget GEMM | F/3 | W/3 + 2M(H+I) |
| up budget GEMM | F/3 | W/3 + 2M(H+I) |
| FP32 gated activation | 额外显式点算预算 | 6MI |
| down budget GEMM | F/3 | W/3 + 2M(H+I) |

三个矩阵的 FLOPs 和原权重读取之和精确保留父预算；最后一项使用浮点余数配平。新增 FP16 中间张量的总 HBM 访问为 `6MH + 12MI` byte，来自 gate/up 的独立输入读取、输出物化、门控操作两读一写以及 down 的输入/输出。它们是明确采用未融合中间张量的实现假设，不是由参数量推导的硬件实测流量。

门控激活单独插在 up 之后、down 之前，使用 FP32 资源，避免在同一原生 trace 内悄悄切换精度或破坏父矩阵 FLOPs 守恒：

- Kimi 使用参考 `SituAndMul` 的运算结构：`beta*tanh(gate/beta)*sigmoid(gate)`，up 可经 `linear_beta*tanh(up/linear_beta)`，然后相乘。这里声明每元素 12 个符号操作的工程代理（9 个表达式操作和3次转换；tanh/sigmoid各按一个代理操作），不是其指令数、时延或已测 FLOP 等价。
- V4.1 使用仓库专用模型的 SwiGLU 预算 `8MI`，同样未校准非线性吞吐。

gate、up、activated 三个 carry 各占 `2MI` byte FP16。适配器保守地将它们保留至 down 完成，activated 缓冲可在 up 开始前预分配；原 coarse expert arena 也保留，不假设复用或提前释放。总新增 carry 容量为 `6MI` byte。

## 共享稠密预算

`shared_projection_pre/post` 各自只展开为一个 `budget_matmul` 子项，完整保留父 FLOPs 和 HBM。共享 D 包含无法逐张量划分的投影、共享专家或其他稠密部分，不能据此编造 Q/K/V 或输出投影矩阵维度，也不任意按百分比拆分。

## 启动和验证

配方的子项启动默认零。0/1 μs 等每 child 启动敏感性由适配器单独输入，独立于通信启动比例；1 μs 是工程假设，不是测量。

原生 trace 校验要求子算子矩阵 FLOPs 总和等于父事件、HBM 不低于父事件。生成器还验证 `P_active_expert = P_expert × k/E` 和父权重读取的均匀专家触达公式。额外点算 FLOPs、物化 HBM 和 carry 容量分列。形状、访存与依赖更细不代表已经提高实机计时精度。

来源：`supernode_sim/components.py` 的 gated FFN 参数恒等式、`docs/sources/modeling_kimi_linear.py.source.txt` 的 `SituAndMul`、`supernode_sim/case_studies/v41_fp4/target.py` 的 SwiGLU 工程预算，以及 `supernode_sim/config.py`/`operator_trace.py` 的原生 trace 契约。

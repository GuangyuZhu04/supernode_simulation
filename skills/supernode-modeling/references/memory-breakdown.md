# 组件级存储与生命周期解释

使用 `model.architecture` 的真实张量维度生成参数账本。`reported_total_params` / `reported_active_params` 保存厂商四舍五入数字，`total_params` / `active_params` 由张量形状推导，禁止缩放账本凑整。旧配置只有总参数预算时保留 `Unassigned`，不伪装成结构级结果。

## 建议输出

1. 组件参数表：WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output layer，以及存在的 AttnRes。列出张量名/形状、总参数、活跃参数、所在层、TP/EP/FSDP 分片或复制。
2. 组件 × 状态矩阵：weight、weight scales、gradient、FP32 master、Adam momentum/variance、Muon momentum、saved activation、recompute/working activation、KV、KDA recurrent/conv state、通信 buffer、optimizer workspace、运行时预留。
3. HBM/DDR 池占用：容量、静态驻留、观察峰值、峰值时刻构成、余量、不可行原因；按设备和层展示最占空间的组成，完整行保留在 CSV。
4. 差异分析：FP8/FP16、Adam/Muon、offload、checkpoint、KV 长度/压缩分别变化时，指出是哪一列变化，保持其他配置一致。

引用引擎输出的 `component_memory.csv`、`state_memory.csv`、`memory_peaks.csv`、`model_parameters.csv` 和 `tensors.csv`；具体文件字段以当前引擎输出为准，不手工编造不存在的结果。报告存在分解模式/状态时同步呈现。

## 数值口径

- 权重容量由总参数决定；MoE top-k 只影响活跃计算和触达专家，不能将所有常驻专家权重乘 top-k / experts。
- Dense/shared 参数通常在 EP rank 间复制，routed 参数沿 EP 分片；norm、MLA latent KV 等也可能在 TP 间复制，不统一除以总卡数。
- embedding 与 output 共享权重时只计一份物理存储，别名仍列出；当前引擎不支持跨 PP 的 tied weight 协议，应保留校验错误。
- MLA 依据 `cache_layout` 按优化 latent cache 或展开 K/V 计算存储；KDA 按每序列固定 recurrent matrix + conv history 计算，不按每 token KV 线性增长。训练 KDA chunk boundary state 与推理 recurrent state 是不同生命周期。结构上可以压缩不等于所选框架实际保存了压缩缓存。
- K3 的 `mla_use_nope=true` 不意味着移除名为 rope 的投影维度；以官方实现为准，缓存向量为 512 + 64 = 576 元素。
- 激活留存由 batch、长度、checkpoint、融合算子和运行时决定。留存为 0 可能表示重计算，不代表该组件无计算工作集。
- 静态逻辑状态、复制后的驻留总量、某设备峰值、各组件独立峰值分别展示。不得把各组件独立峰值相加声称全局同时峰值。

## Adam 与 Muon

`execution.optimizer.kind` 支持 `adam` / `muon`，`muon_variant` 支持 `matrix` / `per_head`。Muon 参数分组需要矩阵形状，因此不能只用参数预算。当前策略是矩阵参数一份 momentum；embedding、output 和向量等辅助组走 Adam m/v。此分组属于显式建模策略，除非用户有真实训练参数组证据，否则不能说是 Kimi 的完整官方配方。

Muon momentum 与 FP32 master、梯度是不同状态；Per-Head Muon 改变 Q/K/V 的正交化块，不能消除 momentum 的存储。`ns_iterations`、`workspace_copies`、状态 dtype 决定计算/暂存假设。需要完整矩阵的分布式重组、通信路径和优化器算子性能若未校准，报告相应限制；不能把通用 CPU offload 数值称为生产 K3 的优化器实测。

OOM 时保留结构分解和明确失败资源。模拟中止前观察到的峰值只是已执行前缀的峰值，应结合尚未执行的申请需求解释，不能将其作为完整任务的安全容量。

# SP/CP 算子细粒度仿真

`scripts/scaling_cp_fine_ops.py` 在原 `scaling_context_parallel.build_cp_layer` 的 owner-star 上下文并行图上，调用原生 `apply_kernel_traces` 生成实际执行的子事件。它不是将旧耗时按百分比重画，也不是 Ascend 指令级仿真或实机校准。

## 执行内容

- Attention：QK → FP32 局部 max / exp / sum → AV；保留稳定 softmax 的跨 CP 归并，显式保存 score/exponential 中间量。
- KDA：卷积与门控、state decay、prediction、delta rank-one update、state readout、输出归一化/门控。已知卷积参数工作从共享预算移入，避免重复计参。
- MoE：gate / up / down 三矩阵、Kimi SiTU 或 V4.1 SwiGLU 点算，以及 gate/up/activated 的跨子算子缓冲。参数矩阵 FLOPs 和原权重读取保持守恒，新增点算与物化访存单列。
- 索引与归并：局部索引分数、head reduction、top-k、候选打包、全局 top-k、softmax 重标定和输出归并。
- 共享稠密 D 仍保留预算 GEMM。没有足够张量结构，不能将它伪拆为已知 Q/K/V 权重。

网络事件的工作量、启动、依赖和原有状态驻留不变。新子算子沿各自 compute/HBM 资源执行 roofline 调度；新增临时缓冲具有申请与释放事件。原 arena 保留，因此内存峰值是保守的实现假设。

## 形状与精度边界

严格整数 `MatmulShape` 只用于有参考代数依据的 QK、AV 和 index-dot；压缩 MLA 的形状标签不代表已知的物理 binary/tiling。MoE 的预算等效中间维可能非整数，只记录 logical geometry；[预算配方](cp-fine-budget-recipes.md) 给出完整公式。

KDA 核心继续使用旧父事件的 compute 资源，以控制本轮比较变量；FP32 持久状态的容量和访存按 FP32 计，新标量事件使用 FP32 资源。该选择不证明实际 KDA kernel 的算术吞吐。非线性操作量、未融合物化及 kernel 启动均须用后续实测校准。

## 入口

已有上轮场景与 CP 结果时，从项目根目录运行：

```powershell
.venv/Scripts/python.exe scripts/run_scaling_cp_fine.py
.venv/Scripts/python.exe scripts/report_scaling_cp_fine.py
.venv/Scripts/python.exe -m pytest tests/test_scaling_cp_fine.py -q
.venv/Scripts/python.exe scripts/verify_scaling_cp_fine_results.py
```

默认读取 `outputs/scaling-10case-20261113/` 的原场景、Attention 模板及 `sp-megakernel/cp-owner-star/summary.json`。用 `--study` 指定另一个同结构实验目录，用 `--output` 改变输出位置；`--case` 可选择一个已有 case ID。

默认执行原 12 个 CP 对照（四个 2.8T 场景 × 网络启动系数 1/0.5/0.3），kernel 显式启动为零；另执行四个网络系数为 1、每真实 kernel 启动 1 μs 的工程敏感性场景。网络启动系数不作用于 kernel 启动。原先被必要下界排除的 10T/100T 场景没有伪造细粒度执行或占比。

API：

```python
graph, audit = build_fine_cp_layer(config, profile, kind, alpha=1.0, launch_s=0.0)
```

`audit['trace_config']` 是实际匹配的原生 trace 配置，其余字段保留守恒、额外工作量、依赖重排和预算未解析项。`graph.metadata.operator_trace` 保留 native 匹配统计；匹配率不是张量形状解析率或计时精度。

## 交付物与验证

默认输出 `outputs/scaling-10case-20261113/sp-megakernel/cp-fine-operators/`：

- 每场景配置、实际 kernel-traces、结果 JSON、正常报告、完整 latency/operator CSV。
- 四个主对照的每 kind 代表图与执行 trace（gzip）、子算子规格 CSV。
- 汇总、与聚合模型的对照图、Attention/MoE 内部成本、完整算子明细与覆盖表。
- 源码 SHA256、参数/权重/FLOPs/HBM/网络/生命周期检查和独立验证结果。

所有时间来自新图实际执行。更细粒度可以暴露串行 kernel、访存和启动开销，但不代表已经提高了实机预测精度。

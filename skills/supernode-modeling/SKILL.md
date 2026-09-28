---
name: supernode-modeling
description: Model supernode training and inference memory, compute, and communication with supernode_sim. Use for component-level weights, activations, gradients and Adam/Muon state breakdowns; Kimi K3 and Ascend 950 scenarios; FP8/FP16 capacity, offload, KV/KDA state, SU/SO comparisons and minimum-SU searches.
---

# 超节点资源与通信建模

把问题转为可复现配置，调用 Python 引擎计算，解释资源需求、瓶颈和适用条件。此 skill 是引擎的调用与分析说明；**不包含独立的 Python 引擎**，需要 `supernode_simulation` 仓库或已安装的 `supernode_sim` 包。

## 找到可运行环境

优先使用用户已给定的项目路径，其次检查 `SUPERNODE_SIM_ROOT` 环境变量，再检查当前工作目录及其父目录中的 `supernode_sim/` 与 `pyproject.toml`。如果本文件位于项目的 `skills/supernode-modeling/`，其上两级目录就是项目根目录。被复制到全局 skills 目录后，不能把该目录视为引擎所在项目。

在项目根目录、项目 Python 解释器下执行：

```text
python -m supernode_sim --help
python -m supernode_sim schema
```

包已安装时也可使用 `python -m supernode_sim`；示例文件仍位于原仓库。若找不到仓库或模块，报告具体缺少的路径/依赖，不伪造执行结果。引用配置文件路径、输出目录时遵循所在 shell 的引号规则。

## 按问题选择资料和命令

- 配置、来源、单位、CLI：读 [configuration.md](references/configuration.md)。
- 按组件分析存储、activation、梯度、Adam/Muon：读 [memory-breakdown.md](references/memory-breakdown.md)。
- Kimi K3 / Ascend 950 样例和真实性边界：读 [ascend950-kimi-k3.md](references/ascend950-kimi-k3.md)。
- 训练/FSDP2/optimizer offload/10T 容量：读 [training.md](references/training.md)。
- 推理/KV 压缩与 offload/Engram：读 [inference.md](references/inference.md)。
- 拓扑/SU 与 SO/链路瓶颈/最小域搜索：读 [topology.md](references/topology.md)。

先检查用户提供的配置与可核实资料，再补齐问题所需的参数。只询问会改变结论且无法从现有资料获得的缺失项，使用明确字段名。模型名称不是结构证据；对未公开结构采用用户认可或明确标注的 synthetic 假设，不把假设写成某模型的公开规格。

保留用户场景，在工作目录创建可重跑配置。先 `validate`，再按任务运行 `estimate`、`simulate`、`sweep` 或 `size-su`；`estimate` 提供必要下界，不能证明 SLO 可行。硬件显然无法容纳的配置也有分析价值，保留失败原因而不自动放大硬件。

## 解释结果时的关键口径

- `total_params`、每 token 的 `active_params` 和中间 activation 张量是三个不同量。
- 独立配置权重驻留、计算、梯度、优化器、KV、通信与权重回传精度。FP8 驻留不意味着 H2D 按 FP8 传输。
- 区分逻辑容量、实际复制/分片容量、生命周期峰值；DRAM 与 DDR 不重复计容。
- 存储报告应给出“组件 × 状态”矩阵及逐层/逐设备明细。Kimi K3 至少展示 WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output layer，并保留实际存在的 AttnRes；未知结构列为 Unassigned，不按百分比伪分配。
- 将 weight、gradient、master、Adam m/v、Muon momentum、activation、KV、KDA recurrent/conv state、量化元数据、通信/计算 workspace 分列；Muon 的正交化暂存和辅助 Adam 参数组单列。
- 表明每张表的加和规则：逻辑静态容量可加；不同组件在不同时刻取得的峰值不能相加当作同一时刻设备峰值。应同时查看设备池峰值时刻的组件构成。
- 区分 transport payload 与逐跳物理字节量；dispatch/combine 是 all-to-all-v 的语义视图，不能再相加。
- 给出配置、版本/来源、命令、结果文件，以及结论成立的约束。执行失败、不可行和缺参分别处理。
- 缓存/MTE 的工作集、访问量和搬运需求不能直接换算成 L1/L2/L3 容量或通道数；需要硬件与 kernel 校准。
- 固定 batch 仿真不提供生产请求排队或 P99 保证；SU “最小”限定为已检查候选集合与调度策略。

正式数值引用引擎生成的 JSON/CSV/报告；容量锚点可用于交叉检查。报告没有的数据应标注缺失或扩展需求，不补写成仿真输出。

继续中断的大型实验时，先检查输出文件与 `checkpoint/`，区分数值执行已完成和明细导出已完成。可用 `python -u scripts/run_reference_experiment.py <配置> -o <原输出目录>` 恢复已提交的匹配检查点。它会校验配置和数值模块指纹，跳过已完成的数值计算。精简结果 JSON 不能补造缺失的逐张量生命周期；若没有原始记录，须重放并保存检查点。

大型场景默认按报告中的导出清单精简原始事件和临时通信缓冲记录，容量统计与真实并发峰值保持完整。优先阅读 `report.md`、`component_memory.csv`、`state_memory.csv` 和 `model_parameters.csv`；需要检查全部逐事件生命周期时，在 `estimate` 或 `simulate` 命令添加 `--full-detail`。不要把导出行数减少解释为仿真忽略了这些资源。

大型 optimizer offload 场景可显式配置 `execution.optimizer.inflight_components_per_rank: 2`，限制同时在途的组件更新并保留搬运/计算重叠；这是执行策略，会影响峰值和耗时，应随结果说明。未设置时保留原有全部提交策略。

用户指出仿真与现实存在差距时，先核对结构、精度、复制/分片与模型范围，再核对平台 SKU、单位、带宽方向、拓扑和实测 kernel/collective。用可核实 profile 替换合成硬件，并说明尚未校准的影响。型号相同不等于已验证性能准确；没有实测结果时不报告误差百分比或“符合真实情况”。

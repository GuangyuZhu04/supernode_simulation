# Marin 535B-A23B 训练校准

同一段 4K 训练提供了一个直接对照。Marin 在固定容量 EP 下约为 21.5% MFU，切换 ragged All-to-All 后约为 23%。两者使用相同的模型、704 张 GB200、每步 token 数和日志 FLOPs 分子。末期 65K 模型的日志分子发生变化，需要单独核算。[EP 技术博客](https://marin.community/blog/2026/09/29/expert-parallelism/) · [W&B 切换记录](https://wandb.ai/marin-community/marin/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng) · [最终训练记录](https://github.com/marin-community/marin/issues/8435)

| 用作输入的实测点 | 用作检验的条件 | 条件外推 MFU | 公开近似值 | 带符号差值 |
| --- | --- | ---: | ---: | ---: |
| 固定容量 EP，4K，21.5% | ragged EP，4K | 22.94% | 23% | −0.06 个百分点 |
| 后期内核，4K，29% | 同等有效计算速率，65K | 33.15% | 约 33% | +0.15 个百分点 |

每张卡每步处理 65,536 个 token。程序用原生结构化编译器计算均衡条件下一张卡 48 层的前反向 FLOPs，共覆盖 384 个路由专家、top-8、两个共享专家、3072 维路由、36 层 2048 滑窗和 12 层全因果注意力。局部和全局层分别使用 12 和 6 个 KV 头。将锚点 MFU 乘以“新／旧日志 FLOPs”和“旧／新执行 FLOPs”，即表中的条件外推。第二行单独使用 kernel stack 更新后的 29% 作为锚点。公开的 27% 中期读数、约 26% 的 FA4 与 mask-free MLP 切换点也记录在结果文件中。[模型结构与并行设置](https://github.com/marin-community/marin/issues/8435) · [中期更新](https://x.com/percyliang/status/2107502170191155633)

这个外推假设两次对照各自保持相同的单位执行 FLOP 时间。它将通信、重计算、优化器及其他开销的影响包含在锚点中。若固定早期内核效率一路外推到 65K，结果约为 26.2%。这些观察值以近似百分比公布，差值的小数位用于复现算术。对独立集群或未见过的模型评估精度，需要额外的按算子测量。

### 可复核的工作量

- 固定容量按 1.15 倍发送槽位与专家容量计费，丢弃率输入为 3.4%。ragged 取零丢弃近似。每层每卡向其他 63 张卡发送的 dispatch 数据从 3.647 GB 降至 3.171 GB，按十进制字节计。共享专家与本地份额均排除在链路字节之外。
- 65K 时，3:1 局部／全局混合注意力的有效 Q/K 位置对为全因果模型的 29.61%。Marin 日志按注意力跨度计算，4K、16K、65K 对应的前反向 GFLOPs/token 分别为 149.736、160.672、204.413。仿真器按有效位置对计费，并提供独立的日志口径。[日志 FLOPs 实现](https://github.com/marin-community/marin/blob/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/lib/levanter/src/levanter/utils/flop_utils.py)
- 4K 的 16 卡切片包含 4 层、EP8、DP2，每卡仍有 6 个专家和 65,536 个输入 token。它完整调度了前反向、整层重计算、集合通信和设备侧 Muon 代理，拿来检查物理开销、显存与时延方向。单步从 4.210 s 降至 4.149 s，峰值 HBM 从 81.71 GB 降至 80.75 GB。链路延迟与算子效率是示例假设。704 卡实跑的分布式 MuonH、选择性重计算、三波 EP 的逐波调度和各轮 FA4/XLA 优化还需要独立 trace。

[机器可读对照结果](comparison.json) · [固定容量 EP 切片图](fixed_capacity.svg) · [ragged EP 切片图](ragged.svg)

### 复现

```bash
python -m examples.marin_training --output outputs/marin/comparison.json --slice-reports outputs/marin --full-config outputs/marin/704-fixed.json
```

`704-fixed.json` 可供常规配置验证。本实验对该规模计算代表卡算子工作量。完整的事件调度和图表分别保存在 `fixed_capacity/` 和 `ragged/`。每个目录包含 `training_summary.json` 和 `training_breakdown.svg`。

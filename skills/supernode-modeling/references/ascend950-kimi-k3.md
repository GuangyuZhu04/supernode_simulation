# Kimi K3 × Ascend 950：资料驱动样例

本项目已保存官方资料；从项目根目录读 `docs/sources/README.md`，必要时查看其中链接的 `kimi_k3_config.json`、`modeling_kimi_linear.py.source.txt` 和官方技术报告。源码快照仅供读，不需要安装或执行模型，也不需要下载模型权重。

## 选择案例

| 配置 | 用途 |
|---|---|
| `examples/kimi_k3_ascend950_fp8.yaml` | 官方 K3 文本 backbone 维度，统一 FP8 权重重精度方案，64 NPU 推理 |
| `examples/kimi_k3_ascend950_fp16.yaml` | 同一结构与负载，FP16 权重/计算方案 |
| `examples/kimi_k3_ascend950_mxfp4.yaml` | 官方发布存储格式参考：routed experts FP4 + group-32 uint8 scale，其余 BF16；计算仍是显式 BF16 假设 |
| `examples/kimi_k3_ascend950_training_muon.yaml` | 64 NPU，FP8 权重、FP16 梯度、FP32 master/Muon momentum、CPU offload；保留真实容量约束下的不可行结果 |
| `examples/kimi_k3_ascend950_training_muon_128.yaml` | 128 NPU，TP8/EP8/DP2/FSDP2；人为分成两个 64 NPU SU 域，评估容量与 SO 敏感性 |

运行 `validate` 再运行 `estimate` / `simulate`。完整模型有 93 层，TP/EP collective 展开后事件数较多；完整仿真需要更多时间，但不能缩层后仍称完整 K3。无 SLO 输入的案例只报告预测时间与资源，不能补造性能目标。

## 模型必须保留的细节

- 93 层：69 KDA + 24 Gated MLA，第一层 Dense，其余 92 层 MoE。MLA 的 1-based 层号为 4, 8, …, 92, 93。
- h=7168、vocab=163840、96 attention heads；896 routed experts、top-16、2 shared experts；routed latent=3584、expert intermediate=3072，shared experts 输入仍为 7168。
- KDA head=128、conv=4；output gate 全秩，decay gate 低秩 128；MLA q_lora=1536、kv_lora=512、qk dimensions=128+64、v=128。
- 五个样例显式设置 `architecture.mla.cache_layout: compressed`，这是优化布局假设，保存 512 + 64 = 576 元素/token/layer 并按 TP 复制。官方 HF 参考实现实际缓存展开的 K/V；使用 `expanded` 比较该布局。NoPE 仅不施加位置编码，不移除 64 维投影。缓存选择不能替代不同 kernel 的运行时校准。
- AttnRes block=12，必须计入参数、跨层共享 block 和工作区。当前 PP>1 的 AttnRes 转发未实现时应明确报错，不绕过校验。
- 文本 backbone 参数约 2.77948T；官方约数 2.8T/104B 不直接充当张量数。此范围排除 MoonViT、projector 和训练 MTP。
- “FP8 vs FP16” 是设计对比。发布 checkpoint 是混合精度，不能声称全参数统一 FP8 就是官方存储。

需要检查缓存实现带来的差距时运行：

```text
python -m supernode_sim sweep examples/kimi_k3_ascend950_fp8.yaml --axes examples/sweep_kimi_k3_kv_layout.yaml --mode estimate --output outputs/kimi-k3-kv-layout
```

axes 文件仅改变 `model.architecture.mla.cache_layout`，其余参数一致。对长序列容量结论同时展示两种布局；配置来源与结果中保留选中的运行时假设。

## 硬件事实与假设

`profiles/hardware/ascend950dt_64.yaml` 使用 **950DT**，不混用 950PR。公开值：每芯片 144 GB HiZQ 2.0、4 TB/s 访存、950 系列 FP8 1 PFLOPS；Atlas 950 每柜 64 NPU、最大 8192 NPU。厂商新闻为发布/路线图资料，不是本项目跑出的硬件 benchmark。

仿真取 144 GB = 144e9 byte；官方新闻另有整域 1152 TB 的单位口径不一致，保留该不确定性。官方 2 TB/s 是 total interconnect，没有说明单向/双向；案例采用 1 TB/s/方向和 80% 有效效率作为显式假设。full_mesh 是有端点注入瓶颈的逻辑交换网络，不表示真实物理全互连线缆。

CPU/DDR、每 host NPU 数、H2D/D2H、SO、FP16/BF16 FLOPs、效率、runtime/workspace 都是标明的假设。128 NPU 案例人为创建两个 SU 域，不能据此声称 Atlas 950 无法跨柜 scale-up。实际 Atlas 950 最大 8192 NPU 的域规模不能被单柜样例中的 `su_size=64` 取代。

## 精度改进与校准

2026-10 的 CANN 优化实现公开了特定 32 卡、DSpark 路径的 TPOT 参考，但接受长度按 4 归一化、只统计大小模型；其 TTFT 未披露。见 [运行时证据与适用范围](kimi-k3-runtime-evidence.md)，不要把该值移植到本项目普通 decode、其他 batch/并行布局或不同规模模型。现有案例的静态容量依据仍有效，运行时精度须独立验证。

先用组件账本对齐参数和精度，再用同硬件同负载的 kernel、HBM/H2D 带宽与 collective 实测校准。至少区分计算效率、有效带宽、启动时间和 host 内存；不能用一个总体系数同时校准所有误差。

真实 K3 训练还使用 virtual PP、MoonEP 动态冗余专家、Pipeline ZeRO-2、FP8 activation offload/remote offload、AttnRes checkpoint 和 P2P Muon 正交化。通用 FSDP2/非交错 1F1B/CPU optimizer baseline 与该生产实现不同；没有端到端实测数据时，交付资料驱动的预测与敏感性，不能声称绝对误差已被校准。

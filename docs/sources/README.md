# 官方来源快照与建模边界

核验日期：2026-09-23。配置和实现通过官方页面核验后直接下载；实现文件仅作为只读资料，未执行远程代码，未下载模型权重。

## Huawei Ascend 950 / Atlas 950

| 来源 | 已核实内容 | 建模注意事项 |
|---|---|---|
| [Huawei Connect 2025：Eric Xu 官方演讲，2025-09-18](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech) | Ascend 950PR 面向 prefill/recommendation，HiBL 1.0；950DT 面向 decode/training，HiZQ 2.0：144 GB、4 TB/s memory access bandwidth；950 系列 1 PFLOPS FP8、2 PFLOPS MXFP4，2 TB/s total interconnect bandwidth | 不能把 950PR、950DT 或芯片与整机柜混用。演讲未说明 2 TB/s 是双向求和还是每方向，也未给 FP16 峰值及 dense/sparse 条件。1 TB/s/方向只能作为对总带宽保守拆分的显式假设，不能称为官方单向规格。 |
| 同上 | Atlas 950 SuperPoD：最多 8,192 个 950DT；128 计算柜 + 32 通信柜；全光互连；计划 Q4 2026 上市；UB 技术描述 2.1 microsecond latency | 此页是路线图数据，不能称为已交付实测性能；2.1 us 是系统互连技术描述，不应在每一条抽象 hop 上重复累加。 |
| [Huawei MWC 2026 官方新闻，2026-03-02](https://www.huawei.com/en/news/2026/3/mwc-superpod-ai) | Atlas 950：每柜 64 NPU，最多 8,192 NPU，UnifiedBus；首次全球亮相 | 64 NPU 示例是该架构的单柜切片，不等同于满配 8,192 NPU 超节点；此页不提供交付/吞吐实测或详细拓扑。 |

144 GB 的二进制/十进制口径在发布稿没有明确界定。2025 演讲另报满配 1,152 TB，与按十进制 `8192 * 144 GB = 1179.648 TB` 不一致。因此配置应明确采用 `144e9 byte/NPU` 的 SI 解释，保留单位不确定性；不要同时使用相互不一致的整域总量。

下列值需要用户 BOM、硬件手册或实测校准，不能从上述新闻稿推导为事实：Host DDR 容量/带宽、CPU 算力、PCIe/H2D/D2H 带宽、SO NIC 端口、有效利用率、交换机级别拓扑/超售比、collective 启动时间、kernel 时延、MTE 与缓存容量。容量、带宽和 FLOPs 需求可据明确假设求解，但 SLO 结果仍是未校准预测。

## Kimi K3 官方模型

- [官方技术博客](https://www.kimi.ai/blog/kimi-k3)：2.8T、KDA + AttnRes、Stable LatentMoE、Per-Head Muon、建议 64 个或更多加速器的超节点部署。
- [官方 GitHub 模型概要与技术报告](https://github.com/MoonshotAI/Kimi-K3)：104B 活跃参数；93 层，69 KDA + 24 Gated MLA；1 Dense 层；96 attention heads；896 routed experts、top-16、2 shared experts；MoonViT-V2 401M。
- [官方 config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)：本地 `kimi_k3_config.json`。文件历史显示初始提交 `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`。
- [官方文本模型实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)：本地 `modeling_kimi_linear.py.source.txt`，保留原始版权声明；对应许可保存为 `Kimi-K3-LICENSE.txt`。
- [官方技术报告 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)：本地 `kimi_k3_tech_report.pdf`。

### 参数与结构

| 项目 | 官方配置与实现 |
|---|---|
| backbone | hidden=7168，layers=93，vocab=163840；embedding 与 lm_head 不共享；dense intermediate=33792 |
| 调度 | `full_attn_layers=[4,8,...,92,93]` 为 1-based；其余层是 KDA；第一层 Dense，其余 92 层 MoE |
| MLA | q_lora=1536，kv_lora=512，qk_nope=128，qk_rope=64，v_head=128，heads=96；q_norm=1536，kv_norm=512；output_gate 为 7168→12288 |
| NoPE 的重要细节 | `mla_use_nope=true` 关闭位置编码操作，但官方实现仍保留 q_head_dim=192 和 kv_a 输出=576；不能因 NoPE 把这些维度减掉 |
| KDA | heads=96、head_dim=128、short_conv_kernel=4；Q/K/V 为 7168→12288；decay 为 7168→128→12288；beta 为 7168→96；output gate 为完整 7168→12288；A_log 96、dt_bias 12288、head-shared output RMSNorm 128；output 12288→7168 |
| LatentMoE | routed 输入/输出 3584、intermediate 3072；latent down/up=7168↔3584；norm 3584 位于专家加权合并后、latent up 前；shared experts 仍是全宽 7168，合并 intermediate=6144；router=7168→896，另有 896 路由纠偏参数 |
| AttnRes | 每层多两个 7168 RMSNorm 和两个 7168→1 projection；尾部另一个 7168 RMSNorm 和 7168→1 projection；block size=12；必须有显式组件或明确归属，不能遗漏 |
| KV/state | 优化 MLA 可按 576 latent elements/token/layer 缓存；官方 HF 参考代码缓存的是展开 K/V，不能把优化布局当作参考实现事实。场景以 `cache_layout: compressed` 显式选择优化假设，`expanded` 可作比较。KDA recurrent 为每序列每层 `96*128*128` 个状态元素，另有 Q/K/V short-conv state。状态 dtype 与物理 conv 缓冲槽数由 runtime 确定；不能对全部 93 层套用标准 KV 公式 |

### 精度与 Muon

官方发布权重使用 MXFP4 分组格式，group_size=32、uint8 scale。`quantization_config.ignore` 排除了 self_attn、shared_experts、Dense MLP、lm_head、vision_tower、mm_projector。不能把整模型所有组件都按 0.5 byte/parameter 处理。FP8/FP16 仿真可以作为重新指定精度的设计方案，但应明确与官方发布的混合精度 checkpoint 不同。

技术报告 §2.5（p10）说明：矩阵参数用 Muon；attention Q/K/V 的 momentum 矩阵按 head 切分，分别进行 Newton–Schulz 正交化。报告未给出训练状态 dtype、非矩阵参数 optimizer 的完整参数组、确切 scratch 内存/通信实现。默认 Muon momentum 为一份状态、辅助参数采用 Adam 时须明示策略；不能用 Adam m/v 两份状态代表所有 Muon 参数，也不能把 Per-Head Muon 当作消除矩阵 momentum 存储。

### 与生产训练的差距

官方报告 §5.2（p18–20）采用 PP+virtual stages、EP、ZeRO-1、Pipeline ZeRO-2、CP；具体内存优化包括：

- 大部分保存 activation 使用 block-wise FP8 + local/remote offload，elementwise recomputation。
- dispatch 重计算避免保存部分路由后的 activation。
- AttnRes block representation 生成一次并共享；AttnRes 整体 checkpoint；PP 增量传送新 block。
- interleaved 1F1B 将 activation 远程 offload 到其他 PP rank，平衡峰值。
- 梯度按 DP sharding 后驻留 CPU，GPU 保留 double gradient buffer。
- Muon 通过 P2P 仅获取 rank 拥有的完整矩阵，避免全参数 all-gather buffer，并按 model chunk 流水重叠。

若仿真仍采用通用 FSDP2、非交错 1F1B、固定 activation 留存系数，必须明示它是可解释的设计基准，不能声称复现上述生产训练。报告表 1 的训练模型含 1 MTP 层；发布 config 的 `num_nextn_predict_layers=0`，应以所选 checkpoint/训练对象分别建模。文本 backbone 仿真不包含 vision encoder、projector 或训练 MTP 时，容量必须注明范围。

## 快照校验

| 文件 | SHA256 |
|---|---|
| kimi_k3_config.json | `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213` |
| modeling_kimi_linear.py.source.txt | `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a` |
| kimi_k3_tech_report.pdf | `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c` |
| Kimi-K3-LICENSE.txt | `20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887` |

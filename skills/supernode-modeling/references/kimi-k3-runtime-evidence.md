# Kimi K3 / Ascend 950DT：运行时证据与适用范围

核验日期：2026-10-19。用户提供的[分享对话](https://chatgpt.com/s/t_6ad5cdf16ab88191966d0d0e7ce95075)是检索线索；下列结论按可读取原始资料核对。动态 master/main 链接不是锁定 revision，后续校准应保存实际 commit、配置与测量日志。

读取 CANN master 页面时分支 commit 为 `3b36eeac4cde5e608f00cdd0560a715ff8925e15`；技术报告页面标注的文件最后变更为 `61193a74093192e82dd8c7b02c789eb2834fd5a3`（2026-10-08）。这是页面元数据，未另读固定 commit URL。提交说明称该轮未做 NPU 性能/精度复测，不能将文档变更当成新的测试记录。

## 公开性能点：不是当前仿真的目标值

| 项目 | 核验结果 |
|---|---|
| 原始来源 | [CANN Kimi K3 推理技术报告](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/kimi_k3/kimi_k3_inference_guide.md) |
| 平台与场景 | 950DT，32 卡，KDA TP16，输入 100K、输出 256；报告用“单卡 Batch 1”描述 |
| 公开 TPOT | 28 ms → 10.3 ms；接受长度统一按 4 折算，只统计主模型与草稿模型耗时 |
| 投机设置 | DSpark，草稿长度 7；归一化分母 4 **不证明实际平均接受长度为 4** |
| TTFT | 上述性能点未披露；保持 null/未知 |
| 证据级别 | `public`，发布方模型执行结果；非本地 `measured`，非独立复测，非客户端服务延迟 |

[社区转载](https://hwcomputing.csdn.net/6ac7907ead85947e68ce8bd0.html)给出约 10 ms，并把接受长度描述为平均 4；页面署名用户且指向[微信原文](https://mp.weixin.qq.com/s/eL91UHgKm_TGxw1aOybkHQ)，本次未能读取微信正文。以 CANN 仓库对归一化的说明为准，不将转载另算一组独立实测，10 与 10.3 也不是置信区间。

这组优化涉及 CCU 通信、KDA 融合与投机状态管理、多流、图执行和低精度路径，改变了算子图与每轮输出 token 数。当前引擎的普通 autoregressive decode 不能通过 `npu.efficiency *= 2.8` 复现。更不能从 decode 改善倍数推出 TTFT。

## batch 与并行布局：必须按原配置解释

[CANN 模型 README](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/kimi_k3/README.md)介绍的默认 `kimi_k3_rank_32P_100k_1batch.yaml` 是：32 卡，Attention TP16/DP2，global batch 32，输入长度上限 102400，`prefill_mini_batch_size=1` 是每 Attention DP group 的 prefill mini-cycle 请求数。Attention/Embedding/OProj 使用 TP16，Dense/LMHead/DSpark 使用 TP8。

这些是所读版本的运行配置说明，不足以证明文章性能图使用的每个字段都完全一致；无锁定 benchmark 配置和日志时保留差异。尤其不能把文件名中的 1batch 或单卡 Batch 1 写成本项目 `workload.batch_size=1` 后声称同负载复现。100K 的具体含义应从实际配置取值，不默认按 100000 或 102400 互换。

本引擎目前使用统一 TP/PP/EP/DP 独立维度；真实框架的组件并行与共用 communicator 不能直接照抄。无法映射的配置标记不完全可比，保留已有 64 卡/1024 卡分析场景，不能为了对齐性能点擅自改为 32 卡。

README 还区分首次 decode 编译/捕获与缓存后的执行，推荐复用 compile cache 后取第二遍；其 `skip_prefill=True` 是零状态/占位 token 的性能路径，不能作为真实 prompt 的 TTFT 或文本精度证据。比较前保留 warmup/图捕获策略，不把冷启动与稳态混为一组。

## 其他资料能证明什么

| 来源 | 可用于本 skill 的证据 | 不能推出的结论 |
|---|---|---|
| [vLLM-Ascend K3 部署指南](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html) | 950DT 部署、W4A8 checkpoint、混部/PD 示例与 benchmark 方法；组件/框架配置需单独对齐 | 与 CANN 优化实现等价、当前配置必定达到 10.3 ms |
| [v0.26.0rc1 测试报告](https://github.com/vllm-project/vllm-ascend/blob/releases/v0.26.0rc/tests/vllm_ascend_v0.26.0rc1_test_conclusion.md) | 列出 K3/950 的验证范围、不同长输入和 prefix cache 场景；该页未列 K3 TTFT/TPOT 数值 | 性能数值已公开，或所有不同平台/版本已消除同样的问题 |
| 同一 rc1 报告的 known issue | 某些长输入场景 DSpark 接受率低，投机收益可能消失 | 任意输入/并发下接受长度恒为 4，或所有后续版本仍有该问题 |

对 vLLM 部署指南中 EP 与 Attention TP/DP 的表达，先查框架 group 含义，不按本引擎独立维度直接相乘。

## 在仿真任务中使用这些证据

先保留本引擎的原始预测，再列出参考值、计时边界、batch、组件并行、精度、缓存布局、融合/图模式、投机等差异。这里的数值只能作为**特定软件路径的外部参考**；不得自动写成 `hardware.kernel_profiles`、硬件峰值或 SLO，不能称“已校准为真实 10 ms”。

目前的 [投机 feature](features.md)提供 draft、target verification、接受提交、快照/回滚与显式 replay 成本的轮次模型。其接受数和 draft 成本仍需来自同负载测量或明确假设，不能直接拿此处 10.3 ms 校准。decode 阶段按整段实际产出计算 `sum(round_time) / sum(committed_output_tokens)`，并明确首 token 与统计窗口。额外核对 bonus token、EOS 截断及客户端 chunk 输出方式，不能简单将普通 TPOT 除以草稿长度 7 或常数 4。

TTFT 仍需独立测量 prefill、调度、首 token 以及选定边界内的前后处理。拿到同条件 kernel/request trace 后，按 [运行时校准流程](runtime-calibration.md) 更新支持的参数，并在留出负载点验证；没有原始数据时仍保留 `uncalibrated`/部分 kernel 校准状态。

# 交互式仿真示意

当前 skill 与引擎的架构、能力边界和函数流程见 [整体架构图](../simulation-architecture.md) 与 [18 张函数级流程图](../simulation-function-flows.md)。

本目录保存可离线打开的 HTML 动画。前四个历史页面来自本地 `outputs/flow-animation/`；新算子动画由当前引擎生成，便于随代码一起版本管理。

| 文件 | 内容 |
| --- | --- |
| [inference-timing-functions.html](inference-timing-functions.html) | 推理时延讲解，标注实际函数与实验保存版本的源码位置 |
| [inference-timing-explained.html](inference-timing-explained.html) | Kimi / DeepSeek 历史案例的推理时延分解 |
| [supernode-case-film.html](supernode-case-film.html) | 一批请求穿过超节点的完整案例与原理拆解 |
| [supernode-simulation-flow.html](supernode-simulation-flow.html) | 从场景输入到仿真分析与决策的全流程动画 |
| [deepseek-v41-operator-simulation.html](deepseek-v41-operator-simulation.html) | V4.1 Pro 10T：64 层结构、四阶段算子回放、精确 trace 展开、资源时钟与 TTFT / TPOT |

下载仓库后用浏览器打开 HTML 文件；GitHub 文件页只显示源码。页面脚本和案例数据均内嵌，无需额外 JSON 或仿真服务。文件采用 UTF-8；历史页面为可嵌入片段，新算子动画为完整独立文档。

时延示例展示已有仿真记录，不代表硬件实测；函数位置对应实验保存版本，可能与后续代码版本存在差异。

## V4.1 算子动画

新页面沿用此前推演的 synthetic V4.1 Pro 10T 结构，使用当前内置 FP4 / DFlash 案例默认配置：512 卡、TP4 / EP128 / SP1、128K 输入、Prefill batch 1、Decode 总 batch 32、block 8 / 净接受 6。可播放或拖动两分钟讲解，切换 Prefill / Draft / Verify / Commit，点击层号或算子查看阶段内时间、依赖和资源需求；支持导出内嵌 JSON。

页面明确保留两个执行范围：整网四阶段由案例适配器交给当前 `engine.simulate()` 执行；新版 `kernel_traces` 在同一模型 L0 共享专家的独立单 rank 微图中实际运行。该案例适配器尚未自动调用通用 trace。微图将三矩阵预算展开为 up / gate / down，保留原 ID 完成点与后继依赖，不改变整网指标，也不宣称重建真实融合 kernel 内部 DAG。三项各加 1 μs 启动与 256 KiB 局部工作区是演示假设；FLOPs 和原 HBM 访问预算守恒。SwiGLU 仍是原图中的独立 FP32 预算。无实测 kernel profile。

在项目根目录重新生成：

```powershell
.\.venv\Scripts\python.exe scripts/render_deepseek_v41_operator_animation.py
```

生成器读取当前模型、运行完整四阶段、运行两个微图并核验时钟、依赖、容量释放、形状不匹配回退和工作量守恒。它还从 AST 提取当前函数行号。数值与来源记录保存到 `outputs/deepseek-v41-operator-animation/`；HTML 模板位于 `scripts/templates/deepseek-v41-operator-animation.html`。讲解动效不按真实时间等速播放，资源需求条及结果归因条单独注明比例口径。

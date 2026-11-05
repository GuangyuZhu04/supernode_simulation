# 交互式仿真示意

当前 skill 与引擎的架构、能力边界和函数流程见 [整体架构图](../simulation-architecture.md) 与 [18 张函数级流程图](../simulation-function-flows.md)。

本目录保存此前更新的四个 HTML 示意文件，原文件来自本地 `outputs/flow-animation/`；保留原始页面内容，便于随代码一起版本管理。

| 文件 | 内容 |
| --- | --- |
| [inference-timing-functions.html](inference-timing-functions.html) | 推理时延讲解，标注实际函数与实验保存版本的源码位置 |
| [inference-timing-explained.html](inference-timing-explained.html) | Kimi / DeepSeek 历史案例的推理时延分解 |
| [supernode-case-film.html](supernode-case-film.html) | 一批请求穿过超节点的完整案例与原理拆解 |
| [supernode-simulation-flow.html](supernode-simulation-flow.html) | 从场景输入到仿真分析与决策的全流程动画 |

下载仓库后用浏览器打开 HTML 文件；GitHub 文件页只显示源码。页面脚本和案例数据均内嵌，无需额外 JSON 或仿真服务。文件采用 UTF-8，原始内容为可嵌入的 HTML 片段。

时延示例展示已有仿真记录，不代表硬件实测；函数位置对应实验保存版本，可能与后续代码版本存在差异。

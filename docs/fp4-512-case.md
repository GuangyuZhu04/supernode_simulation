# FP4 512 卡 / EP128 前端案例

工作台左侧的 **FP4 案例** 或模型预设下的案例链接进入 `/fp4-case.html`。默认是 **950DT 96 GB/卡**，可切换 **950PR 128 GB/卡**。

该案例来自 `outputs/deepseek-v41-pro-10t-pod-dflash-fp4-20261102` 的已完成实验。数值模块及必需的结构/硬件配置已迁入 `supernode_sim/case_studies/v41_fp4/`，安装包包含它们，运行时不依赖原 `outputs/`。原 FP4 / FP8 实验产物保持原样。

默认配置是 512 卡、单个 Full mesh SU POD、TP4 / EP128 / DP1 / SP1，128K（131072）上下文、Prefill batch 1、Decode 总 batch 32、DFlash 净接受 6 / 块长 8。页面支持原实验的 TP1/2/4、SP1/TP、10 档上下文、6 档 Decode batch 和 4 组接受数/块长；DP 由总卡数推导，总 Decode batch 保持固定。

仅路由专家 gate/up/down 权重采用 FP4，末轴每 32 元素计 1 byte scale，显式计入非融合 FP4→FP8 转换。其余模型精度、DFlash 草稿、硬件效率及资源假设沿用原实验。模型为 synthetic V4.1 Pro 10T 外推，不是官方 10T checkpoint，也没有实机精度或质量校准。

TTFT 包含目标 prefill 和 DFlash 上下文 KV 初始化。TPOT 为固定上下文一轮 `draft + verify + commit` 耗时除以净接受 token 数，不是长输出平均 ITL 或服务排队延迟。各阶段调用项目原生调度器，采用原实验的串行阶段与资源包络；不把不同 batch 的 prefill 和 decode 拼成同一请求时间线。容量失败保留阶段诊断与空时延，不填零。

案例输入是独立的 `CaseConfig`，普通场景的 `Config` 不能完整表达此实验的 CED/CSA2/Engram 与分阶段 batch，因此网页明确使用 `pod_dflash` 模式。它共用现有任务队列、取消、输入快照、历史恢复和文件下载；案例历史在案例页面查看，普通历史只显示普通场景。

API：`GET /api/case-presets` 获取两套预设；`POST /api/validate` 和 `POST /api/jobs` 使用 `{"mode":"pod_dflash","config":{...}}`；任务状态、结果和下载仍使用 `/api/jobs/<id>`。同一服务仍只运行一个任务。

可直接复跑：

```powershell
.\.venv\Scripts\python.exe -m supernode_sim.case_studies.v41_fp4 examples/deepseek_v41_fp4_512_ep128_950dt.json -o outputs/fp4-512-950dt
.\.venv\Scripts\python.exe -m supernode_sim.case_studies.v41_fp4 examples/deepseek_v41_fp4_512_ep128_950pr.json -o outputs/fp4-512-950pr
```

结果保存完整 `result.json`（含所有阶段算子）、`config.resolved.json`、`phases.csv`、`latency_breakdown.csv`、`provenance.json` 和中文 `report.md`。每次结果记录源实验指纹及当前案例/引擎源码哈希，方便审计。

`tests/fixtures/v41_fp4_512_reference.json` 保存原实验两个默认案例的指标及阶段检查点。回归检查比较完整阶段的时延、拆解、物理工作量、内存、事件数，并测试真实网页 worker、不可变快照、重启恢复与 4M 容量失败：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_web_fp4_case.py tests/test_web.py tests/test_web_serving.py -q
```

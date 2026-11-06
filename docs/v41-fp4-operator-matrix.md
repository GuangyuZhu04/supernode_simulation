# V4.1 FP4 / POD 算子级矩阵

入口为 `scripts/run_v41_fine_matrix.py`，使用当前 `supernode_sim.operator_trace.apply_kernel_traces` 和原生事件引擎。已有 `CaseConfig` 的固定 512 卡 / EP128 前端案例及其历史数值不变；此矩阵扩展走独立 CLI。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_v41_fine.py tests/test_operator_trace.py tests/test_web_fp4_case.py -q
.venv/Scripts/python.exe scripts/run_v41_fine_matrix.py --workers 4
.venv/Scripts/python.exe scripts/report_v41_fine_matrix.py
```

默认输出位于 `outputs/deepseek-v41-fp4-operator-matrix-20261105/`。每个阶段写入带源码指纹的 SQLite 检查点，重新运行会复用匹配结果。`--prepare-only` 仅生成实验配置；`--export-only` 要求所有当前指纹的检查点已经完成。worker 数仅改变主机仿真并行度。

扫描 10 档上下文、6 档 Decode 总 batch、4 组 DFlash、2 款芯片、8 档 EP、6 档卡数。沿用原实验 Prefill batch1、TP=1/2/4/8、SP=1 或 TP、DP=cards/(TP×EP)；无合法布局和容量失败均保留。

`fine.py` 从原张量清单展开投影矩阵及 grouped expert gate/up/down，并调整 MLP 内 activation/down/combine 的顺序。DFlash Q/K/V 与 gate/up 分离，增加独立读取所需的 HBM 访问。命名注意力 tile、mHC 和逐点融合组仍是声明的 kernel 边界，不是实测可执行文件。精确形状、工作量、阶段、并行度、硬件身份均参与 trace 匹配。

启动参数位于 `supernode_sim/case_studies/v41_fp4/operator_launch_assumptions.json`，是明确的工程敏感性假设，不是 950 的实测 profile。不同算子使用不同参数；0.5×/2× 敏感性只改变算子启动。通信启动继续按原 POD 调度收费。

三项拆解口径为计算（含算子启动）+访存+通信；计算服务与算子启动另有独立列。各算子的计算/HBM roofline 重叠取最大项，不能把两个理论需求时长再相加。Rank 包络、barrier、静态内存保留与拓扑假设均继承原研究，未加入服务排队或物理多流网络竞争。

主表在两阶段同时可行的布局中按 TPOT 选择。另存最优 Prefill 与最优 Decode 表，两表的最优布局可能不同，不能拼接成同一部署的两项成绩。Decode 单独可行不代表 Prefill 或端到端请求可行。

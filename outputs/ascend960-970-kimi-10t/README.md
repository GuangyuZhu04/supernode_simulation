# Kimi 10.171T：Ascend 960DT / 970 推理仿真

本目录保存 288 组固定 batch 实验：两种硬件 × TP8/EP128、TP16/SP16/EP64、TP32/SP32/EP32 × top-32/top-64 × 128k/256k/512k/1M × batch 1/2/4/8/16/32。以 report_status.json 的完成数量为准；运行期间的审计和报告属于阶段快照。

模型是原 Kimi K3 文本骨干的合成专家扩容：3328 个专家、93 层、10.171385035872T 参数。FP8 权重/计算，FP16 激活/通信/压缩 MLA KV，KDA recurrent state 为 FP32。Megatron SP 与 TP 共用 rank，不额外乘入卡数；它切分部分激活并执行 AG/RS，不是 context parallel，KV 不按上下文切分。所有方案使用 1024 张卡、PP=DP=1。

128k/256k/512k/1M 分别是 131072/262144/524288/1048576 token。output_length=2 测量预加载权重后的 TTFT 与首个 decode 间隔 TPOT，未建模请求排队、冷启动和连续批处理。prefill 不可行时保留失败；另跑的 standalone decode 预置持久 KV/KDA 状态，其 TTFT 留空，不能据此宣称端到端可行。

硬件使用截至 2026-10-15 查得的 2026 路线图：960DT 为 FP8 2 PFLOPS、288 GB HBM、9.6 TB/s；970 为 3.6 PFLOPS、288 GB、14.4 TB/s。公开芯片互联总量分别为 2.2/4.4 TB/s，方向未说明，仿真按其一半设置单向注入；2 μs RTT 按 1 μs 单向逻辑路径处理。计算效率 0.5、网络效率 0.8、FP16/FP32 等未核实速率继续沿用旧基线，详见 [来源与假设](profiles/provenance.json)。这些是未实测校准的预测。

拓扑选取 Atlas 960E 公布的 4096-NPU UnifiedBus/NPO 结构中的 1024 卡可比切片，抽象为一个逻辑无阻塞 scale-up 域。970 继承该拓扑是已确认的代理假设，尚无独立公开 A7 物理图。8 卡/主机和 64 卡统计组沿用实验分组，不表示真实机箱 BOM。跨 64 卡组仍有流量，但属于 SU；SO 为零不等于跨机柜流量为零。未补造物理交换路径、超卖率和交换机二分带宽。与原 950 结果比较时，硬件和 SU/SO 边界同时变化，不能把全部收益归因于芯片算力。

冻结源与输入：

- 主运行时：[runtime-synchronous-rounds-manifest.json](code/runtime-synchronous-rounds-manifest.json)，ID sha256:78919546d7eb336972226e9f71cc396718eafeafc5999327c398eeb025c40620。
- 原逐传输运行时：[runtime-manifest.json](code/runtime-manifest.json)，ID sha256:447b4248c25385c9ca971fb1b070aa195ff2397136f5b75b48881bd615ee3645；原生参考结果保存在 audit/native-reference-cases/。
- 输入 ID sha256:ce1202280337bedc62489a0f01b1b444c8677493eacedbeb86c313c7057288c7；[input-manifest.json](input-manifest.json) 包含 580 个文件 SHA，[manifest.json](manifest.json) 绑定全部场景、运行器和结果指纹。

同步轮次版本把具有相同进度的匹配通信流合并为一个调度事件，继续调用原离散事件求解器。严格的模型、精度、内存策略、拓扑和带宽条件保证资源工作量、缓冲峰值、流量与时序一致；不满足时回退。全部 288 配置已通过条件核验，两个原生参考覆盖小 batch 与最大 SP 案例。事件数量因表示压缩而变化，不能用来推断流量。[数学条件和逐项数值验证](audit/synchronous-rounds/README.md) · [来源和指纹审计](audit/final-reproducibility-audit.json)。

在仓库根目录、现有依赖环境中续跑；匹配的完整结果会复用，不需要重新 prepare：

~~~powershell
.\.venv\Scripts\python.exe outputs/ascend960-970-kimi-10t/code/run_experiment_rounds.py --run --cache-mib 64
.\.venv\Scripts\python.exe outputs/ascend960-970-kimi-10t/code/summarize_experiment.py
~~~

建议最多使用 3 个 worker，每进程 --cache-mib 128，并为同一运行器设置 --shards 3 和不同 --shard-index 0/1/2；内存紧张时使用上述串行命令。每进程分别占用图和缓存内存，缓存预算不是进程 RSS 上限。数值复现不要求使用原进程数量或缓存预算。核验工具 verify_synchronous_eligibility.py 只检查条件；benchmark_synchronous_rounds.py 从永久原生参考目录重放比较，会额外执行仿真。

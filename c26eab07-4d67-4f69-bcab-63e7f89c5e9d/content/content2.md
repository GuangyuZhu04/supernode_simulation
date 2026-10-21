# 10T 推理需同时满足显存峰值、HBM 带宽与低时延互联
<!-- 逻辑：方案；版式：V4三问，三个资源列；共4页无封面目录 -->
副标题：02 集群要求 · 以下为已仿真的部署条件，未搜索最小卡数
核心思想：1024卡采用 TP32×EP32，SP32共享TP ranks；PP=DP=1。

## 显存：按物理副本与阶段峰值预算
1M、batch=1、top-32、完整优化F：逻辑FP8权重10.171TB；集群物理权重+scale 19.877TB；每卡完整运行峰值61.436GB。含runtime8GB与workspace256MB。
不能用10.171TB/1024卡替代真实峰值。
## 计算和访存：prefill与decode分别设计
950DT / 960DT 每卡HBM 144 / 288GB，HBM带宽4 / 9.6TB/s；FP8标称峰值1 / 2PFLOP/s，计算效率假设50%。
F、1M、batch1：prefill计算1.611EFLOP；decode集群累计HBM读写8.338TB/步；读写量不是容量。
## 互联：专家分发需低启动时延
1024卡单逻辑非阻塞SU，SO=0；单向带宽假设1 / 1.1TB/s，效率80%；逻辑路径启动/传播2.1 / 1μs。
F、1M、batch1：prefill载荷772.868TB；decode载荷3.314GB/步；decode关键路径通信启动占约86% / 87%。

完整优化F：SP32 + MLA compressed + FP8权重/计算/激活/通信/KV；KDA递归状态FP32。
本页亮点：SP与压缩/量化共同控制峰值；decode首先关注collective启动与HBM服务时间。
页内边界：硬件路线图与效率、拓扑假设，未实机校准；上述容量与流量限本页负载。
来源：最新report.md、summary.csv、phase_metrics.csv、latency_breakdown.csv与配置。

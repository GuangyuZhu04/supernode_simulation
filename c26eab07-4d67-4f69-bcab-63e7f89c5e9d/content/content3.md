# 仿真 Skill 将配置编译为资源事件图，评估容量与时延
<!-- 逻辑：方案；版式：V4三问，三个模块+反馈闭环；共4页无封面目录 -->
副标题：03 仿真方法 · supernode-modeling 编排，supernode_sim 执行
核心思想：Skill负责配置与解释，Python引擎执行可复现的数值计算。

## 1 输入与严格校验
模型结构、精度、负载、并行策略、SU/SO拓扑与SLO。
标注public / user / assumed / measured来源；检查并行乘积、专家整除、拓扑连通与特性兼容。
拓扑子skill将自然语言编译为原生链路图。
## 2 编译与资源竞争
逐组件张量账本：权重、KV/KDA状态、激活、工作区；训练另计梯度/优化器。
计算/访存/collective事件形成DAG，映射NPU、HBM、链路及注入资源。
先做解析下界，再做离散事件调度：roofline重叠、max-min资源共享、内存准入与释放。
本次结构化流式执行复用相同数值语义，保留阶段资源与峰值。
## 3 输出与反馈
TTFT、TPOT、吞吐、逐阶段通信/计算、单卡峰值、容量失败与瓶颈。
输出解析后配置、CSV/JSON和报告；冻结代码指纹以支持复跑。
调节硬件、并行与加速特性，扫描候选，再用实机kernel/collective校准。

本页亮点：同一事件图统一计算、存储与网络；预测带来源和约束，可复现、可校准。
页内边界：固定batch，无在线排队/生产P99；本次无实机kernel校准；下界不等于可行性证明。
来源：skills/supernode-modeling/SKILL.md；docs/simulation-architecture.md；最新report.md。

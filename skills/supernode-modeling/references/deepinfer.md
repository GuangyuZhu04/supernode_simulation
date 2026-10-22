# DeepInfer：阶段部署、batch 与推理加速 feature

用于 P/D 选型、Wide-EP、batch 扫描和 DeepInfer 优化方案分析。先读 [features.md](features.md) 的接口契约。本页是现有 feature 的组合方法及扩展边界，不新增 `deepinfer: true` 一类引擎开关。

## 来源与适用范围

核对日期：**2026-10-21**。依据 DeepSeek-AI 官方 DeepInfer 仓库的部署 [Q&A][qa]、[README][readme]、[benchmark 说明][bench] 和 [sweep 脚本][sweep]；固定版本为 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`（2026-10-20）。Q&A 是官方[推理系统文章][article]的配套部署说明。后续更新需重新核对版本，不能把这里的快照称为永远最新。

这些资料对应 DeepSeek-V4.1-Flash 的实现和硬件条件。机制可用于其他 MoE 模型的选型，模型结构、容量、kernel 耗时和接受率必须使用目标模型资料。官方推荐值、本项目扫描候选、实测数据分别记录，不能把三者混写。

## P 小组、D 大组与独立 batch

| 维度 | Prefill（P） | Decode（D） | 来源/解释 |
| --- | --- | --- | --- |
| EP | 2、4、8 起步，在 HBM 允许时优先较小组 | 64、128 起步，在设备和通信条件允许时优先较大组 | Q&A 的阶段部署建议；实例数和每实例卡数是两个变量 |
| Attention 并行 | 官方基线 TP1、无 CP，Attention DP 与 EP 同规模 | 同样使用 Attention DP 与 Wide-EP | 官方当前生产路径；README 中有 balanced CP 能力不等于生产启用 CP |
| 小 SP+EP 候选 | TP2/SP2 或 TP4/SP4，搭配小 EP | 优先 TP1/SP 关闭；需要 TP 才能容纳模型时再扫描 | 用户提出的本项目候选；本引擎 SP 与 TP 共用 ranks，不独立乘卡数，也不是 CP |
| 请求 batch | 优先 `bs=1` | 优先 `bs=128,256`；用 32/64 等作为延迟对照 | 用户指定的扫描起点；官方 decode sweep 包含 128/256，不是所有生产场景的固定最优值 |
| 优化目标 | 满足 TTFT、峰值 HBM，比较输入 tokens/s 和单位卡效率 | 在 TPOT 约束内比较有效输出 tokens/s、峰值 HBM 和通信 | 大 batch 增加权重复用，也可能增加步延迟、KV 读取和容量压力 |

P 的请求 bs=1 不意味着 GEMM 的 token 维度为 1：一个请求的长 prompt 可提供大量 token。官方 `--prefill_batch_size` 是每步 **token budget**（默认 32,768），`--decode_num_requests` 才是请求数。不要把 `prefill_batch_size=1` 当作单请求 P 配置。当前引擎没有动态 chunk 调度，`input_length` 也不能直接替代 chunk budget。

默认把本页候选的 bs 标为**每 Attention replica 的请求数**，以便对齐官方 CP=1 的 decode benchmark。若用户场景中的 bs 指整个实例/池，保留该口径并改用下表换算；不要默默乘 EP。

| 名称 | 本引擎口径与换算（单实例、PP=DP=1） |
| --- | --- |
| `bs_per_attention_replica` | 概念量，不是 schema 字段。EP lanes 分配请求，同 lane 的 TP ranks 共同处理这些请求 |
| `workload.batch_size` | 一个完整 DP 副本内、所有 EP lanes 的请求总数。等量分配时为 `bs_per_attention_replica × EP`；总是整数 |
| EP64、D bs=128/256（每 replica） | `workload.batch_size=8192/16384`；不是 128/256 |
| EP64、D bs=128/256（整个实例） | `workload.batch_size=128/256`，每 Attention replica 为 2/4 |
| EP2、P bs=1（每 replica） | `workload.batch_size=2`；若整个 P 实例只有一条请求，则填 1，其他 EP lanes 没有独立 attention 请求 |
| 官方 Attention DP=EP | 本引擎通过 EP lanes 的请求归属表达该关系；`execution.dp` 是额外完整模型副本，单实例保持 1，避免把卡数放大为 EP² |

先固定模型、长度分布、精度、SLO 和 batch 口径，再扫描阶段布局。每个候选重新校验 `num_experts % EP == 0`、heads/TP、实际 rank 放置与拓扑。大 EP 优先放在同一 SU 域；跨 SU 时计算实际 dispatch/combine、注入和共享上联成本。当前均匀路由结果不覆盖热点或 EPLB 效果。

P/D 池大小独立规划：`N_total = replicas_P × TP_P × EP_P + replicas_D × TP_D × EP_D`（PP=1）。P 单实例小，不代表整个 P 池总卡数很少。按阶段服务能力和到达负载确定两侧副本数；不能从 EP 比例直接决定 P:D 卡数比，也不能把独立阶段容量预测称为集群 goodput。

## 当前可执行的阶段实验

现有 `prefill_decode_disaggregation` 只支持相同 TP/EP 和同一固定 batch；不会执行异构布局重分片、多个 P 请求批汇聚为 D 大 batch、在线排队或跨 batch 流水。需要上述非对称配置时，保存 **两个独立阶段场景**，用 `run_streaming` 的 `phase` 选择执行阶段；不要在 PD feature 内填写不存在的 `prefill_ep`、`decode_batch_size` 等字段。

仓库示例 `examples/features/synthetic-deepinfer-stages.yaml` 是一层小型 GQA+MoE、128 专家和假设硬件，仅用于验证配置与阶段调用。以下 Python 代码在仓库根目录运行，生成 P 小 SP/EP 与 D EP64/128、bs128/256 的 canonical 配置。`RUN=False` 默认只准备和校验；需要运行时显式改为 `True`。这段程序仅修改该 synthetic 示例的设备池，真实硬件的拓扑应按 [topology.md](topology.md) 独立构造，不能照抄自动扩域。

```python
from copy import deepcopy
import json
from pathlib import Path

from supernode_sim.config import Config, load_config
from supernode_sim.streaming import run_streaming

RUN = False
out = Path('outputs/features/deepinfer-stages')
out.mkdir(parents=True, exist_ok=True)
base = load_config('examples/features/synthetic-deepinfer-stages.yaml').model_dump()
# bs is requests per Attention replica; all model/hardware values are synthetic.
cases = [('prefill', tp, ep, 1) for tp in (1, 2, 4) for ep in (2, 4, 8)]
cases += [('decode', 1, ep, bs) for ep in (64, 128) for bs in (128, 256)]
for phase, tp, ep, bs in cases:
    data = deepcopy(base)
    data['execution'].update(tp=tp, ep=ep, sequence_parallel=(tp > 1))
    cards = tp * ep
    data['hardware'].update(npu_count=cards, su_size=cards,
                            npus_per_host=min(cards, 8))
    data['workload'].update(batch_size=bs * ep,
                           input_lengths=[], output_lengths=[])
    cfg = Config.model_validate(data)
    name = f'{phase}-tp{tp}-ep{ep}-bs{bs}-per-replica'
    (out / f'{name}.json').write_text(
        json.dumps(cfg.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
    if RUN:
        result = run_streaming(cfg, phase=phase)
        (out / f'{name}.result.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
```

真实任务中的输入/输出长度必须保持用户分布；改变 batch 时同步重建逐序列列表。示例 `output_length=2` 只产生一个 decode 间隔，不代表稳态。D 的 `phase='decode'` 预置 KV/state，TTFT 为 null，结果不含 P→D 状态传输或汇聚等待；P 的 TTFT 截止模型侧首 token。分别报告 `metrics`、`phase_metrics`、HBM 峰值、通信与来源。

需要阶段吞吐时，对可行结果用实际执行 token 数除以该阶段耗时：P 使用输入 token 数；普通 D 只计 `sum(output_lengths - 1)`，不把 P 已生成的首 token 算进 D。按实际池卡数归一化 tokens/s/card。只有同一请求流的交接、重分片、队列和资源争用建模完整，才能给出端到端结论。

## 其他加速 feature 如何纳入分析

以下“部分/未建模”是本仿真器边界，与 DeepInfer 是否开源该实现是两个问题。记录方案中的特性及其成本输入；没有对应模型时列为扩展需求，不写入当前严格 schema。

| 特性 | 仿真应保留的关键成本/条件 | 现有入口与支持范围 |
| --- | --- | --- |
| Wide-EP | 专家分片驻留、每 rank 请求数、实际触达专家的权重读取、dispatch/combine 和网络争用 | `parallelism.ep`、`workload.batch_size`、拓扑：已有解析模型，路由均匀假设需注明 |
| 按组件低精度 | 权重/计算/KV/通信分别选精度，保留 scale、转换和额外副本 | `precision.routed_expert_weight: fp4` 可表达 routed experts 的低精度驻留；FP4 KV、不同 KV 区域混合格式尚不支持，不能把所有 KV 改成 fp4 |
| Mega MoE 与融合通信计算 | dispatch→expert GEMM→combine 的真实依赖、通信进度、临时缓存和融合耗时 | `megakernel` 仅支持合法局部串行链；不能跨 collective。具体 kernel 可用 profile 校准，完整 Mega MoE 需扩展 |
| DSpark 置信度驱动变长验证 | draft 成本、每请求每位置接受概率、验证预算、commit/replay、有效提交 token | 现有 `speculative_decode` 是固定预算与显式接受数；Confidence Head、跨请求预算排序和变长 verify 未建模。不能把少验 token 等同少生成 draft |
| EPLB 与冗余专家 | 热点专家副本容量、路由负载 trace、统计窗口、重分配周期与权重迁移流量 | 动态 EPLB 未实现。不能增加逻辑 `num_experts` 冒充冗余副本，也不能把均匀路由称为已模拟 EPLB |
| CPU/GPU 异步调度 | CPU schedule/post-process、GPU forward/sample、同步等待及跨步重叠 | 内核 profile 只覆盖匹配事件；尚无 CPU serving 调度流水，不能通过把 startup 置零声称已实现零开销调度 |
| 前缀 KV 复用、外部存储、异步拉取/GDR | 命中匹配、存储容量/带宽/IOPS、网络、H2D、HBM 驻留和逐层依赖 | `kv_offload` 是静态放置与传输，不代表 prefix cache/3FS/GDR。官方 D 保持在线 KV 在 HBM，持久化复用与 decode offload 分开讨论 |
| SWA bounded replay / YOCO | 模型支持的 encoder/consumer 层依赖、命中前缀、有限重放窗口和质量条件 | 未实现完整路径；不能缩短整个 `input_length` 或给全部 prefill 乘折扣来模拟。普通 SWA/CSA/HCA 状态不自动具备该能力 |
| P 动态 chunk 与 D 请求均衡 | P 按上下文/计算预算分配 chunk；D 同时均衡请求数和累计 KV 长度 | 固定 batch 无动态调度。输入长度列表能表达静态负载，不能替代到达 trace、准入和路由策略 |

上述机制来源见 [Q&A][qa] 的对应章节。EPLB 校准可从报告中的 P 每卡额外 4 专家、D 每卡额外 1 专家出发，记录整除补齐后的物理副本数以及统计/迁移开销；它们是目标实现的参考值，不是当前仿真支持项。

投机解码校准时，官方 benchmark 的满接受吞吐使用 `steps/s × requests × (1 + nextn)`（CP=1）。引用有效输出吞吐必须使用实际提交数/接受口径，不把这个满接受指标当作真实 acceptance。[Q&A][qa] 的动态验证收益依赖并发和硬件，低并发点也可能退化；不设置固定提速倍率。保留本引擎 `accepted_lengths` 包含 bonus/replacement token 的定义。

组合分析先建立普通 decode 基线，再按目标实现引入低精度、kernel、投机等成本，并检查现有组合限制。投机可使用普通 `simulate` 或 `run_streaming` API，但仍须显式 draft/acceptance 成本并检查批次范围；megakernel 要走普通 `simulate`，当前 PD 也不支持同时开启投机。参数看似可校验不等于完整复现 DeepInfer serving。

[qa]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md
[readme]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md
[bench]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/README.md
[sweep]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/sweep/run_sweep.sh
[article]: https://zhuanlan.zhihu.com/p/2095560295930508953

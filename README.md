# 超节点通信与资源需求建模

`supernode_sim` 是面向架构分析的 Python 数值引擎。`supernode-modeling` skill 负责收集配置与解释结果；公式、通信映射和时序由可重跑代码执行。

项目设计与实现：[整体架构设计文档](docs/simulation-architecture.md) · [函数级仿真流程图](docs/simulation-function-flows.md)。

支持训练与固定负载推理、FP8/FP16 权重、FSDP2/CPU optimizer offload、MoE、标准与 latent KV、KV offload、Engram 查询、SU/SO 拓扑和有限候选搜索。Demo 与 10T 容量锚点是明确标注的合成配置；Kimi K3 / Ascend 950DT 案例使用公开结构和硬件资料，并分别标明运行时、拓扑及效率假设，不将型号名称当作实测校准。

训练与推理可通过 `hardware.kernel_traces` 将匹配的组件展开为串行子算子，显式输入 GEMM M/N/K、启动、HBM 访问和局部工作区，再使用现有 `kernel_profiles` 校准。结构化事件新增 `workload_shape`，区分 prefill 长度分布、decode 上下文及训练微批；预算训练的因果 attention 计算量也已与推理口径对齐。[参考仿真器映射](skills/supernode-modeling/references/reference-simulators.md) 和 [输入/复现说明](skills/supernode-modeling/references/operator-traces.md) 说明来源与边界；细化本身不证明实机预测误差下降。

```powershell
python skills/supernode-modeling/scripts/evaluate_operator_traces.py --output outputs/operator-traces
```

该实验生成训练/推理的前后配置、结果和流式一致性比较，全部为 synthetic，不用于硬件准确率声明。

按用户指定的 [msModeling/ServingCast](skills/supernode-modeling/references/msmodeling-servingcast.md) 与 [湛卢训练参考](skills/supernode-modeling/references/zhanlu-training.md) 继续细化：新增 `serve` 请求到达、FCFS 成批与隔离多实例调度；训练新增 `metrics.training_accounting`，分开模型 F+B、重算、optimizer、辅助工作和各精度峰值归一指标。来源已固定到具体 revision，未照搬 mock 硬件参数或论文误差。服务能力及局限见 [serving.md](skills/supernode-modeling/references/serving.md)。

```powershell
python -m supernode_sim serve examples/serving_synthetic_model.yaml --serving examples/serving_synthetic_requests.yaml -o outputs/serving-example
python skills/supernode-modeling/scripts/evaluate_reference_simulators.py --output outputs/reference-simulators
```

第二条命令复跑 checkpoint 开关与 1/2 个隔离实例的 synthetic 对比。增加实例同时增加总硬件，不代表同资源加速；服务 P99 是指定输入和 FCFS 策略的有限样本分位数。

## 仓库内容与本地产物

`main` 保留仿真引擎与网页源码、`skills/` 中的 skill 代码和文档、运行/汇总脚本及 HTML 报告模板、示例配置、硬件配置、测试、架构说明和 `docs/sources/` 中的参考资料。

仿真结果（JSON/CSV、报告、图表、检查点、日志及压缩包）统一写入 `outputs/`；PPT 文件、内容稿、预览和制作中间材料放在 `outputs/presentations/`。这些本地产物不提交到 Git。`.gitignore` 同时排除旧的根目录 UUID 演示工作目录、独立 PowerPoint 文件和 `docs/*-ppt-*.md` 内容稿，取消跟踪时保留本地文件。

## 网页仿真工作台

在项目根目录启动本地网页，无需安装 Node.js 或前端依赖：

```powershell
.\.venv\Scripts\python.exe -m supernode_sim web
```

浏览器打开 **http://127.0.0.1:8765**。选择训练或推理预设，调整负载、并行布局、硬件、精度及 offload 选项，校验后点击开始运行。网页调用现有 Python 引擎，展示真实运行阶段、通信流量、容量、资源使用和事件时间线，并提供历史任务及报告下载。完整配置可通过高级 JSON 编辑器修改或导入。

新版网页提供“请求级服务”模式，支持独立请求表/JSON、隔离实例与成批参数、服务 SLO、双快照恢复、请求分位数和相对批次时间线。训练结果新增 FLOPs/MFU 账本，普通与服务结果均可查看实际算子展开和计时校准覆盖。所有指标来自原引擎；页面区分任务完成、请求失败和 SLO 未达标。详见 [网页使用说明](docs/web-interface.md)。

左侧“输入”页提供模型、拓扑、硬件、并行度和 Workload 的完整总览与编辑。修改同步到当前实验配置，统一校验后可直接运行；历史输入可以复制为新的可编辑配置，原任务快照保留。

也可运行 `snmodel web` 或 `python -m supernode_sim.web`；`--port 8766` 修改端口，`--output outputs/web` 指定任务保存目录。详细操作和指标口径见 [网页工作台使用说明](docs/web-interface.md)。

## 安装与快速运行

要求 Python 3.11+，依赖 Pydantic 2、PyYAML。当前实现也在 Python 3.14 上验证。

```powershell
python -m pip install -e ".[test]"
snmodel validate examples/demo_training.yaml
snmodel simulate examples/demo_training.yaml -o outputs/demo-training
snmodel simulate examples/demo_inference.yaml -o outputs/demo-inference
snmodel size-su examples/demo_inference.yaml --mode both -o outputs/su-search
snmodel feedback examples/demo_inference.yaml --search examples/feedback_inference.yaml -o outputs/demo-feedback
```

未安装 CLI 时，在项目根目录把 `snmodel` 替换为 `python -m supernode_sim`，其余参数不变。

批量搜索支持多核：在 `sweep`、`size-su`、`feedback` 命令后添加 `--workers 4`，用四个独立进程评估候选，保持候选顺序、失败诊断与数值口径。默认 `--workers 1`；小任务可能受进程启动开销影响，大图需按每个进程的内存占用控制并发。流式推理、自动并发与可复跑的性能对比见 [仿真器运行性能与资源](skills/supernode-modeling/references/resource-usage.md)。

当前工作区已在 `.venv` 安装 CLI，可直接运行 `.\.venv\Scripts\snmodel.exe`，也可先执行 `.\.venv\Scripts\Activate.ps1` 再使用 `snmodel`。

```powershell
python -m supernode_sim estimate examples/training_10t_fp8.yaml -o outputs/10t-fp8
python -m supernode_sim estimate examples/training_10t_fp16.yaml -o outputs/10t-fp16
python -m supernode_sim simulate examples/inference_10t.yaml -o outputs/10t-inference
python -m supernode_sim sweep examples/training_10t_fp8.yaml --axes examples/sweep_precision.yaml --mode estimate -o outputs/precision-sweep
python -m supernode_sim sweep examples/inference_10t.yaml --axes examples/sweep_active_params.yaml --mode estimate -o outputs/active-sweep
python -m pytest -q
```

10T 示例采用假设的巨容量设备，以较小 rank 数验证全局容量与流量。示例中的 2/4 TB HBM 和 64 TB 主机 DRAM **不是产品规格**；用于真实选型前应替换硬件和负载配置。大 FSDP communicator 的 ring 展开事件数随 rank 数平方增长，先用小规模案例验证配置，再逐步扩大事件仿真。

## 输出和指标

大型参考实验需要支持中断恢复时，可使用下面的运行器。它输出阶段进度，并在数值仿真结束后提交压缩 JSONL 检查点：

```powershell
python -u scripts/run_reference_experiment.py examples/kimi_k3_ascend950_training_muon_128.yaml -o outputs/kimi-k3-ascend950-training-muon-128
```

再次执行相同命令，会验证配置与数值模块指纹并跳过已完成的数值执行，继续重建图和导出报告。检查点位于输出目录的 `checkpoint/`；恢复报告仍需读取记录，但不会一次性将全部原始日志载入内存。

超过 50,000 条事件或张量分配的大场景，默认导出全部汇总、实际峰值构成及模型明细，并精简重复的通信临时记录。`result.json` 和报告会列出原始、导出及省略行数。要导出完整原始记录，可在 `estimate` / `simulate` 后添加 `--full-detail`；这不会改变数值计算，只改变导出明细。

每次 `estimate`/`simulate` 输出：

- `report.md`：中文结论、容量、网络分类、SLO 和假设。
- `result.json`：版本化结果、完整参数来源、配置哈希与详细指标。
- `config.resolved.json`：包含校准值的独立可重跑配置。
- `resources.csv`、`memory.csv`：资源访问量、容量/速率和峰值。
- `latency_breakdown.csv`：TTFT / TPOT（TPOP）的计算、node 内 / scale-up / scale-out / 其他通信、显式同步、调度槽位、内存准入等待和资源争用，单位秒；搜索输出包含逐候选标识。中文报告和网页使用毫秒展示相同拆解。
- 仿真另有 `events.csv` 和可在 Perfetto/Chrome Trace 中导入的 `timeline.json`。

单位是 byte、second、FLOP；报告容量 TB 使用十进制。`unit` 是资源速率单位，`work_unit` 是累计工作量单位。

推理不模拟请求到达队列：TTFT 从开始执行计时；每请求 TPOT 为首 token 后的平均间隔；批量 SLO 使用请求中最大的 TTFT 和 TPOT。输出长度为 1 时 TPOT 为 null。`throughput_tokens_s` 为本次完整执行期间产生的输出 token 数/耗时，输入吞吐另列；Feedback 另以首 token 后的输出数和对应执行时间窗计算 Decode 吞吐。训练报告首次、后续平均以及逐步耗时。

`tpop_s` 是 `tpot_s` 的同值输出别名。`metrics.latency_breakdown` 沿决定各指标的请求关键依赖路径归因，各分项可加和为总时延，不累加并发 rank 的耗时或重复计入已归因上游依赖的同步等待。通信按实际路径限制资源分类，资源争用独立列出；估算、训练、未完成执行及未定义指标显示不可用。具体字段与口径见 [时延拆解说明](skills/supernode-modeling/references/latency-breakdown.md)。

`estimate` 给出必要下界，不声明 SLO 已满足。其乐观内存时间线忽略争用，不能作为通用容量下界；搜索只用固定驻留内存和严格性能下界剪枝，再仿真候选。

CLI 退出码：0 成功；2 输入缺失/非法或图错误；3 容量不可行/SLO 失败。扫描保留每个失败候选；搜索没有可行候选时在 JSON 中报告 `no_feasible_candidate`。

## 配置与 Python API

完整字段及约束由 schema 导出：

```powershell
python -m supernode_sim schema -o outputs/config.schema.json
```

```python
from supernode_sim.config import load_config
from supernode_sim.analysis import run

config = load_config("examples/demo_inference.yaml")
result = run(config, mode="simulate")
print(result["metrics"])
```

并行关系为 `TP × PP × EP × DP = NPU 数`；`fsdp` 是 DP 的子组且必须整除 DP，不能再次乘进总卡数。默认 rank 顺序为 `[dp, pp, ep, tp]`，`rank_order` 可覆盖物理放置。`batch_size` 是每个 DP 副本的 microbatch/推理 batch；改变 DP 会改变聚合负载。

PD 分离是总卡数规则的显式例外：两个相同 TP/EP 的池，DP=PP=1，物理卡数 `2 × TP × EP`；使用 `prefill_ranks` / `decode_ranks` 指定互斥且覆盖全设备的物理映射，不使用 `execution.rank_order`。

精度分别配置：`weight`、`compute`、`gradient`、`master`、`optimizer`、`kv`、`weight_transfer`、`activation`、`communication`。只有明确修改才改变其他张量精度。FP8 activation 通信当前采用共享静态 scale 假设；FP8 权重/KV 的 scale 与 offload 元数据另计。

10T 基准：FP8/FP16 权重分别为 10/20 TB；FP16 梯度 20 TB、FP32 主权重 40 TB、FP32 Adam m/v 80 TB，基础状态共 150/160 TB。CPU optimizer offload 的主权重与 m/v 共 120 TB。若全参数每步更新，梯度下传 20 TB，FP8/FP16 权重回传 10/20 TB；metadata、复制、暂存与实际路径另计。

## 模型导入与校准

面向不同 Ascend 环境，现支持 [多环境响应曲线](skills/supernode-modeling/references/ascend-runtime-adaptation.md)：`hardware.calibration_context` 区分 SKU、CANN/HCCL、驱动、runtime、transport 和计时模式；`collective_models` 在相同条件的实测 payload 范围内插值，不再要求每个消息量都恰好命中单点 profile。通用 `build_runtime_calibration.py` 可导入 HCCL/其他后端的归一化数据，分环境生成校准包，隔离训练与验证；HCCL transport 数据与 MoE 全调用边界分别进入执行图。kernel profile 也可附环境和结构元数据约束。采集口径见 [Ascend 实测指南](skills/supernode-modeling/references/ascend-measurement-sources.md)。

```powershell
python skills/supernode-modeling/scripts/build_runtime_calibration.py skills/supernode-modeling/assets/runtime-calibration.synthetic.json --output outputs/ascend-runtime-adaptation
```

这份双环境示例全部是 synthetic，只验证泛化接口、插值和超范围回退，不代表 A2/A3/950 实测精度。

希望数值更贴近真实运行时，使用 [执行模型拟合](skills/supernode-modeling/references/runtime-fitting.md)：MoE dispatch/combine 可独立指定通信精度及元数据字节；collective profile 保留实际传输和争用，并提供包含启动/转换边界的服务时间约束。`python skills/supernode-modeling/scripts/fit_deepep_collectives.py --output outputs/deepep-fit` 使用 DeepEP 公开微基准，按 EP16/32 拟合、EP64/128 留出，输出前后预测及可复用 profile；本轮四个留出点平均绝对相对误差为 23.19% → 3.96%。这是特定通信阶段拟合，不代表整网 serving 已达到相同精度。

硬件拓扑进一步支持 `hardware.io_groups`（物理 GPU/NPU 到独立或共享 NIC/PCIe 的约束、SO 与 offload 争用）、显式多路径的 `routing_algorithm: ecmp_hash`，以及需要逐层速率输入的 A3 HiAM/SIO/HCCS 拓扑生成器。新 profile 锁定硬件、rank 和完整 peer 矩阵，避免换接线后误套旧时延。运行 `python skills/supernode-modeling/scripts/evaluate_hardware_topology.py` 可复现路径和容量检查；[配置、公开架构依据及数值边界](skills/supernode-modeling/references/hardware-topology-fitting.md) 区分了模型检查与实机拟合精度。

与真实推理比较时，可使用新增的 [公开 benchmark 证据与差距分析](skills/supernode-modeling/references/public-benchmark-validation.md)：包含 DeepInfer、NVIDIA、Ascend 的固定来源数据和标准库比较脚本，逐项检查模型、硬件、软件、负载与指标口径；条件缺失时输出 `not_comparable`，不自动拟合效率。`public` 来源中的作者估算另标 `evidence_kind: estimate`，不充当实测。

SO 网络新增可选 `hardware.scale_out.host_injection_Bps`，独立表达多 NIC 主机每方向聚合能力，避免主机内所有卡被限制在一份单卡注入带宽中。单卡注入、共享上联和显式链路限制仍然生效；未设置时保留原行为。具体配置和公开规格依据见上述说明。

```powershell
python -m supernode_sim import-hf path/to/config.json --overrides path/to/model-overrides.json -o outputs/imported-model.json
```

导入结果是模型片段与 `missing_fields` 清单，合并到完整场景后再验证。HF 配置未提供的参数总量、活跃量和 MoE 结构不会按名称推测。各参数来源记录在 `sources`，状态可为 `assumed`、`public`、`user`、`measured`。

场景中可设置 `calibration: calibration.example.yaml`，路径相对于场景文件。校准文件仅接受 `hardware` 覆盖项和 `source`；解析后的输出配置已嵌入覆盖值，不依赖原校准文件。

`hardware.kernel_profiles` 支持按 kernel 名称提供实测条目列表，每条包括 `flops`、`memory_bytes`、`duration_s`，可选 `dtype`、`phase` 和 `startup_s`。仅匹配相同工作量；未匹配项保留解析模型并报告。`duration_s` 是不含事件启动延迟的独占服务时间，不能低于物理计算/带宽下界。`startup_s` 给出时替换原启动开销，省略则保留原值，不能把同一启动时间重复计入两个字段。匹配后保留真实 FLOPs、共享 HBM 争用与计算流串行约束；用测量时间替换粗略效率限制。

普通图与流式执行均在 `model_metadata.kernel_calibration` 报告实际匹配数量、覆盖率和 prefill/decode 分项；只有发生匹配才标记部分 kernel 校准。可选 `hardware_name`、`calibration_context`、`metadata_match` 限制适用范围，旧记录仍标 unguarded；没有声明的 shape/版本不会自动验证。事件覆盖率不代表端到端精度。实测对齐流程见 [运行时校准](skills/supernode-modeling/references/runtime-calibration.md)，K3/950DT 的公开投机 decode 参考及计时限制见 [来源核验](skills/supernode-modeling/references/kimi-k3-runtime-evidence.md)。

可用随 skill 提供的标准库工具分析逐 token 实测 trace，输出 TTFT、请求 TPOT、E2E、独立 ITL 分布、吞吐和联合 SLO goodput，并显式计数 warmup、失败与未完成请求。以下示例为 synthetic，只验证工具，不能用于硬件校准：

```powershell
python skills/supernode-modeling/scripts/analyze_inference_trace.py skills/supernode-modeling/assets/inference-trace.example.json -o outputs/trace-metrics.example.json
```

decode 的 KV 问题需要分开分析前缀复用、设备页驻留与硬件缓存命中。当前静态 `kv_offload_fraction` 不代表动态命中率；HBM 驻留的 KV 仍有读带宽成本。自动前缀复用、KV 淘汰/抢占/重算和 L2 命中尚未模拟，不能用一个命中率直接缩放 TTFT/TPOT。

## 设计搜索与建模边界

`targets.candidate_npu_counts`、`candidate_su_sizes`、`parallel_candidates` 定义有限候选集。多域搜索必须给定 `total_npus` 或 `max_total_npus`。普通模式未指定并行候选时固定 TP/PP/EP/FSDP，并按总卡数推导 DP；PD 保持 DP=PP=1，按双池总量校验，物理池映射不会随候选被偷偷改写。候选数量上限由 `max_candidates` 控制，结果只声称该集合及指定调度下的最小值。

`feedback` 在固定模型、硬件和物理拓扑上输入 TTFT/TPOT 上限，联合搜索 TP、SP、EP、Prefill 长度及 Decode batch，优先最大化满足目标的 **Decode 吞吐量**：

```powershell
python -m supernode_sim feedback examples/demo_inference.yaml --ttft-s 1 --tpop-s 0.05 --prefill-lengths 32 64 128 --decode-batch-sizes 1 2 4 8 --output-length 4 -o outputs/demo-feedback
```

`--tpop-s` 是 `--tpot-s` 的别名，单位均为秒。也可用 `--search examples/feedback_inference.yaml` 传入搜索规范。默认固定原 PP/DP 并枚举用满设备的合法 TP/EP；SP 共用 TP rank，开启时 degree=TP，只在当前引擎支持时枚举。精度、offload、rank 放置、硬件链路与容量保持不变。Prefill 和 Decode 共享候选的固定 batch，完整运行后同时检查 TTFT、TPOT、内存和原有完整执行吞吐 SLO。

Decode 吞吐为所有请求首 token 后的输出 token 数，除以最早首 token 到最晚末 token 的时间窗；DP=PP=1 时为 Decode 阶段时间，其他并行布局可能包含与未完成 Prefill 的重叠。吞吐最优候选的输入长度和 batch 不一定是各自最大值。未显式给出长度/batch 时，取 1、2、4 等至原输入长度最大值/原 batch 并包含端点；原配置输出长度不一致时，必须显式给出至少为 2 的 `--output-length`。最优只在声明的有限候选集内成立，超过 `max_candidates` 会在仿真前拒绝，不截断或假定单调。

Feedback 输出候选 CSV、失败诊断、Pareto 与布局可行边界，并在有解时生成可复跑的 `best.config.json`；`--engine auto` 为结构化 DP=PP=1 场景选择流式完整运行，启用投机解码、Megakernel 或 PD 分离时使用普通仿真。PD 布局按双池总卡数校验并保留物理映射。字段、Python API 与报告口径见 [Feedback 使用说明](skills/supernode-modeling/references/feedback.md)。

- 参数总量/活跃量是声明的预算；层级算子是架构近似，并非某个框架的精确执行 trace。
- Dense 参数在 EP 间复制，expert 参数按 EP 划分；专家默认均匀路由。稀疏 decode 的权重读取按预期触达专家量估算。
- 训练采用非交错 1F1B；checkpoint、reshard、同步与预取策略影响实际事件图。
- 推理为固定变长 batch，完成的序列不再 decode；latent KV 按明确配置的 latent/RoPE 张量计算，TP 复制假设记录在报告。
- Clos/Fat-tree 模板是单 spine 抽象；支持显式有向链路、共享容量与路由覆盖，无报文拥塞控制或 ECMP 包级仿真。
- 事件采用 roofline 和归一化 max-min 资源共享。缓存报告是算子工作集及访问量；不据此声称 L1/L2/L3 最优容量或精确 MTE 通道数。
- 投机推理支持显式 draft/profile 与接受数驱动的同步 batch 轮次模型，不模拟接受率算法或具体 DSpark 内核。PD 支持两个独立池、同布局状态交接；当前没有异构池重分片、expert offload、生产排队尾延迟和周期级仿真。

## 可选加速特性

在场景顶层 `features` 中选择策略，或用 `--features <文件>` 传入独立对象。统一入口整合量化、checkpoint、optimizer/KV offload、FSDP、SP、Adam/Muon，以及结构化 CSA/HCA/KDA、DSpark 类 draft/verify 轮次。旧 canonical 配置继续有效，冲突会明确报错；resolved 配置只保留 canonical 字段。

```powershell
python -m supernode_sim list-features
python -m supernode_sim simulate examples/features/synthetic-training.yaml -o outputs/features/training
python -m supernode_sim simulate examples/features/synthetic-attention-base.yaml --features examples/features/synthetic-attention-features.yaml -o outputs/features/attention
python -m supernode_sim simulate examples/features/synthetic-dspark.yaml -o outputs/features/dspark
```

这些小示例全部是 synthetic，不代表真实硬件 benchmark。报告和 JSON 的 `model_metadata.feature_plan` 列出实际选择与限制。CSA/HCA 计入压缩池、滑窗、索引/选择、评分暂存及索引 TP 通信；训练保留/重算相关状态，但不包含完整 DeepSeek V4 的 mHC 或 indexer 辅助训练目标。投机模式计入 draft prompt 准备、draft 成本、完整 verify query、接受/回滚与临时状态，并按实际输出 token 计指标；同步等长 batch、DP=PP=1，支持普通 `simulate` 与 `run_streaming` API；后者保留原生投机轮次并按组件边界分块。

开启特性不保证更快：offload 可能增加传输，checkpoint 增加重算，投机有 draft/验证成本。融合与图执行通过具体 kernel profile 表达，不能用通用倍数。参数、可用组合、示例与证据边界见 [feature 输入说明](skills/supernode-modeling/references/features.md)。

`features.megakernel` 将合法的同 rank/层/阶段串行 kernel 链合成事件，显式建模融合启动、受限 HBM 访问节省和可选 fused profile；保留所有分配，按保守生命周期计算峰值。`features.prefill_decode_disaggregation` 将 prefill 与 decode 放置到不同 NPU 池，计入双份驻留及实际状态/首 token ID 交接。两者可组合，普通 `simulate` 支持，streaming 会明确拒绝。

```powershell
python -m supernode_sim simulate examples/features/synthetic-megakernel.yaml -o outputs/features/megakernel
python -m supernode_sim simulate examples/features/synthetic-pd.yaml -o outputs/features/pd
```

新增示例仍为 synthetic。报告列出融合匹配覆盖率，以及 PD prefill、交接、首个 decode 间隔和交接后 decode 时间；固定 batch 并不模拟 P/D 池的在线队列或调度收益。

## 使用 Skill

可复用 skill 源码在 `skills/supernode-modeling/`。使用该 skill 的运行环境必须能访问本项目或已安装的 `supernode_sim` 包；安装 skill 时复制完整目录到个人 skills 目录，包括 `subskills/`。项目不会自动修改全局 Codex 配置。

调用 skill 默认先自动准备并打开现有仿真网页：检查页面与 Python 环境，启动或复用匹配的本地服务，然后进入可编辑的“输入”页。启动脚本位于 `skills/supernode-modeling/scripts/launch_frontend.py`，复制 skill 时一并保留 `scripts/`。前端使用项目内同一份静态资源，无需 npm 构建；自动启动过程不会自行提交仿真。用户明确要求只用 CLI 时继续使用原命令。详细流程见 [Skill 前端自动启动说明](skills/supernode-modeling/references/frontend.md)。

```powershell
python skills/supernode-modeling/scripts/launch_frontend.py --project-root .
```

命令输出经过检查的实际页面 URL、输出目录与日志位置；端口占用时在有限范围内避让，默认使用独立的 `outputs/skill-web/<端口>/` 保存任务。

可以直接在任务中引用 `skills/supernode-modeling/SKILL.md`，请求分析、调整场景或运行参数扫描。Skill 接受模型、负载、并行策略、scale-out 域划分与 SU/SO 网络拓扑，数值分析调用现有引擎。自然语言生成拓扑由内置子 skill [supernode-topology](skills/supernode-modeling/subskills/supernode-topology/SKILL.md) 处理：先生成独立 topology spec，再编译为原生硬件片段或完整场景，并按生成链路给出拓扑图。

例如：“4 个超节点，每个 2 张卡，前两个在 SO 域 A、后两个在域 B；SU 用 full-mesh，SO 用 Clos，两域双向 400 Gbit/s 连接。”子 skill 会区分 4 个 SU 域与 2 个 SO 域，补齐会影响计算的链路参数及其来源，不从型号推测拓扑。完整输入契约见 [拓扑输入说明](skills/supernode-modeling/references/topology.md)。

从项目根目录运行示例：

```powershell
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py --schema -o outputs/topology.schema.json
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py skills/supernode-modeling/subskills/supernode-topology/examples/two_domains.yaml -o outputs/topology.hardware.yaml
# 与已有完整场景合并时添加 --base path/to/scenario.yaml；并行乘积必须匹配新卡数。
python skills/supernode-modeling/subskills/supernode-topology/scripts/build_topology.py path/to/topology.yaml --base path/to/scenario.yaml -o outputs/scenario.topology.yaml
python -m supernode_sim validate outputs/scenario.topology.yaml
python -m supernode_sim simulate outputs/scenario.topology.yaml -o outputs/topology-simulation
```

`topology.hardware.yaml` 仅为硬件/来源/假设片段。SO 域成员是编译器输入，不能直接添加到严格的引擎配置中；编译器展开完整有向链路并使用 `routing_policy: su_so`，同 SU 仅走 SU 边、跨 SU 仅走 SO 边。生成图绑定卡数与域成员，改变域划分的搜索候选需逐个重新编译。子 skill 也可整目录单独安装后以 `$supernode-topology` 调用；仅随父 skill 安装时，由父 skill 读取其入口。


## Kimi K3 组件存储与 Ascend 950DT 样例

新增结构驱动的存储账本：WordEmbedding、PreNorm、PostNorm、MLA、KDA、Dense、MoE、Output layer 及实际存在的 AttnRes；逐组件区分权重、梯度、master、Adam m/v、Muon momentum、activation、KV/递归状态、量化元数据和工作区。`model.architecture` 从张量形状推导参数，旧的纯参数预算模型保留 Unassigned，不伪分配组件。

案例基于 [Kimi K3 官方配置快照与来源](docs/sources/README.md) 和 Huawei 950DT 公开规格：单芯片 144 GB、4 TB/s；64 NPU 是 Atlas 950 单柜切片，最大整域为 8192 NPU。单位、带宽方向、Host DDR/H2D/SO 和算子效率中的假设分别记录；不是已校准的实机性能复现。

```powershell
python -m supernode_sim validate examples/kimi_k3_ascend950_fp8.yaml
python -m supernode_sim simulate examples/kimi_k3_ascend950_fp8.yaml -o outputs/kimi-k3-ascend950-fp8
python -m supernode_sim estimate examples/kimi_k3_ascend950_fp16.yaml -o outputs/kimi-k3-ascend950-fp16
python -m supernode_sim estimate examples/kimi_k3_ascend950_mxfp4.yaml -o outputs/kimi-k3-ascend950-mxfp4
python -m supernode_sim estimate examples/kimi_k3_ascend950_training_muon.yaml -o outputs/kimi-k3-ascend950-muon
python -m supernode_sim estimate examples/kimi_k3_ascend950_training_muon_128.yaml -o outputs/kimi-k3-ascend950-muon-128
```

FP8/FP16 是对同一官方模型结构重新指定精度的设计对比；MXFP4 样例单独表达发布模型的 routed expert 混合存储，计算仍是显式假设。文本 backbone 排除 vision encoder、projector 和训练 MTP。64 NPU 训练案即使 optimizer offload 也可能因权重和梯度超出 144 GB 而不可行，保留该结果；128 NPU 案增加 FSDP2 作为容量替代方案，其两个 SU 域的划分是分析假设。

MLA 缓存默认采用明确标注的 `compressed` 优化布局：每 token 保存 512 + 64 元素。官方 Hugging Face 参考实现保存展开后的 K/V；使用 `examples/sweep_kimi_k3_kv_layout.yaml` 可比较 `compressed` 和 `expanded` 容量/访存。布局选择目前改变存储账本与访存需求，不能据此声称已经校准两种 kernel 的真实耗时。NoPE 不会移除配置中的 64 维 key 投影。

[Skill 的组件分析说明](skills/supernode-modeling/references/memory-breakdown.md) 和 [样例选择及校准边界](skills/supernode-modeling/references/ascend950-kimi-k3.md) 给出报告的加和口径、Muon 参数组策略和实机校准需要的数据。不同组件的独立峰值不能相加当作同一时刻设备峰值。

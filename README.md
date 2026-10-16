# 超节点通信与资源需求建模

`supernode_sim` 是面向架构分析的 Python 数值引擎。`supernode-modeling` skill 负责收集配置与解释结果；公式、通信映射和时序由可重跑代码执行。

项目设计与实现：[整体架构设计文档](docs/simulation-architecture.md) · [函数级仿真流程图](docs/simulation-function-flows.md)。

支持训练与固定负载推理、FP8/FP16 权重、FSDP2/CPU optimizer offload、MoE、标准与 latent KV、KV offload、Engram 查询、SU/SO 拓扑和有限候选搜索。Demo 与 10T 容量锚点是明确标注的合成配置；Kimi K3 / Ascend 950DT 案例使用公开结构和硬件资料，并分别标明运行时、拓扑及效率假设，不将型号名称当作实测校准。

## 网页仿真工作台

在项目根目录启动本地网页，无需安装 Node.js 或前端依赖：

```powershell
.\.venv\Scripts\python.exe -m supernode_sim web
```

浏览器打开 **http://127.0.0.1:8765**。选择训练或推理预设，调整负载、并行布局、硬件、精度及 offload 选项，校验后点击开始运行。网页调用现有 Python 引擎，展示真实运行阶段、通信流量、容量、资源使用和事件时间线，并提供历史任务及报告下载。完整配置可通过高级 JSON 编辑器修改或导入。

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
- 仿真另有 `events.csv` 和可在 Perfetto/Chrome Trace 中导入的 `timeline.json`。

单位是 byte、second、FLOP；报告容量 TB 使用十进制。`unit` 是资源速率单位，`work_unit` 是累计工作量单位。

推理不模拟请求到达队列：TTFT 从开始执行计时；每请求 TPOT 为首 token 后的平均间隔；批量 SLO 使用请求中最大的 TTFT 和 TPOT。输出长度为 1 时 TPOT 为 null。`throughput_tokens_s` 为本次完整执行期间产生的输出 token 数/耗时，输入吞吐另列；Feedback 另以首 token 后的输出数和对应执行时间窗计算 Decode 吞吐。训练报告首次、后续平均以及逐步耗时。

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

精度分别配置：`weight`、`compute`、`gradient`、`master`、`optimizer`、`kv`、`weight_transfer`、`activation`、`communication`。只有明确修改才改变其他张量精度。FP8 activation 通信当前采用共享静态 scale 假设；FP8 权重/KV 的 scale 与 offload 元数据另计。

10T 基准：FP8/FP16 权重分别为 10/20 TB；FP16 梯度 20 TB、FP32 主权重 40 TB、FP32 Adam m/v 80 TB，基础状态共 150/160 TB。CPU optimizer offload 的主权重与 m/v 共 120 TB。若全参数每步更新，梯度下传 20 TB，FP8/FP16 权重回传 10/20 TB；metadata、复制、暂存与实际路径另计。

## 模型导入与校准

```powershell
python -m supernode_sim import-hf path/to/config.json --overrides path/to/model-overrides.json -o outputs/imported-model.json
```

导入结果是模型片段与 `missing_fields` 清单，合并到完整场景后再验证。HF 配置未提供的参数总量、活跃量和 MoE 结构不会按名称推测。各参数来源记录在 `sources`，状态可为 `assumed`、`public`、`user`、`measured`。

场景中可设置 `calibration: calibration.example.yaml`，路径相对于场景文件。校准文件仅接受 `hardware` 覆盖项和 `source`；解析后的输出配置已嵌入覆盖值，不依赖原校准文件。

`hardware.kernel_profiles` 支持按 kernel 名称提供实测条目列表，每条包括 `flops`、`memory_bytes`、`duration_s`，可选 `dtype` 和 `phase`。仅匹配相同工作量；未匹配项保留解析模型并报告。`duration_s` 是不含事件启动延迟的独占服务时间，不能低于物理计算/带宽下界。匹配后保留真实 FLOPs、共享 HBM 争用与计算流串行约束；用测量时间替换粗略效率限制。匹配信息在 `model_metadata.kernel_calibration`。

## 设计搜索与建模边界

`targets.candidate_npu_counts`、`candidate_su_sizes`、`parallel_candidates` 定义有限候选集。多域搜索必须给定 `total_npus` 或 `max_total_npus`。未指定并行候选时固定 TP/PP/EP/FSDP，并按总卡数推导 DP。候选数量上限由 `max_candidates` 控制，结果只声称该集合及指定调度下的最小值。

`feedback` 在固定模型、硬件和物理拓扑上输入 TTFT/TPOT 上限，联合搜索 TP、SP、EP、Prefill 长度及 Decode batch，优先最大化满足目标的 **Decode 吞吐量**：

```powershell
python -m supernode_sim feedback examples/demo_inference.yaml --ttft-s 1 --tpop-s 0.05 --prefill-lengths 32 64 128 --decode-batch-sizes 1 2 4 8 --output-length 4 -o outputs/demo-feedback
```

`--tpop-s` 是 `--tpot-s` 的别名，单位均为秒。也可用 `--search examples/feedback_inference.yaml` 传入搜索规范。默认固定原 PP/DP 并枚举用满设备的合法 TP/EP；SP 共用 TP rank，开启时 degree=TP，只在当前引擎支持时枚举。精度、offload、rank 放置、硬件链路与容量保持不变。Prefill 和 Decode 共享候选的固定 batch，完整运行后同时检查 TTFT、TPOT、内存和原有完整执行吞吐 SLO。

Decode 吞吐为所有请求首 token 后的输出 token 数，除以最早首 token 到最晚末 token 的时间窗；DP=PP=1 时为 Decode 阶段时间，其他并行布局可能包含与未完成 Prefill 的重叠。吞吐最优候选的输入长度和 batch 不一定是各自最大值。未显式给出长度/batch 时，取 1、2、4 等至原输入长度最大值/原 batch 并包含端点；原配置输出长度不一致时，必须显式给出至少为 2 的 `--output-length`。最优只在声明的有限候选集内成立，超过 `max_candidates` 会在仿真前拒绝，不截断或假定单调。

Feedback 输出候选 CSV、失败诊断、Pareto 与布局可行边界，并在有解时生成可复跑的 `best.config.json`；`--engine auto` 为结构化 DP=PP=1 场景选择流式完整运行。字段、Python API 与报告口径见 [Feedback 使用说明](skills/supernode-modeling/references/feedback.md)。

- 参数总量/活跃量是声明的预算；层级算子是架构近似，并非某个框架的精确执行 trace。
- Dense 参数在 EP 间复制，expert 参数按 EP 划分；专家默认均匀路由。稀疏 decode 的权重读取按预期触达专家量估算。
- 训练采用非交错 1F1B；checkpoint、reshard、同步与预取策略影响实际事件图。
- 推理为固定变长 batch，完成的序列不再 decode；latent KV 按明确配置的 latent/RoPE 张量计算，TP 复制假设记录在报告。
- Clos/Fat-tree 模板是单 spine 抽象；支持显式有向链路、共享容量与路由覆盖，无报文拥塞控制或 ECMP 包级仿真。
- 事件采用 roofline 和归一化 max-min 资源共享。缓存报告是算子工作集及访问量；不据此声称 L1/L2/L3 最优容量或精确 MTE 通道数。
- 当前没有 PD 分离、投机推理、expert offload、生产排队尾延迟和周期级仿真。

## 使用 Skill

可复用 skill 源码在 `skills/supernode-modeling/`。使用该 skill 的运行环境必须能访问本项目或已安装的 `supernode_sim` 包；安装 skill 时复制完整目录到个人 skills 目录，包括 `subskills/`。项目不会自动修改全局 Codex 配置。

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

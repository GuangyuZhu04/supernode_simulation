# 函数级仿真流程图

配套说明：[超节点仿真架构](simulation-architecture.md)。依据当前源码整理，日期 2026-10-16。以下 14 张函数级图覆盖 CLI/Web 入口、训练/推理构图、优化器、通信、完整图与流式执行、归因导出、搜索和恢复。

阅读约定：带 `()` 的节点标注源码中的真实函数/方法；标注“内部”的节点表示函数体内逻辑块，不是新函数。实线箭头表示调用顺序或控制流。完整图路径中的循环在 Python 中编译事件图，运行依赖由 `Event.deps` 表达；图 13 的 streaming 在结构屏障处交替构图与数值执行，均不执行真实模型算子。源码链接定位文件，函数/方法名用于检索，以免代码迭代使行号链接失效。

2026-10-19 补充主链中的可选 PD/投机构图与 megakernel 变换；特性输入和具体支持边界见 [features.md](../skills/supernode-modeling/references/features.md)。

Mermaid 代码块可在支持 Mermaid 的 Markdown 预览器中显示；每张图另存为 [`diagrams/`](diagrams/) 下的 `.mmd` 源文件。总体分层图位于架构文档，并提供 [`00-architecture.mmd`](diagrams/00-architecture.mmd)。

## 1. CLI 与单次仿真主链

源码：[`cli.main()`](../supernode_sim/cli.py)、[`cli._main()`](../supernode_sim/cli.py)、[`load_config()`](../supernode_sim/config.py)、[`analysis.run()`](../supernode_sim/analysis.py)、[`compile_graph()`](../supernode_sim/analysis.py)。独立图：[01-entry-and-run.mmd](diagrams/01-entry-and-run.mmd)。

```mermaid
flowchart TD
    CLI["snmodel / python -m supernode_sim"] --> MAIN["cli.main() → _main() → parser()"]
    MAIN --> CMD{"内部：命令类型"}
    CMD -->|web| WEB["web.main() → create_server()<br/>见图 14"]
    CMD -->|schema| SCHEMA["Config.model_json_schema() → 写出 schema"]
    CMD -->|import-hf| HF["import_hf_config() → read_data()<br/>模型片段 + missing_fields"]
    CMD -->|其余命令| LOAD["load_config() → read_data()<br/>可选校准 deep_merge() → Config.model_validate()"]
    LOAD --> SUB{"内部：命令分派"}
    SUB -->|validate| VALID["返回 valid；不构图"]
    SUB -->|feedback| FB["FeedbackSpec.model_validate() → feedback()<br/>见图 12"]
    SUB -->|sweep / size-su| SEARCH["sweep() / size_su()<br/>见图 10"]
    SUB -->|estimate / simulate| RUN["analysis.run(config, mode, full_detail)"]
    RUN --> COMPILE["compile_graph()<br/>规范化 Config → Topology.__init__()"]
    COMPILE --> SPECIAL{"内部：PD / 投机特性？"}
    SPECIAL -->|PD| PD["build_disaggregated()<br/>P/D 阶段 + 状态交接"]
    SPECIAL -->|投机| SPEC["build_speculative()<br/>draft / verify / commit"]
    SPECIAL -->|普通| ARCH{"内部：model.architecture 存在？"}
    ARCH -->|是| STRUCT["build_structured()<br/>Builder.training_graph() / inference()"]
    ARCH -->|否，training| TRAIN["build_training() → _TrainingBuilder.build()"]
    ARCH -->|否，inference| INFER["build_inference()"]
    STRUCT --> FUSE["可选 apply_megakernel()<br/>安全局部链融合"]
    TRAIN --> FUSE
    INFER --> FUSE
    PD --> FUSE
    SPEC --> FUSE
    FUSE --> CAL["apply_kernel_profiles()<br/>内部：assumptions / fidelity / feature_plan"]
    CAL --> EST["graph_estimate(graph)<br/>DAG 与资源下界、驻留容量、通信总量"]
    EST --> REQ["device_requirements()<br/>内部：快照、SHA-256、目标窗口、requirements"]
    REQ --> MODE{"内部：mode == simulate？"}
    MODE -->|是| SIM["engine.simulate(graph, retain_event_payload)<br/>analysis 中别名 simulate_graph"]
    SIM --> MET["execution_metrics() → evaluate_targets()<br/>内部：汇总最终 status"]
    MODE -->|否| MEM["build_memory_breakdown()<br/>见图 9"]
    MET --> MEM
    MEM --> RESULT["run() 返回结果字典"]
    RESULT --> WRITE["CLI：write_report(result, output)"]
    SEARCH --> WRITE
    FB --> WRITE
    WRITE --> EXIT["打印摘要并返回退出码<br/>非 web 的 main() finally 恢复 GC 状态"]
```

`run()` 是完整构图路径的共享入口，Python API 默认 `full_detail=True`；CLI 的 estimate/simulate 默认传 `False`，可用 `--full-detail` 保留全部详情。精简计算仍执行所有事件：事件数超过 50,000 时，仅省略 trace 行中 `work/allocations/releases` 的副本；allocation 数超过 50,000 时，仅省略临时 `state=buffers` 的宽张量明细行。内存日志、指标、通信总量和精确峰值仍完整计算。这与图 13 的分块 streaming 是两种保留策略。

`run()` 不写文件，调用者决定是否调用 `write_report()`。simulate 失败仍保留诊断与已观察的内存账本；输入/构图异常由 CLI 捕获并返回 `invalid_input`。`validate` 只运行配置层校验。Web CLI 入口保留 HTTP 服务的 GC 策略，worker 自行临时禁用并恢复 GC。

## 2. 参数预算版训练构图

源码：[`pipeline_schedule()`](../supernode_sim/training.py)、[`_TrainingBuilder.build()`](../supernode_sim/training.py)、[`.phase()`](../supernode_sim/training.py)、[`.optimizer()`](../supernode_sim/training.py)。独立图：[02-training-budget.mmd](diagrams/02-training-budget.mmd)。

```mermaid
flowchart TD
    ENTRY["training.build_training(config, topology)"] --> INIT["_TrainingBuilder.__init__()<br/>内部：rank / stage 映射与 dense / expert 参数预算"]
    INIT --> BUILD["build() → initial_state()<br/>驻留 weights / gradients / master / Adam / runtime"]
    BUILD --> STEP["内部：遍历 training step"]
    STEP --> STAGE["内部：遍历 PP stage<br/>pipeline_schedule(stage, stages, microbatches)"]
    STAGE --> PHASE["phase(step, stage, F或B, micro)<br/>phase_previous() 建立前驱"]
    PHASE --> LAYER["内部：按 F 正序 / B 逆序遍历层"]
    LAYER --> GATHER["按 FSDP / reshard 条件 add_gather()<br/>可选一层前瞻预取"]
    GATHER --> DISPATCH["可选 collective(EP dispatch)<br/>内部：权重精度转换、activation 生命周期"]
    DISPATCH --> CK{"内部：B 且 activation checkpointing？"}
    CK -->|是| RECOMP["kernel(forward=True)<br/>重放 TP / EP 通信"]
    CK -->|否| KERNEL["kernel(..., forward=F)<br/>创建计算与 HBM work 的 Event"]
    RECOMP --> KERNEL
    KERNEL --> COMM["collective(TP attention / MLP all_reduce)<br/>可选 EP combine"]
    COMM --> SYNC{"内部：B 到达梯度同步点？"}
    SYNC -->|是| GRAD["collective()<br/>dense EP AR → FSDP RS → replica DP AR"]
    SYNC -->|否| DONE["barrier(layer done)<br/>释放本层已消费临时分配"]
    GRAD --> DONE
    DONE --> MOREL{"内部：还有层？"}
    MOREL -->|是| LAYER
    MOREL -->|否| PDONE["barrier(phase done)"]
    PDONE --> SEND["有相邻 PP stage 时：<br/>内部可选 pack → Topology.transfer() → 可选 unpack<br/>send.done 供接收方依赖"]
    SEND --> MOREP{"内部：还有 phase / stage？"}
    MOREP -->|是| STAGE
    MOREP -->|否| OPT["optimizer(step)<br/>图 4：Adam 与可选 CPU offload"]
    OPT --> MORES{"内部：还有 step？"}
    MORES -->|是| STEP
    MORES -->|否| RETURN["内部：写 schedule / logical_memory / step_end_events<br/>返回 Graph"]
```

`phase.done` 与 PP `send.done` 是不同标记。发送方后续计算可以只依赖 phase 完成，接收方必须等待 incoming send；因此图中“创建 send 后继续构图”不表示运行时发送阻塞发送方计算。checkpoint 小段画的是运行时依赖：重算 kernel 与 TP/EP 重放完成后才能执行 backward；实际 Python 构图先创建 backward Event，引用稍后才追加的 `recompute.done`。`kernel()` 构造 Event；预算版不支持 Muon。

## 3. 结构化训练构图

源码：[`build_structured()`](../supernode_sim/structured.py)、[`Builder.training_graph()`](../supernode_sim/structured.py)、[`.gather()`](../supernode_sim/structured.py)、[`.recompute_layer()`](../supernode_sim/structured.py)。独立图：[03-training-structured.mmd](diagrams/03-training-structured.mmd)。

```mermaid
flowchart TD
    ENTRY["build_structured() → Builder.__init__()<br/>Topology.populate() + parameter_inventory()"]
    ENTRY --> INIT["training_graph() → initial()<br/>states() / state_dtype() / local_elements()"]
    INIT --> STEP["内部：step → PP stage<br/>pipeline_schedule() → F或B / microbatch"]
    STEP --> LAYER["内部：按阶段方向遍历 stage layers"]
    LAYER --> GATHER["按 FSDP / reshard 条件 gather()<br/>collective(all_gather)"]
    GATHER --> ACT["activation_rows() → component_activations()<br/>内部：B 时建立或复用 gradient buffers"]
    ACT --> CK{"内部：B 且 checkpoint？"}
    CK -->|是| RECOMP["recompute_layer()<br/>kernel() + TP / EP + 临时状态"]
    CK -->|否| ORDER["component_order(layer)<br/>F 正序 / B 逆序遍历组件"]
    RECOMP --> ORDER
    ORDER --> DISPATCH["MoE 时 collective(EP dispatch)"]
    DISPATCH --> KERNEL["逐 rank kernel()<br/>权重转换、FLOPs/HBM、activation/state allocs/releases"]
    KERNEL --> COMM["MoE 时 collective(combine)<br/>attention / FFN 后 collective(TP all_reduce)"]
    COMM --> MOREC{"内部：还有组件？"}
    MOREC -->|是| ORDER
    MOREC -->|否| SYNC{"内部：B 到达梯度同步点？"}
    SYNC -->|是| GRAD["collective()<br/>FSDP RS → replica AR → dense EP AR"]
    SYNC -->|否| DONE["barrier(layer.done)<br/>释放 gather / gradient / recompute 临时项"]
    GRAD --> DONE
    DONE --> MOREL{"内部：还有层？"}
    MOREL -->|是| LAYER
    MOREL -->|否| PHASE["barrier(phase 完成)<br/>释放消费完的 PP receive 等状态"]
    PHASE --> SEND["有下游时 pipeline_transfer()<br/>可选 pack → Topology.transfer() → 可选 unpack<br/>内部：stage_previous 指向发送完成"]
    SEND --> MOREP{"内部：还有 phase / stage？"}
    MOREP -->|是| STEP
    MOREP -->|否| OPT["optimizer(step, stage tails)<br/>见图 4"]
    OPT --> MORES{"内部：还有 step？"}
    MORES -->|是| STEP
    MORES -->|否| ANNOTATE["annotate_collective_allocations()<br/>返回 Graph"]
```

结构化路径没有 FSDP lookahead prefetch；配置阶段会拒绝该开关。其 stage 后继依赖可能等待 PP send，因此与图 2 的异步发送语义有区别。checkpoint 仍产生前向完整临时工作集，只改变保存与重计算策略。

## 4. 训练优化器与 offload

源码：[`_TrainingBuilder.optimizer()`](../supernode_sim/training.py)、[`Builder.optimizer()`](../supernode_sim/structured.py)、[`optimizer_states()`](../supernode_sim/components.py)。独立图：[04-optimizer-offload.mmd](diagrams/04-optimizer-offload.mmd)。

```mermaid
flowchart TD
    ENTRY["optimizer(step)<br/>依赖本步 backward / stage 尾部完成"] --> LOOP["内部：预算版遍历 rank × layer<br/>结构化版遍历 rank × layer × component"]
    LOOP --> LIMIT["结构化可选 inflight_components_per_rank<br/>为组件接入有界在途链的前驱"]
    LIMIT --> OFF{"内部：optimizer_offload？"}
    OFF -->|是| D2H["Topology.transfer(D2H gradients)<br/>CPU gradient buffer"]
    OFF -->|否| NPU["选择 NPU 本地更新资源"]
    D2H --> KIND["结构化：optimizer_states()<br/>预算版：固定 Adam"]
    NPU --> KIND
    KIND --> UPDATE["内部：创建 update Event<br/>Adam / Muon 工作量、状态访问与 scratch"]
    UPDATE --> PACK["内部：创建 pack Event<br/>master 转为传输 / 驻留权重精度"]
    PACK --> BACK{"内部：optimizer_offload？"}
    BACK -->|是| H2D["Topology.transfer(H2D weights)<br/>必要时另外传输 scale"]
    BACK -->|否| REFRESH["内部：refresh Event<br/>更新持久权重，释放临时 buffer"]
    H2D --> REFRESH
    REFRESH --> MORE{"内部：还有更新项？"}
    MORE -->|是| LOOP
    MORE -->|否| DONE["barrier(train.stepN.done)<br/>外层 build() / training_graph() 汇总 step_end_events"]
```

图中的更新项依赖形成事件链；Python 构图循环本身不表示各项运行时完全串行。Muon 正交化是解析工作量与 workspace 近似，当前没有跨 rank 全矩阵组装与 distributed Muon；优化器也不与本步 backward 重叠执行。

## 5. 结构化推理：prefill / decode、SP 与状态生命周期

源码：[`Builder.inference()`](../supernode_sim/structured.py)、[`.kernel()`](../supernode_sim/structured.py)、[`attention_state()`](../supernode_sim/components.py)、[`attnres_state()`](../supernode_sim/components.py)。独立图：[05-inference-structured.mmd](diagrams/05-inference-structured.mmd)。

```mermaid
flowchart TD
    ENTRY["build_structured() → Builder.__init__()"] --> INIT["Builder.inference() → initial()<br/>驻留权重 / runtime / 可选 Engram"]
    INIT --> DP["内部：遍历 DP replica"]
    DP --> TOKEN["内部：遍历 output step<br/>0 = prefill；其余 = decode"]
    TOKEN --> ACTIVE["内部：筛选 output_length 大于 step 的序列<br/>按 EP lane 分配 tokens；SP 计算 padded rows"]
    ACTIVE --> LAYER["内部：遍历 layer → 确定 PP stage"]
    LAYER --> STATE["attention_state() / attnres_state()<br/>KDA 固定状态 或 paged KV 增量<br/>可选残差状态"]
    STATE --> COMPONENT["component_order(layer)<br/>内部可选 streaming component-before hook"]
    COMPONENT --> SP["SP 时 attention / FFN / Output 前<br/>barrier() 分配 padded input → collective(all_gather)"]
    SP --> DISPATCH["MoE 时 collective(EP dispatch)"]
    DISPATCH --> ACT["逐 rank activation_rows()<br/>norm / AttnRes 用 SP 局部 rows<br/>prefill Output 仅算每序列末位"]
    ACT --> READ["内部可选：旧 KV fetch → decode conversion<br/>PreNorm 前 Engram lookup"]
    READ --> KERNEL["kernel()<br/>权重转换 + component Event"]
    KERNEL --> WRITE["内部可选：新 KV encode → offload write"]
    WRITE --> COMM["MoE 时 collective(combine)<br/>attention / FFN 后 TP AR；SP 时改为 RS<br/>SP WordEmbedding 后也执行 RS"]
    COMM --> RELEASE["barrier() 释放 SP gather buffer<br/>内部可选 streaming component-after hook"]
    RELEASE --> MOREC{"内部：还有组件？"}
    MOREC -->|是| COMPONENT
    MOREC -->|否| DONE["barrier(layer_done)<br/>释放上个 stage 的 receive buffer"]
    DONE --> PP{"内部：跨 PP stage？"}
    PP -->|是| SEND["pipeline_transfer()<br/>pack → Topology.transfer() → unpack"]
    PP -->|否| MOREL{"内部：还有层？"}
    SEND --> MOREL
    MOREL -->|是| LAYER
    MOREL -->|否| COMPLETE["barrier(token_complete)<br/>记录 sequence_token_completion_events"]
    COMPLETE --> MORET{"内部：还有 output step？"}
    MORET -->|是| TOKEN
    MORET -->|否| MORED{"内部：还有 replica？"}
    MORED -->|是| DP
    MORED -->|否| RETURN["build_structured() 调用<br/>annotate_collective_allocations() → 返回 Graph"]
```

SP 共用 TP ranks，度数为 TP，不能额外乘入设备数；配置只接受结构化推理、TP > 1、PP = 1。SP 显式计入 padding、all-gather 输入与 reduce-scatter 输出；无 SP 时保持 TP all-reduce。图中 hooks 是 `run_streaming()` 安装的回调；普通完整构图不安装这些 hooks。状态分配后也有可选的 streaming state hook（见图 13）。

KDA recurrent/conv state 不随上下文线性增长；MLA cache 可用 compressed/expanded 布局。非 KDA 的旧 KV 转换具有显式解码 buffer。各序列完成后停止后续计算和 KV 读取，cache 保留至本次固定批次结束。同一 replica 按 token→layer 构图，不使用训练的 1F1B。

## 6. 参数预算版推理

源码：[`build_inference()`](../supernode_sim/inference.py)、[`kv_layout_bytes()`](../supernode_sim/inference.py)。独立图：[06-inference-budget.mmd](diagrams/06-inference-budget.mmd)。

```mermaid
flowchart TD
    ENTRY["inference.build_inference(config, topology)"] --> INIT["Topology.populate()<br/>内部：rank / layer 映射，驻留权重与缓存设施"]
    INIT --> LOOPS["内部：DP → output step → layer → EP → TP<br/>step0 prefill，后续 decode，仅 active 序列"]
    LOOPS --> LAYOUT["闭包 layout() → kv_layout_bytes()<br/>data / scale / page table"]
    LAYOUT --> READY["闭包 alloc() / add()<br/>ready Event：KV 增量、activation 与临时状态"]
    READY --> FETCH["内部可选：KV offload fetch<br/>Topology.transfer()"]
    READY --> ENGRAM["内部可选：Engram lookup / cache hit<br/>Topology.transfer() 与资源 Event"]
    READY --> CAST["内部可选：weight conversion Event"]
    FETCH --> DENSE["内部：dense Event<br/>attention + dense 计算，读侧 KV 转换融合建模"]
    ENGRAM --> DENSE
    CAST --> DENSE
    DENSE --> KVCONVERT["内部可选：新 KV conversion"]
    KVCONVERT --> KVWRITE["内部可选：KV offload write"]
    KVCONVERT --> MOE["可选 collective(dispatch)<br/>expert Event → collective(combine)"]
    MOE --> TP["闭包 collective()<br/>层末 TP attention AR → TP FFN AR"]
    TP --> PP["内部可选 PP：pack → Topology.transfer() → unpack"]
    PP --> DONE["内部：layer.done<br/>等待层输出与 KV write；释放临时 buffer"]
    KVWRITE --> DONE
    DONE --> MOREL{"内部：还有层？"}
    MOREL -->|是| LOOPS
    MOREL -->|否| TOKEN["内部：token barrier<br/>记录逐序列 token 完成事件"]
    TOKEN --> MORE{"内部：还有 step / replica？"}
    MORE -->|是| LOOPS
    MORE -->|否| RETURN["返回 Graph"]
```

从 ready 发出的可选分支在依赖满足时可重叠执行；关闭时相应节点退化为直通依赖。该路径把读侧 KV 转换融合于 attention，不分配完整旧 KV 解压缓存；两次 TP reduction 都位于层末。两条推理路径的事件粒度和通信顺序有区别。

## 7. Collective 与拓扑路由展开

源码：[`Builder.collective()`](../supernode_sim/structured.py)、[`Topology.collective()`](../supernode_sim/topology.py)、[`.transfer()`](../supernode_sim/topology.py)、[`.path()`](../supernode_sim/topology.py)。独立图：[07-collective-topology.mmd](diagrams/07-collective-topology.mmd)。

```mermaid
flowchart TD
    ENTRY["构图器 collective() 封装<br/>结构化 / 预算训练 / 预算推理"] --> CAST["内部：必要时 pack Event<br/>activation → communication dtype"]
    CAST --> TC["Topology.collective(graph, kind, ranks, size, deps)"]
    TC --> SINGLE{"内部：rank 数为 1？"}
    SINGLE -->|是| IDENTITY["直接返回传入 deps，不产生通信事件"]
    SINGLE -->|否| KIND{"内部：collective 类型"}
    KIND -->|AG / RS| RING["内部：ring，N-1 轮<br/>每 rank 每轮 size/N"]
    KIND -->|AR| AR["内部：RS → AG，两阶段 ring"]
    KIND -->|A2AV| A2A["内部：按 payload_matrix<br/>pairwise N-1 轮，对角项无网络传输"]
    RING --> TRANS["Topology.transfer(src, dst, bytes, deps)"]
    AR --> TRANS
    A2A --> TRANS
    TRANS --> PATH["Topology.path(src, dst)"]
    PATH --> ROUTE["内部：pair route cache → 验证显式 routes<br/>→ 同 SU full_mesh 懒建直连<br/>→ 按允许域读取/建立 BFS source tree<br/>有序边保证确定性最少 hop"]
    ROUTE --> WORK["transfer() 内部：端点内存、逐跳链路<br/>injection / shared group / NIC / bridge 工作量<br/>startup = 路径 latency 之和"]
    WORK --> EVENT["返回 Event<br/>payload 与 physical bytes 分别记 metadata"]
    EVENT --> BUFS["collective() 内部：添加 send / recv allocation<br/>完成时 release，设置 tx / rx slots<br/>追加到 graph.events"]
    BUFS --> NEXT{"内部：还有轮次 / 阶段？"}
    NEXT -->|是，前轮 terminals 作为 deps| TRANS
    NEXT -->|否| TERMINALS["返回末轮 terminal event IDs"]
    TERMINALS --> UNPACK["构图器：必要时 unpack 与 wire buffer 释放"]
    IDENTITY --> UNPACK
```

`Topology.path()` 对静态拓扑的 pair route cache 使用 16 MiB / 65,536 项预算，每种路由模式（不限域、SU、SO）的 BFS source-tree cache 各使用 16 MiB / 128 项预算。超预算项仍服务本次查询但不保留。full_mesh 的 pair cache 使用普通字典，以保留懒建链路后已有 pair 的历史路由；缓存预算不等同于拓扑总内存上限。零字节 transfer 不查询路径。

`transfer()` 本身返回资源化的 Event；collective 的 staging buffer/slot 在 `collective()` 内添加，普通点对点 buffer 由对应构图器管理。AG/RS/AR 的 `size` 是每 rank 完整张量大小。所有同轮 transfer 的完成 ID 作为下一轮依赖，图表达的是明确的通信算法，而非一个总字节量常数。

## 8. 离散事件执行器与 max-min 资源共享

源码：[`engine.simulate()`](../supernode_sim/engine.py)、[`max_min_rates()`](../supernode_sim/engine.py)。独立图：[08-event-engine.mmd](diagrams/08-event-engine.mmd)。

```mermaid
flowchart TD
    ENTRY["engine.simulate(graph, retain_event_payload)"] --> CHECK["内部：校验唯一 ID、资源、allocation、依赖和 DAG"]
    CHECK --> VALID{"内部：图合法？"}
    VALID -->|否| ERROR["闭包 fail(error, code, detail)<br/>result() 带 diagnostics"]
    VALID -->|是| INIT["闭包 allocate(initial_allocations)<br/>记录 sequence / time / 占用 / 峰值"]
    INIT --> CAP{"内部：驻留内存超容量？"}
    CAP -->|是| OOM["fail(infeasible, initial_memory_exceeded)"]
    CAP -->|否| FIN["循环内部：先完成事件<br/>release 内存与 slots，更新 ready，写 trace"]
    FIN --> ALL{"内部：全部事件完成？"}
    ALL -->|是| RESULT["排序 trace → result(feasible)"]
    ALL -->|否| ADMIT["内部：按 ID 扫描 ready<br/>slot 空闲且内存可容纳 → 原子 allocate<br/>active = startup + remaining"]
    ADMIT --> CHANGE{"内部：刚完成或准入了事件？"}
    CHANGE -->|是| FIN
    CHANGE -->|否| ACTIVE{"内部：存在 active？"}
    ACTIVE -->|否，内存阻塞| DEAD["fail(infeasible, memory_admission_deadlock)"]
    ACTIVE -->|否，其他情况| ERROR
    ACTIVE -->|是| RATE["max_min_rates(startup 已结束的 active work)"]
    RATE --> SOLO["函数内部：solo = min(capacity / work)<br/>归一化需求 = work × solo"]
    SOLO --> FAIR["函数内部：等量提高未冻结事件的速度比例<br/>冻结瓶颈资源相关事件，重复至全部分配"]
    FAIR --> SPEED["返回 rate = fraction × solo"]
    SPEED --> DELTA["simulate 内部：delta = min<br/>下一 startup 结束、remaining/rate"]
    DELTA --> OK{"内部：时间步可推进且合法？"}
    OK -->|否| ERROR
    OK -->|是| ADVANCE["内部：更新 remaining / work_done / busy<br/>now += delta"]
    ADVANCE --> FIN
```

`retain_event_payload=False` 仅省略 trace 行的 work/allocations/releases 副本，仍保留每个已完成事件的 ID、时间、metadata、slots，以及完整 memory_log；它不改变调度、内存准入与数值结果，也不将完整图执行变成 streaming。

事件启动时原子分配内存并占用 slot，事件结束时执行 release。相同时间点先释放再准入新事件。资源同时服务，计算/访存耗时取 roofline 瓶颈。图中为便于阅读将若干校验错误合并到 `fail(error, ...)`，具体 code 以 diagnostics 为准。

## 9. 内存归因与报告导出

源码：[`build_memory_breakdown()`](../supernode_sim/memory_breakdown.py)、[`write_report()`](../supernode_sim/report.py)、[`_prepare_export()`](../supernode_sim/report.py)、`_write_event_artifacts()`、`_write_result_json()`。独立图：[09-memory-and-report.mmd](diagrams/09-memory-and-report.mmd)。

```mermaid
flowchart TD
    ENTRY["analysis.run() 的 graph / simulation / estimate"] --> MEM["build_memory_breakdown()"]
    MEM --> INDEX["闭包 add_tensor() → _labels()<br/>索引全部 allocations 与释放事件<br/>可省略临时 buffers 的宽明细行"]
    INDEX --> LOG{"内部：存在 memory_log 字段？"}
    LOG -->|是| REPLAY["内部：按 sequence 有序回放<br/>校验 ID、pool、bytes、时间与生命周期"]
    REPLAY --> CHANGE["闭包 change()<br/>更新池/组件/状态与集群并发峰值"]
    LOG -->|否| RESIDENT["对 initial residency 调用 change()<br/>mode = resident_only"]
    CHANGE --> STATUS["内部：完整执行为 simulated<br/>提前停止为 observed_partial"]
    STATUS --> TABLES["内部：生成 tensors / pools / component_state_rows<br/>state_rows / cluster / logical states"]
    RESIDENT --> TABLES
    TABLES --> GROUPS["_equivalent_groups()<br/>返回 memory_breakdown"]
    GROUPS --> RUNRESULT["run() 返回完整结果对象"]
    RUNRESULT --> WRITE["write_report(result, directory, full_detail)"]
    FB["feedback() 返回 mode=feedback"] --> WRITE
    WRITE --> MODE{"内部：mode == feedback？"}
    MODE -->|是| FBR["write_feedback_report()<br/>见图 12；独立候选报告"]
    MODE -->|否| PREP["_prepare_export()<br/>检测大图/已精简明细，生成 artifact_export"]
    PREP --> DETAIL{"内部：请求 full detail 但明细已丢失？"}
    DETAIL -->|是| ERROR["ValueError：必须从计算阶段重新保留完整明细"]
    DETAIL -->|否| CONFIG["内部：写 config.resolved.json"]
    CONFIG --> CSV["_csv()<br/>resources.csv / memory.csv"]
    CSV --> SIM{"内部：存在 simulation？"}
    SIM -->|是| TRACE["_write_event_artifacts()<br/>一遍遍历写 events.csv 与 timeline.json<br/>更新 artifact_export 实际导出条数"]
    SIM -->|否| MEMORY["_csv()<br/>组件/状态/张量/峰值/可选参数表"]
    TRACE --> MEMORY
    MEMORY --> JSON["_write_result_json()<br/>临时文件 → replace result.json<br/>markdown_report() → report.md"]
    JSON --> OUT["返回输出目录"]
    FBR --> OUT
```

`write_report()` 的标准输入是仍持有可迭代 trace 与张量列表的 `run()` 结果；已写出且引用过滤产物的 `result.json` 不能直接重新导出。checkpoint 的 `JSONLRecords` 可流式导出；`run_streaming()` 的紧凑结果不是该接口的完整输入。

自动导出阈值为 50,000 行，比较 trace、原始 tensor、memory_log 的最大条数。大表在非 full 模式下保留模型 kernel、token/step marker、保留模型张量的分配事件，省略临时 buffer 张量行与原始内存日志；保留张量表仍超过 50,000 行时使用 `tensors.csv.gz`。事件 CSV 与 Perfetto timeline 共用一遍遍历，计数后再写 `result.json`，对 JSON 使用临时文件替换；这不是整个输出目录的原子事务。

过滤只修改浅层导出视图，不改 API 所有的数组及完整计算得到的指标、通信/资源总量与并发峰值。完整详情要求计算阶段保留事件 payload 和全部张量行。不同组件/不同池的独立峰值不允许直接相加。feedback 使用独立格式，且无胜者时删除旧 `best.config.json`。

## 10. 参数扫描与最小 SU 搜索

源码：[`sweep()`](../supernode_sim/search.py)、[`size_su()`](../supernode_sim/search.py)。独立图：[10-sweep-and-su-search.mmd](diagrams/10-sweep-and-su-search.mmd)。

```mermaid
flowchart TD
    CMD{"CLI 分派"} -->|sweep| SW["sweep(config, axes, mode)<br/>校验 axes 与 max_candidates"]
    SW --> PRODUCT["内部：itertools.product 枚举组合<br/>copy 配置 → set_field()"]
    PRODUCT --> SWVALID["Config.model_validate()"]
    SWVALID --> SWRUN["run(candidate, mode, full_detail=False)<br/>compact_result()"]
    SWVALID -->|异常| SWBAD["内部：保留 invalid_config 与原因"]
    SWRUN --> SWNEXT{"内部：还有组合？"}
    SWBAD --> SWNEXT
    SWNEXT -->|是| PRODUCT
    SWNEXT -->|否| SWOUT["返回 completed + candidates"]
    CMD -->|size-su| SU["size_su(config, single / multi / both)"]
    SU --> ENUM["内部：枚举 NPU count × SU size × parallel layout<br/>多域需 total_npus 或 max_total_npus"]
    ENUM --> VALID["Config.model_validate(candidate)"]
    VALID -->|异常| BAD["记录 invalid_config"]
    VALID --> EST["run(candidate, estimate, full_detail=False)"]
    EST --> MEM{"内部：resident_memory 超容量？"}
    MEM -->|是| MP["记录 infeasible<br/>resident_memory_lower_bound 剪枝"]
    MEM -->|否| PERF{"内部：duration 下界超过有效全图 deadline？"}
    PERF -->|是| PP["记录 slo_failed<br/>performance_lower_bound 剪枝"]
    PERF -->|否| SIM["run(candidate, simulate, full_detail=False)<br/>_payload_summary()"]
    BAD --> NEXT{"内部：还有候选？"}
    MP --> NEXT
    PP --> NEXT
    SIM --> NEXT
    NEXT -->|是| VALID
    NEXT -->|否| BEST["内部：过滤 feasible<br/>按 SU size / NPU count / duration 排序"]
    BEST --> OUT["返回 best / minimum_su_size / 全部候选<br/>或 no_candidates / no_feasible_candidate"]
```

图中合并了 single/multi 两个子搜索的外层循环。无并行候选时保留 TP/PP/EP/FSDP 并推导 DP。只使用必要下界剪枝；忽略争用的乐观内存峰值和全图 TTFT/TPOT deadline 不用于这种剪枝。`run(estimate)` 与 `run(simulate)` 各自构图，当前未共享候选 Graph。

启用 PD 时搜索按双池总卡数校验，保持 DP=PP=1；显式 P/D 映射随候选重新验证，不自动派生新的 DP 或物理放置。

## 11. 大型参考实验检查点与恢复

源码：[`scripts/run_reference_experiment.py`](../scripts/run_reference_experiment.py)、[`fingerprint()`](../scripts/run_reference_experiment.py)、[`load_checkpoint()`](../scripts/run_reference_experiment.py)。独立图：[11-reference-checkpoint.mmd](diagrams/11-reference-checkpoint.mmd)。

```mermaid
flowchart TD
    ENTRY["run_reference_experiment.main()"] --> CONFIG["load_config() → fingerprint()<br/>resolved config + 指定数值源码"]
    CONFIG --> WRAP["内部：保存 analysis 原函数并安装进度包装<br/>checkpoint/fingerprint 前20位目录<br/>--full-detail 追加 -full；可选 gzip 压缩级别"]
    WRAP --> RUN["analysis.run(config, simulate,<br/>full_detail=args.full_detail)"]
    RUN --> COMPILE["包装 compile_graph() → 原 compile_graph()<br/>包装 graph_estimate() → 原 graph_estimate()"]
    COMPILE --> SIM["包装 simulate_graph(graph, **kwargs)"]
    SIM --> EXISTS{"内部：已提交 manifest.json？"}
    EXISTS -->|否| PHYSICS["原 engine.simulate(graph, **kwargs)<br/>传递 retain_event_payload 策略"]
    PHYSICS --> SAVE["save_records(trace) + save_records(memory_log)<br/>gzip JSONL 临时文件完成后替换"]
    SAVE --> COMMIT["内部：最后提交 manifest.json<br/>fingerprint / count / size / simulation 摘要"]
    EXISTS -->|是| LOAD["load_checkpoint(directory, fingerprint)"]
    COMMIT --> LOAD
    LOAD --> CHECK["内部：校验 fingerprint 与文件大小<br/>大记录使用 JSONLRecords，否则加载 list"]
    CHECK --> METRIC["run() 继续 execution_metrics() / evaluate_targets()"]
    METRIC --> MEM["包装 breakdown() → build_memory_breakdown()<br/>按 sequence 回放日志，流式记录省临时明细"]
    MEM --> EXPORT["write_report(result, output,<br/>full_detail=args.full_detail)"]
    EXPORT --> FINALLY["内部 finally：恢复 analysis 原函数与 GC 状态"]
```

`fingerprint()` 覆盖已解析配置与指定数值源码（含 `cache.py`）。加载时核验 fingerprint 和文件大小；trace、allocation、memory_log 任一数量超过 50,000 时使用 `JSONLRecords`，其 `__iter__()` 在迭代结束时核验记录条数。`--full-detail` 使用独立 `-full` 目录，避免从精简 checkpoint 恢复已经省略的 payload。

检查点在一次数值调用返回后提交，也可以保留 infeasible/error 结果。恢复只跳过已完成的数值调用，仍需重建图、计算下界与指标并回放日志；不支持在数值执行中途恢复 active/ready/remaining 状态。精简 `result.json` 不能替代原始检查点；JSONL 流式读取也不同于图 13 的分块数值执行。

## 12. Feedback：固定模型与硬件的联合推理搜索

源码：[`feedback.py`](../supernode_sim/feedback.py)（`FeedbackSpec`、`feedback()`、`_layouts()`、`_candidate_config()`、`_decode_metrics()`、`_rank()`、`_frontier()`）。独立图：[12-feedback-search.mmd](diagrams/12-feedback-search.mmd)。

```mermaid
flowchart TD
    ENTRY["feedback(config, search)"] --> SPEC["FeedbackSpec.model_validate()<br/>tpop_alias() → canonical_axes()"]
    SPEC --> CHECK["内部：只接受 inference；要求 TTFT 与 TPOT<br/>固定 output_length ≥ 2；ragged 输出需显式长度"]
    CHECK --> ENGINE["内部：选择 engine<br/>auto 在 structured 且 DP=PP=1 时选 streaming<br/>其余选完整 simulate"]
    ENGINE --> AXES["_bounded_grid() / _layouts()<br/>length × batch × TP/EP/SP 有限候选网格"]
    AXES --> LIMIT{"内部：候选数 ≤ max_candidates？"}
    LIMIT -->|否| ERROR["ValueError：缩小轴或显式提高上限"]
    LIMIT -->|是| LOOP["内部：itertools.product 遍历全部候选"]
    LOOP --> CAND["_candidate_config()<br/>复制基线，替换执行轴、统一输入长度/批次/输出长度<br/>Config.model_validate()"]
    CAND -->|校验失败| BAD["记录 invalid_config 与原因"]
    CAND --> CHOICE{"内部：选定引擎"}
    CHOICE -->|streaming| STREAM["run_streaming(candidate, phase=all)<br/>见图 13"]
    CHOICE -->|simulate| RUN["analysis.run(candidate, simulate, full_detail=False)"]
    STREAM --> STATUS{"内部：物理执行成功？"}
    RUN --> STATUS
    STATUS -->|否| FAILURE["保留 infeasible / error、memory peaks、diagnostics"]
    STATUS -->|是| METRIC["_decode_metrics() → evaluate_targets()<br/>解码窗口吞吐量 + TTFT / TPOT 与已有吞吐目标"]
    METRIC --> ROW["记录 feasible 或 slo_failed"]
    BAD --> MORE{"内部：还有候选？"}
    FAILURE --> MORE
    ROW --> MORE
    MORE -->|是| LOOP
    MORE -->|否| RANK["_rank() 排序 feasible<br/>构造 best 与实际 best.config"]
    RANK --> FRONT["_frontier()<br/>内部：每 layout 的可行边界与全部状态计数"]
    FRONT --> OUT["返回 mode=feedback<br/>best / candidates / frontier / boundaries<br/>search_complete = 无 error 候选"]
    OUT --> REPORT["write_report() → write_feedback_report()<br/>result.json / config.resolved.json / 可选 best.config.json<br/>candidates.csv / report.md"]
```

模型、硬件/拓扑、PP/DP/FSDP、rank order、精度与 features 固定，只搜索 TP、EP、SP、prefill 长度与固定批次。默认长度/批次网格是“不超过基线端点的 2 的幂 + 端点”，并非所有整数。TP/EP 默认由设备槽位与专家整除关系推导；显式不合法轴仍保留为 `invalid_config`。SP 共用 TP ranks。

每个候选完整联合执行 prefill 与 decode，不使用 estimate 下界剪枝、分离阶段择优或 chunked prefill。decode throughput = 全部请求首 token 之后的输出 token 总数 /（最晚末 token − 最早首 token）；DP/PP > 1 时该窗口可能与其他请求 prefill 重叠。TTFT 取各请求首 token 时延的最大值，TPOT 取各请求平均输出间隔的最大值；output_length=2 只测一个 decode 间隔。除两项延迟目标外，已有 throughput 目标也必须通过。

最优项先最大化 decode throughput，同分依次偏向更长输入、更大批次、更小 TTFT/TPOT、较小 TP/EP/SP。frontier 同时最大化长度、批次、decode throughput；同一 layout 的最大长度与最大批次可能来自不同候选，实际 `best` 才是联合验证配置。出现 `error` 候选时整体 `status=error`、`search_complete=False`，即使已有 best 也不宣称搜索完成。

## 13. Streaming：结构屏障分块、精确准入与缓存

源码：[`streaming.py`](../supernode_sim/streaming.py)（`run_streaming()` 及其闭包）、[`cache.py`](../supernode_sim/cache.py)（`BoundedCache`、`_size_bytes()`）、[`engine.py`](../supernode_sim/engine.py)（`simulate()`）。独立图：[13-streaming-execution.mmd](diagrams/13-streaming-execution.mmd)。

```mermaid
flowchart TD
    ENTRY["run_streaming(config, phase, progress, cache)"] --> CHECK["内部：structured inference 且 DP=PP=1<br/>phase 为 all / prefill / decode"]
    CHECK --> BUILDER["Builder.__init__() + Topology.__init__()<br/>默认 timing cache = BoundedCache()"]
    BUILDER --> HOOKS["内部：安装 cached_initial() 与 state/component hooks<br/>graph.events 替换为 _EventSink"]
    HOOKS --> COMPILE["Builder.inference() → initial()<br/>无 Engram 时可复用折叠初始驻留缓存<br/>按现有 token → layer → component 生成事件"]
    COMPILE --> DECODE{"内部：standalone decode 且 step=0？"}
    DECODE -->|是| SEED["skip_prefill_kernel() / skip_prefill_collective()<br/>accept_event() 仅记录永久 KV/KDA 等状态"]
    SEED --> COMPILE
    DECODE -->|否| MACRO["before_component() → macro_key()<br/>尝试本 token 内同形组件宏重放"]
    MACRO --> ACCEPT["accept_event()<br/>state_hook() / after_component() 及 layer/token 屏障<br/>触发 flush()"]
    ACCEPT --> FLUSH["flush()<br/>验证跨块依赖均为已完成屏障<br/>带存活分配建立 chunk → apply_kernel_profiles()"]
    FLUSH --> KEY["内部：规范化层前缀构造 timing key<br/>保留容量/work、依赖、slot、dtype、alloc/release"]
    KEY --> HIT{"内部：已有 template 或宏？"}
    HIT -->|否且可缓存| TEMPLATE["simulate(unconstrained, retain_event_payload=False)<br/>记录 duration / work / peak与final delta"]
    HIT -->|是| FIT{"内部：当前驻留 + template peak delta<br/>满足所有真实 pool capacity？"}
    TEMPLATE --> FIT
    HIT -->|否且不缓存| SIM["simulate(chunk, retain_event_payload=False)<br/>完整 slot / memory admission"]
    FIT -->|否| SIM
    FIT -->|是| REPLAY["内部：复用时间与资源结果<br/>按当前驻留恢复真实 peak / final / 分类"]
    SIM --> MERGE["内部：累计 duration / work / busy / peaks / phase / chunks<br/>可选 progress 回调"]
    REPLAY --> MERGE
    MERGE --> STATUS{"内部：chunk feasible？"}
    STATUS -->|否| STOP["内部：以 _Stop 结束构图<br/>失败保留诊断；prefill 到目标后正常停止"]
    STATUS -->|是| CARRY["内部：累计本块通信；按 releases 更新 named 存活项<br/>永久状态按 pool/category/dtype 合并<br/>满足条件时保存组件宏"]
    CARRY --> TOKEN{"内部：token complete？"}
    TOKEN -->|是| MARK["内部：记录 marker；清理本 token 的宏与旧屏障"]
    TOKEN -->|否| CONT["内部：清空 sink，继续构图"]
    MARK --> PREFILL{"内部：phase=prefill 且 token0 完成？"}
    PREFILL -->|是| STOP
    PREFILL -->|否| CONT
    CONT --> COMPILE
    STOP --> CLEAN["内部 finally：移除 hooks 与闭包引用，清空 sink"]
    COMPILE -->|正常结束| CLEAN
    CLEAN --> METRIC["可行且有 token 标记时派生指标<br/>all 用 execution_metrics()；prefill 仅 TTFT<br/>decode 仅 TPOT；失败 TTFT/TPOT=null"]
    METRIC --> OUT["返回紧凑数值结果<br/>simulation / phase_metrics / chunks / marker派生指标"]
```

分块使用既有 state/component/layer/token 屏障，保留原 kernel、collective、placement、slot 竞争与内存准入；跨块不能引用尚未完成的依赖。永久分配按 pool/category/dtype 折叠，仍保持各池容量约束；临时存活分配按 ID 跨块携带。时序模板只在当前驻留加峰值增量适配真实容量时复用，否则回退正常准入调度；无约束模板本身不是可行性结论。组件宏仅在没有新存活项且未释放已有 named 项时保存，每个 token 后清理。

默认初始驻留与 timing 模板共用 LRU，预算为 64 MiB / 256 项；超大项跳过保留。调用者可传 `BoundedCache` 跨实验共用预算，普通 dict 则延续无界行为。`_size_bytes()` 按 Python 对象估算条目占用并计入 LRU 额外额度，并非进程 RSS 上限；宏、拓扑、当前 chunk、metadata 和摘要另有占用。

`phase=all` 完整执行 prefill+decode，也是 feedback 使用的唯一 streaming 模式。`phase=prefill` 在首 token 屏障后结束；`phase=decode` 直接播种 prefill 完成后留下的永久 KV/KDA 等状态，跳过 prefill 执行与瞬时内存，要求至少两个输出 token，TTFT 为 null。暖缓存 decode 可用于 prefill 不可行时的独立实验，不能证明联合请求可行。

chunk 失败会保留已观察 duration/work/peak、该 chunk 摘要和诊断并停止；失败 chunk 通信不并入成功执行的 phase 汇总，`memory_final_bytes` 保留最后成功 chunk 的值。progress 回调异常向外传播，finally 仍拆除引用环。返回值没有逐事件 trace/张量生命周期，不是完整 `run()` 或标准 `write_report()` 输入。

## 14. Skill 前端入口、Web 作业与隔离 worker

源码：[`launch_frontend.py`](../skills/supernode-modeling/scripts/launch_frontend.py)（`launch()`）、[`web.py`](../supernode_sim/web.py)（`create_server()`、`WebHandler._route()`、`validate_config()`、`JobManager`）、[`web_worker.py`](../supernode_sim/web_worker.py)（`main()`、`compact_result()`）。独立图：[14-web-worker.mmd](diagrams/14-web-worker.mmd)。

```mermaid
flowchart TD
    SKILL["Skill 调用 launch_frontend.launch()"] --> DISCOVER["discover_project() → select_python() → _preflight()<br/>定位引擎与解释器，检查依赖与静态文件"]
    DISCOVER --> REUSE["_ready() / _identity()<br/>匹配 engine root、output 与健康状态<br/>复用服务或在可用端口启动"]
    REUSE --> URL["_result()<br/>返回 url / inputs_url / workspace_url<br/>由 Skill 打开 inputs_url"]
    URL --> EDIT["内部：前端可编辑输入页<br/>载入预设/JSON/历史副本 → 编辑 → 校验/提交"]
    ENTRY["cli._main() → web.main()<br/>或 python -m supernode_sim.web"] --> SERVER["create_server() → WebServer.__init__()<br/>本机 IPv4 loopback / localhost"]
    REUSE -->|需要新服务| SERVER
    SERVER --> RESTORE["JobManager.__init__()<br/>读取 job.json；遗留 active 标记 failed"]
    RESTORE --> HTTP["WebHandler.do_GET() / do_HEAD() / do_POST()<br/>_handle() → _route() → _check_request()"]
    EDIT --> HTTP
    HTTP --> ROUTE{"内部：API / 静态资源"}
    ROUTE -->|schema / presets / health / static| READ["内部：只读响应 / _file()"]
    ROUTE -->|POST validate| VALID["validate_config()<br/>拒绝 calibration 路径、batch > 100000<br/>Config.model_validate()"]
    ROUTE -->|POST jobs| SUBMIT["JobManager.submit() → validate_config()<br/>仅 estimate / simulate；一次一个 active job"]
    SUBMIT --> SAVE["内部：创建目录，保存 input.config.json / job.json<br/>线程调用 _execute()"]
    SAVE --> PROCESS["JobManager._execute()<br/>启动 python -u -m supernode_sim.web_worker<br/>隔离子进程，Windows CREATE_NO_WINDOW"]
    PROCESS --> WORKER["web_worker.main()<br/>禁用 GC → 读取并校验 input.config.json"]
    WORKER --> RUN["analysis.run(config, mode, full_detail=False)<br/>完整构图路径，见图 1 / 8"]
    RUN --> EXPORT["write_report(result, directory, full_detail=False)<br/>见图 9"]
    EXPORT --> UI["compact_result()<br/>有界展示摘要 → 临时文件 replace ui.result.json"]
    UI --> END["内部：输出 completed/result_status 进度行<br/>finally 恢复 GC；异常输出 failed 并返回1"]
    PROCESS --> PROGRESS["内部：逐行读取 stdout → worker.log<br/>解析 _phase() JSON 更新 job phase/log"]
    END --> PROGRESS
    PROGRESS --> COMPLETE["内部：未 cancelled 时判断进程结果<br/>退出0且 ui.result.json 存在则 completed，否则 failed<br/>独立保留 result_status"]
    ROUTE -->|GET jobs / result / artifacts| QUERY["JobManager.get() / list() / directory()<br/>completed 后读取 UI 结果；产物按列出的文件名下载"]
    ROUTE -->|POST cancel| CANCEL["JobManager.cancel()<br/>标记 cancelled，terminate 后必要时 kill"]
    CANCEL --> PROCESS
    SERVER -->|服务关闭| CLOSE["server_close() → JobManager.close()<br/>取消并回收活动 worker"]
```

launcher 只负责定位、检查、复用/启动并返回地址，不打开浏览器、不启动仿真；Skill 打开返回的 `inputs_url`（`#inputs`）后，用户可修改输入再校验/提交。历史配置复制为当前编辑副本，不覆写旧任务的输入快照；提交时统一经过服务端配置校验。

HTTP 层检查 Host 与修改请求来源，JSON 请求体上限 2 MiB，输入必须已解析，不能通过 calibration 路径让服务端读取文件。Web 仅提供 estimate/simulate 作业，不提供 feedback/独立 streaming 入口。

`completed` 表示 worker 已完成报告和 UI 摘要；物理不可行或 SLO 不满足通过独立 `result_status` 呈现。UI 最多展示 240 个 memory pools、160 个 resources、均匀采样的 300 个 events，解释 metadata 也有上限；报告保留完整数值汇总，下载明细仍遵循图 9 的默认 compact 规则。服务重启不会恢复正在执行的作业。

## 核对范围

函数名称、关键分支、支持范围与接口按本工作区源码静态核对。流程图为可读性省略字段构造、部分 barrier 与重复 rank 循环，不能替代精确执行 trace。内嵌图与独立 Mermaid 文件同步生成；本次为文档更新，未据此声称新增数值验证结果。

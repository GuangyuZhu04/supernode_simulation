# 函数级仿真流程图

配套说明：[超节点仿真 Skill 架构](simulation-architecture.md)。依据当前源码整理，日期 2026-09-28。以下 11 张函数级图覆盖命令入口、两类训练/推理构图、优化器、通信、执行、归因导出、搜索和恢复。

2026-10-19 补充主链中的可选 PD/投机构图与 megakernel 变换；特性输入和具体支持边界见 [features.md](../skills/supernode-modeling/references/features.md)。

阅读约定：带 `()` 的节点标注源码中的真实函数/方法；标注“内部”的节点表示该函数体内的逻辑块，不是新函数。实线箭头表示调用顺序或控制流；循环是在 Python 中**编译事件图**，不是执行真实模型算子。运行时事件依赖由 `Event.deps` 表达，图 8 描述它们如何被执行。

Mermaid 代码块可在支持 Mermaid 的 Markdown 预览器中直接显示；每张图另存为 [`diagrams/`](diagrams/) 下的 `.mmd` 源文件，可继续编辑或导出 SVG/PDF。总体分层图位于架构文档，并提供 [`00-architecture.mmd`](diagrams/00-architecture.mmd)。

## 1. CLI 与单次仿真主链

源码：[`cli.main()`](../supernode_sim/cli.py#L84)、[`cli._main()`](../supernode_sim/cli.py#L42)、[`load_config()`](../supernode_sim/config.py#L453)、[`analysis.run()`](../supernode_sim/analysis.py#L244)、[`compile_graph()`](../supernode_sim/analysis.py#L21)。独立图：[01-entry-and-run.mmd](diagrams/01-entry-and-run.mmd)。

```mermaid
flowchart TD
    CLI["snmodel / python -m supernode_sim"] --> MAIN["cli.main() → _main() → parser()"]
    MAIN --> CMD{"内部：命令类型"}
    CMD -->|schema| SCHEMA["Config.model_json_schema() → 写出 schema"]
    CMD -->|import-hf| HF["import_hf_config() → read_data()<br/>模型片段 + missing_fields"]
    CMD -->|其余命令| LOAD["load_config() → read_data()<br/>可选校准 deep_merge() → Config.model_validate()"]
    LOAD --> SUB{"内部：命令分派"}
    SUB -->|validate| VALID["返回 valid；不构图"]
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
    MODE -->|是| SIM["engine.simulate(graph)<br/>analysis 中别名 simulate_graph"]
    SIM --> MET["execution_metrics() → evaluate_targets()<br/>内部：汇总最终 status"]
    MODE -->|否| MEM["build_memory_breakdown()<br/>见图 9"]
    MET --> MEM
    MEM --> RESULT["run() 返回结果字典"]
    RESULT --> WRITE["CLI：write_report(result, output)"]
    SEARCH --> WRITE
    WRITE --> EXIT["打印摘要并返回退出码<br/>main() finally 恢复 GC 状态"]
```

`run()` 是共享计算入口；Python 调用者自行决定是否调用 `write_report()`。simulate 失败也保留诊断与已观察的内存账本；输入/构图异常由 CLI 捕获并返回 `invalid_input`。`validate` 只运行配置层校验。

## 2. 参数预算版训练构图

源码：[`pipeline_schedule()`](../supernode_sim/training.py#L19)、[`_TrainingBuilder.build()`](../supernode_sim/training.py#L521)、[`.phase()`](../supernode_sim/training.py#L267)、[`.optimizer()`](../supernode_sim/training.py#L445)。独立图：[02-training-budget.mmd](diagrams/02-training-budget.mmd)。

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

源码：[`build_structured()`](../supernode_sim/structured.py#L792)、[`Builder.training_graph()`](../supernode_sim/structured.py#L577)、[`.gather()`](../supernode_sim/structured.py#L192)、[`.recompute_layer()`](../supernode_sim/structured.py#L335)。独立图：[03-training-structured.mmd](diagrams/03-training-structured.mmd)。

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

源码：[`_TrainingBuilder.optimizer()`](../supernode_sim/training.py#L445)、[`Builder.optimizer()`](../supernode_sim/structured.py#L720)、[`optimizer_states()`](../supernode_sim/components.py#L262)。独立图：[04-optimizer-offload.mmd](diagrams/04-optimizer-offload.mmd)。

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

## 5. 结构化推理：prefill / decode 与状态生命周期

源码：[`Builder.inference()`](../supernode_sim/structured.py#L386)、[`.kernel()`](../supernode_sim/structured.py#L216)、[`attention_state()`](../supernode_sim/components.py#L410)、[`attnres_state()`](../supernode_sim/components.py#L386)。独立图：[05-inference-structured.mmd](diagrams/05-inference-structured.mmd)。

```mermaid
flowchart TD
    ENTRY["build_structured() → Builder.__init__()"] --> INIT["Builder.inference() → initial()<br/>驻留权重 / runtime / 可选 Engram"]
    INIT --> DP["内部：遍历 DP replica"]
    DP --> TOKEN["内部：遍历 output step<br/>0 = prefill；其余 = decode"]
    TOKEN --> ACTIVE["内部：筛选 output_length 大于 step 的序列<br/>按 EP lane 分配 tokens"]
    ACTIVE --> LAYER["内部：遍历 layer → 确定 PP stage"]
    LAYER --> STATE["attention_state() / attnres_state()<br/>KDA 固定状态 或 paged KV 增量<br/>可选残差状态"]
    STATE --> COMPONENT["component_order(layer)<br/>遍历 embedding / norm / attention / FFN / output"]
    COMPONENT --> DISPATCH["MoE 时 collective(EP dispatch)"]
    DISPATCH --> ACT["逐 rank activation_rows()<br/>prefill Output 仅算每序列末位"]
    ACT --> READ["内部可选：旧 KV fetch → decode conversion<br/>PreNorm 前 Engram lookup"]
    READ --> KERNEL["kernel()<br/>权重转换 + component Event"]
    KERNEL --> WRITE["内部可选：新 KV encode → offload write"]
    WRITE --> COMM["MoE combine<br/>attention / FFN 后 TP all_reduce"]
    COMM --> MOREC{"内部：还有组件？"}
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
    MORED -->|否| RETURN["annotate_collective_allocations()<br/>返回 Graph"]
```

KDA recurrent/conv state 不随上下文线性增长；MLA cache 可用 compressed/expanded 布局。非 KDA 的旧 KV 转换具有显式解码 buffer。各序列完成后停止后续计算和 KV 读取，cache 保留至本次固定批次结束。同一 replica 按 token→layer 构图，不使用训练的 1F1B。

## 6. 参数预算版推理

源码：[`build_inference()`](../supernode_sim/inference.py#L46)、[`kv_layout_bytes()`](../supernode_sim/inference.py#L18)。独立图：[06-inference-budget.mmd](diagrams/06-inference-budget.mmd)。

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

源码：[`Builder.collective()`](../supernode_sim/structured.py#L102)、[`Topology.collective()`](../supernode_sim/topology.py#L254)、[`.transfer()`](../supernode_sim/topology.py#L206)、[`.path()`](../supernode_sim/topology.py#L163)。独立图：[07-collective-topology.mmd](diagrams/07-collective-topology.mmd)。

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
    PATH --> ROUTE["内部依次：缓存 → 按 SU/SO 策略验证显式 routes<br/>→ 同 SU full_mesh 懒建直连<br/>→ 确定性最少 hop 搜索（按策略筛选链路）"]
    ROUTE --> WORK["transfer() 内部：端点内存、逐跳链路<br/>injection / shared group / NIC / bridge 工作量<br/>startup = 路径 latency 之和"]
    WORK --> EVENT["返回 Event<br/>payload 与 physical bytes 分别记 metadata"]
    EVENT --> BUFS["collective() 内部：添加 send / recv allocation<br/>完成时 release，设置 tx / rx slots<br/>追加到 graph.events"]
    BUFS --> NEXT{"内部：还有轮次 / 阶段？"}
    NEXT -->|是，前轮 terminals 作为 deps| TRANS
    NEXT -->|否| TERMINALS["返回末轮 terminal event IDs"]
    TERMINALS --> UNPACK["构图器：必要时 unpack 与 wire buffer 释放"]
    IDENTITY --> UNPACK
```

`transfer()` 本身返回资源化的 Event；collective 的 staging buffer/slot 在 `collective()` 内添加，普通点对点 buffer 由对应构图器管理。AG/RS/AR 的 `size` 是每 rank 完整张量大小。所有同轮 transfer 的完成 ID 作为下一轮依赖，图表达的是明确的通信算法，而非一个总字节量常数。

## 8. 离散事件执行器与 max-min 资源共享

源码：[`engine.simulate()`](../supernode_sim/engine.py#L56)、[`max_min_rates()`](../supernode_sim/engine.py#L18)。独立图：[08-event-engine.mmd](diagrams/08-event-engine.mmd)。

```mermaid
flowchart TD
    ENTRY["engine.simulate(graph)"] --> CHECK["内部：校验唯一 ID、资源、allocation、依赖和 DAG"]
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

事件启动时原子分配内存并占用 slot，事件结束时执行 release。相同时间点先释放再准入新事件。资源同时服务，计算/访存耗时取 roofline 瓶颈。图中为便于阅读将若干校验错误合并到 `fail(error, ...)`，具体 code 以 diagnostics 为准。

## 9. 内存归因与报告导出

源码：[`build_memory_breakdown()`](../supernode_sim/memory_breakdown.py#L118)、[`write_report()`](../supernode_sim/report.py#L355)、[`_prepare_export()`](../supernode_sim/report.py#L249)。独立图：[09-memory-and-report.mmd](diagrams/09-memory-and-report.mmd)。

```mermaid
flowchart TD
    ENTRY["analysis.run() 的 graph / simulation / estimate"] --> MEM["build_memory_breakdown()"]
    MEM --> INDEX["闭包 add_tensor() → _labels()<br/>索引全部 allocations 与释放事件<br/>可省略临时 buffers 的宽明细行"]
    INDEX --> LOG{"内部：存在 memory_log？"}
    LOG -->|是| REPLAY["内部：按 sequence 有序回放<br/>校验 ID、pool、bytes、时间与生命周期"]
    REPLAY --> CHANGE["闭包 change()<br/>更新池/组件/状态与集群并发峰值"]
    LOG -->|否| RESIDENT["对 initial residency 调用 change()<br/>mode = resident_only"]
    CHANGE --> STATUS["内部：完整执行为 simulated<br/>提前停止为 observed_partial"]
    STATUS --> TABLES["内部：生成 tensors / pools / component_state_rows<br/>state_rows / cluster / logical states"]
    RESIDENT --> TABLES
    TABLES --> GROUPS["_equivalent_groups()<br/>返回 memory_breakdown"]
    GROUPS --> RUNRESULT["run() 返回完整结果对象"]
    RUNRESULT --> WRITE["write_report(result, directory, full_detail)"]
    WRITE --> PREP["_prepare_export()<br/>检测大图/已精简明细，生成 artifact_export"]
    PREP --> DETAIL{"内部：请求 full detail 但明细已丢失？"}
    DETAIL -->|是| ERROR["ValueError：必须从计算阶段重新保留完整明细"]
    DETAIL -->|否| JSON["内部：写 result.json / config.resolved.json<br/>markdown_report() → report.md"]
    JSON --> CSV["_csv()<br/>资源、内存、组件、状态、张量、峰值、参数"]
    CSV --> SIM{"内部：存在 simulation？"}
    SIM -->|是| TRACE["_csv(events.csv)<br/>_write_timeline(timeline.json)"]
    SIM -->|否| OUT["返回输出目录"]
    TRACE --> OUT
```

此图按职责组织输出节点；`write_report()` 的具体写文件顺序是 result/Markdown/config → resources/memory → events/timeline → 组件等内存 CSV。明细过滤不改变所有事件参与计算得到的资源总量或并发峰值。不同组件/不同池的独立峰值不允许直接相加。

## 10. 参数扫描与最小 SU 搜索

源码：[`sweep()`](../supernode_sim/search.py#L42)、[`size_su()`](../supernode_sim/search.py#L71)。独立图：[10-sweep-and-su-search.mmd](diagrams/10-sweep-and-su-search.mmd)。

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

源码：[`scripts/run_reference_experiment.py`](../scripts/run_reference_experiment.py)、[`fingerprint()`](../scripts/run_reference_experiment.py#L55)、[`load_checkpoint()`](../scripts/run_reference_experiment.py#L64)。独立图：[11-reference-checkpoint.mmd](diagrams/11-reference-checkpoint.mmd)。

```mermaid
flowchart TD
    ENTRY["run_reference_experiment.main()"] --> CONFIG["load_config() → fingerprint()<br/>resolved config + 指定数值源码"]
    CONFIG --> WRAP["内部：保存 analysis 原函数并安装进度包装<br/>准备 checkpoint/fingerprint 前20位目录"]
    WRAP --> RUN["analysis.run(config, simulate, full_detail=False)"]
    RUN --> COMPILE["包装 compile_graph() → 原 compile_graph()<br/>包装 graph_estimate() → 原 graph_estimate()"]
    COMPILE --> SIM["包装 simulate_graph(graph)"]
    SIM --> EXISTS{"内部：已提交 manifest.json？"}
    EXISTS -->|否| PHYSICS["原 engine.simulate(graph)"]
    PHYSICS --> SAVE["save_records(trace) + save_records(memory_log)<br/>gzip JSONL 临时文件完成后替换"]
    SAVE --> COMMIT["内部：最后提交 manifest.json<br/>fingerprint / count / size / simulation 摘要"]
    EXISTS -->|是| LOAD["load_checkpoint(directory, fingerprint)"]
    COMMIT --> LOAD
    LOAD --> CHECK["内部：校验 fingerprint 与文件大小<br/>大记录使用 JSONLRecords，否则加载 list"]
    CHECK --> METRIC["run() 继续 execution_metrics() / evaluate_targets()"]
    METRIC --> MEM["包装 breakdown() → build_memory_breakdown()<br/>按 sequence 回放日志，流式记录省临时明细"]
    MEM --> EXPORT["write_report(result, output)<br/>继续生成汇总与导出"]
    EXPORT --> FINALLY["内部 finally：恢复 analysis 原函数与 GC 状态"]
```

`JSONLRecords.__iter__()` 在迭代结束时核验记录条数。恢复只跳过已完成的数值执行，仍需重建图和回放日志；不支持在数值执行中途恢复 active/ready/remaining 状态。精简 `result.json` 不能替代原始检查点。

## 核对范围

函数名称、关键分支及源文件位置已按本工作区源码核对。流程图按可读性省略字段构造、少量 barrier 和重复 rank 循环，不能用作精确框架执行 trace。此次只生成与静态核对文档，未执行仿真或测试，也未通过 Mermaid 渲染器验证排版。

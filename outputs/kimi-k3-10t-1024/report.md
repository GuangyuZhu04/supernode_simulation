# Kimi K3 专家扩容至约10T：1024卡推理流量与延迟

**报告状态：已完成。已读取 48/48 组结果。**

生成时间：2026-10-15T16:43:01+08:00（Asia/Shanghai）。Prefill状态：feasible=12；infeasible=36。

## 场景与指标

只增加routed experts和相应router输出维度，保留93层、69 KDA+24 MLA、92 MoE层、hidden7168、latent3584、专家宽3072、2个shared experts。当前严格引擎要求EP整除专家数；3328专家是最接近10T的合法EP128配置，每EP26专家，精确总参数10.171385035872T（较10T多1.71385%）。Top32活跃154.414012512B，Top64活跃251.654756448B。该扩容模型是合成设计，不是官方发布的10T Kimi模型。

1024张Ascend950DT假设卡：8卡/节点、64卡/SU机柜、16柜/超节点SO域。TP8/EP128/PP1/DP1，TP完全留在节点内；序列按i%128分配到EP，B≤32时每个活跃EP持有一条完整序列，不按1024卡切开长上下文。每卡144GB HBM、4TB/s标称HBM带宽；FP8权重/计算，FP16通信、activation与KV，FP32 KDA状态。SU为单向1TB/s/卡、80%效率；SO为400Gb/s/节点、80%效率（有效40GB/s/节点），柜间共享400GB/s/方向上联，显式路径4个SO hops。

TTFT为固定batch执行开始到首token完成；TPOT为首token后的decode间隔，单位分别为秒和毫秒。未完成prefill时TTFT为空；独立decode从已驻留完整KV的状态开始，只给出条件性的单步decode时间，不代表该配置能完成端到端请求。只模拟一个首token后decode步，无排队、continuous batching或P99保证。输入128k/256k/512k/1M分别为131072/262144/524288/1048576 tokens/序列。

## 上下文与Top-k概览

Decode栏的*表示独立KV驻留decode，未标*为端到端请求中的TPOT。完整batch明细随后列出。

| Top-k | 输入长度 | Prefill组数 | TTFT B1 s | TTFT B32 s | Decode B1 ms | Decode B32 ms |
|---|---|---|---|---|---|---|
| 32 | 128k | feasible:6 | 1,046.782 | 1,109.548 | 463.531 | 487.698 |
| 32 | 256k | infeasible:6 | — | — | 465.796* | 489.963* |
| 32 | 512k | infeasible:6 | — | — | 470.326* | 494.493* |
| 32 | 1M | infeasible:6 | — | — | 479.386* | 503.553* |
| 64 | 128k | feasible:6 | 2,084.445 | 2,208.881 | 471.472 | 496.575 |
| 64 | 256k | infeasible:6 | — | — | 473.737* | 498.840* |
| 64 | 512k | infeasible:6 | — | — | 478.267* | 503.370* |
| 64 | 1M | infeasible:6 | — | — | 487.326* | 512.430* |

Decode-step吞吐为batch/单步decode时间，单列于summary.csv；端到端输出吞吐包括TTFT，另列full_run_output_throughput_tokens_s，二者不能互换。

## 全部48组延迟与容量

“独立decode”与端到端TPOT分别列出；—表示未定义、未执行或尚无结果。prefill失败时，已记录的耗时和流量只属于中断前执行，不能作为完整prefill预测。

### Top32 · 128k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | feasible | 1,046.782 | 463.531 | — | 118.971 | — | — |
| 2 | feasible | 1,055.459 | 466.915 | — | 118.971 | — | — |
| 4 | feasible | 1,072.814 | 473.681 | — | 118.971 | — | — |
| 8 | feasible | 1,107.523 | 487.210 | — | 118.971 | — | — |
| 16 | feasible | 1,108.198 | 487.386 | — | 118.971 | — | — |
| 32 | feasible | 1,109.548 | 487.698 | — | 118.971 | — | — |

### Top32 · 256k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 465.796 | 126.462 | 145.257 | 49 |
| 2 | infeasible | — | — | 469.179 | 126.462 | 145.257 | 49 |
| 4 | infeasible | — | — | 475.946 | 126.462 | 145.257 | 49 |
| 8 | infeasible | — | — | 489.475 | 126.462 | 145.257 | 49 |
| 16 | infeasible | — | — | 489.651 | 126.462 | 145.257 | 49 |
| 32 | infeasible | — | — | 489.963 | 126.462 | 145.257 | 49 |

### Top32 · 512k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 470.326 | 141.483 | 179.071 | 25 |
| 2 | infeasible | — | — | 473.709 | 141.483 | 179.071 | 25 |
| 4 | infeasible | — | — | 480.475 | 141.483 | 179.071 | 25 |
| 8 | infeasible | — | — | 494.005 | 141.483 | 179.071 | 25 |
| 16 | infeasible | — | — | 494.180 | 141.483 | 179.071 | 25 |
| 32 | infeasible | — | — | 494.493 | 141.483 | 179.071 | 25 |

### Top32 · 1M

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 479.386 | 62.663 | 167.914 | 1 |
| 2 | infeasible | — | — | 482.769 | 62.663 | 167.914 | 1 |
| 4 | infeasible | — | — | 489.535 | 62.663 | 167.914 | 1 |
| 8 | infeasible | — | — | 503.065 | 62.663 | 167.914 | 1 |
| 16 | infeasible | — | — | 503.240 | 62.663 | 167.914 | 1 |
| 32 | infeasible | — | — | 503.553 | 62.663 | 167.914 | 1 |

### Top64 · 128k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | feasible | 2,084.445 | 471.472 | — | 118.971 | — | — |
| 2 | feasible | 2,101.761 | 474.945 | — | 118.971 | — | — |
| 4 | feasible | 2,136.394 | 481.887 | — | 118.971 | — | — |
| 8 | feasible | 2,205.661 | 495.762 | — | 118.971 | — | — |
| 16 | feasible | 2,206.734 | 496.076 | — | 118.971 | — | — |
| 32 | feasible | 2,208.881 | 496.575 | — | 118.971 | — | — |

### Top64 · 256k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 473.737 | 126.462 | 145.257 | 49 |
| 2 | infeasible | — | — | 477.210 | 126.462 | 145.257 | 49 |
| 4 | infeasible | — | — | 484.152 | 126.462 | 145.257 | 49 |
| 8 | infeasible | — | — | 498.027 | 126.462 | 145.257 | 49 |
| 16 | infeasible | — | — | 498.341 | 126.462 | 145.257 | 49 |
| 32 | infeasible | — | — | 498.840 | 126.462 | 145.257 | 49 |

### Top64 · 512k

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 478.267 | 141.483 | 179.071 | 25 |
| 2 | infeasible | — | — | 481.739 | 141.483 | 179.071 | 25 |
| 4 | infeasible | — | — | 488.682 | 141.483 | 179.071 | 25 |
| 8 | infeasible | — | — | 502.557 | 141.483 | 179.071 | 25 |
| 16 | infeasible | — | — | 502.871 | 141.483 | 179.071 | 25 |
| 32 | infeasible | — | — | 503.370 | 141.483 | 179.071 | 25 |

### Top64 · 1M

| Batch | Prefill状态 | TTFT s | 端到端TPOT ms | 独立decode ms | 已准入峰值 GB | 失败申请所需 GB | 失败层 |
|---|---|---|---|---|---|---|---|
| 1 | infeasible | — | — | 487.326 | 62.663 | 167.914 | 1 |
| 2 | infeasible | — | — | 490.799 | 62.663 | 167.914 | 1 |
| 4 | infeasible | — | — | 497.742 | 62.663 | 167.914 | 1 |
| 8 | infeasible | — | — | 511.617 | 62.663 | 167.914 | 1 |
| 16 | infeasible | — | — | 511.931 | 62.663 | 167.914 | 1 |
| 32 | infeasible | — | — | 512.430 | 62.663 | 167.914 | 1 |

## 网络流量

以下为全超节点聚合的逻辑transport payload：每个有向发送计一次，不把发送与接收重复相加。dispatch/combine是all_to_all_v的语义视图，不能再次叠加all_to_all_v。柜内payload与SU hopbytes相等；柜间4-hop物理字节为柜间payload的4倍。HBM工作量是读写流量，不是HBM容量。

完整prefill流量由独立公式给出；不可行配置的完整流量是计划需求，未声称实际执行完成。失败阶段的通信只累计已完成组件，communication_scope=completed_components_only；HBM工作量还包括失败组件中已执行的部分事件，两者范围不同。Decode实际流量已逐项与公式校验（总payload、dispatch、combine、TP、SU、SO，容差1B或1e-9相对误差）。

| Top-k | Batch | Decode总 MB/步 | 柜间 MB/步 | 柜间占比 | 128k prefill TB | 256k TB | 512k TB | 1M TB |
|---|---|---|---|---|---|---|---|---|
| 32 | 1 | 372.335 | 316.539 | 85.01% | 48.803 | 97.605 | 195.211 | 390.421 |
| 32 | 2 | 744.669 | 633.078 | 85.01% | 97.605 | 195.211 | 390.421 | 780.842 |
| 32 | 4 | 1,489.338 | 1,266.156 | 85.01% | 195.211 | 390.421 | 780.842 | 1,561.684 |
| 32 | 8 | 2,978.677 | 2,532.311 | 85.01% | 390.421 | 780.842 | 1,561.684 | 3,123.369 |
| 32 | 16 | 5,957.353 | 5,064.622 | 85.01% | 780.842 | 1,561.684 | 3,123.369 | 6,246.738 |
| 32 | 32 | 11,914.707 | 10,129.244 | 85.01% | 1,561.684 | 3,123.369 | 6,246.738 | 12,493.476 |
| 64 | 1 | 707.338 | 633.078 | 89.50% | 92.712 | 185.424 | 370.849 | 741.698 |
| 64 | 2 | 1,414.676 | 1,266.156 | 89.50% | 185.424 | 370.849 | 741.698 | 1,483.396 |
| 64 | 4 | 2,829.353 | 2,532.311 | 89.50% | 370.849 | 741.698 | 1,483.396 | 2,966.792 |
| 64 | 8 | 5,658.706 | 5,064.622 | 89.50% | 741.698 | 1,483.396 | 2,966.792 | 5,933.583 |
| 64 | 16 | 11,317.412 | 10,129.244 | 89.50% | 1,483.396 | 2,966.792 | 5,933.583 | 11,867.166 |
| 64 | 32 | 22,634.824 | 20,258.488 | 89.50% | 2,966.792 | 5,933.583 | 11,867.166 | 23,734.333 |

Top64的MoE和SO流量严格为Top32的2倍；TP all-reduce不变，因此总network payload增至1.899738倍。通信量随prefill总token和decode batch线性增加。序列长度影响本地KV读和attention计算，当前布局下单步decode网络payload不随上下文长度变化。

当前pairwise EP128每次dispatch/combine有127轮，92层共两次；TP使用ring。仅collective启动延迟的算法下界已约450–473ms/步，解释低batch decode为何不能仅按总字节/总带宽估算。该数值来自所选算法和路径延迟，未用真实collective测量校准。

## Prefill实际流量与完整计划

| 案例 | 状态 | 实际已执行 TB | 完整计划 TB | HBM工作量 TB | 单卡最大HBM工作量 TB |
|---|---|---|---|---|---|
| top32-128k-b1 | feasible | 48.803 | 48.803 | 198.321 | 15.480 |
| top32-128k-b2 | feasible | 97.605 | 97.605 | 377.299 | 15.534 |
| top32-128k-b4 | feasible | 195.211 | 195.211 | 735.255 | 15.643 |
| top32-128k-b8 | feasible | 390.421 | 390.421 | 1,451.166 | 15.861 |
| top32-128k-b16 | feasible | 780.842 | 780.842 | 2,882.988 | 16.296 |
| top32-128k-b32 | feasible | 1,561.684 | 1,561.684 | 5,746.632 | 17.167 |
| top32-256k-b1 | infeasible | 49.915 | 97.605 | 171.830 | 13.192 |
| top32-256k-b2 | infeasible | 99.830 | 195.211 | 333.779 | 13.247 |
| top32-256k-b4 | infeasible | 199.660 | 390.421 | 657.676 | 13.358 |
| top32-256k-b8 | infeasible | 399.320 | 780.842 | 1,305.470 | 13.581 |
| top32-256k-b16 | infeasible | 798.641 | 1,561.684 | 2,601.057 | 14.026 |
| top32-256k-b32 | infeasible | 1,597.281 | 3,123.369 | 5,192.232 | 14.915 |
| top32-512k-b1 | infeasible | 48.960 | 195.211 | 153.201 | 11.639 |
| top32-512k-b2 | infeasible | 97.921 | 390.421 | 301.566 | 11.693 |
| top32-512k-b4 | infeasible | 195.842 | 780.842 | 598.295 | 11.802 |
| top32-512k-b8 | infeasible | 391.684 | 1,561.684 | 1,191.755 | 12.020 |
| top32-512k-b16 | infeasible | 783.368 | 3,123.369 | 2,378.673 | 12.455 |
| top32-512k-b32 | infeasible | 1,566.735 | 6,246.738 | 4,752.511 | 13.326 |
| top32-1M-b1 | infeasible | 0.000 | 390.421 | 0.248 | 0.031 |
| top32-1M-b2 | infeasible | 0.000 | 780.842 | 0.496 | 0.031 |
| top32-1M-b4 | infeasible | 0.000 | 1,561.684 | 0.992 | 0.031 |
| top32-1M-b8 | infeasible | 0.000 | 3,123.369 | 1.984 | 0.031 |
| top32-1M-b16 | infeasible | 0.000 | 6,246.738 | 3.969 | 0.031 |
| top32-1M-b32 | infeasible | 0.000 | 12,493.476 | 7.937 | 0.031 |
| top64-128k-b1 | feasible | 92.712 | 92.712 | 295.636 | 20.978 |
| top64-128k-b2 | feasible | 185.424 | 185.424 | 571.929 | 21.085 |
| top64-128k-b4 | feasible | 370.849 | 370.849 | 1,124.514 | 21.298 |
| top64-128k-b8 | feasible | 741.698 | 741.698 | 2,229.684 | 21.726 |
| top64-128k-b16 | feasible | 1,483.396 | 1,483.396 | 4,440.025 | 22.581 |
| top64-128k-b32 | feasible | 2,966.792 | 2,966.792 | 8,860.707 | 24.292 |
| top64-256k-b1 | infeasible | 94.779 | 185.424 | 271.261 | 18.809 |
| top64-256k-b2 | infeasible | 189.558 | 370.849 | 532.640 | 18.918 |
| top64-256k-b4 | infeasible | 379.117 | 741.698 | 1,055.397 | 19.137 |
| top64-256k-b8 | infeasible | 758.234 | 1,483.396 | 2,100.913 | 19.574 |
| top64-256k-b16 | infeasible | 1,516.467 | 2,966.792 | 4,191.943 | 20.448 |
| top64-256k-b32 | infeasible | 3,032.934 | 5,933.583 | 8,374.004 | 22.195 |
| top64-512k-b1 | infeasible | 92.870 | 370.849 | 250.516 | 17.137 |
| top64-512k-b2 | infeasible | 185.740 | 741.698 | 496.195 | 17.244 |
| top64-512k-b4 | infeasible | 371.480 | 1,483.396 | 987.555 | 17.458 |
| top64-512k-b8 | infeasible | 742.961 | 2,966.792 | 1,970.273 | 17.885 |
| top64-512k-b16 | infeasible | 1,485.921 | 5,933.583 | 3,935.711 | 18.741 |
| top64-512k-b32 | infeasible | 2,971.842 | 11,867.166 | 7,866.586 | 20.451 |
| top64-1M-b1 | infeasible | 0.000 | 741.698 | 0.248 | 0.031 |
| top64-1M-b2 | infeasible | 0.000 | 1,483.396 | 0.496 | 0.031 |
| top64-1M-b4 | infeasible | 0.000 | 2,966.792 | 0.992 | 0.031 |
| top64-1M-b8 | infeasible | 0.000 | 5,933.583 | 1.984 | 0.031 |
| top64-1M-b16 | infeasible | 0.000 | 11,867.166 | 3.969 | 0.031 |
| top64-1M-b32 | infeasible | 0.000 | 23,734.333 | 7.937 | 0.031 |

## 内存峰值与失败定位

已准入峰值只包含成功分配的内存，不含被拒绝的工作区，因此可能低于144GB仍发生OOM。“失败申请所需”取阻塞事件的144GB容量+shortage_bytes，表示该事件在失败时加入所需的单卡容量下界。不是完整执行最终峰值，也不能将它补成TTFT。组件表取同一张卡同一峰值时刻的构成，可以相加；不把不同组件各自独立的峰值相加。EP使routed权重分片，但dense/shared/router和部分投影复制：逻辑10.171T参数不能直接除1024得到每卡驻留量。以配置为准，权重约23.603GB/卡、FP8 scale约0.738GB/卡，加8GB runtime和0.256GB workspace，静态约32.597GB/卡。

| 案例 | 峰值卡 | 峰值 GB | 同一峰值构成 GB |
|---|---|---|---|
| top32-128k-b1 | npu0.hbm | 118.971 | page_table=0.002；workspace=0.256；recurrent_state=0.054；kv_cache=3.624；activation=67.660；quantization_metadata=0.738；weight=23.603；convolution_state=0.003；attnres_residual=15.032；runtime=8.000 |
| top32-256k-b1 | npu0.hbm | 126.462 | page_table=0.002；workspace=0.256；recurrent_state=0.028；kv_cache=3.624；activation=75.178；quantization_metadata=0.738；weight=23.603；convolution_state=0.001；attnres_residual=15.032；runtime=8.000 |
| top32-256k-b2 | npu0.hbm | 126.462 | quantization_metadata=0.738；recurrent_state=0.028；convolution_state=0.001；runtime=8.000；attnres_residual=15.032；activation=75.178；workspace=0.256；weight=23.603；page_table=0.002；kv_cache=3.624 |
| top32-256k-b4 | npu0.hbm | 126.462 | recurrent_state=0.028；convolution_state=0.001；kv_cache=3.624；page_table=0.002；quantization_metadata=0.738；attnres_residual=15.032；workspace=0.256；activation=75.178；runtime=8.000；weight=23.603 |
| top32-256k-b8 | npu0.hbm | 126.462 | page_table=0.002；workspace=0.256；recurrent_state=0.028；kv_cache=3.624；activation=75.178；quantization_metadata=0.738；weight=23.603；convolution_state=0.001；attnres_residual=15.032；runtime=8.000 |

| 案例 | 失败层(1-based) | 组件 | 失败组件已准入 GB | 失败申请所需 GB | 诊断（完整见failure_diagnostics.json） |
|---|---|---|---|---|---|
| top32-256k-b1 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-256k-b2 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-256k-b4 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-256k-b8 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-256k-b16 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-256k-b32 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top32-512k-b1 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-512k-b2 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-512k-b4 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-512k-b8 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-512k-b16 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-512k-b32 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top32-1M-b1 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top32-1M-b2 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top32-1M-b4 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top32-1M-b8 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top32-1M-b16 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top32-1M-b32 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-256k-b1 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-256k-b2 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-256k-b4 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-256k-b8 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-256k-b16 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-256k-b32 | 49 | AttnRes | 55.043 | 145.257 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l48.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 1256538388.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l48.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 1256538388.0}}]}] |
| top64-512k-b1 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-512k-b2 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-512k-b4 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-512k-b8 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-512k-b16 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-512k-b32 | 25 | AttnRes | 58.787 | 179.071 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l24.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 35070877972.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l24.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 35070877972.0}}]}] |
| top64-1M-b1 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 8, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-1M-b2 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 16, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-1M-b4 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 32, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-1M-b8 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 64, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-1M-b16 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 128, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |
| top64-1M-b32 | 1 | AttnRes | 47.630 | 167.914 | [{&quot;code&quot;: &quot;memory_admission_deadlock&quot;, &quot;message&quot;: &quot;Ready events cannot fit and no running event can release memory&quot;, &quot;blocked_event_count&quot;: 256, &quot;blocked_sample&quot;: [{&quot;event&quot;: &quot;infer.d0.s0.l0.npu0.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu0.hbm&quot;: 23914414356.0}}, {&quot;event&quot;: &quot;infer.d0.s0.l0.npu1.AttnRes&quot;, &quot;shortage_bytes&quot;: {&quot;npu1.hbm&quot;: 23914414356.0}}]}] |

## Compressed / expanded MLA缓存敏感性

当前主实验使用compressed MLA优化假设。官方参考布局expanded每层每卡3840个FP16元素/token，compressed为576，前者KV为后者6.667倍。下表只给缓存/状态容量，不是对expanded进行过延迟仿真。B≤32时每活跃EP一条序列，单卡容量不随batch变化。

| 输入tokens | Compressed MLA KV GB/卡 | Expanded MLA KV GB/卡 | KDA状态 GB/卡 | 页表 GB/卡 |
|---|---|---|---|---|
| 131072 | 3.624 | 24.159 | 0.056807 | 0.001573 |
| 262144 | 7.248 | 48.318 | 0.056807 | 0.003146 |
| 524288 | 14.496 | 96.637 | 0.056807 | 0.006291 |
| 1048576 | 28.991 | 193.274 | 0.056807 | 0.012583 |

Expanded 1M仅MLA KV就193.274GB/卡，已超过144GB；compressed 1M KV为28.991GB，但prefill activation、AttnRes和通信临时缓冲仍可能导致OOM。独立decode无需同时持有prefill临时张量，所以其可行性必须单独解释。

## 适用范围与复现

预测使用均匀专家路由、均衡专家计算、通用融合/流式attention和粗粒度kernel。当前MoE组件把routed与shared路径合并近似，并按均衡EP token数量计时；实际shared/latent/router计算可能受源序列EP集中影响，缺少生产kernel和路由trace校准。B≤32时attention、dense和KV集中在少数EP组，1024卡并不意味着所有算子获得1024卡并行。完整prefill不按chunk重新调度，未优化长序列切分。绝对TTFT/TPOT是此配置、调度和算法的分析预测，未宣称真实硬件误差。

节点/机柜流量见[node_traffic.csv](node_traffic.csv)与[rack_traffic.csv](rack_traffic.csv)：分别记录128节点、16机柜的每阶段SO tx/rx逻辑NIC payload、SU tx/rx、平均B/s及有效容量利用率。tx/rx分开，不能把两者相加当作全局payload。可行阶段已校验sum(host SO tx)=SO hopbytes/4、sum(host SU tx)=SU hopbytes；失败阶段使用实际resource work，含失败组件部分事件，不能与仅完整组件通信计数强行对齐。节点SO有效40GB/s/方向，机柜上联有效320GB/s/方向。

结果：summary.csv（所有48组，原始byte/s），phase_traffic.csv（实际各阶段语义payload、SU/SO hopbytes、HBM总工作量和最大单卡），planned_phase_traffic.csv（全部48组×prefill/decode的完整计划流量、语义payload及hopbytes），peak_components.csv（同峰值组件），cases/*.json（仿真原始结果），audit/traffic-formulas.json（独立公式），configs/*.json与hardware.profile.json（可重跑配置）。report.html提供离线交互筛选和图表。

本次主任务验证记录：269项测试与7项子测试通过。报告生成器另外核对实际通信与独立公式、节点注入计数，以及离线HTML脚本语法/筛选/SVG生成；脚本检查不等同于浏览器视觉检查。

生成报告：`python scripts/summarize_kimi_10t_inference.py outputs/kimi-k3-10t-1024`。

逐案例配置+数值模块指纹已核对当前源码，共48份；详见report_status.json。不同负载配置的指纹自然不同。

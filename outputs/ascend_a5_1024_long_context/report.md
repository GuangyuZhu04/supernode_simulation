# Ascend 950DT × Kimi K3：1024 卡长上下文流量仿真

运行日期：2026-10-08。用户确认 A5 按 Ascend 950DT 计算，k=1024、1M=1,048,576 tokens，保留全部超窗口压力场景。

**结果：12 组请求的完整流量需求已输出；128k Prefill 在当前执行模型下可完成，256k/512k/1M 的整段 Prefill 在 AttnRes 暂存分配时失败。长 Decode 的网络流量按引擎公式精确累计，时间来自已执行的上下文快照和插值，属于预测。**

## 场景与硬件

| 项目 | 本次配置 |
|---|---|
| 模型 | Kimi K3 文本 backbone，完整 93 层；69 KDA、24 MLA、92 MoE、896 专家 top-16、AttnRes block=12 |
| 参数 | 张量形状推导约 2.77948T；不包含视觉 encoder、projector、训练 MTP |
| 拓扑 | 128 主机 × 8 卡；8 SU × 128 卡；一个 SO 域共 1024 卡；SU 和 SO 为显式 Clos 流体图 |
| 并行 | TP8 × EP128 × PP1 × DP1；单请求 batch=1，TP 组在每台 8 卡主机内 |
| 存储/计算精度 | 权重 FP8、计算 FP8；activation/通信/MLA KV FP16；KDA recurrent FP32 |
| 每卡硬件 | 144 GB HBM、4 TB/s HBM、FP8 1 PFLOPS；计算效率假设 50% |
| SU | 每卡每方向假设 1 TB/s × 80%=800 GB/s；每 leaf 共享有效 6.4 TB/s |
| SO | 每主机共享假设 400 Gbit/s × 80%=40 GB/s/方向；每 SU 聚合有效 640 GB/s/方向 |
| 路径 | TP 同主机 2 SU hop / 1.05 µs；跨主机同 SU 4 SU hop / 2.1 µs；跨 SU 6 SO hop / 5 µs |
| 负载 | Prefill 128k/256k/512k/1M × 输出 4k/16k/128k，共 12 组；首输出由 Prefill 产生 |

官方 2 TB/s 互联的方向尚不明确，SU 单向速率、SO NIC、CPU、启动延迟与效率均是单独标注的未实测假设。FP8 是重精度设计，不是官方 MXFP4 混合 checkpoint 的完全复现。容量与带宽均使用十进制 GB/TB/PB。

## 12 组流量结果

下表为**整请求需求**，即完整模型执行所需的传输载荷。失败场景不表示这些字节已传输；已完成字节保存在每个 `raw/prefill_*.json` 的 `completed_communication`。SU/SO physical 是逐链路 hop 累计，不能与 transport 相加。

| Prefill | 输出 | Prefill载荷 TB | Decode载荷 TB | 总载荷 TB | SU载荷 TB | SO载荷 TB | SU逐跳 TB | SO逐跳 TB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128k | 4k | 26.848 | 0.839 | **27.687** | 7.720 | 19.967 | 20.788 | 119.800 |
| 128k | 16k | 26.848 | 3.356 | **30.204** | 8.422 | 21.782 | 22.678 | 130.691 |
| 128k | 128k | 26.848 | 26.848 | **53.695** | 14.972 | 38.723 | 40.317 | 232.340 |
| 256k | 4k | 53.696 | 0.839 | **54.534** | 15.206 | 39.328 | 40.947 | 235.970 |
| 256k | 16k | 53.696 | 3.356 | **57.051** | 15.908 | 41.143 | 42.837 | 246.861 |
| 256k | 128k | 53.696 | 26.848 | **80.543** | 22.458 | 58.085 | 60.475 | 348.510 |
| 512k | 4k | 107.391 | 0.839 | **108.230** | 30.178 | 78.052 | 81.264 | 468.311 |
| 512k | 16k | 107.391 | 3.356 | **110.747** | 30.880 | 79.867 | 83.153 | 479.201 |
| 512k | 128k | 107.391 | 26.848 | **134.239** | 37.431 | 96.808 | 100.792 | 580.850 |
| 1M | 4k | 214.783 | 0.839 | **215.622** | 60.123 | 155.499 | 161.897 | 932.992 |
| 1M | 16k | 214.783 | 3.356 | **218.138** | 60.825 | 157.314 | 163.787 | 943.883 |
| 1M | 128k | 214.783 | 26.848 | **241.630** | 67.375 | 174.255 | 181.426 | 1045.532 |

每处理 token 的期望载荷：204,832,768 B；其中 SU=57,114,624 B，SO=147,718,144 B（72.12%）。逐跳物理字节 SU=153,796,608 B、SO=886,308,864 B。

```text
TP allreduce = 93 × 2 × 2 × (8−1) × 7168 × 2 = 37,330,944 B/token
MoE dispatch = MoE combine = 92 × 8 × 127 × (16 × 3584 × 2 / 128)
             = 83,750,912 B/token
整请求载荷 = 204,832,768 × (Prefill长度 + 输出长度 − 1)
```

Dispatch/combine 是 all-to-all-v 的两个语义分项；不能再加到 all-to-all-v 统计上。没有启用 KV offload，因此 KV 增长和读取主要计入 HBM，不额外产生 KV 的 SU/SO 传输。

## 执行状态与时间

| Prefill | Prefill状态 | TTFT 秒 | 首次失败位置/最小当次分配需求 |
|---|---|---:|---|
| 128k | 可完成 | 493.248 | 完成；实测仿真峰值 109.87 GB/卡 |
| 256k | 当前策略内存失败 | — | 第 61 层 AttnRes；最热卡当次至少需 155.87 GB |
| 512k | 当前策略内存失败 | — | 第 25 层 AttnRes；最热卡当次至少需 169.97 GB |
| 1M | 当前策略内存失败 | — | 第 1 层 AttnRes；最热卡当次至少需 158.82 GB |

以下 Decode 时间**以 cache 已建立为条件**。Prefill 失败的组合没有端到端 TTFT/吞吐；这里只给 Decode 需求与时间预测。1M Prefill 的三组同时超过官方 1M 总窗口，按用户确认保留为压力外推。

| Prefill | 输出 | Decode平均 TPOT ms | Decode预测分钟 | 整请求预测分钟 | 模型1M窗口 |
|---|---|---:|---:|---:|---|
| 128k | 4k | 120.717 | 8.239 | 16.460 | 窗口内 |
| 128k | 16k | 120.824 | 32.991 | 41.212 | 窗口内 |
| 128k | 128k | 121.814 | 266.106 | 274.327 | 窗口内 |
| 256k | 4k | 122.982 | 8.394 | — | 窗口内 |
| 256k | 16k | 123.089 | 33.609 | — | 窗口内 |
| 256k | 128k | 124.079 | 271.054 | — | 窗口内 |
| 512k | 4k | 127.512 | 8.703 | — | 窗口内 |
| 512k | 16k | 127.618 | 34.846 | — | 窗口内 |
| 512k | 128k | 128.609 | 280.949 | — | 窗口内 |
| 1M | 4k | 136.572 | 9.321 | — | 超窗口压力场景 |
| 1M | 16k | 136.678 | 37.320 | — | 超窗口压力场景 |
| 1M | 128k | 137.669 | 300.740 | — | 超窗口压力场景 |

## 结果含义与边界

- 单请求由 EP lane 0 所属的 TP8 承载 attention、KV 和 AttnRes，EP128 主要分担专家。总 HBM 147.456 TB 不能直接用于这 8 张热卡的缓存和暂存；本引擎没有 context parallelism。
- 长 Prefill 失败源于整段执行及未融合 AttnRes 的 FP32 暂存。它是本次执行布局的结果，不能据此判定生产 Kimi K3 在 1024 卡上无法处理长上下文。评估 fused/streaming AttnRes、chunked Prefill 或 CP 需要实现或提供对应 kernel/allocator trace 后重跑。
- MLA 压缩缓存的逻辑量为 24 × 576 × 2=27,648 B/token/序列；每个 TP rank 保存一份。1M 输入时每热卡 KV 数据约 28.991 GB，8 卡合计约 231.928 GB，另有分页表和 KDA 固定状态。Expanded 参考布局是压缩布局的 6.667 倍，本场景不能把压缩布局当作官方 HF 缓存事实。
- SO 共享 NIC 使发起请求的单台主机成为重要通信瓶颈；专家虽然放在 128 台主机上，源请求的吞吐不会自动获得所有 NIC 的聚合带宽。
- 128k Prefill 中，源 host0 NIC TX/RX 各访问 9.680856 TB；在本次 dispatch 与 combine 顺序执行的 DAG 中，两阶段 NIC 服务约 484.043 秒，占 TTFT 约 98.13%。TX/RX 容量独立；一般场景不能随意将两个双工资源下界相加。
- 专家路由为均匀期望模型。batch1/top16 在真实运行中只触达有限专家，而本引擎向 EP128 分数分摊 token，pairwise collective 的启动轮次与负载均衡因此是模型假设。它不覆盖稀疏路由 trace、热点、ECMP、包级拥塞或 P99。
- CSV 中 HBM_access_PB_demand 是全 1024 卡、整个请求的重复读写访问量，并非存储容量。当前 MoE kernel 按均匀分数 token 执行，非专家/shared 权重在 EP 间复制并按该 kernel 布局读取；此字段属于引擎需求账本，不是生产稀疏路由的实测访存 trace。
- 所有时间均未进行 Ascend 实机 kernel/collective 校准；固定请求仿真不包含服务排队、连续 batching、PD 分离、投机推理或 prefix cache。
- 用户允许 Kimi K3 或 DeepSeek v4.1 Flash，本次选择已有完整引擎适配的 Kimi。DeepSeek v4.1 Flash 的 CED、CSA2 跨层 KV 共享及 FP4 cache 尚未适配，未用通用 MLA 冒充该模型。

## 方法、验证与复跑

1. `prepare_cases.py` 从官方结构案例和拓扑编译产物创建 12 个严格原生配置，并通过 `Config.model_validate`。
2. `streaming_runner.py` 在原引擎的 `layer_done` / `token_complete` 完整障碍处分段执行，保持资源争用及跨层驻留；静态分配仅按池、状态、组件、精度合并。原引擎代码未修改。
3. 四个完整 93 层 Prefill 已执行；内存失败后继续编译未执行层，只统计完整需求，保留真实停止位置及已完成流量。
4. 9 个 Decode 上下文快照已执行。D 个输出实际包含 D−1 Decode 步骤；网络字节按精确常量求和，resource work 经所有资源仿射检验，最大相对残差 0。耗时按整数位置对快照分段线性积分。
5. 快照耗时相对首尾直线的最大已观察偏差 0.000000 ms。该值仅反映采样点曲率，不是实机误差或未采样点的严格误差上界。
6. 流式执行与原引擎在 93 层小负载、TP2/EP2 跨 SU、容量失败场景交叉验证；资源访问量、通信量、峰值和耗时一致。另用独立整数公式审核全部 12 组网络字节。

```powershell
.\.venv\Scripts\python.exe outputs/ascend_a5_1024_long_context/prepare_cases.py
.\.venv\Scripts\python.exe outputs/ascend_a5_1024_long_context/run_suite.py
.\.venv\Scripts\python.exe outputs/ascend_a5_1024_long_context/render_outputs.py
```

`run_suite.py` 会检查原生配置及引擎/流式执行器 SHA256，复用匹配的已完成 raw 文件。长 Decode 原生配置用于表达完整需求；直接用原 CLI 做逐 token 完整仿真会展开极大事件图，本实验使用这里明确记录的快照积分流程。最后输出位置为零基编号 L+D−1，总输入加输出长度为 L+D，官方窗口判断使用后者。

关键文件：`summary.csv`、`summary.json`、`results.html`、`raw/*.json`、`configs/*.json`、`audit_calculations.json`、`streaming_validation.json`、`memory_capacity.csv`、`timing_sampling_audit.json`、`topology/`。

## 官方来源

- [Huawei 950DT 官方规格与互联方向边界](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)
- [Huawei Atlas 950 规模](https://www.huawei.com/en/news/2026/3/mwc-superpod-ai)
- [Kimi K3 官方配置](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json)
- [Kimi K3 官方实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)
- [DeepSeek V4.1 Flash 官方模型说明](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)

来源核验和结构说明另见 `source_verification.txt` 以及仓库 `docs/sources/README.md`。

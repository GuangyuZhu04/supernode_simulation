# DavidV100 微架构服务需求

用于依据《基于 MicroBenchmark 揭秘 DavidV100 芯片性能特性》V1.0（2026 年 11 月）改善子算子资源约束。报告环境为 **DavidV100、32 个 AI Core、CANN 9.1.T560、AIBench 与随包 msprof**；物理架构的 36 个 AI Core 与实测启用配置不同，一个 AI Core 含一个 AIC、两个 AIV。这里的 DavidV100 是 Ascend 芯片，不是 NVIDIA V100。

这是有来源的解析细化，须用目标负载的留出实测评估预测误差。没有同条件实测时报告“资源约束已细化、精度待验证”，不承诺加速比或误差下降。DavidV121、DavidV200、Ascend 950 及其他机型必须独立校准，不由名称推定参数等价。

## 从路径到服务需求

先标记实际输入输出、生产者与消费者、tile、缓冲生命周期和每条路径的字节数：

```text
矩阵：GM(HBM/L2) → MTE2 → L1 → MTE1 → L0A/L0B → Cube → L0C → FixPipe
SIMD：GM ↔ MTE ↔ UB ↔ VLD/VST ↔ 向量寄存器
SIMT：GM ↔ DCache/GSU ↔ 寄存器；访问 UB 也经过 GSU
控制：Scalar 下发、pipe 事件、STARS 供给、核间/流间同步
```

区分有效数据、实际传输字节、重复搬运、填充计算和驻留容量；Scalar DCache、SIMT DCache、L2 与软件管理的 L1 不是同一层。不能仅因工作集小于标称容量就保证缓存命中。

统一为秒后计算：

```text
计算服务秒 = 实际执行 ops / (同模式 ops/cycle × 对应频率 Hz × 参与 AIC 数)
搬运服务秒 = 对应路径实际 bytes / 同粒度、同方向和同并发范围的有效 bytes/s
发射服务秒 = 指令数 / (可比有效 IPC × 对应频率 Hz)
子算子服务下界 = max(各共享资源累计服务秒, 必须串行的依赖路径秒, 原计算/HBM下界)
```

矩阵乘加计 2 ops。逻辑 FLOPs 与补齐后的执行工作量分别核算；不能为了使用 HF32 的吞吐而静默改变严格 FP32 的精度语义。每项资源先汇总，再比较独立资源；只有证据支持的并发才按最大值处理。外部启动成本仍单独按原执行契约计算，内部不能掩盖的依赖用 `dependency_floor_s` 表达，不重复添加。

使用 [子算子接口](operator-traces.md#可选的微架构服务需求) 的 `operations[].resource_demands`。trace 绑定同一硬件、phase、dtype、工作量和结构元数据；软件、实现或缓存条件影响结果时，加入 `calibration_context` 与 `implementation` 身份。服务需求本身不会推断这些条件。

## 生成可绑定的资源需求

项目内辅助器将显式工作量转换为服务需求；它不修改现有场景，不创建实测 kernel profile：

```powershell
python skills/supernode-modeling/scripts/build_david_v100_demands.py skills/supernode-modeling/assets/david-v100-workload.synthetic.json -o outputs/david-v100-demands.json
python skills/supernode-modeling/scripts/build_david_v100_demands.py path/to/workload.json -o outputs/david-v100-demands.json --operators-output outputs/david-v100-operators.json --kernel explicit_kernel --trace-id david-v100-scoped
```

首条命令使用合成输入：64³ 严格 FP32 的有效工作量为 524288 ops，**1 GHz 只是算术验证假设，不是 DavidV100 时钟规格**。其中 Cube 与 SIMT 访存预算仅验证计算公式，不代表真实 MMAD kernel 的数据路径。实际任务须替换为编译实现和测量得到的工作量、频率与搬运路径。

输入是 JSON 对象，固定参考身份为 `device_sku: "DavidV100"`、`cann_version: "9.1.T560"`、`enabled_ai_cores: 32`。这些字段声明所用参考环境，不能替代当前设备检测。按实际工作量提供以下内容：

| 字段 | 契约 |
| --- | --- |
| `frequency_hz` | Cube 周期换算所需的实际频率；没有时钟证据时不生成数值计算服务需求 |
| `cube.mode` | `fp32`、`hf32`、`fp16`、`bf16`、`int8`、`fp8` 或 `hif8`；实际实现必须遵守对应精度语义 |
| `cube.useful_ops`, `cube.executed_ops` | 有效与实际执行 ops；执行量包含已核实的 padding/尾块额外工作，不由工具自动猜测 |
| `cube.active_aic` | 实际参与该工作量的 AIC 数，范围 1–32；不能填 AIV 数或 Block 数 |
| `memory` | 每项描述一个 `level: hbm/l2`；同一 level 最多一项，先合并共享层级的实际读写量 |
| `memory[].access_path` | `simt`、`mte` 或 `mixed`；MTE/SIMT 共用一个层级时用 `mixed`，不能各拿一份整芯片带宽 |
| `memory[].read_bytes`, `write_bytes` | 该层级真实读写 bytes；L2 流量与 HBM 流量分别提供，工具不推断命中率 |
| `memory[].effective_Bps`, `source` | MTE 或混合路径必须提供可比有效带宽及来源；显式带宽应保留测量对象、并发范围和读写组合 |
| `memory[].evidence_kind` | 显式带宽可标 `measured_reference` 或 `assumed`，默认后者；使用自动 SIMT 参考时不要覆盖来源或证据字段 |
| `dependency_floor_s`, `dependency_source` | 可选依赖路径下界及其计时来源；非零下界必须说明来源，不含外层启动 |

SIMT 未覆盖有效带宽时，工具按读、写或混合选择下表中的报告参考值；这是对应微基准条件下的服务下界，不能推定少核、小粒度请求也达到满核平台。MTE 与混合路径没有统一自动系数，需要显式带宽。工具不自动填充设备频率、UB/DCache 配置或 HBM 容量。

输出包含 `resource_demands`、`accounting`、`reference_context` 与 `limitations`。将需求绑定到对应 operation，保存核算和限制作为来源记录；不要把整个辅助器输出直接塞进严格 operation schema。核对 `accounting` 的有效/实际执行 ops 与各层级字节，原 operation 的 `flops` 仍守恒，`memory_bytes` 仍代表实际 HBM 访问。若需要不同芯片或软件版本，先建立独立证据与适配器，不修改参考身份来强行通过。

`david_hbm`、`david_l2` 只由声明相应需求的算子共享；offload 与通信仍通过原 `npu.hbm` 参与争用。辅助器不替换整芯片全局带宽，`hardware.npu.hbm_Bps` 仍须按真实设备及相应流量条件独立配置。不同方向和路径的归一化服务需求是固定预算近似，不模拟并发时动态变化的混合带宽曲线。

第二条命令额外输出 `{id, source, operations}`，可交给 [build_operator_trace.py](../scripts/build_operator_trace.py) 绑定具体父事件。生成的单 operation 使用 `useful_ops` 和 HBM 读写总量作为预算，并在 `implementation.david_v100_service_model` 保留完整输入；不是完整场景，也不自动完成 kernel 身份测量。先用绑定工具的 `--list` 检查目标，确认工作量、shape、dtype 与实现一致。生成标签要求 trace 的运行环境明确匹配 DavidV100、CANN 9.1.T560，Cube 模式还须与计算 dtype 一致；不能绑定到 950 场景。HIF8/HF32 等模式是否能进入当前执行配置，还受引擎精度支持和实际数值契约约束。

## 报告参数的采用范围

| 报告章节 | 可用的证据 | 建模边界 |
| --- | --- | --- |
| §2.2.1–§2.2.3 | 64³ MMAD：严格 FP32 512、HF32 4068、FP16/BF16 8192、INT8/FP8/HIF8 16384 ops/cycle | 绑定模式、参与 AIC 数与实际时钟；92 cycle 不能当作整个 MMAD 服务时间。严格 FP32 的 64³ 工作量按吞吐约需 1024 cycle，而非 92 cycle |
| §2.2.3 | FP32 M/N 越过 16 元素边界有计算台阶，报告推断计算分形 16×1×16 | 计算补齐与 L0 存储格式分开，不能由计算 K 步长推定存储 K 对齐为 1；其他 dtype 的合法 tile 查当前 SDK |
| §3.1.2 | MTE/FixPipe 核外路径：单核读约 154、写约 196 GB/s；满核 HBM 读 1512–1524、写约 1521 GB/s；L2 读 4574–4755、写 4800–5006 GB/s | GB/s 为十进制；满核结果不能乘 32，也不能把 AIC/AIV 测试峰值相加；小粒度固定开销另测，不由路径延迟倒数替代稳态带宽 |
| §3.4.2、§3.4.4.2 | SIMT HBM 读/写/混合总带宽 1506/1546/1289 GB/s；L2 为 5305/5235/5445 GB/s | 混合是总带宽，不是读写两个单向峰值之和；MTE 与 SIMT 仍可能共享核外资源。图中 55/46“核”需先核实 AIV/Block 口径 |
| §3.1.1、§3.4.1 | 单核有效 L2 约 64 MB、多核约 128 MB，涉及两个 Die | 不给每个核分配 128 MB 缓存；没有确证的地址交织函数和放置策略不能补造 |
| §3.3.1 | SIMD UB：两读 512、两写 256、一读一写合计 512 B/cycle；报告架构说明为 16 bank、32 B 交织 | 绑定访问组合、偏移、stride 和同拍 bank，不能将所有 512 B 间隔统一加罚时；这些速率不自动适用于 SIMT/GSU |
| §3.3.2、§3.4.3 | SIMD 与 FixPipe 无 bank 冲突时可基本重叠；SIMT 经 GSU 时，即使 bank 分开也有明显争用 | 不统一假定 Cube/Vector/FixPipe 完全重叠，也不把样例的近两倍拖尾设为通用惩罚系数 |
| §3.4.1、§2.3.2.2.3 | 动态 UB 与 DCache 的测试点及 spill 变化 | `U≤120 KB` 时 DCache 约 128 KB，`120<U≤216 KB` 时约 `248−U KB` 是对观测的推断。不能据此枚举任意合法容量、保证缓存命中或推导固定 spill 罚时 |
| §2.1.2 | pipe 下发队列深度与 outstanding 分别测量 | 两者不是 tile 数或内存请求并发数；不能把 N+K 当作推荐展开度，或将 AIV_VEC 的 K=1 当作内部仅能执行一条指令 |
| §4.1–§4.5 | Device Task 边界、供给状态、核间/流间同步 | 22 ns 仅是已准备任务的同流 Gap；空任务 PMU 对照不可机械扣除。barrier 协议增量不含全部负载不均衡等待，各同步模式的计时定义也不同 |

上述数字是报告实测或读图参考，不是任意算子的有效效率。ND2NZ 非对齐用例的平均降幅、DCache 指针追逐与逐指令的不同延迟、ATOM/RED 热点串行趋势，都应保留各自方法与语义；不合成为统一延迟或固定加速倍率。完整报告未确定的频率、L0/L1 容量、event 数、寄存器 API 和控制位不填默认常量。

## 证据与结果核对

`resource_demands.evidence_kind` 区分报告测量参考 `measured_reference`、架构说明 `architecture`、机制推断 `inferred` 与工程假设 `assumed`。由参考吞吐推算的服务秒仍是解析模型，不能升级成 `kernel_profiles.evidence_kind=measurement`；后者需要目标 kernel 同条件真机测量。报告资源建模覆盖率和精度验证分开。

验收优先选择能区分机制的边界：M/N=15/16/17、ND2NZ 行宽 128 B 附近、tile/缓冲/UB/DCache 容量点、读写混合与单向、空任务与连续异步供给。比较条件包含 dtype、layout/stride、启用 AIC/AIV、编译选项、线程与 Block、缓存冷热、PMU 开关、时间边界和重复次数。用同条件 A/B 及未参与拟合的目标负载报告误差和离散程度。

先检查执行明细中的资源需求、来源与依赖下界，再检查 `model_metadata.operator_trace.resource_modeled_operations`、`dependency_modeled_operations` 及实际匹配范围。新增约束应保留逻辑 FLOPs、HBM 总字节、依赖与容量语义；片上资源服务量不得被统计为额外 HBM 容量或网络流量。资源需求仅细化子算子服务下界，没有实现逐指令调度、自动缓存命中、bank 仲裁或动态 warp 占用。

报告来源为 `ascend-david-v100-operator-optimization` 技能的 `references/compute.md`、`memory.md`、`scheduling.md` 和 `evidence-and-measurement.md`；本页保留了执行建模所需口径，复制此 skill 后不要求通过用户目录绝对路径读取来源技能。

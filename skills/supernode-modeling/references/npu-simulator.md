# 将 NPUSim 算子仿真接入超节点模型

本入口读取 CANN NPUSim 已生成的单次 kernel 报告，将 CA 模式的设备执行窗口绑定到已展开子算子。它不编译 Ascend C、不运行 NPU/仿真器，也不从指令轨道重建硬件调度。先按 [operator-traces.md](operator-traces.md) 描述真实算子链和实现身份，再导入对应报告；一个 Add/MatMul kernel 的时间不能直接替换整个 Dense、Attention 或 MoE 组件。

## 选择证据和运行模式

- NPUSim 的 CA 模式提供时序仿真；当前导入器只接受 manifest 的 `mode: ca`。PV（`npusim record --pv`）仅用于快速功能验证，没有精确时序，不能导入性能 profile。
- `evidence_kind: simulator` 表示真实仿真器报告；示例与人为构造报告必须使用 `synthetic`。这两类都不能标为硬件实测：导入器拒绝 `source.status: measured`。
- `source.status` 描述来源渠道（`assumed/public/user`），与 `evidence_kind` 分开。公开示例不因 `public` 自动变成实测；`source` 的结构为 `{status, description, uri?, revision?}`。
- `--analyze` 只解析字段和单位，证据类型保持 `unspecified`，不证明 CA/PV 模式、正确性、采集完整性或算子身份。不能把其中换算出的 duration 当成已经验证的校准数据。

## 在 CANN 环境采集

记录实际 toolkit、NPUSim 版本/修订、目标 SoC、编译选项及设备 ELF 的 SHA-256。保持输入 shape、dtype、format、tiling key、blockDim、kernel binary 与拟绑定子算子一致。正确性由调用程序对 golden 结果验证，保留输入及比对日志；命令返回零或生成 summary 本身不能证明数值正确。

以下命令在安装了对应 CANN/NPUSim 的主机执行，`/path/to/operator_app` 是已经编译的算子应用：

~~~bash
npusim --help
npusim record --help
npusim report --help
npusim record /path/to/operator_app -s Ascend950 -o ./capture -n all --gen-report
# 未自动解析时，显式指定本次运行目录：
npusim report -e ./capture/npusim_<本次运行目录> -o ./analysis -n all
~~~

先核对当前版本支持的选项；旧版本命令名为 `cannsim`。保留完整命令与日志，单独选择目标 launch 的 `report/results/kernel_*_reports/summary.json`，不要用通配符隐式选择多次运行、把多个 kernel 求和，或拿工具的墙钟时间代替设备时间。

`record -n` 控制采集核范围；`report -n` 控制报告核范围。指定 `--gen-report` 却省略 `-n` 时，官方流程可能默认只报告 core 0。固定参考版本中，`kernel_info` 来自全输入 DB 的 KernelInfo，而部分 pipe/scalar/unit 指标按报告核过滤；单核 trace 不必然表示 kernel 只使用一个核。导入器要求声明 `all_active_cores` 并核对活跃核数，但不读取日志证实采集范围；单凭 `ai_core_active`、`per_core` 长度或 `imbalance_ratio=1` 也不能证明全核采集/均衡。使用 `-n all` 并保留日志可避免这类歧义。

用户提供的 ops-math 文档描述 Ascend950PR/950DT、单卡 0、AI Core 计算场景，不覆盖 MC2/HCCL，并写明不支持 ARM；较新的 npu-simulator README 已列出 aarch64/x86_64，源码还出现独立的通信 mock 选项。不要把不同版本能力混成统一承诺。本入口只接收本地计算子算子的报告，不导入通信、混合设备或多 rank 时间；运行架构和芯片支持以安装版本的帮助与配套 CANN 为准。

## summary 的单位与边界

官方 summary 的关键结构如下；数字取自上游两核 synthetic 测试 fixture，仅展示格式：

~~~json
{
  "kernel_info": {
    "kernel_total_clocks": 99,
    "frequency_ghz": 1.65,
    "ai_core_active": 2
  },
  "pipe_utilization": {
    "pipeline_util_summary": {}
  }
}
~~~

导入时只采用：

~~~text
duration_s = kernel_total_clocks / (frequency_ghz * 1e9)
duration_us = kernel_total_clocks / (frequency_ghz * 1e3)
~~~

`kernel_total_clocks` 是官方 `sim.KernelInfo.exec_time_ticks` 的转出值，导入器不以各核或指令时长重算它。它表示该 launch 的设备 kernel 窗口，已包含内部流水重叠；不乘活跃核数、不累加 MTE/VECTOR/CUBE 时长。示例的 99 clocks / 1.65 GHz 对应 0.06 μs，不是实测时延。

`pipe_utilization.pipeline_util_summary` 内各流水的 `mean/min/max/per_core` 为 0–1 比例，`per_core` 是数值数组。`top_level_diagnosis`、`pipeline_overlap`、`bandwidth`、`scalar_instructions`、`aiv_vector_instructions`、`cache`、`simd_vf_metrics` 保留作诊断；缺失、null 和真实零值不能互换。流水重叠率不是利用率相加，上游算法先合并区间，再取交集/并集。

指令文件为 Chrome Trace JSON，可能分片或 gzip，见 `core_*/trace_core*.json`。参考版本的 `ts/dur` 始终是微秒，`displayTimeUnit: ns` 仅控制显示；不能当成纳秒重复换算。summary 不包含本入口要求的 shape、dtype、实现哈希、CA/PV、正确性等身份字段，因此须另附 manifest。上游 `.npu-rep` 包中的 CSV 是 DB 表导出，不是本入口的统一 summary CSV；不要将 msprof 的 `OPPROF_*/simulator` 文件套作 NPUSim summary。

## 实现身份与 manifest

在每个 `hardware.kernel_traces.*[].operations[]` 中提供可选的 `implementation` 对象；普通 trace 可省略，NPUSim 绑定必须提供下列字段：

| 字段 | 内容 |
|---|---|
| `op_type`, `kernel_name`, `tiling_key` | 非空字符串，来自实际编译与调用；tiling key 即使为数字也用字符串 |
| `binary_sha256` | 编译后设备 kernel 的小写 64 位 SHA-256，不能填源码或 summary 哈希 |
| `block_dim` | 正整数，实际 launch 参数；不直接等同 `ai_core_active` |
| `inputs`, `outputs` | 非空数组；每项含 `shape`（正整数数组，标量用 `[]`）、`dtype`、`format` |

可附实际 `tiling_data`、stride、编译标志等 JSON 字段，整份 implementation 参与精确匹配。不要省略会改变执行路径的布局/尾块信息。导入器校验身份字段并与配置匹配，但不重算设备二进制哈希、不反向验证 tensor shape 对应的 FLOPs，也不执行 golden；这些证据由采集者提供。

manifest 完整字段如下。先用下方 `--list` 获取 `events` 中的完整目标对象，原样放入 `target`：

| 字段 | 要求 |
|---|---|
| `schema_version` | `"1.0"`，可省略 |
| `mode` | `"ca"`；拒绝 PV，命令含 `--pv` 也拒绝 |
| `correctness` | `"passed"`，有对应精度比对证据 |
| `capture_scope` | `"all_active_cores"`，有完整采集证据 |
| `expected_active_cores` | 正整数，必须等于 summary 的活跃核数 |
| `hardware_name` | 精确等于场景的 `hardware.name` |
| `simulator_version`, `simulator_revision` | 本次采集实际版本/修订，非空字符串 |
| `command` | 本次采集完整命令，非空字符串 |
| `summary_sha256` | 输入 summary 原始字节的小写 SHA-256；编辑空白也需更新 |
| `calibration_context` | 完整运行环境，精确等于场景中的同名硬件字段 |
| `target` | `--list` 中选中事件的完整对象，不能只填 event ID |
| `evidence_kind` | `"simulator"`（默认）或 `"synthetic"` |
| `startup_s` | 可省略/null：保留原启动成本；非负数：替换完整启动成本，需独立依据 |
| `source` | `{status, description, uri?, revision?}`；不能使用 `measured` |

`calibration_context` 必须包含非空字符串：`device_sku`、`cann_version`、`hccl_version`、`driver_version`、`runtime`、`runtime_version`、`transport`、`mode`、`timer_boundary`。其中 `timer_boundary` 固定为 `device_only`。不适用的组件记录具体事实，例如未安装驱动且使用 camodel、未调用 HCCL；不能拿 `unknown`/`n/a` 冒充已知环境。保持 profile 与场景一致，不能为命中校准而改写真实环境。

`target` 包含 `event_id/kernel/phase/dtype/flops/memory_bytes/original_startup_s/metadata_match`。`metadata_match` 绑定组件、layer、结构签名、tokens、workload shape、TP/EP/SP、trace ID、原 kernel、operation index、可选 matmul 维度与 implementation；以 `--list` 实际输出为准。同 FLOPs 或算子名不足以复用报告。

## 本项目入口与 synthetic 复现

从项目根目录执行。示例是人为构造的接口验证数据，不要求安装 CANN，不能据此声称得到 Ascend 实测/仿真性能提升。

~~~powershell
# 列出已经展开且有 implementation 的可绑定子算子
python skills/supernode-modeling/scripts/build_npu_sim_calibration.py examples/npu_sim/synthetic.config.json --list
# 只解析报告和时间单位，不生成校准
python skills/supernode-modeling/scripts/build_npu_sim_calibration.py --analyze --summary examples/npu_sim/summary.synthetic.json
# 校验 manifest 并生成可叠加的 kernel_profiles 片段
python skills/supernode-modeling/scripts/build_npu_sim_calibration.py examples/npu_sim/synthetic.config.json --summary examples/npu_sim/summary.synthetic.json --manifest examples/npu_sim/manifest.synthetic.json -o outputs/npu-sim/calibration.json
# 完整运行基线与导入后模型，保存配置、结果和对比
python skills/supernode-modeling/scripts/evaluate_npu_sim.py --output outputs/npu-sim
~~~

`--list` 默认最多 20 条，可用 `--limit` 调整，检查 `total_matching_events/truncated`。`--analyze -o <报告>` 可保存归一化结果。导入默认同时写出 `outputs/npu-sim/calibration.report.json`，也可用 `--report-output` 指定。输出是校准片段；将其与已有条目合并或通过场景顶层 `calibration` 引用，路径按场景所在目录解析，再运行 `validate` 与 `simulate`。重复/冲突的 profile 不会按列表顺序静默取一个。

只有已展开、有实现身份、具备结构和 workload shape、仅占用本地 compute/HBM 与 compute stream 的子算子可导入。未拆分组件、缺少 implementation、collective、megakernel 不在此入口支持范围。导入前检查 `--list`，不要绕过选择器将单 kernel 时间硬塞到整个组件。

## 算子内部流水与外层执行

外层仍按声明的子算子顺序执行。NPUSim kernel 时间作为子算子的独占服务时间约束，原 FLOPs、HBM 访问量、资源争用、workspace 与依赖继续生效，不将工作量清零。无争用时不能比导入 kernel 窗口更短；外部 HBM/计算争用可能延长完成时间。不要把内部 MTE2/MTE3 再添加成外层串行事件，否则会重复计算已含在 kernel 窗口中的搬运与重叠。片上路径带宽、工作集和流水占用率不能直接换成 HBM 字节或缓存容量。

官方 AddExample 展示了 shape 和 tiling 身份的重要性：host tiling 先根据 AIV 核数做 ceil 分块，再按 UB 容量为两个输入和一个输出的双缓冲预留六块空间并对齐，最后设置 blockDim 和 dtype 对应 tiling key。device 端裁剪尾核/尾 tile，用两深度队列组织 CopyIn → Compute → CopyOut。源码调用顺序不证明 DMA 与计算完全串行，队列及同步允许内部流水重叠，应以 CA 窗口和 trace 诊断确认。

检查导入报告的 target、provenance、startup policy 和匹配数，再查看完整运行中 `model_metadata.kernel_calibration` 的覆盖率与 evidence kind。仅将同形状、同实现、同环境报告用于当前子算子；prefill/decode、训练 forward/dgrad/wgrad/recompute 分开。性能改善结论仍需未参与校准的同条件验证；synthetic 用例只证明导入、单位与调度语义。

## 固定来源

于 2026-11-02 核读公开源码，不复制上游实现。以下链接固定到核读 revision：

- ops-math `c47568d3b5d22b1570bfbaad40dc63681ff5ca30`：[用户指定的 NPUSim 文档](https://gitcode.com/cann/ops-math/blob/c47568d3b5d22b1570bfbaad40dc63681ff5ca30/docs/zh/debug/npu_sim.md)、[构建/安装/执行脚本](https://gitcode.com/cann/ops-math/blob/c47568d3b5d22b1570bfbaad40dc63681ff5ca30/scripts/ci/run_cannsim_example.sh)、[AddExample tiling](https://gitcode.com/cann/ops-math/blob/c47568d3b5d22b1570bfbaad40dc63681ff5ca30/examples/add_example/op_host/add_example_tiling.cpp)、[AddExample kernel](https://gitcode.com/cann/ops-math/blob/c47568d3b5d22b1570bfbaad40dc63681ff5ca30/examples/add_example/op_kernel/add_example.h)。
- npu-simulator `8dc64db200f497c55db31d3f970e4cc054786a25`：[架构支持](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/README.md)、[使用指南](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/docs/zh/user_guide/01_user_guide.md)、[CLI 与 CA/PV](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/npusim/main.py)、[summary 聚合](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/npusim/prof_demo/reports/summary.py)、[DB 核范围](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/npusim/prof_demo/database.py)、[时钟单位](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/npusim/prof_demo/utils.py)、[两核 fixture 与采样测试](https://gitcode.com/cann/npu-simulator/blob/8dc64db200f497c55db31d3f970e4cc054786a25/tests/test_core/test_prof_demo.py)。

关键内容 SHA-256：NPUSim `summary.py` 为 `ae5fcad32e64b8f3a3e3d4b754d55a5054e1d51b32f74b2fa6181d68762c3537`；ops-math `npu_sim.md` 为 `f5c5178be56854f3aee1f6cfbae9cd2de39ab5d8e6401ed4f707b63e4f20146d`。这些是参考源码哈希，不可填作 manifest 的 summary/binary 哈希。

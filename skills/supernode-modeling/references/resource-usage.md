# 仿真器运行性能、内存与磁盘占用

本页控制运行 Python 仿真器的主机资源；不改变所模拟设备的 HBM/DDR、精度、并行布局、负载或调度策略。不能为了降低主机开销而缩小用户场景、跳过失败案例或把独立 decode 当作端到端成功。

## 执行方式

### Web 的主机资源保护

Web worker 在构图前应用操作系统内存硬上限，默认取主机总内存与当前可用内存各一半的较小值；Linux 额外遵守 cgroup v1/v2 容器额度，Windows 额外考虑可用提交额度，macOS 读取可回收内存。无法探测时回退到 1 GiB；不能建立保护或预算不足时明确失败。超限保留输入并释放任务槽位，不等到整机内存耗尽。

此预算按运行 Python 的主机确定，不按模拟的 NPU 型号/卡数决定，也不缩小模型或 token 数。更强的机器自动获得更大预算，不固定为本机的 GB 数。可在服务启动前用 `SUPERNODE_WEB_MEMORY_MIB` 指定 MiB 预算；必须为正整数，仍保留至少当前可用内存的 25%。未设置时保留一半可用内存。任务日志记录预算，完成后记录主机峰值；Windows 为进程提交内存，POSIX 上限为地址空间而峰值为 RSS，口径不同。

Web 大规模结构化推理按卡数、层数、最长输出和 TP/EP 估计图存储压力，结合主机预算自动选择流式引擎；只对满足原有 DP=PP=1 等支持条件的配置启用，不绕过 megakernel/PD 限制。流式时序缓存预算为进程预算的 10%，大主机可使用更大缓存；分配按实际需要增长，淘汰不改变数值。训练、估算、服务及不兼容的推理继续走原路径，同样受硬上限保护；预算不足明确报错，不伪造完成。

流式网页结果保留数值和分块时间，但不提供完整事件回放、张量生命周期或组件峰值构成。详细操作与产物见项目 `docs/web-interface.md`。CLI/批量搜索资源策略仍按下面的规则，不声称 Web 的进程限制自动覆盖外部 CLI。

- 结构化推理（`model.architecture`，DP=PP=1）只需要性能、流量和设备容量峰值时，使用 `supernode_sim.streaming.run_streaming`。它逐组件执行并释放事件图；支持 SP，以及显式 draft/verify/commit 投机轮次；投机分块与完整图共用轮次 hooks，按实际输出 token 数计时。Megakernel/PD 仍不支持此路径。返回数值 JSON，不提供普通报告的完整组件/张量生命周期。阶段解释见 [inference.md](inference.md#大规模结构化推理)。
- 训练、其他并行布局、完整组件/张量审计使用普通 `estimate` / `simulate`。默认大型导出精简事件与临时缓冲记录，完整汇总仍保留。
- 需要中断后恢复时使用 `scripts/run_reference_experiment.py`。检查点需要额外磁盘；普通 CLI 不额外保存检查点。先检查已有结果和指纹，避免重复运行同一场景。

流式 API 示例（从项目根目录、项目解释器运行）：

```python
import json
from pathlib import Path
from supernode_sim.cache import BoundedCache
from supernode_sim.config import load_config
from supernode_sim.streaming import run_streaming

config = load_config('path/to/scenario.yaml')
cache = BoundedCache(max_bytes=64 * 1024 * 1024, max_entries=256)
result = run_streaming(config, cache=cache)
output = Path('outputs/my-streaming-case')
output.mkdir(parents=True, exist_ok=True)
snapshot = config.model_dump()
snapshot['calibration'] = None  # profile 已解析进快照
for name, value in [('config.resolved.json', snapshot), ('result.json', result)]:
    with (output / name).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
```

## 缓存和批量实验

`sweep`、`size-su`、`feedback` 将独立候选分配给多个 Python 进程，适用于 CPU 密集的批量仿真。默认 `--workers 1`；`--workers 4` 最多使用四个子进程；`--workers 0` 按可用 CPU 自动选择，最多四个，并受实际候选数限制。结果仍按原候选规则排序，非法配置和不可行候选保留原诊断。主进程有界提交任务，子进程返回候选摘要，避免把全部事件图在进程间传输。

```text
python -m supernode_sim sweep examples/demo_training.yaml --axes examples/sweep_precision.yaml --workers 4 -o outputs/parallel-sweep
python -m supernode_sim size-su examples/demo_inference.yaml --workers 4 -o outputs/parallel-su
python -m supernode_sim feedback examples/demo_inference.yaml --search examples/feedback_inference.yaml --workers 4 -o outputs/parallel-feedback
```

进程数是仿真器主机的执行选项，不改变所模拟系统的 TP/EP/DP、事件依赖、资源争用或数值精度。单次 `simulate` 的事件调度仍在一个进程执行；重复计算优化自动生效。Windows 启动进程与导入模块有固定开销，只有少量短任务时串行通常更快。先测一个代表性候选的主机耗时和内存，再选择并发；自动模式限制 CPU 数量，但不探测可用内存，也不保证内存足够。

Python API 的 `workers` 为关键字参数。多进程脚本放在可导入的 `.py` 文件中，并保护入口；交互式控制台可使用 `workers=1` 或改用 CLI：

```python
from supernode_sim.config import load_config
from supernode_sim.search import sweep

if __name__ == '__main__':
    result = sweep(load_config('examples/demo_training.yaml'),
                   {'execution.precision.weight': ['fp8', 'fp16']}, workers=2)
```

`size_su(config, workers=4)` 和 `feedback(config, search_options, workers=4)` 使用相同接口；`workers` 不写入场景配置或 `FeedbackSpec`。

流式时序缓存默认最多 64 MiB 的 Python 对象计量、256 项；淘汰只触发重新计算，不放宽显存准入。跨场景共享时使用 `BoundedCache`；显式传入普通 `dict` 仍沿用调用者管理的无界行为。预算为 0 时关闭时序缓存。

两个 Kimi 批处理脚本支持 `--cache-mib`，例如：

```text
python scripts/run_kimi_10t_inference.py --run --cache-mib 32
python scripts/run_kimi_sp_inference.py --run --cache-mib 32
```

这些 Kimi 脚本要求对应目录已有 manifest、配置和硬件 profile，仍沿用各自的 shard 入口。缓存预算按每个进程计，不是进程 RSS 上限；事件、拓扑、结果和 Python 运行时另占内存。增加搜索 workers 或 shard 进程数前按各进程峰值评估总内存，避免在多个 shard 内重复开启满额并发。

静态拓扑的 pair 路由缓存最多 16 MiB / 65,536 项，各路由模式的源树缓存最多 16 MiB / 128 项。内置 `full_mesh` 会惰性增加 SU 边，其 pair 缓存保留历史语义，暂不限制；源树仍有界。拓扑本身的边和资源不受这些缓存预算约束。

带签名的 collective profile / 响应曲线自动复用相同硬件、物理 rank 和完整 payload 矩阵的签名，每个 Topology 最多 8 MiB / 1024 项；曲线模型在初始化时校验一次。phase、dtype、操作元数据和计时边界仍逐次匹配，缓存淘汰只触发重算。校准拓扑持有初始化快照；直接使用底层 API 修改硬件或 profile 后，应新建 Topology。

## 可复跑的主机性能对比

`scripts/benchmark_simulation.py` 记录实际主机墙钟时间与数值摘要；默认重复三次、取中位数。计时包含图编译和数值汇总，批量运行包含子进程启动；不含配置读取和文件导出。流式每轮使用新的时序缓存，避免用跨轮热缓存冒充引擎加速。

```text
python scripts/benchmark_simulation.py path/to/scenario.yaml --repeats 3 -o outputs/performance/before.json
python scripts/benchmark_simulation.py path/to/scenario.yaml --repeats 3 --compare outputs/performance/before.json -o outputs/performance/after.json
```

第一条在修改引擎前运行，第二条在修改后运行；也可用 `--implementation-root path/to/saved-revision` 指向包含 `supernode_sim/` 的旧版本目录。比较要求相同 resolved config、模式和 axes，并检查状态、指标、通信、资源工作量与内存等数值摘要（相对容差 `1e-10`，绝对容差 `1e-12`）。不一致时返回非零退出码，不输出有效加速比；原始事件顺序及未保留的明细仍由回归测试验证。

使用 `--mode streaming` 测流式完整推理，或 `--mode sweep --axes path/to/axes.yaml --workers 1/4` 比较批量串并行。比较时关闭其他重型任务，并报告 Python 版本、CPU 数、场景规模与重复次数；合成案例的仿真器加速比不代表真实 NPU/GPU 的模型推理加速比。

## 文件大小与恢复

大型默认导出中，超过 50,000 行的保留张量表写入 `tensors.csv.gz`，保留 CSV 的列与全部导出行；实际文件名、导出和省略行数见 `result.json` 的 `artifact_export` 和报告开头。小表仍是 `tensors.csv`。按行读取压缩表：

```python
import csv, gzip
with gzip.open('outputs/my-case/tensors.csv.gz', 'rt', encoding='utf-8-sig', newline='') as handle:
    for row in csv.DictReader(handle):
        # 在此筛选或累计，不把整个文件转成 list。
        pass
```

`--full-detail` 明确请求完整逐事件和张量记录，仍输出普通 `tensors.csv`，会明显增加内存和磁盘需求。大型 JSON 使用紧凑编码，批处理 JSON 逐块写入，不先创建整份字符串。

参考实验检查点默认 gzip level 6，可用 `--checkpoint-compression 1` 换取更快写入；其 `--full-detail` 使用独立的 `-full` 检查点。恢复支持流式读取压缩日志，代码/配置指纹不匹配时重新建检查点，不能强行复用。修改引擎后旧实验结果仍可阅读，但不属于当前代码的已验证缓存。

本次优化只影响新运行和导出，不自动清理历史输出。重新导出优先使用新目录，避免把旧 `tensors.csv` 与新 `tensors.csv.gz` 混为同一次结果。需要清理时先列出大小、用途和可重建性，保留配置、来源、摘要及仍需恢复的检查点，按用户指定范围处理。

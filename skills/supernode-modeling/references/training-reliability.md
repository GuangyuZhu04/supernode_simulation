# 预训练故障、恢复与工期

以完成 `dataset_tokens × epochs` 个有效 token 的墙钟时间比较方案。每个优化步结束后进度才可提交，最后一步向上取整，净吞吐的分子只计目标预算内的 token。两次外部事件之间可批量推进优化步和完整检查点周期。

## 运行入口

```powershell
python -m supernode_sim validate examples/training_reliability_atlas950.yaml
python -m supernode_sim simulate examples/training_reliability_atlas950.yaml -o outputs/reliability/atlas950
python -m supernode_sim training-reliability examples/training_reliability_scaling.yaml -o outputs/reliability/scaling
python -m supernode_sim training-reliability examples/training_reliability_ft_hsdp.yaml -o outputs/reliability/ft_hsdp
```

- 原生场景在 `workload.training.pretrain.reliability` 填故障试验。健康步时、每步有效 token、并行组和链路直接来自当前配置。启用 `local_spare` 时 `hardware.npu_count` 仍表示参与计算的卡，每个 SU 末尾另计预留位置。Web 可选“预训练故障 · Atlas 950 保护端口降速与掉卡”预设，或在高级 JSON 中编辑。[原生配置](../../../examples/training_reliability_atlas950.yaml)
- 规模研究的 `clusters` 填装机卡数、物理 SU 大小、每主机卡数、完整副本卡数、每副本每步 token、该规模的步时与来源。热备从装机预算中扣除，剩余位置按物理顺序组成完整副本。模型在缺副本时保持每个健康副本的批量和步时，全局 batch 随之变化。固定全局 batch 的部署需要提供对应的执行模型。示例对 10,240 与 102,400 卡各比较两种 SU 大小、三种策略和七组故障率或恢复时长。规模示例中的步时、故障率和维修时长均为假设。[规模配置](../../../examples/training_reliability_scaling.yaml)

## 故障日历

| 输入 | 单位与含义 |
|---|---|
| `horizon_s`、`trials`、`seed` | 观察墙钟窗口、重复次数、伪随机种子。停机期间继续计时和采样 |
| `faults.<名称>.scope` | `card`、`host`、`su` 或 `link`。主机与 SU 按物理成员一次性影响其承载的所有副本 |
| `rate_per_unit_hour` | 每个物理单元每小时的独立泊松根因事件率。分母含装机备卡。每个命名链路组算一个单元。默认仅执行显式注入 |
| `trace` | 每行含 `at_s`、`fault` 和从零起的物理 `target`。可填该条日志对应的 `detect_s`、`repair_s` 覆盖类别默认值。命名链路组的 target 为 0。显式事件与随机事件叠加 |
| `detect_s`、`repair_s` | 从故障到检测、从检测到该根因物理维修完成的时间，均为秒 |
| `resources`、`bandwidth_fraction` | 原生链路事件指定受影响的带宽资源及剩余比例。引擎重新运行同一张训练图。多个故障命中同一资源时取最低容量上限 |
| `checkpoint_interval_steps`、`checkpoint_write_s` | 按已提交优化步周期同步保存，写完才更新可用于恢复的 token 游标。省略间隔时仅有初始恢复点。与激活重计算各自配置 |

只有独立卡故障能够打断全作业时，健康状态的平均连续运行时间为 `1 / (在用卡数 × 单卡每小时率)` 小时。`stochastic_fault_rates` 给出这个基线。另按实际观测时间统计物理事件数和全停次数。多卡同时掉线以其共同根因注入一次，若将生产作业中断率用于拟合，需先按根因去重并核对在役卡时。

## 恢复策略与计时

| 策略 | 执行方式 |
|---|---|
| `global_restart` | 退回最近一次写完的检查点。待设备可用，重建通信组 `rebuild_s`、恢复训练状态 `restore_s`，再开始优化步 |
| `local_spare` | 同样回滚。在故障发生时预留同 SU 的健康备卡，检测后用 `swap_s` 切换。备卡不足或原卡修好更快时等待原卡。被换下的卡修好后可用于以后替换 |
| `replica_isolation` | 故障单元退出，所有副本丢弃故障瞬间尚未完成的步。检测和 `isolate_s` 结束后，存活副本继续提交。恢复单元在后续优化步加入。若全部完整状态副本都受损，回退为全局检查点恢复 |

原生训练按 `rank_order` 放置完整 PP 阶段，通过稠密及专家 FSDP 分片组归并共享状态的外层 DP 副本。降容阶段按剩余完整副本计算每步贡献。快速恢复所需的软件协议及状态传输时间由策略输入给出。独立故障率与固定恢复时长可换成基于日志的分类型事件和时长。原生链路降速保持通信路由。每组降速状态重算代表步，发生在步内的链路切换按剩余步进度折算。交换机失效还可作为 SU 级中断单独建模。

## 读取结果与校准

`training_reliability.csv` 给出每种规模、策略、敏感性条件的汇总。`training_reliability.json` 保存每次试验、输入、首轮试验的全部事件记录和按故障类型汇总的恢复分位。相同装机拓扑、种子和故障率下，核对各策略的 `trace_sha256` 即可确认使用同一物理轨迹。降低单卡率时在基准轨迹中按概率保留事件，升高时叠加独立事件。其余根因和显式注入沿用原时刻与物理位置。半恢复时间场景缩放检测、物理维修、替换、隔离、重建和加载耗时，检查点写入时间单独控制。

- `wall_s_p50/p95` 按所有试验求最近秩分位。观察结束时尚未完工记为右删失，超出可观测范围的工期分位为 `null`。净吞吐与卡时按每次试验实际观察到的进度计算。
- `rollback_tokens` 是已提交后又被回滚的 token。故障时未完成的步以 `discarded_partial_token_equivalent` 记录。净吞吐的分子只计最终保留的有效 token。完工时刻取最后一次优化步提交，周期性保存只在后面还有训练工作时执行。
- `lost_card_hours` 等于总装机卡时减去产生已保留 token 所需的等效健康在用卡时，包含预留卡时与活跃卡损失。`effective_training_fraction` 按健康吞吐折算有效训练时间。最后一批超出预算的 token 不计收益。
- 事件的 `first_commit_at_s` 记录故障后首次成功提交优化步，`full_capacity_at_s` 记录当前全部受损资源和通信恢复就绪，`full_throughput_at_s` 记录首个满编、健康步的开始。物理维修另计，各 `*_due_s` 是单次事件的计划时刻。按根因汇总的 `observed_p50_s/p95_s` 仅对观测到的终点计算，并列出尚未观测到的个数。

规模示例以 100 亿有效 token 为目标，每组 32 条轨迹。在 102,400 卡、每 SU 1,024 卡的基准场景，全局重启、域内热备、按副本续训的 P50 工期依次为 17.94、11.48、9.96 小时。全局重启时，256 卡 SU 基准为 19.88 小时。只将大 SU 方案的各项恢复时长增加 20% 或 40%，对应 P50 为 19.59 或 21.42 小时。这两个样本点展示了恢复代价如何改变排序，实际选型应替换两种拓扑各自的步时和故障参数。[参数与随机种子](../../../examples/training_reliability_scaling.yaml)

**Atlas 950** 的保护参数面光模块故障可使链路带宽减半。随仓例只借用这项机制，四卡拓扑及时间参数为合成值。天翼云 9,216 卡报告可用来分别核对 910B2 的检测和恢复边界。950DT 的发生率需要本机型根因日志与在役卡时。[Atlas 950 故障快恢](https://support.huaweicloud.com/resmgmt-modelarts/resmgmt-a5_0007.html) · [天翼云测试](https://www.ctyun.cn/document/10541165/10724629)

**FT-HSDP** 算例复现论文 §6.1 的 18 分钟记账窗口。12 个完整副本各占 8,192 张 H100，全局策略停 600 秒，副本策略停 180 秒，再用 11 个副本工作 420 秒。用 `effective_training_fraction` 核对 `8/18` 和 `(8+7×11/12)/18`。此例的 token 单位归一化，目标设得大于窗口产出，检验的是窗口内的有效训练占比。[论文](https://arxiv.org/html/2602.00277) · [复现配置](../../../examples/training_reliability_ft_hsdp.yaml)

**生产校准**可参考 Aegis 的单卡统计与 Llama 3 的作业中断统计。AIReSim 提供了服务器、备机和维修队列的离散事件参考实现。[Aegis](https://www.usenix.org/system/files/nsdi25-dong.pdf) · [Llama 3 §3.3.4](https://arxiv.org/html/2407.21783v3#S3.SS3.SSS4) · [AIReSim](https://github.com/karthikp-ubc/airesim)

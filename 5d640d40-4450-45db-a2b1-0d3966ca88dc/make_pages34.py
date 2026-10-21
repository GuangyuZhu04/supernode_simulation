from deck_style import *

SOURCE=DATA["source"]
ref=DATA["ar_cases"][0]
hw=DATA["experiment"]["platforms"]

def page3():
    p=Slide(3,"10T 推理需分开预算 HBM 与主机内存，并按阶段校验峰值",
        "03 硬件与容量 · 参考布局 TP8 / EP128 / DP1 / SP8 · 1M · AR · 每平台 1024 卡",
        [("同一布局：",BLACK,True),("prefill 峰值高于 decode",RED,True),("；Engram 表驻留主机，随 DP 副本数增长。",BLACK,True)],
        [
            [("参考布局峰值：prefill bs1 ",BLACK,False),(f"{ref['prefill_bs1_peak_GB']:.2f} GB",RED,True),("；decode bs32 ",BLACK,False),(f"{ref['hbm_peak_GB']:.2f} GB",RED,True),("，两种负载分别核对。",BLACK,False)],
            [("主机侧 ngram 每 DP 副本 ",BLACK,False),(f"{ref['cluster_ngram_DRAM_TB']:.4f} TB",RED,True),("；扩 DP 时同时增加权重与查表容量。",BLACK,False)],
        ],
        ["口径：TTFT 对应 prefill 全局 bs1；decode 全局 bs32。单个逻辑非阻塞 SU；普通通信不走 SO。",
         "来源：2026-10-20 experiment.json / prefill.csv / summary.csv；容量、带宽为十进制；硬件与效率均未实测校准。"])
    left=panel(50,172,380,366,"HBM：按阶段与物理副本计容","database")
    left+=text(66,246,f"{ref['prefill_bs1_peak_GB']:.2f}",38,RED,True)+text(210,244,"GB / 卡",15,BLACK,True)
    left+=text(66,273,"prefill bs1 峰值 · 两平台容量账本相同",12)
    left+=text(66,318,f"{ref['hbm_peak_GB']:.2f}",38,RED,True)+text(210,316,"GB / 卡",15,BLACK,True)
    left+=text(66,345,"decode bs32 峰值 · 不能替代 prefill 预算",12)
    rows=[
        ["每卡权重 + scale",f"{ref['weight_GB_per_card']:.2f} GB"],
        ["每卡运行时","8 GB"],
        ["每卡工作区","256 MB"],
        ["prefill bs8 峰值",f"{ref['prefill_bs8_peak_GB']:.2f} GB"],
    ]
    left+=table(66,364,[210,138],["组成 / 阶段","容量"],rows,28,26)
    left+=text(66,521,"峰值包含最终 KV 预留、激活与暂存空间。",12)
    p.add("phase-capacity",left)
    mid=panel(450,172,380,366,"算力与带宽：分精度建模","server")
    rows=[
        ["HBM 容量","144 GB","288 GB"],
        ["HBM 带宽","4 TB/s","9.6 TB/s"],
        ["FP8 峰值","1 PFLOP/s","2 PFLOP/s"],
        ["FP32 峰值","50 TFLOP/s","50 TFLOP/s"],
        ["计算效率","50%","50%"],
    ]
    mid+=table(466,215,[138,105,105],["每卡规格","950DT","960DT"],rows,31,28,size=12)
    mid+=line(466,418,814,418)
    mid+=text(466,442,"主对照统一 FP8",15,RED,True)
    mid+=text(466,466,"权重 / 计算 / 激活 / 通信 / KV",12)
    mid+=text(466,490,"router、mHC、norm 与标量：显式 FP32",12)
    mid+=text(466,520,"公开 FP4 格式单列；未假设其算力倍率。",12)
    p.add("hardware-inputs",mid)
    right=panel(850,172,380,366,"互联与主机：查表成本不省略","arrow-left-arrow-right")
    rows=[
        ["SU / 方向","1 TB/s","1.1 TB/s"],
        ["SU 启动","2.1 μs","1 μs"],
        ["SU 效率","80%","80%"],
    ]
    right+=table(866,215,[138,105,105],["每卡互联","950DT","960DT"],rows,31,28,size=12)
    right+=text(866,374,"Engram：每副本主机驻留",15,RED,True)
    right+=text(866,419,f"{ref['cluster_ngram_DRAM_TB']:.4f}",38,RED,True)+text(1070,416,"TB",15,BLACK,True)
    right+=text(866,445,"每主机 4 TB DRAM · 512 GB/s · 10⁸ IOPS",12)
    right+=text(866,469,"host link：64 GB/s × 85% · 2 μs",12)
    right+=text(866,493,"表按行分片，地址和返回行均进入账本。",12)
    right+=text(866,521,"无 cache 命中；无免费预取或传输重叠。",12)
    p.add("host-and-network",right)
    speech=(
        "这一页保留原版材料的显存、算力和互联三条主线，并补上DeepSeek条件记忆的主机内存路径。"
        "所有数字以一百零四万八千五百七十六token、1024卡、TP8、EP128、DP1、SP8的AR参考布局为锚点。"
        "首先要区分阶段：prefill全局批大小为1时，单卡预留峰值是62.0232GB；decode全局批大小为32时，峰值只有27.3956GB。"
        "它们是两个独立阶段的负载口径，不能把后者作为整批32个请求同时prefill的容量证明。prefill批大小8另有78.7364GB记录。"
        "峰值包括权重物理副本、最终KV预留、每lane激活arena、通信暂存、8GB运行时和256MB工作区，不能用逻辑参数简单均分来替代。"
        "两平台的容量账本一致，但容量上限、HBM带宽、FP8计算速率和SU启动时间不同；效率分别固定为计算50%和互联80%，FP32峰值沿用50TFLOP/s假设。"
        "实验把主要矩阵、激活、通信和KV统一为FP8，显式计入router、mHC及norm的FP32工作量。公开参考中的FP4专家和KV只是来源记录，没有凭空换算为性能倍率。"
        "Engram的3.4398T逻辑参数另存主机，FP8与每32元素一个scale使每DP副本占约3.5473TB；该表按行分片到副本所属主机，而非每台主机都放整表。"
        "冷查表经过CPU、DRAM或IOPS、host链路、SU和TP通信，所有流量和启动均入账。扩大DP会复制权重和整张逻辑表，即使副本闲置仍保留容量。"
        "这些配置是未校准模型的输入，不是硬件规格实测，也不承诺实际物理布线。"
    )
    p.save(speech,"明确资源边界后，下一页说明如何将新架构接入已有仿真内核并校验结果。",
        [
            ("prefill 峰值高于 decode","summary.csv参考布局记录prefill bs1峰值62.023152704GB、decode bs32峰值27.395639616GB；负载不同，分别展示。"),
            (f"{ref['prefill_bs1_peak_GB']:.2f}","prefill.csv中参考布局、1M、AR、全局bs1的每卡峰值；不是decode容量。"),
            (f"{ref['hbm_peak_GB']:.2f}","summary.csv中参考布局、1M、AR、全局bs32的decode阶段每卡峰值。"),
            ("主对照统一 FP8","experiment.json规定矩阵存储/计算/激活/通信/KV统一FP8，FP32操作另计。"),
            ("Engram：每副本主机驻留","methodology.md定义每DP副本完整ngram表，按行分散在副本所属主机。"),
            (f"{ref['cluster_ngram_DRAM_TB']:.4f}","3.4398T查表参数按FP8和每32元素1字节scale计算为3.54729375TB，每DP副本一份。"),
        ],
        [SOURCE+"/experiment.json",SOURCE+"/summary.csv",SOURCE+"/prefill.csv",SOURCE+"/methodology.md"],
        glossary_extra=[
            {"term":"scale","full":"Quantization Scale，量化缩放因子","explain":"随FP8权重、KV和Engram表存储的额外字节。"},
            {"term":"router","full":"Expert Router，专家路由器","explain":"以FP32计算显式路由工作量。"},
            {"term":"norm","full":"Normalization，归一化","explain":"本实验显式计入FP32归一化计算。"},
            {"term":"host link","full":"Host Link，主机到加速卡链路","explain":"64GB/s名义带宽、85%效率与2μs启动的继承假设。"},
            {"term":"cache","full":"Cache，缓存","explain":"主场景不假设Engram查表缓存命中。"},
        ])

def page4():
    status=DATA["status_counts"]
    p=Slide(4,"新增架构用独立解析模块接入原生资源内核，结果可追溯",
        "04 仿真方法 · CED / CSA2 / mHC / Engram / DSpark 扩展 · 冻结 supernode_sim 内核",
        [("保留原生调度与容量语义；",BLACK,True),("17 组不变量验证",RED,True),("通过，",BLACK,True),("性能精度仍待实机校准",RED,True),("。",BLACK,True)],
        [
            [("覆盖 ",BLACK,False),("15,120",RED,True),(" 行 decode 与 ",BLACK,False),("1,440",RED,True),(" 行 prefill；失败与不兼容行完整保留。",BLACK,False)],
            [("TTFT 单列 bs1 / bs8；decode 扫描 bs1–32，",BLACK,False),("固定全局负载，不以新增请求制造 DP 收益",RED,True),("。",BLACK,False)],
        ],
        ["边界：解析资源预测；无 kernel 实测 profile、请求排队、动态 batching、客户端交付与 P99；不与 Kimi 直接作实测加速比较。",
         "来源：2026-10-20 methodology.md / verification.json / result-verification.json / provenance.json；校验通过不等于精度校准。"])
    left=panel(50,172,365,306,"01 固定输入与兼容性","file")
    left+=text(66,225,"结构与精度",15,BLACK,True)
    left+=text(66,249,"逐矩阵参数 · FP8 / FP32 分账",12)
    left+=text(66,274,"CED / CSA2 / mHC / Engram / DSpark",12)
    left+=text(66,310,"并行与负载",15,BLACK,True)
    left+=text(66,334,"TP × EP × DP = 1024；PP = 1",12)
    left+=text(66,358,"SP 共用 TP；活跃 DP = min(DP, batch)",12)
    left+=text(66,394,"45 个候选布局 · 4 种上下文",15,BLACK,True)
    left+=text(66,418,"decode bs = 1 / 2 / 4 / 8 / 16 / 32",12)
    left+=text(66,442,"TP16 / TP32 与 output groups=8 不兼容",12)
    left+=text(66,466,"从配置入口保留不兼容结果。",12)
    p.add("fixed-inputs",left)
    mid=panel(455,172,365,306,"02 原生内核执行资源图","code")
    mid+=text(471,225,"组件 → 阶段 → 原生资源封装",15,BLACK,True)
    for i,(label,ic) in enumerate([("计算","calculator"),("HBM","database"),("网络 / 主机","server")]):
        x=471+i*113
        mid+=rect(x,247,105,57,BG,LINE)
        mid+=icon(ic,x+43,254,18)+text(x+52.5,293,label,12,BLACK,True,"middle")
    mid+=arrow(637,310,637,328)
    mid+=rect(471,334,333,40,PALE,"#F1B4B7")+text(637.5,359,"依赖屏障 + 带宽 + 容量准入",15,BLACK,True,"middle")
    mid+=text(471,398,"计算 / HBM 依 roofline 取主导项",12)
    mid+=text(471,422,"collective 轮次、端点注入与启动显式计费",12)
    mid+=text(471,446,"各算子阶段串行；未假设计通重叠",12)
    mid+=text(471,469,"资源封装保留总物理 FLOPs / bytes 账本",12)
    p.add("native-engine",mid)
    right=panel(860,172,370,306,"03 验证与证据留存","chart-bar")
    right+=text(876,228,"17",38,RED,True)+text(944,225,"组结构 / 内核校验",15,BLACK,True)
    right+=text(876,255,"2 / 4 / 8 卡通信图核对 · 异构 rank 计算",12)
    right+=text(876,280,"路由守恒 · KV owner · DP 资源独立",12)
    right+=text(876,316,"数值与容量账本",15,BLACK,True)
    right+=text(876,340,"6 类时延可加 · 阶段工作量守恒",12)
    right+=text(876,364,"真实失败保留 null；不以零延迟替代",12)
    right+=text(876,400,"来源与复现",15,BLACK,True)
    right+=text(876,424,"CSV / JSON · 原始阶段 · SHA256 指纹",12)
    right+=text(876,448,"锁定公开 Flash 配置 / 代码版本",12)
    right+=text(876,470,"缩约验证不证明硬件或模型质量。",12)
    p.add("checks-and-evidence",right)
    strip=rect(50,492,1180,46,BG,LINE)
    vals=[("可行",status["feasible"]),("容量不可行",status["infeasible"]),("结构不兼容",status["incompatible"])]
    x=66
    for i,(label,v) in enumerate(vals):
        strip+=text(x,520,label,15,BLACK,True)+text(x+135,520,f"{v:,}",15,RED,True)
        if i<2: strip+=line(x+330,501,x+330,528)
        x+=392
    p.add("matrix-status",strip)
    speech=(
        "这一页承接原版材料的仿真Skill流程，但需要把本次适配范围讲清楚。DeepSeek的CED、CSA2、mHC、Engram和三层DSpark"
        "通过独立的解析模块编译为阶段图，调用冻结的supernode_sim原生Graph、Event、Allocation、Resource以及engine.simulate，"
        "没有把这些尚未进入通用streaming接口的特性写成通用API已原生支持。"
        "输入包括逐矩阵结构、精度、1024卡拓扑和固定全局batch；DP仅分摊已有请求，活跃副本数量不超过全局batch。"
        "扫描包含45个并行候选、四种二进制上下文和六档decode batch，TP16与TP32因为保留output groups=8而明确标为不兼容。"
        "各组件按barrier串行；计算与HBM在单个kernel内共同约束，只把时间归给主导资源，通信每轮同时受端点和启动约束。"
        "仿真使用资源封装缩约相同速率的rank组，仍保留全物理工作量；封装中代表性工作量不能被误认成全卡流量。"
        "17组校验核对小规模原生通信图、异构计算、DP主机独立、整数路由守恒、缓存owner、投机状态和容量失败等。"
        "全量共有15120行decode，7560行可行、3528行容量不可行、4032行结构不兼容，另有1440行prefill。"
        "这些行包括AR和六档条件提交数，不是15120次不同模型训练或独立硬件实测。"
        "失败项保留空时延，不能在图表中当成零延迟。prefill按全局bs1和bs8另列，decode批次扫描不改写首token的含义。"
        "输入、原始阶段结果、代码指纹、矩阵审计及六类资源时延账本均可追溯。验证证明了方程、加和与缩约一致性，"
        "并未证明实际kernel精度、学习路由、接收率、排队、动态批处理或P99；下一步仍需要实机校准。"
    )
    p.save(speech,"在统一的硬件和负载口径下，下一页比较参考布局与解码优先布局的收益及代价。",
        [
            ("17 组不变量验证","verification.json记录17组结构和原生内核缩约核对，范围不是硬件性能准确度。"),
            ("17","verification.json的count=17，覆盖张量、容量、通信及状态账本。"),
            ("性能精度仍待实机校准","methodology.md明确未使用kernel实测profile，没有性能误差承诺。"),
            ("15,120","summary.csv行数，包含AR和DSpark的六个条件C值；不是独立实测次数。"),
            ("1,440","prefill.csv独立行数，prefill全局批次为1和8。"),
            ("固定全局负载，不以新增请求制造 DP 收益","experiment.json规定active_DP=min(DP,global_batch)，吞吐按实际已给定batch计算。"),
            ("7,560","result-verification.json和summary.csv状态计数中的可行decode行。"),
            ("3,528","summary.csv中容量不满足当前预留策略的decode行；时延为空。"),
            ("4,032","summary.csv中TP16/32与output groups=8不兼容的decode行。"),
        ],
        [SOURCE+"/methodology.md",SOURCE+"/verification.json",SOURCE+"/result-verification.json",SOURCE+"/provenance.json",SOURCE+"/experiment.json"],
        glossary_extra=[
            {"term":"roofline","full":"Roofline Model，屋顶线模型","explain":"同一矩阵阶段以计算和HBM服务时间中的较慢者约束完成。"},
            {"term":"collective","full":"Collective Communication，集合通信","explain":"本实验ring/pairwise轮次、载荷和启动均显式计入。"},
            {"term":"min","full":"Minimum，最小值函数","explain":"活跃DP副本数是DP数量与全局batch的较小值。"},
            {"term":"output groups","full":"Output Projection Groups，输出投影分组","explain":"保留8组结构导致TP16/32不兼容。"},
            {"term":"bytes","full":"Bytes，字节数","explain":"物理内存和链路账本的流量单位。"},
            {"term":"null","full":"Null Value，空值","explain":"不可行结果未执行或没有有效时延，不能当作零。"},
            {"term":"kernel","full":"Compute Kernel，计算内核","explain":"本实验未用实测kernel启动或吞吐profile校准。"},
            {"term":"profile","full":"Measured Performance Profile，实测性能档案","explain":"用于把解析模型校准到真实算子的测量资料。"},
            {"term":"batching","full":"Batching，请求批处理","explain":"动态请求合批尚未纳入模型。"},
        ])

if __name__=="__main__":
    page3()
    page4()

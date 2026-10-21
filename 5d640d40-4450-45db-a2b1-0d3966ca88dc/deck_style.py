"""Shared Huawei SVG style; agents own only their assigned page builders."""
from pathlib import Path
from html import escape
import json, re
from xml.etree import ElementTree as ET
WS=Path(__file__).resolve().parent
SKILL=Path(r"C:\Users\Guangyu\.codex\skills\ppt-master-huawei-v2")
DATA=json.loads((WS/"input/deck-data.json").read_text(encoding="utf8"))
RED="#C00000"; BLACK="#1D1D1A"; GRAY="#666666"; LINE="#D9D9D9"; PALE="#FFF1EF"; BG="#F2F2F2"
FONT="微软雅黑, Arial, sans-serif"
def esc(v): return escape(str(v),quote=True)
def text(x,y,v,size=12,color=BLACK,bold=False,anchor="start"):
    """v may be string or [(text,color,bold), ...], with no positioning on spans."""
    inside=esc(v) if isinstance(v,(str,int,float)) else "".join(f'<tspan fill="{c}" font-weight="{"bold" if b else "normal"}">{esc(t)}</tspan>' for t,c,b in v)
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" xml:space="preserve">{inside}</text>'
def rect(x,y,w,h,fill="white",stroke=LINE,rx=0,sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def line(x1,y1,x2,y2,color=LINE,dash=None):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1"'+(f' stroke-dasharray="{dash}"' if dash else "")+'/>'
def arrow(x1,y1,x2,y2,color=GRAY):
    assert x1==x2 or y1==y2
    import math
    d=math.hypot(x2-x1,y2-y1); ux=(x2-x1)/d; uy=(y2-y1)/d
    points=[(x2,y2),(x2-ux*7-uy*3.5,y2-uy*7+ux*3.5),(x2-ux*7+uy*3.5,y2-uy*7-ux*3.5)]
    return line(x1,y1,x2,y2,color)+f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{color}"/>'
def icon(name,x,y,size=20,color=RED):
    assert (SKILL/f"assets/icons/chunk-filled/{name}.svg").exists(),name
    return f'<use data-icon="chunk-filled/{name}" x="{x}" y="{y}" width="{size}" height="{size}" fill="{color}"/>'
def group(id,content): return f'<g id="{id}">{content}</g>'
def panel(x,y,w,h,title,ic="chart-bar"):
    return rect(x,y,w,h)+rect(x,y,w,28,BG,"none")+icon(ic,x+12,y+4)+text(x+40,y+20,title,15,RED,True)
def table(x,y,widths,headers,rows,row_h=29,head_h=28,red_rows=(),size=12,aligns=None):
    """Rows items strings or dict(text,color,bold). No fake data bars."""
    out=rect(x,y,sum(widths),head_h,BG,"none")
    xx=x
    for i,(w,h) in enumerate(zip(widths,headers)):
        out+=text(xx+8,y+19,h,size,BLACK,True); xx+=w
    yy=y+head_h
    for j,row in enumerate(rows):
        if j in red_rows: out+=rect(x,yy,sum(widths),row_h,PALE,"none")
        xx=x
        for i,(w,v) in enumerate(zip(widths,row)):
            d=v if isinstance(v,dict) else {"text":v}
            a=(aligns[i] if aligns else "start")
            out+=text(xx+(w-8 if a=="end" else w/2 if a=="middle" else 8),yy+row_h*.65,d["text"],size,d.get("color",BLACK),d.get("bold",j in red_rows),a)
            xx+=w
        out+=line(x,yy+row_h,x+sum(widths),yy+row_h)
        yy+=row_h
    return out

GLOSSARY={
"DeepSeek":("（公司名）深度求索","本材料所采用的公开架构参考来源。"),
"V4.1":("Version 4.1，4.1版本","指本地锁定的DeepSeek-V4.1-Flash公开资料版本。"),
"Flash":("（产品名）DeepSeek-V4.1-Flash","公开架构参考；与本材料合成Pro-10T规模分开。"),
"Pro-10T":("（研究配置名）合成Pro万亿级结构","由Flash机制推演的研究配置，不是公开Pro规格。"),
"10T":("10 Trillion，十万亿参数","本实验名义骨干参数规模，实际审计为10.1225T。"),
"MoE":("Mixture of Experts，混合专家","按token路由到部分专家的稀疏前馈层。"),
"CED":("Causal Encoder-Decoder，因果编码器—解码器","批量prompt先经过编码器，再投影解码器全局KV。"),
"CSA2":("Compressed Sparse Attention 2，第二代压缩稀疏注意力","跨层共享KV和索引，并复用稀疏位置。"),
"mHC":("Manifold-constrained Hyper-Connections，流形约束超连接","以四路残差流及显式FP32操作建模的跨层连接。"),
"Engram":("（组件名）条件记忆模块","从主机侧ngram表查找并传回选中行。"),
"DSpark":("（机制名）DeepSeek DSpark","三层草稿网络并行提出五个位置，主模型验证六个query。"),
"DSpark3":("DSpark with three draft layers，三层DSpark","本实验草稿网络层数为三。"),
"SWA":("Sliding Window Attention，滑动窗口注意力","本实验窗口长度为128 token。"),
"KV":("Key/Value cache，键值缓存","推理中驻留的注意力历史状态。"),
"Q":("Query，查询向量","当前注意力位置对应的查询。"),
"Full":("Full Mode，完整模式","生成共享KV/index，并计算稀疏索引的层模式。"),
"Reindex":("Reindex Mode，重索引模式","复用共享KV并重新计算索引。"),
"Reuse":("Reuse Mode，复用模式","复用已有KV及稀疏位置。"),
"ngram":("N-gram，N元词组","本实验条件记忆表的索引组织方式。"),
"AR":("Autoregressive decoding，自回归解码","每轮提交一个输出token的基线。"),
"TP":("Tensor Parallelism，张量并行","对注意力头、专家中间维等张量分片。"),
"EP":("Expert Parallelism，专家并行","在不同rank间分布专家权重。"),
"DP":("Data Parallelism，数据并行","在独立权重副本上分配已存在的全局请求。"),
"SP":("Sequence Parallelism，序列并行","与TP共用卡，分片pointwise和残差等token状态。"),
"PP":("Pipeline Parallelism，流水线并行","本实验固定PP=1。"),
"HBM":("High Bandwidth Memory，高带宽内存","NPU卡上存储，容量和带宽分别约束。"),
"DRAM":("Dynamic Random Access Memory，动态随机存取存储器","主机内存，用于分片存放Engram表。"),
"CPU":("Central Processing Unit，中央处理器","执行hash与主机侧处理成本。"),
"NPU":("Neural Processing Unit，神经网络处理器","仿真中的加速卡。"),
"FP8":("8-bit Floating Point，8位浮点","主对照的权重、计算、激活、通信和KV精度。"),
"FP4":("4-bit Floating Point，4位浮点","公开参考采用的部分存储格式，本实验未假设其加速倍率。"),
"FP32":("32-bit Floating Point，32位浮点","router、mHC、norm等显式标量运算精度。"),
"SU":("Scale Up，域内互联","单逻辑1024卡无阻塞互联及端点注入模型。"),
"SO":("Scale Out，域间互联","本实验普通通信不经过此路径。"),
"H2D":("Host to Device，主机到设备传输","主机内存与加速卡间传输方向。"),
"D2H":("Device to Host，设备到主机传输","加速卡回传主机内存的方向。"),
"IOPS":("Input/Output Operations Per Second，每秒输入输出操作数","主机查表随机访问能力的约束。"),
"TTFT":("Time To First Token，首token时间","本稿统一单独标示prefill全局bs=1的执行时间。"),
"TPOT":("Time Per Output Token，每输出token时间","AR为轮时延；DSpark为轮时延除以提交数C。"),
"ITL":("Inter-Token Latency，客户端相邻token到达间隔","DSpark的折算TPOT不等同客户端ITL。"),
"SLO":("Service Level Objective，服务等级目标","本实验没有证明线上端到端服务指标。"),
"P99":("99th Percentile，99百分位","本模型未模拟请求排队与尾时延。"),
"bs":("Batch Size，批大小","decode为全局同时参与序列数；TTFT的prefill批次另列。"),
"batch":("Batch Size，批大小","在相应阶段同时处理的序列数量。"),
"prefill":("Prompt Prefill，提示词预填充","含编码器bulk、解码器KV投影及有界replay。"),
"decode":("Autoregressive Decode，自回归生成","生成阶段；可选择AR或DSpark轮调度。"),
"draft":("Draft Proposal，草稿提出","DSpark三层草稿网络和Markov校正阶段。"),
"verify":("Target Verification，主模型验证","主模型对六个输入位置执行完整验证。"),
"commit":("State Commit，状态提交","接受并写回状态，包含非零恢复与追赶开销。"),
"query":("Attention Query，注意力查询","草稿验证中的输入位置与对应查询。"),
"token":("Token，词元","模型输入输出的基本离散单位。"),
"tokens/s":("Tokens Per Second，每秒词元数","global batch乘C除以轮时延的系统输出吞吐。"),
"tok/s":("Tokens Per Second，每秒词元数","全系统输出吞吐。"),
"rank":("Parallel Rank，并行进程编号","模型中独立资源端点编号。"),
"owner":("Cache Owner，缓存持有层","仅指定层生成并驻留共享全局KV/index。"),
"top-k":("Top-k Selection，前k项选择","每个token路由至的专家数或索引保留数。"),
"top-24":("Top-24 Expert Routing，24选路由","每个token选择24个路由专家。"),
"top-512":("Top-512 Sparse Index，512项稀疏索引","每层稀疏注意力索引选中位置数量。"),
"GEMM":("General Matrix Multiplication，通用矩阵乘","计算与HBM按roofline同时计费的矩阵操作。"),
"FPLOP":("Floating Point Operation，浮点运算","计算操作计量。"),
"FLOPs":("Floating Point Operations，浮点运算次数","工作量而非利用率或时延。"),
"EFLOP":("Exa Floating Point Operations，百京次浮点运算","10的18次方次浮点运算。"),
"TFLOP":("Tera Floating Point Operations，万亿次浮点运算","10的12次方次浮点运算。"),
"PFLOP/s":("Peta FLOPs Per Second，每秒千万亿次浮点运算","硬件名义计算速率。"),
"GB":("Gigabyte，十进制吉字节","本稿GB=10^9字节。"),
"TB":("Terabyte，十进制太字节","本稿TB=10^12字节。"),
"MB":("Megabyte，十进制兆字节","本稿MB=10^6字节。"),
"B":("Billion，十亿","在参数计数中为10^9个参数。"),
"T":("Trillion，万亿","在参数计数中为10^12个参数。"),
"ms":("Millisecond，毫秒","10^-3秒。"),
"μs":("Microsecond，微秒","10^-6秒。"),
"s":("Second，秒","时延单位。"),
"1M":("1,048,576 Tokens，一百万级词元上下文","本稿1M采用二进制长度。"),
"128k":("131,072 Tokens，十三万级词元","本稿k采用二进制长度。"),
"256k":("262,144 Tokens，二十六万级词元","本稿k采用二进制长度。"),
"512k":("524,288 Tokens，五十二万级词元","本稿k采用二进制长度。"),
"C":("Committed Tokens Per Round，每轮提交词元数","条件输入C=1至6，包含replacement或bonus，不是接受率。"),
"950DT":("（硬件配置名）Ascend 950DT","本实验采用的144GB、4TB/s、1PFLOP/s假设配置。"),
"960DT":("（硬件配置名）Ascend 960DT","本实验采用的288GB、9.6TB/s、2PFLOP/s假设配置。"),
"Skill":("（工具流程名）技能编排","组织配置、执行、结果解释与复现材料。"),
"Python":("（编程语言名）Python","数值引擎与制图代码使用的语言。"),
"supernode_sim":("（软件名）超节点流量仿真引擎","本次使用冻结的原生资源和调度语义。"),
"Graph":("Graph，资源事件图","将组件计算和传输组织为依赖图。"),
"Event":("Event，事件","仿真中的计算或通信步骤。"),
"Allocation":("Allocation，内存分配","用于容量准入与阶段峰值账本。"),
"Resource":("Resource，资源","计算、HBM、端点及主机等约束。"),
"CSV":("Comma-Separated Values，逗号分隔数据","全量结构化结果格式。"),
"JSON":("JavaScript Object Notation，结构化对象格式","模型、实验与验证账本格式。"),
"SHA256":("Secure Hash Algorithm 256-bit，256位安全散列","源文件固定版本及审计指纹。"),
"OOM":("Out of Memory，容量不足","给定调度、精度与预留方案的容量失败。"),
"FP32 all-reduce":("FP32 All-Reduce，32位浮点全归约","索引打分跨TP归约后再执行top-k。"),
"Huawei Proprietary - Restricted Distribution":("华为专有资料—限制分发","沿用用户要求的华为模板固定页脚。"),
}

class Slide:
    def __init__(self,n,title,subtitle,core,highlights,footnotes):
        self.n=n; self.title=title; self.parts=[]; self.panels=[]
        self.parts.append(group("page-header",text(50,50,title,28,RED,True)+text(50,77,subtitle,12,GRAY)))
        self.parts.append(group("core-band",rect(50,96,1180,62,PALE,RED,4,1.5)+icon("target-arrow",70,115,22)+text(104,132,"核心思想",15,RED,True)+line(180,110,180,146,"#F1B4B7")+text(200,132,core,15,BLACK,True)))
        self.highlights=highlights;self.footnotes=footnotes
    def add(self,id,content): self.parts.append(group(id,content))
    def save(self,speech,transition,red_notes,sources,content=None,glossary_extra=None):
        h=rect(50,574,1180,68,BG,LINE,4)+text(72,600,"本页亮点",15,RED,True)
        for i,s in enumerate(self.highlights): h+=text(180,600+25*i,s,15)
        self.parts.append(group("highlight-strip",h))
        for i,f in enumerate(self.footnotes): self.parts.append(group(f"source-note-{i}",text(50,552+13*i,f,10.5,GRAY)))
        self.parts.append('<g id="footer-group" font-family="Arial, sans-serif" font-size="10.5" fill="#1D1D1A"><text x="77" y="683">'+str(self.n)+'</text><text x="125" y="683">Huawei Proprietary - Restricted Distribution</text><image href="logo.png" x="1070" y="663" width="133" height="28" preserveAspectRatio="xMidYMid meet"/></g>')
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720"><rect width="1280" height="720" fill="#FFFFFF"/>'+''.join(self.parts)+'</svg>'
        (WS/f"svg/page{self.n}.svg").write_text(svg,encoding="utf8")
        plain=" ".join(ET.fromstring(svg).itertext())
        glossary=[]
        for term,(full,explain) in GLOSSARY.items():
            if re.search(r"(?<![A-Za-z])"+re.escape(term)+r"(?![A-Za-z])",plain,re.I):
                glossary.append({"term":term,"full":full,"explain":explain})
        glossary+=glossary_extra or []
        notes={"page":self.n,"title":self.title,"speech":speech,"transition":transition,"glossary":glossary,"red_notes":[{"quote":q,"explain":e} for q,e in red_notes],"sources":sources}
        (WS/f"notes/page{self.n}.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding="utf8")
        (WS/f"content/content{self.n}.md").write_text(content or f"<!-- 逻辑套路：问题分析 -->\n# {self.title}\n\n{speech}",encoding="utf8")

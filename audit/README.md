# 巡店 AI · 端到端框架 v0.2（单图版·五类新口径）

一句话：**一张图进来 → 规则筛 → YOLO 检测 → 1:N 检索 → VLM 判定 → 融合结论（含是否转人工）**。
每个阶段都可降级：缺权重、VLM 服务没起、没建索引，都不会崩，只在证据链里标 `unavailable`。

## 五类问题判定口径（业务共识）

| 类别 | 口径 | 实现位置 |
|---|---|---|
| 摆拍-空瓶 | 无液瓶冒充陈列凑数，**出现即命中**不设数量门槛；孤立 1~2 支低置信(0.5)转人工，≥3 支高置信 | `rules.rule_empty_bottle` + VLM `empty_bottle_count` |
| 摆拍-纵深不足 | 前排元气瓶后方无产品或只有他牌→命中；**后方有元气即合格**（不限排数）；遮挡 unknown 不判。VLM 只报 `behind` 自由文本观察，**depth 由代码关键词推导** | `vlm._normalize` 推导 + `rules.rule_depth` |
| 摆拍-场景错误 | **白名单制**：饮料货架/饮料冰柜(含他司)/割箱堆箱/端架 合法，其余（零食货架、收银台、餐区、路边等）全判错 | `vlm.SCENE_WHITELIST` 确定性推导，不信任模型自报布尔 |
| 套拍-相似图 | 拍屏幕/翻拍（VLM `is_screen`）+ 拼接双图（左右半幅相关 ≥0.85） | `rules.rule_screen` / `rule_composite` |
| 套拍-同店多门店 | 跨记录检索，**本阶段保留骨架不增强**（输入形态待业务确认） | `rules.rule_reuse` |

融合优先级：套拍-相似图 → 套拍-同店多门店 → 空瓶 → 场景错误 → 纵深不足 → 正常陈列。
多标签共现（如空瓶+场景错误）不再降权；**任一命中标签置信度 < 0.6 即转人工**。
拍屏幕时场景/纵深/空瓶判定自动不适用（画面位置无意义，主问题归套拍）。
`single_row`（单排等距）只作摆拍形态证据，不再直接定罪为空瓶。

**观察→结论确定性推导（V4，实测 2B 结论字段不可信后加）**：
- `depth` 不再让模型自报：截取 behind 文本“后方/背后/后面”之后片段，含元气→yes；遮挡/看不清→unknown（优先，宁可不判）；**割箱场景且后方为纸箱/饮料箱/包装袋→unknown**（后方通常是元气自己的纸箱，品牌不可读）；其他品牌/他牌/空层板/无产品→no；其余 unknown；
- `scene` 白名单推导之外加 around 交叉校验：观察出现强非饮料信号（洗衣/洗涤/文具/烟酒/母婴/餐具/冰柜顶/盖上/餐桌/桌面/收银）但 scene 报白名单时覆写为非白名单场景，**置信度封顶 0.55 转人工**。**“零食”不做覆写信号**（零食常邻接饮料陈列，实测正常割箱被误伤）；
- 空瓶自矛盾护栏：`empty_bottle_count>0` 但 empty_obs/reason 明确“无空瓶”→ 命中但 0.5 低置信转人工。
- **路由安全阀（空瓶/纵深共用）**：合法场景 + YOLO 多排陈列（rows>1）时，空瓶/纵深主张缺乏摆拍上下文佐证
  （2B 液面线与 behind 观察在正常货架上实测不可靠、跨轮漂移）→ 置信度封顶 0.55 转人工。
  标签保留、召回不降；**正常图绝不自动定罪**，最坏交人工复核。单排摆拍上下文/非白名单场景不受影响。

## 阈值校准记录（10 张真实样本，`scripts/_calibrate.py`）

- `composite_corr`：正常图最高 0.627、真拼接图 0.952/0.959 → 阈值 **0.85**（旧 0.6 会把正常割箱误报为拼接）。
- `moire`：128px 下拍屏幕 1.40~1.55 与正常图 1.41~1.49 完全重叠、无区分度 → **只记录不定罪**，旧 `moire_threshold=0.35` 恒命中隐患已移除。

## 目录

```
audit/
├── config.json          # 所有路径与阈值
├── requirements.txt
├── audit/               # 核心包
│   ├── config.py
│   ├── logutil.py       # 统一日志（实时阶段进度，只打日志不配 handler）
│   ├── imageops.py      # 清晰度 / 拼接检测 / 摩尔纹 / 感知哈希
│   ├── detect.py        # YOLO 检测（瓶体），带 target 语义声明
│   ├── embed.py         # 特征向量（ResNet / 退化 pHash）
│   ├── retrieve.py      # numpy 1:N 检索
│   ├── vlm.py           # VLM 客户端（OpenAI 兼容 / 原生 Ollama）+ 协议归一
│   ├── compensate.py    # 针对 2B 的缺陷补偿层（可整层关闭）
│   ├── rules.py         # 规则命中（单排 / 拼接 / 屏幕 / 场景 / 纵深 / 复用）
│   ├── fusion.py        # 证据融合 + 是否转人工
│   └── pipeline.py      # 编排
├── scripts/
│   ├── run_single.py    # 单图入口
│   ├── build_index.py   # 建检索索引
│   ├── build_table.py   # 生成训练表（单表结构）
│   └── smoke_test.py    # 冒烟测试
└── outputs/
```

## 环境

用 conda 的 `yolo` 环境（已含 torch/ultralytics/cv2）：

```powershell
C:\Users\YQSL\miniconda3\envs\yolo\python.exe -V
```

VLM 服务需在跑。默认走 **OpenAI 兼容接口** `/v1/chat/completions`，Ollama 本身就提供这个端点，
不需要额外部署；后期换 vLLM / SGLang 只改 `config.json → vlm.base_url`。
模型必须是 **instruct 变体**（thinking 变体不可用，见下方坑记录）。
本机实际用派生模型 `qwen3-vl:2b-instruct-ctx3072`（把上下文长度烧进模型，见下方 500 坑记录）：

```powershell
ollama list          # 应有 qwen3-vl:2b-instruct-ctx3072
curl http://127.0.0.1:11434/v1/models    # 兼容层探活，应返回模型列表
```

## 坑记录：为什么必须用 instruct 变体

- `qwen3-vl:2b`（thinking 变体）**不是混合思考模型，不支持关闭思考**
  （`think:false` 与 `/no_think` 均无效，ollama/ollama#16945）。
  CPU 上思考耗尽 token 预算 → content 恒空（实测 1500/2500 tokens 全被 thinking 吃掉，
  甚至退化重复循环）；换 `qwen3-vl:2b-instruct` 后单图 61s、JSON 完整。
- 旧版把空响应当 `status=ok`，融合层会拿全 unknown 字段静默输出“正常陈列/auto”。
  现在空响应/解析失败标 `empty_response`/`parse_error` 降级态，一律转人工。

## 坑记录：VLM 报 500（两类根因，都不是模型能力问题）

16GB 纯 CPU 机器上跑完 YOLO（imgsz=1280）后剩余物理内存常不足 2GB，服务端会因内存不足失败。
两种表现要分清：

**A. 启动期申请 KV cache 失败**（错误体写明 buffer 大小）：

```
500 {"error":"llama-server startup failed ... reported out-of-memory during startup:
failed to allocate CPU buffer of size 822521856"}
```

**B. runner 进程推理途中崩溃**（错误体是内部 TCP 连接被强制关闭）：

```
500 {"error":"an error was encountered while running the model: read tcp 127.0.0.1:xxxxx->127.0.0.1:yyyyy:
wsarecv: An existing connection was forcibly closed by the remote host."}
```

B 的根因要去 **事件查看器 → Windows 日志 → 应用程序** 里找 `Application Error`：
实测 `llama-server.exe` 在 `ggml-base.dll` 里报 `0xc0000005`（访问违例，分配失败后空指针解引用）、
在 `ucrtbase.dll` 里报 `0xc0000409`（fail-fast）。同期连 `dwm.exe` 都报 `0xc00001ad`
（STATUS_FATAL_MEMORY_EXHAUSTION）——说明是**整机内存耗尽**，不是 Ollama 一家的问题。
注意 `ollama ps` 可能仍显示"已加载 1.7GB"，那是服务端状态残留，进程列表里其实没有 llama-server。

A/B 都表现为 `degraded: ["vlm:error"]`，与思考变体的"content 恒空"是不同类问题。

**修复：把上下文长度压到 3072**（实测 1024px 图 + 全量 PROMPT 需 ≥3072，
本机实测 `prompt_tokens=1469`；压到 2048 时 image token 挤占预算，content 反而恒空）。
**改哪里取决于 `vlm.api`**：

| `vlm.api` | 上下文长度怎么设 |
|---|---|
| `openai`（默认） | 兼容层**无法逐请求设置**（Ollama 官方限制）。本机用 `ollama create` 派生模型（见下）；也可启动服务前设 `$env:OLLAMA_CONTEXT_LENGTH="3072"`；vLLM 用 `--max-model-len` |
| `ollama` | `config.json → vlm.num_ctx = 3072` 直接逐请求下发 |

**本机在用的派生模型做法**——不重启服务、不需要管理员权限，官方对兼容层就是这么建议的：

```powershell
# Modelfile 只两行：FROM qwen3-vl:2b-instruct  /  PARAMETER num_ctx 3072
ollama create qwen3-vl:2b-instruct-ctx3072 -f Modelfile
ollama show qwen3-vl:2b-instruct-ctx3072 --modelfile   # 应看到 PARAMETER num_ctx 3072
```

复用同一份权重 blob（输出刷 `using existing layer`），**不占额外磁盘**；
`config.json → vlm.model` 指向派生名即可，跑完用 `ollama ps` 核实 `CONTEXT` 列真的是 3072。
此时 config 里的 `num_ctx` 应删掉：上下文已由模型携带，留着只会触发"不生效"告警。

`api=openai` 时若 config 里还留着 `num_ctx`，初始化会告警提示它不生效（不做静默死参数）。
排查命令：`ollama ps`、`Get-Process llama-server`、`Get-CimInstance Win32_OperatingSystem`
（看 `FreePhysicalMemory` 与提交内存 `TotalVirtualMemorySize - FreeVirtualMemory`）。

**放内存时的权限坑**：占用大户若是提权进程（实测 Marvis 全家 15 个进程占 8.41GB 提交内存，
外加一个父控制台已消失的孤儿 `ollama.exe`），非管理员 shell 里 `Stop-Process -Force` 与
`taskkill /F` 都只返回"拒绝访问"，而 `-ErrorAction SilentlyContinue` 会把失败吞掉、
造成"已清理"的假象。先确认自身权限：

```powershell
([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)
```

要杀就得 `Start-Process powershell -Verb RunAs` 走 UAC，或在管理员任务管理器里结束。
另外**关窗口 ≠ 退后台服务**：Marvis 窗口关闭后 `MarvisSvr` / `MarvisHost` / `MarvisAgent` /
`MarvisKnowledgebase` 仍在跑，必须按进程名核实，不能看窗口。清掉后实测
`free` 0.48GB → 3.87GB、提交内存 40.26GB → 30.79GB，单图链路随即跑通。

## 坑记录：2B 结论字段与自身观察系统性矛盾

实测 qwen3-vl:2b-instruct 的**观察字段反而正确、结论字段被“整齐=合格”锚定**：
- 洗涤货架前摆元气：around=洗衣液✓，但 scene=drink_shelf✗、depth=yes✗（behind 观察为他牌）；
- 冰柜顶摆瓶：around=冰柜顶✓，但 scene=drink_freezer✗；
- 正常货架：reason 说“无液体空瓶”但 empty_bottle_count=5✗（empty_obs 鹦鹉学舌抄提示词+幻觉计数）。

对策即上方 V4 三条：模型只负责观察，业务映射交给代码；矛盾证据低置信转人工。
另一实测坑：**2B 会鹦鹉学舌字段描述里的例句**——给空瓶字段加否定例句后所有图都抄“无空瓶”、计数全 -1；
改回问句式描述（不给例句）后才恢复真实计数。提示词字段描述只写问句、不写可抄的答案句。
空瓶判据用物理先验：**密封透明瓶默认有液体**（饮料本身透明），仅液面线清晰可见/瓶口敞开/瓶身压瘪才计空瓶
（否则 2B 会把正常货架的密封透明瓶误计为空瓶）。
精度上限仍受 2B 观察能力限制，按共识后期原位替换微调模型。

## 跑

```powershell
cd D:\Users\Desktop\巡店AI\audit
$py = "C:\Users\YQSL\miniconda3\envs\yolo\python.exe"

# 1) 冒烟测试（默认全链路：检测+VLM，会打印降级项）
& $py scripts\smoke_test.py

# 2) 单图端到端
& $py scripts\run_single.py --image images\20251004\central\00987_cur.jpg

# 3) 严格模式：任何环节降级即失败（CI/验收用）
& $py scripts\run_single.py --image images\... --strict

# 4) 只跑规则（0.1 秒，离线）
& $py scripts\run_single.py --image images\... --no-vlm --no-detect

# 5) 建检索索引
& $py scripts\build_index.py --limit 500

# 6) 生成训练表
& $py scripts\build_table.py --labels labels.csv --out outputs\train.csv

# 7) 判定路径回归（不依赖 VLM 服务/YOLO，秒级，80 项：五类口径 + 换模型/换后端契约）
& $py scripts\path_check.py

# 8) 看实时过程日志（排查用）
& $py scripts\run_single.py --image images\... --log-level debug
& $py scripts\run_single.py --image images\... --quiet     # 只要结论
```

## 实时日志（每到一个阶段边界就输出一行，不等全部跑完）

```
[+   0.0s] INFO    vlm       [vlm] 传输=openai endpoint=http://127.0.0.1:11434/v1/chat/completions model=qwen3-vl:2b-instruct-ctx3072 max_tokens=1500
[+   0.0s] INFO    pipeline  初始化：VLM=开(qwen3-vl:2b-instruct-ctx3072 via openai) | 检测=开(yolo26n.pt) | 索引=120 条
[+   0.0s] INFO    pipeline  模型语义开关：detector.target=bottle | VLM 缺陷补偿=around_override, behind_to_depth
[+   0.1s] INFO    pipeline  [quality] 0.10s -> 1440x1304 blur=860.17 composite=False(corr=0.397) moire=1.523
[+   0.1s] INFO    pipeline  [detect] YOLO 推理中（imgsz=1280 device=cpu，CPU 上约 20~60s）...
[+  10.9s] INFO    pipeline  [detect] 10.72s -> status=ok target=bottle 瓶=25 行=4
[+  11.5s] INFO    vlm       [vlm] 请求 .../v1/chat/completions api=openai 图长边=1024 max_tokens=1500 num_ctx=服务端决定 ...
[+  75.9s] INFO    vlm       [vlm] 64.3s finish_reason=stop completion_tokens=174 prompt_tokens=1469 content_len=311 thinking_len=0
[+  75.9s] INFO    compensate [compensate] behind='前排元气瓶正后方是洗衣液大瓶装，仍是元气产品' -> 截取片段='是洗衣液大瓶装，仍是元气产品' -> depth=yes
```

| 级别 | 内容 |
|---|---|
| INFO | 阶段边界 + 该阶段决定了什么（命中/推导/封顶/耗时） |
| DEBUG | 中间量：模型原始输出头部、YOLO 类别分布、每个框统计、解析后的原始 JSON |
| WARNING | 降级与兜底：`degraded` 逐项、pHash 退化、权重加载失败、检出非零但类名过滤后为 0 |
| ERROR | 本图必转人工的失败：VLM 空响应（含 finish_reason/completion_tokens/thinking_len 指纹）、HTTP 非 200（含服务端错误体原文）、JSON 解析失败 |

约定（[audit/logutil.py](audit/logutil.py) 顶部有详细说明）：

- 库代码只 `get_logger(__name__)` 打日志，**不配置 handler**；handler 只由 `scripts/` 入口
  调 `setup_logging()`。所以 `path_check.py` 这类不调 setup 的脚本默认是安静的（只留 WARNING 以上）。
- 日志走 **stdout**，不走 stderr：PowerShell 会把 stderr 渲染成红色错误文本，正常进度看起来像报错。
  只要结论时用 `--quiet`；要落文件自己 `*> log.txt`。
- 长耗时环节（YOLO / VLM）**在开始之前**就先打一条，避免 20~80s 期间终端一片空白。
- `--json-out` 的产物不变（机器可读结果），日志只给人看；两者不混：结论块仍用 print。

## 换模型操作口径（后期替换必读）

模块依赖是星形的：只有 `pipeline.py` import 各环节，环节之间靠 dict 传值、互不 import，
`rules.py` / `fusion.py` 是纯函数。所以**换实现不需要动其他模块**，但有几个开关必须跟着改，
否则会出现"环节跑通了、口径却错了"的静默劣化（`degraded` 不会报警）。

### 1. 换微调 YOLO 权重

改 `config.json` 三项，**缺一不可**：

```json
"detector": {
  "weights": "D:/models/genki_yolo.pt",
  "target": "genki",
  "bottle_class_names": ["genki"]
}
```

- `target` 是语义声明（`bottle` = COCO 通用瓶，`genki` = 元气专用）。它决定安全阀怎么解释 `rows`：
  - `bottle`：`rows>1`（任意瓶多排）才封顶转人工，单排可能是摆拍，不封顶；
  - `genki`：**去掉 `rows>1` 条件**，合法场景即封顶。因为正常货架上元气常只占顶层一排
    （实测 00021），沿用旧条件会让安全阀整体失效、空瓶误主张重新走 auto。
  - 判据集中在 [rules._weak_display_context](audit/rules.py)，两条规则都只调它。
- 加载期有两道体检会直接告警：权重类别与 `bottle_class_names` 无交集；`target=genki` 但类名仍是通用 `bottle`。
- **auto 放开的前置是微调 VLM 到位，不是检测器升级。**

### 2. 换 VLM 推理后端 / 服务地址

客户端默认说 **OpenAI 兼容协议**，这是 vLLM / SGLang / LMDeploy / TGI / Xinference / Ollama / LM Studio
的共同接口。换后端只改地址，**代码与判定口径一行不动**：

```json
"vlm": {
  "api": "openai",
  "base_url": "http://127.0.0.1:11434/v1",
  "api_key": "ollama",
  "model": "qwen3-vl:2b-instruct-ctx3072",
  "max_tokens": 1500,
  "temperature": 0.1,
  "image_max_side": 1024,
  "timeout": 300
}
```

| 键 | 说明 |
|---|---|
| `api` | `openai`（默认，`/v1/chat/completions`）或 `ollama`（原生 `/api/chat`，回退/调试用） |
| `base_url` | 服务地址。带不带 `/v1`、带不带尾斜杠都行，内部会归一；旧 `host` 键仍兼容 |
| `api_key` | Ollama 忽略但要求非空；vLLM 启动带 `--api-key` 或走云端网关时填真值 |
| `max_tokens` | 输出预算（旧名 `num_predict` 仍兼容读取） |
| `num_ctx` | **只在 `api=ollama` 时逐请求生效**；`openai` 模式下会告警。本机走 `openai`，上下文由派生模型 `-ctx3072` 携带，故此键已删 |

搬到服务器（例：vLLM 起微调模型）只改这几行：

```json
"vlm": { "base_url": "http://gpu-server:8000/v1", "model": "genki-vl-7b", "compensations": [] }
```

图片传输两边都是 base64（`image_url` 里塞 `data:image/jpeg;base64,...`），
所以迁移不涉及图像通道改造。注意 Ollama 兼容层**只接受 base64，不接受 http 图片 URL**。

两种传输的能力差异（已在 `path_check.py` 锁成离线契约）：

| | `openai` | `ollama` |
|---|---|---|
| JSON mode | `response_format={"type":"json_object"}` | `format="json"` |
| 逐请求上下文长度 | ✗ 服务端决定 | ✓ `options.num_ctx` |
| 思考内容字段 | `reasoning_content`(vLLM) / `reasoning`(Ollama) | `thinking` |
| 结束原因 / token 数 | `finish_reason` / `usage.*` | `done_reason` / `eval_count` |

`_extract()` 把这些差异归一成同一组字段（`text` / `thinking` / `finish_reason` /
`completion_tokens` / `prompt_tokens`），**下游的空响应降级、JSON 解析、补偿层、规则融合完全与传输无关**。
所以换后端不会动到判定口径，也不会让"content 恒空"这类静默误判复活。

### 3. 换微调 VLM

```json
"vlm": { "model": "genki-vl:7b", "compensations": [] }
```

`compensations` 控制 [audit/compensate.py](audit/compensate.py) 这一层：

| 值 | 含义 |
|---|---|
| `["around_override", "behind_to_depth"]` | 当前 2B 的完整补偿（默认，等价于改造前行为） |
| `[]` | 微调模型模式，直接采信模型结论字段 |
| 只留一个 | 支持；未知名字告警后忽略，不会静默当成开启 |

为什么必须关：这两条补丁的理由都是"2B 结论被『整齐=合格』锚定"。微调模型没这毛病后，
补丁反而有害——模型正确报 `drink_shelf`、`around` 如实提到邻接洗涤区，覆写会把 scene 改成
`other_shelf` 并把置信度压到 0.55，**把修好的模型拖回复核**。

⚠️ **关掉 `behind_to_depth` 必须同步改 PROMPT**：当前 PROMPT 只让模型输出 `behind` 自由文本，
没有 `depth` 字段。补偿一关，`depth` 走"模型自报枚举"分支，模型没报就恒为 `unknown` →
纵深规则不判（不会误定罪，但召回会掉）。新模型要直接输出 `depth=yes/no/unknown`。

场景白名单推导（`scene_error`）是业务共识，**不受开关影响**，永远生效。

### 4. 改判重逻辑

定罪范围只由 [rules.rule_reuse](audit/rules.py) 的 `ok` 表达，`fusion` 不再按规则名二次剔除：

- 同店命中 → `ok=False`（同一门店多拜访重复上传属正常），证据文本仍入 `evidence` 保留记录；
- 异店命中 → `ok=True` 定罪；
- 门店归属不明（未传 `store_id`，或索引元数据缺网点编码）→ 命中但封顶 0.55 转人工，
  不敢断言"另一门店"。跑单图时记得带 `--store-id`，否则判重只能转人工。

改成多命中 / 时间窗 / cur+match 成对比对时，只需动 `rule_reuse` 与 `matches` 结构；
换 embed 后端则必须重跑 `build_index.py`（向量维度变了 `vecs @ v` 会直接维度不匹配）。

### 回归

```powershell
python scripts/path_check.py     # 80 项：26 项五类口径 + 23 项换模型契约 + 31 项换后端传输契约
```

中间 23 项锁的不是"当前模型判得准不准"，而是"换权重/换 VLM/换判重方案时口径不会静默漂移"：
判重定罪范围 6 项、安全阀随 `target` 切换 8 项、补偿层开/关/部分开 9 项。

最后 31 项锁传输协议，**全程不发网络请求**（内存紧张、服务没起时也能跑）：
`base_url` 归一 5 项、endpoint 拼接 4 项、OpenAI 请求信封 6 项、原生信封与 `num_ctx` 语义 6 项、
响应归一与空响应降级指纹 10 项。其中关键一条是"两种信封喂同一组语义值，`_extract` 结果必须完全相等"。

## 降级约定（重要）

任何环节没真正跑起来，都会写在输出 JSON 的 `degraded` 数组里，例如：

```json
"degraded": ["detect:unavailable", "vlm:disabled_by_flag", "retrieval:no_index"]
```

检索报错也会入数组（`retrieval:error:...`）——换 embed 后端或索引维度不匹配时，
跨图规则会静默失效，必须让它显式降级而不是看着像全跑通。

`--strict` 模式下，只要 `degraded` 非空就返回退出码 2。
**看到 `degraded: []` 才代表这条链路是真的全跑通了。**

## 这一版做了什么 / 没做什么

| 阶段 | 状态 | 说明 |
|---|---|---|
| 质量 + 拼接 + 摩尔纹 | ✅ 可用 | 纯 numpy，零依赖 |
| YOLO 瓶体检测 | ⚠️ 需权重 | 首次运行会尝试下载 `yolo26n.pt`；无网则自动降级 |
| 1:N 检索 | ⚠️ 需索引 | 先跑 `build_index.py`；`resnet18` 权重下载失败时退化 pHash |
| VLM 判定 | ✅ OpenAI 兼容接口 | Ollama / vLLM 均可，不依赖 transformers |
| 融合 + 转人工 | ✅ 可用 | 置信度低于阈值自动路由人工 |

**已知缺口（下一版补）**：
1. 2B 空瓶/behind 观察在正常货架上不可靠（密封透明瓶误计空瓶、behind 跨轮漂移）→ 安全阀压人工；
   精度按共识留待微调模型原位替换；
2. 门头 OCR 未接入（PaddleOCR）；
3. 阈值仅在 10 张样本上校准，扩量标注集后需重校（尤其空瓶计数置信度曲线）；
4. 检索索引的 store_id 判重还没接业务库，同店/跨店的区分只做了字面比较；
5. 跨图两类（同店多门店、相册复用）待系统输入形态确认后实现。

## 单图跑通结果（6 图·qwen3-vl:2b-instruct·CPU·全链路无降级）

| 图 | 真值 | 主标签 | 多标签 | conf | 路由 |
|---|---|---|---|---|---|
| 20251009/southwest/05025 | 摆拍·冰柜顶 | 空瓶 | 空瓶+场景错误 | 0.9 | human |
| 20251004/central/00987 | 摆拍·洗涤货架 | 空瓶 | 空瓶+场景错误+纵深不足 | 0.8 | human |
| 20251002/north/00582 | 摆拍·餐区 | 空瓶 | 空瓶+场景错误 | 0.9 | auto |
| 20251016/southwest/15164 | 拍屏幕 | 套拍-相似图 | 套拍-相似图 | 0.95 | auto |
| 20251001/north/00002 | 正常·割箱 | 空瓶(2B 误主张) | 空瓶 | 0.55 | human（安全阀） |
| 20251001/northeast/00021 | 正常·货架 | 空瓶(2B 误主张) | 空瓶 | 0.55 | human（安全阀） |

四条单图判定路径（空瓶/纵深不足/场景错误/拍屏幕）均在真实图触发；
正常图零自动定罪（2B 空瓶误主张被安全阀压到 0.55 交人工）。
跨图两类（同店多门店）保留骨架未增强。单图端到端约 30~80s（VLM 占大头）。

**OpenAI 兼容协议复跑验证**（`api=openai` + 派生模型 `qwen3-vl:2b-instruct-ctx3072`）：
`00987` 单图 `--strict` 退出码 0、`degraded` 为空，`finish_reason=stop`、
`completion_tokens=174`、`prompt_tokens=1469`、`content_len=311`、`thinking_len=0`，
`ollama ps` 的 `CONTEXT` 列为 3072，`path_check` 80/80。
即兼容信封在**真实模型推理**下返回完整 JSON，不只是离线契约通过。

⚠️ 同一张 `00987` 这次多标签是"空瓶+场景错误"，比上表少一个"纵深不足"：
`behind` 观察本轮返回"正后方是洗衣液大瓶装，仍是元气产品"→ 补偿层推 `depth=yes`。
这正是已记录的 **2B behind 观察跨轮漂移**，不是协议改造引入的回归——
传输层只换信封，`PROMPT` / `_parse` / `_normalize` / 补偿层一行未动。

## 输出示例

```json
{
  "verdict": {
    "label_main": "套拍-相似图",
    "label_all": ["套拍-相似图", "场景错误"],
    "confidence": 0.88,
    "route": "auto",
    "evidence": ["[composite] 左右半幅相关 0.91，判定为拼接双图", "..."],
    "reason": "左右半幅相关 0.91，判定为拼接双图"
  },
  "stages": {"quality": {...}, "detect": {...}, "retrieval": {...}, "vlm": {...}},
  "timings": {"quality": 0.02, "detect": 1.8, "vlm": 6.4, "total": 8.3}
}
```

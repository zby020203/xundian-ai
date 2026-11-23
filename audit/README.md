# 巡店 AI · 端到端框架 v0.2（单图版·五类新口径）

一句话：**一张图进来 → 规则筛 → YOLO 检测 → 1:N 检索 → VLM 判定 → 融合结论（含是否转人工）**。
每个阶段都可降级：缺权重、没开 Ollama、没建索引，都不会崩，只在证据链里标 `unavailable`。

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
│   ├── imageops.py      # 清晰度 / 拼接检测 / 摩尔纹 / 感知哈希
│   ├── detect.py        # YOLO 检测（瓶体）
│   ├── embed.py         # 特征向量（ResNet / 退化 pHash）
│   ├── retrieve.py      # numpy 1:N 检索
│   ├── vlm.py           # Ollama qwen3-vl 调用
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

Ollama 需在跑，且已拉取 **instruct 变体**（thinking 变体不可用，见下方坑记录）：

```powershell
ollama list          # 应有 qwen3-vl:2b-instruct
```

## 坑记录：为什么必须用 instruct 变体

- `qwen3-vl:2b`（thinking 变体）**不是混合思考模型，不支持关闭思考**
  （`think:false` 与 `/no_think` 均无效，ollama/ollama#16945）。
  CPU 上思考耗尽 token 预算 → content 恒空（实测 1500/2500 tokens 全被 thinking 吃掉，
  甚至退化重复循环）；换 `qwen3-vl:2b-instruct` 后单图 61s、JSON 完整。
- 旧版把空响应当 `status=ok`，融合层会拿全 unknown 字段静默输出“正常陈列/auto”。
  现在空响应/解析失败标 `empty_response`/`parse_error` 降级态，一律转人工。

## 坑记录：VLM 报 500 out-of-memory（非模型问题，是 num_ctx 默认值）

16GB 纯 CPU 机器上跑完 YOLO（imgsz=1280）后剩余物理内存常不足 2GB，
Ollama 按默认 `num_ctx=4096` 申请 KV cache 会失败：

```
500 {"error":"llama-server startup failed ... reported out-of-memory during startup:
failed to allocate CPU buffer of size 822521856"}
```

表现为 `degraded: ["vlm:error"]`，与思考变体的“content 恒空”是两类问题。
修复：`config.json → vlm.num_ctx = 3072`（实测 1024px 图 + 全量 PROMPT 需 ≥3072；
压到 2048 时 image token 挤占预算，content 反而恒空）。可用 `tasklist | findstr llama` /
`ollama ps` 确认无残留进程、`Get-CimInstance Win32_OperatingSystem` 看剩余内存。

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

# 7) 判定路径回归（不依赖 Ollama/YOLO，秒级，26 条用例：15 规则融合 + 11 归一化推导）
& $py scripts\path_check.py
```

## 降级约定（重要）

任何环节没真正跑起来，都会写在输出 JSON 的 `degraded` 数组里，例如：

```json
"degraded": ["detect:unavailable", "vlm:disabled_by_flag", "retrieval:no_index"]
```

`--strict` 模式下，只要 `degraded` 非空就返回退出码 2。
**看到 `degraded: []` 才代表这条链路是真的全跑通了。**

## 这一版做了什么 / 没做什么

| 阶段 | 状态 | 说明 |
|---|---|---|
| 质量 + 拼接 + 摩尔纹 | ✅ 可用 | 纯 numpy，零依赖 |
| YOLO 瓶体检测 | ⚠️ 需权重 | 首次运行会尝试下载 `yolo26n.pt`；无网则自动降级 |
| 1:N 检索 | ⚠️ 需索引 | 先跑 `build_index.py`；`resnet18` 权重下载失败时退化 pHash |
| VLM 判定 | ✅ 走 Ollama | 不依赖 transformers |
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
跨图两类（同店多门店）保留骨架未增强。单图端到端约 30~50s（VLM 占大头）。

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

"""VLM 判定层：通过 Ollama HTTP API 调用 qwen3-vl。

要点（实测得出）：
1. **必须用 instruct 变体（qwen3-vl:2b-instruct）**：thinking 变体不是混合思考模型，
   不支持关闭思考（think:false 与 /no_think 均无效，ollama/ollama#16945），
   CPU 上思考会耗尽 token 预算导致 content 恒空、甚至退化重复循环；
2. /api/chat 比 /api/generate 快约一倍；
3. 缩略图分辨率由 image_max_side 配置（默认 1024）；
4. 模型输出的 JSON 可能缺字段，这里统一补齐并做归一化；
5. 空响应/解析失败一律标降级状态（empty_response/parse_error），
   由融合层转人工，绝不允许静默输出“正常陈列”。
"""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Dict

import requests
from PIL import Image

# 场景白名单（共识定义）：只有这四类是合法陈列位，其余一律场景错误。
SCENE_WHITELIST = {"drink_shelf", "drink_freezer", "case_display", "endcap"}

# 模型可能输出的旧枚举/别名 -> 新枚举。_generic shelf 无法区分饮料/非饮料，保守归 unknown。
_SCENE_ALIASES = {
    "freezer": "drink_freezer", "fridge": "drink_freezer", "icebox": "drink_freezer",
    "cooler": "drink_freezer", "shelf": "unknown", "snack": "snack_shelf",
    "checkout": "counter", "cashier": "counter", "dining": "table",
    "roadside": "outdoor", "street": "outdoor", "monitor": "screen",
    "case": "case_display", "box_display": "case_display", "end_cap": "endcap",
    "warehouse": "storage",
}

PROMPT = """你是零售巡店的图像审核员。先观察后结论，看不到的填 unknown，禁止猜测。
注意：摆拍陈列往往恰恰表现为“单排、等距、正对镜头”，整齐不是合格依据。
元气森林产品：气泡水、外星人电解质水、维C水、维B水、冰茶、好望水。
空瓶判据：饮料本身是透明液体；密封透明瓶默认判有液体，仅当液面线清晰可见（液面低于瓶肩）或瓶口敞开/瓶身压瘪才计空瓶。

只输出 JSON，字段按序：
1. around：一句话，陈列面周围是什么非元气商品（零食/文具/洗涤/烟酒/母婴/餐具/无…）
2. behind：一句话，前排元气瓶正后方是什么（仍是元气产品/其他品牌产品/空层板/被挡看不清）
3. empty_obs：一句话，陈列面有无无液体的透明空瓶、约几支；写实际观察
4. scene：枚举 drink_shelf(饮料货架内)/drink_freezer(冰柜柜体内部,含他司;冰柜顶/盖上摆瓶不算)/case_display(割箱堆箱)/endcap/snack_shelf/other_shelf(文具/洗涤/烟酒/母婴等非饮料货架)/counter/table(餐区或冰柜顶桌面)/outdoor/storage/screen/other/unknown
5. is_screen：拍屏幕/翻拍=true（UI、状态栏、机身边框、摩尔纹、拼接缝）
6. empty_bottle_count：整数，依据 empty_obs；没有=0，看不清=-1
7. single_row：元气只摆一排等距正对镜头=true
8. confidence：0到1
9. reason：一句中文总结"""

_JSON_HINT = '{"around":"...","behind":"...","empty_obs":"...","scene":"drink_shelf","is_screen":false,"empty_bottle_count":0,"single_row":false,"confidence":0.8,"reason":"..."}'


def _b64(im: Image.Image, max_side: int = 768) -> str:
    if max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaVLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.status = "not_called"

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.cfg.get('host')}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def judge(self, im: Image.Image) -> Dict[str, object]:
        host = self.cfg.get("host", "http://127.0.0.1:11434")
        payload = {
            "model": self.cfg.get("model", "qwen3-vl:2b-instruct"),
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [_b64(im, int(self.cfg.get("image_max_side", 1024)))]}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(self.cfg.get("temperature", 0.1)),
                "num_predict": int(self.cfg.get("num_predict", 1500)),
            },
        }
        try:
            r = requests.post(f"{host}/api/chat", json=payload,
                              timeout=int(self.cfg.get("timeout", 300)))
            r.raise_for_status()
            data = r.json()
            text = (data.get("message") or {}).get("content", "") or data.get("response", "")
        except Exception as e:
            self.status = "error"
            return {"status": "error", "error": str(e)}

        if not (text or "").strip():
            # 思考型模型可能把 token 预算全耗在 thinking 上导致 content 为空。
            # 绝不能标 ok：否则融合层会拿全 unknown 字段输出“正常陈列/auto”的静默误判。
            self.status = "empty_response"
            return {"status": "empty_response",
                    "error": "模型输出为空（思考耗尽 token 预算），本图转人工",
                    "raw": ""}
        parsed = self._normalize(self._parse(text))
        if parsed.get("parse_error"):
            self.status = "parse_error"
            parsed["status"] = "parse_error"
            parsed["error"] = "模型输出无法解析为 JSON，本图转人工"
        else:
            self.status = "ok"
            parsed["status"] = "ok"
        parsed["raw"] = text[:300]
        return parsed

    # ------------------------------------------------------------ 解析

    @staticmethod
    def _parse(text: str) -> Dict[str, object]:
        t = (text or "").strip()
        t = re.sub(r"^```(json)?|```$", "", t, flags=re.MULTILINE).strip()
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"parse_error": True}

    @staticmethod
    def _normalize(d: Dict[str, object]) -> Dict[str, object]:
        out = dict(d)
        scene = str(out.get("scene", "unknown")).lower().strip()
        scene = _SCENE_ALIASES.get(scene, scene)
        known = scene in SCENE_WHITELIST or scene in (
            "snack_shelf", "other_shelf", "counter", "table", "outdoor",
            "storage", "screen", "other",
        )
        if not known:
            scene = "unknown"
        out["scene"] = scene
        out["scene_known"] = scene != "unknown"

        for k in ("is_screen", "single_row"):
            v = out.get(k, False)
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "yes", "1", "是")
            out[k] = bool(v)

        # 场景错误由白名单确定性推导，不信任模型自报的布尔值；
        # 拍屏幕时画面位置无意义，不做场景判定（主问题归套拍-相似图）。
        if out["is_screen"] or not out["scene_known"]:
            out["scene_error"] = False
        else:
            out["scene_error"] = scene not in SCENE_WHITELIST

        # around 交叉校验：观察出现强非饮料信号但 scene 报白名单时，按白名单规则覆写。
        # 实测 2B 的 scene 结论会被“整齐=合格”锚定（洗涤货架报 drink_shelf、冰柜顶报 drink_freezer），
        # 而其 around 观察反而正确。覆写置信度封顶 0.55 转人工，防邻接商品误伤。
        around = str(out.get("around") or "")
        override = None
        if not out["is_screen"] and out["scene_known"] and scene in SCENE_WHITELIST and around:
            if any(k in around for k in ("冰柜顶", "盖上", "餐桌", "桌面")):
                override = "table"
            elif "收银" in around:
                override = "counter"
            elif any(k in around for k in ("洗衣", "洗涤", "文具", "烟酒", "母婴", "餐具")):
                override = "other_shelf"
        # 注意：“零食”不做覆写信号——零食常邻接饮料陈列（实测正常割箱被误伤）。
        if override:
            out["scene_override_from"] = scene
            out["scene"] = override
            out["scene_error"] = True
            try:
                cur = float(out.get("confidence", 0.6))
            except Exception:
                cur = 0.6
            out["confidence"] = min(cur, 0.55)

        # behind 自由文本观察 -> depth 确定性推导（共识：后方有元气即合格）。
        # 不信任模型自报 depth/枚举：实测 2B 结论被“整齐=合格”锚定，自由文本观察反而正确。
        # 截取“后方/背后/后面”之后的片段，避免“前排元气瓶”字样干扰；
        # 遮挡优先判 unknown（宁可不判，不误判）；纸箱不算“无产品”（割箱陈列后方是元气自己的纸箱）。
        behind = str(out.get("behind") or "")
        seg = behind
        for anchor in ("后方", "背后", "后面"):
            i = behind.find(anchor)
            if i >= 0:
                seg = behind[i + len(anchor):]
                break
        low = behind.strip().lower()
        if low in ("genki", "yes"):
            depth = "yes"
        elif low in ("other", "empty", "no"):
            depth = "no"
        elif low in ("occluded", "unknown"):
            depth = "unknown"
        elif "元气" in seg:
            depth = "yes"
        elif any(k in behind for k in ("看不清", "被挡", "遮挡")):
            depth = "unknown"
        elif out.get("scene") == "case_display" and any(k in behind for k in ("纸箱", "饮料箱", "包装袋")):
            # 割箱陈列后方通常是元气自己的纸箱，品牌不可读 → 宁可不判。
            depth = "unknown"
        elif any(k in seg for k in ("其他品牌", "他牌", "空层板", "无产品")):
            depth = "no"
        else:
            depth = "unknown"
        out["depth"] = depth

        # 空瓶计数：缺省/非法一律 -1（不可判），兼容旧字段 has_empty_bottle。
        cnt = out.get("empty_bottle_count", None)
        if cnt is None and isinstance(out.get("has_empty_bottle"), bool):
            cnt = 1 if out["has_empty_bottle"] else 0
        try:
            cnt = int(cnt)
        except Exception:
            cnt = -1
        out["empty_bottle_count"] = max(-1, min(cnt, 999))

        try:
            out["confidence"] = float(out.get("confidence", 0.6))
        except Exception:
            out["confidence"] = 0.6
        if out["confidence"] <= 0:
            out["confidence"] = 0.6
        return out

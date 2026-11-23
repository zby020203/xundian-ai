"""VLM 判定层：默认走 OpenAI 兼容 /v1/chat/completions，原生 Ollama /api/chat 可回退。

为什么默认 OpenAI 兼容（config.vlm.api="openai"）：
- vLLM / SGLang / LMDeploy / TGI / Xinference / Ollama / LM Studio 都暴露这个接口，
  后期换微调模型或换推理后端只改 config.vlm.base_url，客户端与判定口径一行不动；
- 图片传输两边其实一样（base64），差别只在外层信封，迁移没有额外代价；
- 与有没有显卡无关：CPU/GPU 是服务端的事，协议选型是客户端的事。

两种传输的能力差异（Ollama 官方文档 + 本机实测）：
1. **OpenAI 兼容层无法逐请求设置上下文长度**。Ollama 侧要用 OLLAMA_CONTEXT_LENGTH
   环境变量或 Modelfile 派生模型，vLLM 侧是启动参数 --max-model-len；
   config.vlm.num_ctx 只在 api="ollama" 时真正下发，openai 模式下会告警提示。
2. 兼容层只接受 base64 图片，不接受 http 图片 URL —— 本模块一直用 base64，无影响。
3. 原生层多给 thinking / done_reason / eval_count，诊断“思考耗尽预算”更直接；
   兼容层对应 reasoning_content / finish_reason / usage.completion_tokens。
   _extract() 把两者归一到同一组字段，所以下游判定与降级逻辑与传输无关。

判定口径要点（实测得出，与传输无关）：
1. **必须用 instruct 变体（qwen3-vl:2b-instruct）**：thinking 变体不是混合思考模型，
   不支持关闭思考（think:false 与 /no_think 均无效，ollama/ollama#16945），
   CPU 上思考会耗尽 token 预算导致 content 恒空、甚至退化重复循环；
2. 缩略图分辨率由 image_max_side 配置（默认 1024）；
3. 模型输出的 JSON 可能缺字段，这里统一补齐并做归一化；
4. 空响应/解析失败一律标降级状态（empty_response/parse_error），
   由融合层转人工，绝不允许静默输出“正常陈列”。
"""
from __future__ import annotations

import base64
import json
import re
import time
from io import BytesIO
from typing import Dict

import requests
from PIL import Image

from . import compensate
from .logutil import get_logger

log = get_logger(__name__)

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


# 传输协议：openai=业界通用 /v1/chat/completions（默认）；ollama=原生 /api/chat（回退/调试）。
APIS = ("openai", "ollama")


def _root_url(cfg: dict) -> str:
    """把 base_url 归一成服务根（不带 /v1），再由各传输自己拼路径。

    兼容两种常见写法：OpenAI SDK 习惯带 /v1（http://host:8000/v1），
    服务根习惯不带（http://host:11434）。也兼容旧配置里的 host 键。
    """
    raw = str(cfg.get("base_url") or cfg.get("host") or "http://127.0.0.1:11434").strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


class VLMClient:
    """图像判定客户端。传输可换，判定口径（PROMPT/_parse/_normalize/补偿层）不换。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.status = "not_called"
        # 缺陷补偿开关：缺省全开（当前 2B）；微调模型上线置 config.vlm.compensations=[]。
        self.compensations = compensate.resolve(cfg.get("compensations"))

        api = str(cfg.get("api", "openai")).strip().lower()
        if api not in APIS:
            log.warning("[vlm] 未知 api=%r，回退 openai（可选：%s）", api, "/".join(APIS))
            api = "openai"
        self.api = api
        self.root = _root_url(cfg)
        self.model = str(cfg.get("model", "qwen3-vl:2b-instruct"))
        # max_tokens 是 OpenAI 通用名；旧配置的 num_predict 仍兼容读取。
        self.max_tokens = int(cfg.get("max_tokens", cfg.get("num_predict", 1500)))
        self.temperature = float(cfg.get("temperature", 0.1))
        self.timeout = int(cfg.get("timeout", 300))
        self.image_max_side = int(cfg.get("image_max_side", 1024))
        # Ollama 忽略 Authorization，但 OpenAI SDK / vLLM / 云端网关要求非空，统一带上。
        self.api_key = str(cfg.get("api_key") or "ollama")
        self.num_ctx = cfg.get("num_ctx")

        if self.api == "openai" and self.num_ctx is not None:
            log.warning("[vlm] api=openai 无法逐请求设置上下文长度，config.vlm.num_ctx=%s 不会生效；"
                        "Ollama 请在启动服务前设 OLLAMA_CONTEXT_LENGTH=%s（或 ollama create 派生模型），"
                        "vLLM 请用 --max-model-len；要逐请求控制就改 api=ollama",
                        self.num_ctx, self.num_ctx)
        log.info("[vlm] 传输=%s endpoint=%s model=%s max_tokens=%s",
                 self.api, self.endpoint, self.model, self.max_tokens)

    @property
    def endpoint(self) -> str:
        return f"{self.root}/v1/chat/completions" if self.api == "openai" else f"{self.root}/api/chat"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def available(self) -> bool:
        url = f"{self.root}/v1/models" if self.api == "openai" else f"{self.root}/api/tags"
        try:
            r = requests.get(url, headers=self._headers(), timeout=3)
            return r.status_code == 200
        except Exception as e:
            log.debug("[vlm] %s 探活失败：%s", url, e)
            return False

    # ------------------------------------------------------------ 请求/响应信封

    def _payload(self, b64: str) -> Dict[str, object]:
        """同一张图 + 同一段 PROMPT，按传输协议包成不同信封。"""
        if self.api == "openai":
            return {
                "model": self.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]}],
                "stream": False,
                # JSON mode：Ollama 与 vLLM 都支持，等价于原生层的 format:"json"。
                "response_format": {"type": "json_object"},
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        options: Dict[str, object] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = int(self.num_ctx)
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "format": "json",
            "options": options,
        }

    def _extract(self, data: Dict[str, object]) -> Dict[str, object]:
        """把两种响应信封归一到同一组字段，下游判定/降级逻辑与传输无关。"""
        if self.api == "openai":
            choices = data.get("choices") or []
            first = choices[0] if choices else {}
            msg = first.get("message") or {}
            usage = data.get("usage") or {}
            return {
                "text": str(msg.get("content") or ""),
                # vLLM 用 reasoning_content，Ollama 兼容层用 reasoning，原生层用 thinking。
                "thinking": str(msg.get("reasoning_content") or msg.get("reasoning") or ""),
                "finish_reason": first.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
            }
        msg = data.get("message") or {}
        return {
            "text": str(msg.get("content") or data.get("response") or ""),
            "thinking": str(msg.get("thinking") or ""),
            "finish_reason": data.get("done_reason"),
            "completion_tokens": data.get("eval_count"),
            "prompt_tokens": data.get("prompt_eval_count"),
        }

    def judge(self, im: Image.Image) -> Dict[str, object]:
        payload = self._payload(_b64(im, self.image_max_side))
        ctx = (f"num_ctx={self.num_ctx}" if self.api == "ollama" and self.num_ctx is not None
               else "num_ctx=服务端决定")
        log.info("[vlm] 请求 %s api=%s model=%s 图长边=%s max_tokens=%s %s timeout=%ss"
                 "（无 GPU 时单图 20~80s，期间无输出属正常）...",
                 self.endpoint, self.api, self.model, self.image_max_side,
                 self.max_tokens, ctx, self.timeout)
        t = time.time()
        try:
            r = requests.post(self.endpoint, json=payload, headers=self._headers(), timeout=self.timeout)
            if r.status_code != 200:
                # 服务端错误体里才有真正根因（例：500 out-of-memory 会写明申请失败的
                # buffer 大小），框架层只留 "500 Server Error" 会让人在参数层反复误试。
                log.error("[vlm] HTTP %s 响应体：%s", r.status_code, r.text[:400])
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self.status = "error"
            log.error("[vlm] 调用失败（%.1fs）：%s", time.time() - t, e)
            return {"status": "error", "error": str(e)}
        took = time.time() - t
        got = self._extract(data)
        text, thinking = got["text"], got["thinking"]
        log.info("[vlm] %.1fs finish_reason=%s completion_tokens=%s prompt_tokens=%s "
                 "content_len=%d thinking_len=%d",
                 took, got["finish_reason"], got["completion_tokens"], got["prompt_tokens"],
                 len(text), len(thinking))
        log.debug("[vlm] raw head: %r", text[:200])

        if not text.strip():
            # 思考型模型可能把 token 预算全耗在 thinking 上导致 content 为空。
            # 结出的指纹三件套：finish_reason=length + completion_tokens≈max_tokens + thinking 非空。
            # 绝不能标 ok：否则融合层会拿全 unknown 字段输出“正常陈列/auto”的静默误判。
            self.status = "empty_response"
            log.error("[vlm] content 为空（%.1fs）：finish_reason=%s completion_tokens=%s thinking_len=%d "
                      "——疑似思考耗尽 token 预算，本图转人工",
                      took, got["finish_reason"], got["completion_tokens"], len(thinking))
            return {"status": "empty_response",
                    "error": "模型输出为空（思考耗尽 token 预算），本图转人工",
                    "raw": ""}
        parsed = self._normalize(self._parse(text), self.compensations)
        if parsed.get("parse_error"):
            self.status = "parse_error"
            parsed["status"] = "parse_error"
            parsed["error"] = "模型输出无法解析为 JSON，本图转人工"
            log.error("[vlm] 输出无法解析为 JSON（%.1fs），本图转人工 | raw head: %r", took, text[:200])
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
    def _normalize(d: Dict[str, object], compensations=None) -> Dict[str, object]:
        """协议归一（换模型也要留）+ 可选缺陷补偿层（换微调模型要整层关）。

        compensations=None 表示全开，保持 path_check 等直接调用的老语义；传 [] 即微调模型模式。
        业务口径（场景白名单推导 scene_error）不受开关影响，永远生效。
        """
        out = dict(d)
        scene_raw = str(out.get("scene", "unknown")).lower().strip()
        scene = _SCENE_ALIASES.get(scene_raw, scene_raw)
        if scene != scene_raw:
            log.info("[vlm.norm] scene 别名映射：%s -> %s", scene_raw, scene)
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
        log.info("[vlm.norm] scene=%s 属白名单=%s 拍屏=%s -> scene_error=%s",
                 scene, scene in SCENE_WHITELIST, out["is_screen"], out["scene_error"])

        # depth 协议归一：先按模型自报枚举兜底（换任何模型都要有这一步）；
        # 补偿层开启时会被 behind 自由文本推导覆盖，关闭时直接采信模型结论。
        raw_depth = str(out.get("depth") or "").strip().lower()
        out["depth"] = raw_depth if raw_depth in ("yes", "no", "unknown") else "unknown"

        # 缺陷补偿（around 覆写 / behind->depth 推导）已抽到 compensate.py，可整层关闭。
        # 换微调 VLM 时把 config.vlm.compensations 置为 []，这里就只剩协议归一与业务白名单。
        names = compensate.resolve(compensations)
        out = compensate.apply(out, names, SCENE_WHITELIST)
        log.debug("[vlm.norm] depth=%s 来源=%s", out["depth"],
                  "behind 推导" if "behind_to_depth" in names else "模型自报枚举")

        # 空瓶计数：缺省/非法一律 -1（不可判），兼容旧字段 has_empty_bottle。
        cnt = out.get("empty_bottle_count", None)
        if cnt is None and isinstance(out.get("has_empty_bottle"), bool):
            cnt = 1 if out["has_empty_bottle"] else 0
        try:
            cnt = int(cnt)
        except Exception:
            cnt = -1
        out["empty_bottle_count"] = max(-1, min(cnt, 999))
        if cnt == -1 and d.get("empty_bottle_count") is not None:
            log.warning("[vlm.norm] 空瓶计数字段无法转成整数（原始=%r），按不可判处理", d.get("empty_bottle_count"))
        else:
            log.debug("[vlm.norm] 空瓶计数=%s（原始=%r）", out["empty_bottle_count"], d.get("empty_bottle_count"))

        try:
            out["confidence"] = float(out.get("confidence", 0.6))
        except Exception:
            out["confidence"] = 0.6
        if out["confidence"] <= 0:
            out["confidence"] = 0.6
        return out

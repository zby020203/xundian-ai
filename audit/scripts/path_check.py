"""判定路径回归：不依赖 Ollama/YOLO，直接验证 rules+fusion 的五类口径与降级路由。

    python scripts/path_check.py

覆盖：空瓶(多支/孤立)、纵深不足、场景错误、拍屏幕、拼接仅记录、正常、
VLM 空响应/不可用转人工、纵深 unknown 不判、拍屏幕时陈列判定不适用。

另锁四组"换模型/换后端不漂口径"契约（见文件下半）：
    REUSE_CASES  判重定罪范围（同店/异店/归属不明）
    VALVE_CASES  安全阀随 detector.target 切换的语义
    COMP_CASES   VLM 缺陷补偿层开/关/部分开
    传输契约     OpenAI 兼容 / 原生 Ollama 两种信封与响应归一（离线，不发请求）
这些组测的不是"当前模型判得准不准"，而是"换权重/换 VLM/换推理后端/换判重方案时口径不会静默漂移"。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit import fusion, rules  # noqa: E402
from audit.vlm import PROMPT, VLMClient, _root_url  # noqa: E402

CFG = {"human_review_confidence": 0.6,
       "label_priority": ["套拍-相似图", "套拍-同店多门店", "空瓶", "场景错误", "纵深不足", "正常陈列"]}

CLEAN_QUALITY = {"composite": False, "composite_corr": 0.3, "moire": 1.45}
DET_OK = {"status": "ok", "n_bottles": 12, "rows": 1,
          "bottles": [{"cx": 100 + i * 60} for i in range(12)]}


def vlm(**kw):
    base = {"status": "ok", "is_screen": False, "single_row": False,
            "scene": "drink_shelf", "scene_known": True, "scene_error": False,
            "depth": "yes", "empty_bottle_count": 0, "confidence": 0.85}
    base.update(kw)
    return base


def hits_of(v, quality=CLEAN_QUALITY, det=DET_OK):
    return {
        "composite": rules.rule_composite(quality, 0.85),
        "single_row": rules.rule_single_row(det),
        "screen": rules.rule_screen(v),
        "empty_bottle": rules.rule_empty_bottle(v, det),
        "scene_error": rules.rule_scene_error(v),
        "depth": rules.rule_depth(v, det),
        "reuse_other_store": rules.rule_reuse([], 0.92, None),
    }


CASES = [
    # (名称, vlm字段, quality, 期望主标签, 期望路由, 期望包含标签)
    ("空瓶多支+场景+纵深", vlm(empty_bottle_count=5, depth="no", scene="other_shelf",
                          scene_known=True, scene_error=True), CLEAN_QUALITY,
     "空瓶", "auto", {"空瓶", "场景错误", "纵深不足"}),
    ("孤立空瓶转人工", vlm(empty_bottle_count=1), CLEAN_QUALITY,
     "空瓶", "human", {"空瓶"}),
    ("纵深不足-他牌后排", vlm(depth="no"), CLEAN_QUALITY,
     "纵深不足", "auto", {"纵深不足"}),
    ("纵深unknown不判", vlm(depth="unknown"), CLEAN_QUALITY,
     "正常陈列", "auto", set()),
    ("场景错误-餐区", vlm(scene="table", scene_known=True, scene_error=True, depth="no"),
     CLEAN_QUALITY, "场景错误", "auto", {"场景错误", "纵深不足"}),
    ("场景白名单-冰柜", vlm(scene="drink_freezer", scene_known=True, scene_error=False),
     CLEAN_QUALITY, "正常陈列", "auto", set()),
    ("拍屏幕", vlm(is_screen=True, scene="screen", scene_known=True, scene_error=False,
               depth="no", empty_bottle_count=4), CLEAN_QUALITY,
     "套拍-相似图", "auto", {"套拍-相似图"}),
    ("拼接仅记录不定罪", vlm(), {"composite": True, "composite_corr": 0.95, "moire": 1.5},
     "正常陈列", "auto", set()),
    ("正常陈列", vlm(), CLEAN_QUALITY, "正常陈列", "auto", set()),
    ("VLM空响应转人工", {"status": "empty_response"}, CLEAN_QUALITY,
     "正常陈列", "human", set()),
    ("VLM不可用+拼接不定罪仍转人工", {"status": "unavailable"},
     {"composite": True, "composite_corr": 0.95, "moire": 1.5},
     "正常陈列", "human", set()),
    ("空瓶计数自矛盾转人工", vlm(empty_bottle_count=5,
                          empty_obs="透明瓶均有液体，无空瓶"), CLEAN_QUALITY,
     "空瓶", "human", {"空瓶"}),
    ("合法多排空瓶转人工", vlm(empty_bottle_count=5), CLEAN_QUALITY,
     "空瓶", "human", {"空瓶"}, dict(DET_OK, rows=3)),
    ("摆拍单排空瓶auto", vlm(empty_bottle_count=5), CLEAN_QUALITY,
     "空瓶", "auto", {"空瓶"}),
    ("合法多排纵深转人工", vlm(depth="no"), CLEAN_QUALITY,
     "纵深不足", "human", {"纵深不足"}, dict(DET_OK, rows=3)),
]

# _normalize 确定性推导用例：(名称, 模型原始输出, 期望字段子集, 置信度上限可选)
NORM_CASES = [
    ("behind他牌推导depth=no", {"scene": "drink_shelf",
                       "behind": "前排元气瓶正后方是其他品牌产品。"},
     {"depth": "no"}, None),
    ("behind元气推导depth=yes", {"scene": "drink_shelf",
                        "behind": "后方仍是元气产品"},
     {"depth": "yes"}, None),
    ("behind遮挡优先unknown", {"scene": "drink_shelf",
                        "behind": "后方是其他品牌产品，但因遮挡看不清。"},
     {"depth": "unknown"}, None),
    ("behind空层板推导no", {"scene": "drink_shelf", "behind": "后方是空层板"},
     {"depth": "no"}, None),
    ("割箱纸箱不判depth", {"scene": "case_display",
                     "behind": "前排元气瓶正后方是其他品牌产品，包括包装袋和饮料箱。"},
     {"depth": "unknown"}, None),
    ("behind缺失推导unknown", {"scene": "drink_shelf"},
     {"depth": "unknown"}, None),
    ("around洗涤覆写白名单scene", {"scene": "drink_shelf", "behind": "genki",
                          "around": "周围是洗衣液和洗衣皂", "confidence": 0.9},
     {"scene": "other_shelf", "scene_error": True}, 0.55),
    ("around冰柜顶覆写freezer", {"scene": "drink_freezer", "behind": "后方是其他品牌",
                        "around": "冰柜顶上摆瓶，周围零食", "confidence": 0.9},
     {"scene": "table", "scene_error": True}, 0.55),
    ("零食邻接不覆写", {"scene": "case_display", "behind": "后方是元气纸箱",
                   "around": "货架上陈列饮料以及一些零食和包装袋", "confidence": 0.9},
     {"scene": "case_display", "scene_error": False}, None),
    ("拍屏幕不覆写scene", {"scene": "drink_shelf", "is_screen": True,
                      "around": "周围是洗衣液"},
     {"scene_error": False}, None),
    ("正常货架不覆写", {"scene": "drink_shelf", "behind": "genki",
                   "around": "周围是饮料货架，摆放阿萨姆、大红袍", "confidence": 0.9},
     {"scene": "drink_shelf", "scene_error": False, "depth": "yes"}, None),
]


# ---------------------------------------------------------------- 换模型契约用例

# 判重定罪范围：(名称, matches, 当前 store_id, 期望 ok, 期望主标签, 期望路由, 置信度上限)
REUSE_CASES = [
    ("同店命中不定罪", [{"similarity": 0.95, "store_id": "WD001"}], "WD001",
     False, "正常陈列", "auto", None),
    ("异店命中定罪", [{"similarity": 0.95, "store_id": "WD002"}], "WD001",
     True, "套拍-同店多门店", "auto", None),
    ("未传store_id转人工", [{"similarity": 0.95, "store_id": "WD002"}], None,
     True, "套拍-同店多门店", "human", 0.55),
    ("索引缺网点编码转人工", [{"similarity": 0.95, "store_id": ""}], "WD001",
     True, "套拍-同店多门店", "human", 0.55),
    ("未达阈值不定罪", [{"similarity": 0.86, "store_id": "WD002"}], "WD001",
     False, "正常陈列", "auto", None),
    ("无命中不定罪", [], "WD001", False, "正常陈列", "auto", None),
]

# 安全阀语义：(名称, target, det.status, rows, vlm 覆写字段, 期望路由)
VALVE_CASES = [
    ("bottle单排空瓶auto", "bottle", "ok", 1, dict(empty_bottle_count=5), "auto"),
    ("bottle多排空瓶human", "bottle", "ok", 3, dict(empty_bottle_count=5), "human"),
    ("genki单排空瓶仍human", "genki", "ok", 1, dict(empty_bottle_count=5), "human"),
    ("genki多排空瓶human", "genki", "ok", 3, dict(empty_bottle_count=5), "human"),
    ("genki无检测仍human", "genki", "skipped", 0, dict(empty_bottle_count=5), "human"),
    ("genki场景错误不封顶", "genki", "ok", 1,
     dict(empty_bottle_count=5, scene="other_shelf", scene_error=True), "auto"),
    ("bottle单排纵深auto", "bottle", "ok", 1, dict(depth="no"), "auto"),
    ("genki单排纵深仍human", "genki", "ok", 1, dict(depth="no"), "human"),
]

# 补偿层开关：(名称, compensations, 模型原始输出, 期望字段子集, 置信度下限)
COMP_CASES = [
    ("补偿关-邻接洗涤不再误覆写", [],
     {"scene": "drink_shelf", "around": "周围是洗衣液", "confidence": 0.9, "depth": "yes"},
     {"scene": "drink_shelf", "scene_error": False, "depth": "yes"}, 0.9),
    ("补偿关-采信模型depth", [], {"scene": "drink_shelf", "depth": "no"},
     {"depth": "no"}, None),
    ("补偿关-未报depth则unknown", [],
     {"scene": "drink_shelf", "behind": "后方是其他品牌产品"},
     {"depth": "unknown"}, None),
    ("补偿关-非法depth归unknown", [], {"scene": "drink_shelf", "depth": "maybe"},
     {"depth": "unknown"}, None),
    ("补偿关-白名单口径仍生效", [], {"scene": "table"},
     {"scene_error": True}, None),
    ("只留behind推导", ["behind_to_depth"],
     {"scene": "drink_shelf", "around": "周围是洗衣液",
      "behind": "后方是其他品牌产品", "confidence": 0.9},
     {"scene": "drink_shelf", "scene_error": False, "depth": "no"}, 0.9),
    ("只留around覆写", ["around_override"],
     {"scene": "drink_shelf", "around": "周围是洗衣液", "depth": "yes", "confidence": 0.9},
     {"scene": "other_shelf", "scene_error": True, "depth": "yes"}, None),
    ("未知补偿名忽略", ["no_such_patch"], {"scene": "drink_shelf", "depth": "no"},
     {"depth": "no", "scene_error": False}, None),
    ("全开-与改造前一致", None,
     {"scene": "drink_shelf", "around": "周围是洗衣液",
      "behind": "后方是其他品牌产品", "confidence": 0.9},
     {"scene": "other_shelf", "scene_error": True, "depth": "no"}, None),
]


def check_reuse() -> int:
    bad = 0
    for name, matches, sid, want_ok, want_main, want_route, conf_max in REUSE_CASES:
        hit = rules.rule_reuse(matches, 0.92, sid)
        h = hits_of(vlm())
        h["reuse_other_store"] = hit
        out = fusion.fuse(h, True, CFG)
        ok = (hit[0] == want_ok and out["label_main"] == want_main
              and out["route"] == want_route)
        if conf_max is not None:
            ok = ok and out["confidence"] <= conf_max
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22s} -> ok={hit[0]} "
              f"main={out['label_main']} conf={out['confidence']} route={out['route']}")
        if not ok:
            print(f"       期望: ok={want_ok} main={want_main} route={want_route} conf<={conf_max}")
    return bad


def check_valve() -> int:
    bad = 0
    for name, target, status, rows, kw, want_route in VALVE_CASES:
        det = dict(DET_OK, target=target, status=status, rows=rows)
        v = vlm(**kw)
        h = hits_of(v, det=det)
        out = fusion.fuse(h, True, CFG)
        ok = out["route"] == want_route
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22s} -> target={target} rows={rows} "
              f"conf={out['confidence']} route={out['route']}")
        if not ok:
            print(f"       期望: route={want_route}")
    return bad


def check_comp() -> int:
    bad = 0
    for name, comps, raw, want, conf_min in COMP_CASES:
        out = VLMClient._normalize(dict(raw), comps)
        ok = all(out.get(k) == v for k, v in want.items())
        if conf_min is not None:
            ok = ok and out.get("confidence", 0) >= conf_min
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} -> scene={out.get('scene')} "
              f"depth={out.get('depth')} scene_error={out.get('scene_error')} "
              f"conf={out.get('confidence')}")
        if not ok:
            print(f"       期望: {want} conf>={conf_min}")
    return bad


# ---------------------------------------------------------------- 传输协议契约（离线）

B64 = "QUJD"  # 占位 base64：只验信封形状，不验图像内容

# base_url 归一：(名称, cfg, 期望服务根)
URL_CASES = [
    ("带/v1剥掉", {"base_url": "http://127.0.0.1:8000/v1"}, "http://127.0.0.1:8000"),
    ("带尾斜杠", {"base_url": "http://127.0.0.1:11434/"}, "http://127.0.0.1:11434"),
    ("vLLM服务器带/v1/", {"base_url": "http://10.0.0.5:8000/v1/"}, "http://10.0.0.5:8000"),
    ("旧host键兼容", {"host": "http://127.0.0.1:11434"}, "http://127.0.0.1:11434"),
    ("缺省本机", {}, "http://127.0.0.1:11434"),
]

# endpoint 拼接：(名称, cfg, 期望 URL)
ENDPOINT_CASES = [
    ("默认走openai", {}, "http://127.0.0.1:11434/v1/chat/completions"),
    ("显式openai", {"api": "openai"}, "http://127.0.0.1:11434/v1/chat/completions"),
    ("ollama原生回退", {"api": "ollama"}, "http://127.0.0.1:11434/api/chat"),
    ("未知api回退openai", {"api": "grpc"}, "http://127.0.0.1:11434/v1/chat/completions"),
]

# 两种信封喂同一组语义值，_extract 必须归一到完全相同的结果。
# 这就是"换推理后端不改判定逻辑"的契约：下游只认这五个字段。
OPENAI_RESP = {
    "choices": [{"message": {"content": '{"scene":"drink_shelf"}',
                             "reasoning_content": "想了想"},
                 "finish_reason": "stop"}],
    "usage": {"completion_tokens": 194, "prompt_tokens": 1465},
}
OLLAMA_RESP = {
    "message": {"content": '{"scene":"drink_shelf"}', "thinking": "想了想"},
    "done_reason": "stop", "eval_count": 194, "prompt_eval_count": 1465,
}
WANT_EXTRACT = {"text": '{"scene":"drink_shelf"}', "thinking": "想了想",
                "finish_reason": "stop", "completion_tokens": 194, "prompt_tokens": 1465}


def check_transport():
    """返回 (失败数, 总数)。全程不发网络请求，内存紧张时也能跑。"""
    bad, names = 0, []

    def t(name, ok, detail=""):
        nonlocal bad
        names.append(name)
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")

    for name, cfg, want in URL_CASES:
        got = _root_url(cfg)
        t(name, got == want, f"-> {got}")

    for name, cfg, want in ENDPOINT_CASES:
        got = VLMClient(cfg).endpoint
        t(name, got == want, f"-> {got}")

    # ---- OpenAI 兼容请求信封必须是标准多模态形状
    p = VLMClient({"max_tokens": 1500})._payload(B64)
    msg = p["messages"][0]
    parts = msg.get("content")
    t("openai-content是parts数组", isinstance(parts, list) and len(parts) == 2,
      f"-> {type(parts).__name__}")
    t("openai-text part原样PROMPT",
      isinstance(parts, list) and parts[0].get("type") == "text" and parts[0].get("text") == PROMPT)
    iu = parts[1].get("image_url") if isinstance(parts, list) and len(parts) > 1 else None
    t("openai-image_url是对象且dataURL",
      parts[1].get("type") == "image_url" and isinstance(iu, dict)
      and str(iu.get("url", "")).startswith("data:image/jpeg;base64,")
      and str(iu.get("url", "")).endswith(B64), f"-> {str(iu)[:48]}")
    t("openai-JSON mode", p.get("response_format") == {"type": "json_object"})
    t("openai-max_tokens下发", p.get("max_tokens") == 1500 and p.get("stream") is False)
    t("openai-不夹带原生键",
      "options" not in p and "format" not in p and "images" not in msg)

    # ---- 原生回退信封保持旧形状，且 num_ctx 只在这里生效
    o = VLMClient({"api": "ollama", "num_predict": 1500, "num_ctx": 3072})._payload(B64)
    om = o["messages"][0]
    t("ollama-content是字符串+images",
      om.get("content") == PROMPT and om.get("images") == [B64])
    t("ollama-format=json", o.get("format") == "json")
    t("ollama-num_predict兼容旧键", o["options"].get("num_predict") == 1500)
    t("ollama-num_ctx下发", o["options"].get("num_ctx") == 3072)
    t("ollama-未配num_ctx则不传",
      "num_ctx" not in VLMClient({"api": "ollama"})._payload(B64)["options"])
    t("openai-num_ctx不下发",
      "num_ctx" not in VLMClient({"num_ctx": 3072})._payload(B64)
      and "num_ctx" not in str(VLMClient({"num_ctx": 3072})._payload(B64)))

    # ---- 响应归一：两种信封 -> 同一组字段
    go = VLMClient({})._extract(OPENAI_RESP)
    gn = VLMClient({"api": "ollama"})._extract(OLLAMA_RESP)
    t("openai响应归一", go == WANT_EXTRACT, f"-> {go}")
    t("ollama响应归一", gn == WANT_EXTRACT, f"-> {gn}")
    t("两传输归一结果相等", go == gn)

    # ---- 降级指纹在两种传输下都能取到（空 content 绝不能当成 ok）
    empty_o = VLMClient({})._extract({"choices": [{"message": {"content": ""},
                                                   "finish_reason": "length"}],
                                      "usage": {"completion_tokens": 1500}})
    empty_n = VLMClient({"api": "ollama"})._extract(
        {"message": {"content": "", "thinking": "x" * 100},
         "done_reason": "length", "eval_count": 1500})
    t("openai空响应指纹", empty_o["text"] == "" and empty_o["finish_reason"] == "length"
      and empty_o["completion_tokens"] == 1500)
    t("ollama空响应指纹", empty_n["text"] == "" and empty_n["finish_reason"] == "length"
      and empty_n["completion_tokens"] == 1500 and len(empty_n["thinking"]) == 100)
    t("缺choices不崩且归空", VLMClient({})._extract({})["text"] == "")
    t("缺message不崩且归空", VLMClient({"api": "ollama"})._extract({})["text"] == "")
    t("Ollama兼容层reasoning别名",
      VLMClient({})._extract({"choices": [{"message": {"content": "{}", "reasoning": "r"}}]})
      ["thinking"] == "r")
    t("content=null归空串",
      VLMClient({})._extract({"choices": [{"message": {"content": None}}]})["text"] == "")

    # ---- 换后端只改 base_url：同一份 cfg 指向 vLLM 与指向 Ollama 的信封必须一致
    a = VLMClient({"base_url": "http://127.0.0.1:11434/v1", "model": "m"})._payload(B64)
    b = VLMClient({"base_url": "http://gpu-server:8000/v1", "model": "m"})._payload(B64)
    t("换服务器信封不变", a == b)

    return bad, len(names)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    bad = 0
    for case in CASES:
        name, v, q, want_main, want_route, want_labels = case[:6]
        det = case[6] if len(case) > 6 else DET_OK
        h = hits_of(v, q, det)
        out = fusion.fuse(h, str(v.get("status")) == "ok", CFG)
        got_labels = set(out["label_all"])
        ok = (out["label_main"] == want_main and out["route"] == want_route
              and got_labels == want_labels)
        # 拍屏幕时陈列类规则必须不适用
        if name == "拍屏幕":
            ok = ok and not h["empty_bottle"][0] and not h["depth"][0] and not h["scene_error"][0]
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:18s} -> {out['label_main']:<8} "
              f"conf={out['confidence']:<5} route={out['route']:<5} labels={sorted(got_labels)}")
        if not ok:
            print(f"       期望: main={want_main} route={want_route} labels={sorted(want_labels)}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} 通过")
    for name, raw, want, conf_max in NORM_CASES:
        out = VLMClient._normalize(dict(raw))
        ok = all(out.get(k) == v for k, v in want.items())
        if conf_max is not None:
            ok = ok and out.get("confidence", 1) <= conf_max
        bad += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:22s} -> "
              f"scene={out.get('scene')} depth={out.get('depth')} "
              f"scene_error={out.get('scene_error')} conf={out.get('confidence')}")
        if not ok:
            print(f"       期望: {want} conf<={conf_max}")
    total = len(CASES) + len(NORM_CASES)
    print(f"\n{total - bad}/{total} 通过（五类口径 + 归一推导）")

    print("\n---- 换模型契约：判重定罪范围 ----")
    bad += check_reuse()
    print("\n---- 换模型契约：安全阀随 detector.target ----")
    bad += check_valve()
    print("\n---- 换模型契约：VLM 缺陷补偿层开关 ----")
    bad += check_comp()
    print("\n---- 换后端契约：OpenAI 兼容 / 原生 Ollama 传输 ----")
    t_bad, t_total = check_transport()
    bad += t_bad

    total += len(REUSE_CASES) + len(VALVE_CASES) + len(COMP_CASES) + t_total
    print(f"\n{total - bad}/{total} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
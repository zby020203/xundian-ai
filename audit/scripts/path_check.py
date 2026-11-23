"""判定路径回归：不依赖 Ollama/YOLO，直接验证 rules+fusion 的五类口径与降级路由。

    python scripts/path_check.py

覆盖：空瓶(多支/孤立)、纵深不足、场景错误、拍屏幕、拼接、正常、
VLM 空响应/不可用转人工、纵深 unknown 不判、拍屏幕时陈列判定不适用。

另锁三组"换模型不漂口径"契约（见文件下半）：
    REUSE_CASES  判重定罪范围（同店/异店/归属不明）
    VALVE_CASES  安全阀随 detector.target 切换的语义
    COMP_CASES   VLM 缺陷补偿层开/关/部分开
这三组测的不是"当前模型判得准不准"，而是"换权重/换 VLM/换判重方案时口径不会静默漂移"。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit import fusion, rules  # noqa: E402
from audit.vlm import OllamaVLM  # noqa: E402

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
    ("拼接双图", vlm(), {"composite": True, "composite_corr": 0.95, "moire": 1.5},
     "套拍-相似图", "auto", {"套拍-相似图"}),
    ("正常陈列", vlm(), CLEAN_QUALITY, "正常陈列", "auto", set()),
    ("VLM空响应转人工", {"status": "empty_response"}, CLEAN_QUALITY,
     "正常陈列", "human", set()),
    ("VLM不可用+拼接仍转人工", {"status": "unavailable"},
     {"composite": True, "composite_corr": 0.95, "moire": 1.5},
     "套拍-相似图", "human", {"套拍-相似图"}),
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
        out = OllamaVLM._normalize(dict(raw), comps)
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
        out = OllamaVLM._normalize(dict(raw))
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

    total += len(REUSE_CASES) + len(VALVE_CASES) + len(COMP_CASES)
    print(f"\n{total - bad}/{total} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
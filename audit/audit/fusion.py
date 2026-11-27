"""证据融合：把各阶段命中合成最终结论。

规则优先级（可在 config.fusion.label_priority 覆盖）：
    套拍-相似图 -> 套拍-同店多门店 -> 空瓶 -> 场景错误 -> 纵深不足 -> 正常
理由：套拍属提交层作弊、证据最强排最前；空瓶出现即命中且直接凑数排第二；
场景错误（白名单违例）次之；纵深不足最轻排最后。与 labels.csv 人工习惯一致
（空瓶+场景错误共现时主标签取空瓶）。
single_row 只作摆拍形态证据，不定罪（新口径空瓶看有无液体）。
composite（拼接双图）同款仅证据不定罪：导师五类口径无拼接，
检测值只入 evidence 供人工识别重复录入/拼图提交，不产生标签。
"""
from __future__ import annotations

from typing import Dict, List

RULE_TO_LABEL = {
    "screen": "套拍-相似图",
    "reuse_other_store": "套拍-同店多门店",
    "empty_bottle": "空瓶",
    "scene_error": "场景错误",
    "depth": "纵深不足",
}


def fuse(hits: Dict[str, tuple], vlm_ok: bool, cfg: Dict[str, object]) -> Dict[str, object]:
    """hits: {rule_name: (bool, confidence, evidence)}"""
    priority: List[str] = cfg.get("label_priority", [
        "套拍-相似图", "套拍-同店多门店", "空瓶", "场景错误", "纵深不足", "正常陈列",
    ])
    human_thr = float(cfg.get("human_review_confidence", 0.6))

    found: Dict[str, float] = {}
    evidence: List[str] = []
    for rule, (ok, conf, ev) in hits.items():
        if ev:
            evidence.append(f"[{rule}] {ev}")
        if ok:
            label = RULE_TO_LABEL.get(rule)
            if label is None:
                continue
            found[label] = max(found.get(label, 0.0), float(conf))

    # 定罪范围已由规则层表达完毕，这里不再按规则名做二次剔除。
    # 旧版曾在此处用 hits["reuse_other_store"][0] 为 False 去 pop 同店标签，
    # 但 rule_reuse 同店命中时返回的是 True，该分支永远不触发（死代码），
    # 导致同店重复上传被以高置信自动定罪。现同店/归属不明均由 rule_reuse 直接处理。
    # 未命中规则的证据文本仍会在上方循环里入 evidence，所以"仅记录"不丢。

    labels = [k for k in priority if k in found]
    if not labels:
        return {
            "label_main": "正常陈列",
            "label_all": [],
            "confidence": 0.5 if vlm_ok else 0.3,
            "route": "human" if not vlm_ok else "auto",
            "evidence": evidence,
            "reason": "各阶段均未命中异常规则" if vlm_ok else "VLM 不可用，仅完成规则初筛",
        }

    main = labels[0]
    conf = found[main]
    # 多标签共现是摆拍常态（如空瓶+场景错误），不再降权：
    # 旧版 -0.05*(n-1) 会把高置信共现压到人工阈值以下，造成不必要转人工。

    # 任一命中标签置信度不足即转人工（覆盖“孤立 1~2 支空瓶低置信”的共识口径）。
    route = "human" if (not vlm_ok or min(found.values()) < human_thr) else "auto"
    reason = "；".join(
        ev.split("] ", 1)[-1] for r, (ok, _, ev) in hits.items() if ok
    ) or "命中规则"

    return {
        "label_main": main,
        "label_all": labels,
        "confidence": round(float(conf), 2),
        "route": route,
        "evidence": evidence,
        "reason": reason,
    }

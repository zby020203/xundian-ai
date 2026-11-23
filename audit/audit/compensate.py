"""VLM 缺陷补偿层：针对当前 qwen3-vl:2b-instruct 已知毛病打的补丁，可整层关闭。

为什么单独成一层（换 VLM 前必读）：
    vlm._normalize 里原本混着两类性质相反的逻辑——
      (a) 协议归一：字段补齐、枚举别名映射、类型转换、越界钳位。换任何模型都要留。
      (b) 缺陷补偿：2B 的结论字段被"整齐=合格"锚定、与自身观察系统性矛盾，
          于是绕开结论字段、改用 around/behind 自由文本反推业务结论。
          这是为 2B 打的补丁，**不是业务口径**。
    微调模型上线后 (b) 必须整层关掉，否则会出现静默劣化：模型已正确报 drink_shelf，
    around 如实提到邻接的洗涤区，覆写补丁反而把 scene 改成 other_shelf、置信度压到 0.55，
    把修好的模型拖回复核——而 degraded 数组不会有任何警报，只有指标下降。

开关：config.vlm.compensations（字符串数组）
    ["around_override", "behind_to_depth"]  当前 2B 的完整补偿（默认，等价于改造前行为）
    []                                      微调模型模式，直接采信模型结论字段
    只留其中一个也支持；未知名字告警后忽略，绝不静默当成开启。

关掉 behind_to_depth 的前置条件（很容易漏）：
    当前 PROMPT 只让模型输出 behind 自由文本，**没有 depth 字段**。补偿层一关，
    depth 就只能走协议归一的"模型自报枚举"分支，而模型没报 -> 恒为 unknown ->
    纵深规则不判（宁可不判不误判，不会误定罪，但召回会掉）。
    所以换微调模型时必须同步改 PROMPT，让它直接输出 depth=yes/no/unknown 枚举。

不在这一层的东西：场景白名单推导（scene_error）是业务共识，永远生效，留在 vlm._normalize。
"""
from __future__ import annotations

from typing import Dict, List

from .logutil import get_logger

log = get_logger(__name__)

ALL = ("around_override", "behind_to_depth")

# around 出现这些强非饮料信号，说明模型 scene 结论被"整齐"锚定，按观察覆写。
# 顺序即优先级。"零食"故意不在内：零食常邻接饮料陈列，实测正常割箱会被误伤。
_AROUND_SIGNALS = (
    (("冰柜顶", "盖上", "餐桌", "桌面"), "table"),
    (("收银",), "counter"),
    (("洗衣", "洗涤", "文具", "烟酒", "母婴", "餐具"), "other_shelf"),
)


def around_override(out: Dict[str, object], whitelist) -> Dict[str, object]:
    """around 交叉校验：观察出现强非饮料信号但 scene 报白名单时，按观察覆写并封顶置信度。"""
    scene = out.get("scene")
    around = str(out.get("around") or "")
    if (out.get("is_screen") or not out.get("scene_known")
            or scene not in whitelist or not around):
        return out
    override = None
    for keys, target in _AROUND_SIGNALS:
        if any(k in around for k in keys):
            override = target
            break
    if not override:
        return out
    log.info("[compensate] around 出现非饮料信号，scene 由白名单覆写：%s -> %s｜around=%r",
             scene, override, around[:60])
    out["scene_override_from"] = scene
    out["scene"] = override
    out["scene_error"] = True
    try:
        cur = float(out.get("confidence", 0.6))
    except Exception:
        cur = 0.6
    out["confidence"] = min(cur, 0.55)
    log.info("[compensate] 置信度封顶：%s -> %s（低于 0.6 即转人工）", cur, out["confidence"])
    return out


def behind_to_depth(out: Dict[str, object], whitelist=None) -> Dict[str, object]:
    """behind 自由文本观察 -> depth 确定性推导（共识：后方有元气即合格）。

    不信任模型自报 depth：实测 2B 结论被"整齐=合格"锚定，自由文本观察反而正确。
    截取"后方/背后/后面"之后的片段，避免"前排元气瓶"字样干扰；
    遮挡优先判 unknown（宁可不判，不误判）；纸箱不算"无产品"（割箱陈列后方是元气自己的纸箱）。
    """
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
        # 割箱陈列后方通常是元气自己的纸箱，品牌不可读 -> 宁可不判。
        depth = "unknown"
    elif any(k in seg for k in ("其他品牌", "他牌", "空层板", "无产品")):
        depth = "no"
    else:
        depth = "unknown"
    out["depth"] = depth
    log.info("[compensate] behind=%r -> 截取片段=%r -> depth=%s", behind[:70], seg[:40], depth)
    return out


_REGISTRY = {
    "around_override": around_override,
    "behind_to_depth": behind_to_depth,
}


def resolve(enabled) -> List[str]:
    """把配置里的开关解析成有序名字列表。None=全开（兼容直接调 _normalize 的老用法）。"""
    if enabled is None:
        return list(ALL)
    if isinstance(enabled, str):
        enabled = [enabled]
    names: List[str] = []
    for item in enabled or []:
        n = str(item).strip()
        if not n:
            continue
        if n not in _REGISTRY:
            log.warning("[compensate] 未知补偿名 %r 已忽略（可用：%s）", n, ", ".join(ALL))
        elif n not in names:
            names.append(n)
    return names


def apply(out: Dict[str, object], names: List[str], whitelist) -> Dict[str, object]:
    """按 names 顺序施加补偿。names 为空即整层关闭，out 原样返回。

    顺序有意义：around_override 可能改写 scene，behind_to_depth 的割箱特判要读改写后的值，
    与改造前 _normalize 内的先后一致，不可调换。
    """
    if not names:
        log.info("[compensate] 缺陷补偿层：关（微调模型模式，直接采信模型结论字段）")
        return out
    log.info("[compensate] 缺陷补偿层：开（%s）", ", ".join(names))
    for n in names:
        out = _REGISTRY[n](out, whitelist)
    return out

"""规则层：把检测框 / 图像统计量 / VLM 字段翻译成可解释的证据。

每条规则返回 (命中与否, 置信度 0-1, 一句话证据)。

五类问题判定口径（与业务共识一致）：
- 空瓶：无液瓶冒充陈列凑数，出现即命中、不设数量门槛；
  孤立 1~2 支给低置信（低于人工阈值自动转人工），>=3 支给高置信。
- 纵深不足：前排元气瓶后方无产品或只有他牌 -> 命中；后方有元气即合格（不限排数）。
- 场景错误：白名单制，合法位仅 饮料货架/饮料冰柜/割箱堆箱/端架（白名单推导在 vlm._normalize）。
- 套拍-相似图：拍屏幕/翻拍（VLM is_screen）与拼接双图（左右半幅相关）。
- 套拍-同店多门店：跨记录检索，本阶段保留骨架不增强。

阈值校准记录（10 张真实样本，scripts/_calibrate.py）：
- composite_corr：正常图最高 0.627，真拼接图 0.952/0.959 -> 阈值 0.85（旧 0.6 会误报正常割箱）。
- moire：128px 下拍屏幕 1.40~1.55 与正常图 1.41~1.49 完全重叠，无区分度，
  只记录在 quality 阶段作证据，不参与定罪（旧 moire_threshold=0.35 为恒命中隐患，已移除）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import re

RuleHit = Tuple[bool, float, str]

# 空瓶阈值：>=3 支才给可自动通过的置信度；1~2 支命中但置信度低于人工阈值 0.6。
EMPTY_AUTO_COUNT = 3
EMPTY_LOW_CONF = 0.5
EMPTY_HIGH_BASE = 0.65
EMPTY_HIGH_STEP = 0.05
EMPTY_HIGH_CAP = 0.90

# 拼接双图阈值（校准见模块 docstring）。
COMPOSITE_CORR_THRESHOLD = 0.85


def _spacing_cv(bottles: List[dict]) -> float:
    if len(bottles) < 3:
        return 9.9
    xs = sorted(b["cx"] for b in bottles)
    gaps = np.diff(xs)
    gaps = gaps[gaps > 1]
    if gaps.size < 2:
        return 9.9
    return float(np.std(gaps) / (np.mean(gaps) + 1e-6))


def _obs(vlm: Dict[str, object], key: str) -> str:
    """取 VLM 观察先行字段（around/behind/empty_obs）作证据附注。"""
    s = str(vlm.get(key) or "").strip()
    return f"（观察：{s}）" if s else ""


def rule_empty_bottle(vlm: Dict[str, object], det: Optional[Dict[str, object]] = None) -> RuleHit:
    """空瓶：陈列面出现无液瓶即命中；孤立少量给低置信转人工。

    路由安全阀：合法场景 + CV 多排陈列时，空瓶主张缺乏摆拍上下文佐证
    （2B 液面线判断在正常货架上不可靠，五轮提示词迭代实测），置信度封顶转人工；
    标签保留、召回不降，但正常图绝不自动定罪。
    """
    if vlm.get("status") != "ok":
        return False, 0.0, "VLM 不可用"
    if vlm.get("is_screen"):
        return False, 0.0, "拍屏幕照片，空瓶判定不适用"
    n = int(vlm.get("empty_bottle_count", -1))
    if n < 0:
        return False, 0.0, "空瓶数量不可判（unknown），不判"
    if n == 0:
        return False, 0.0, "陈列面未见空瓶"
    # 自矛盾护栏：计数>0 但观察/结论明确否定空瓶（2B 实测会鹦鹉学舌+幻觉计数）→ 低置信转人工。
    obs = str(vlm.get("empty_obs") or "") + str(vlm.get("reason") or "")
    if re.search(r"无(液体|液)?空瓶|没有空瓶|未见空瓶", obs):
        return True, EMPTY_LOW_CONF, f"计数={n} 与观察/结论‘无空瓶’自相矛盾，低置信转人工复核"
    if n < EMPTY_AUTO_COUNT:
        return True, EMPTY_LOW_CONF, f"见 {n} 支空瓶（孤立少量，低置信转人工复核）{_obs(vlm, 'empty_obs')}"
    conf = min(EMPTY_HIGH_CAP, EMPTY_HIGH_BASE + EMPTY_HIGH_STEP * n)
    ev = f"陈列面见 {n} 支无液空瓶，冒充陈列凑数{_obs(vlm, 'empty_obs')}"
    if (det or {}).get("status") == "ok" and int((det or {}).get("rows", 0)) > 1 \
            and vlm.get("scene_known") and not vlm.get("scene_error"):
        conf = min(conf, 0.55)
        ev += "（合法场景多排陈列，空瓶主张无摆拍上下文佐证，低置信转人工）"
    return True, round(conf, 2), ev


def rule_single_row(det: Dict[str, object]) -> RuleHit:
    """摆拍形态线索：单排、等距、正对镜头。只作证据，不直接定罪。

    新口径下空瓶看有无液体、纵深看后方有无元气，单排本身不是五类问题之一。
    """
    if det.get("status") != "ok":
        return False, 0.0, "检测不可用"
    n = int(det.get("n_bottles", 0))
    rows = int(det.get("rows", 0))
    if n < 4:
        return False, 0.0, f"仅检出 {n} 个瓶体，不足以判断排面"
    if rows != 1:
        return False, 0.0, f"检出 {rows} 排，非单排"
    cv = _spacing_cv(det.get("bottles", []))
    if cv > 0.35:
        return False, 0.2, f"单排但间距不均（CV={cv:.2f}）"
    conf = float(np.clip(1.0 - cv / 0.35, 0.5, 0.95))
    return True, round(conf, 2), f"单排 {n} 支、间距均匀（CV={cv:.2f}），摆拍形态线索"


def rule_composite(quality: Dict[str, object], threshold: float = COMPOSITE_CORR_THRESHOLD) -> RuleHit:
    corr = float(quality.get("composite_corr", 0) or 0)
    if quality.get("composite"):
        return True, round(min(0.99, 0.6 + corr * 0.4), 2), f"左右半幅相关 {corr}，判定为拼接双图"
    return False, 0.0, f"左右半幅相关 {corr}，未达拼接阈值 {threshold}"


def rule_screen(vlm: Dict[str, object]) -> RuleHit:
    if vlm.get("status") != "ok":
        return False, 0.0, "VLM 不可用"
    if vlm.get("is_screen") is True:
        return True, float(vlm.get("confidence", 0.7) or 0.7), "VLM 判定为拍屏幕/翻拍"
    return False, 0.0, "VLM 未判定为屏幕"


def rule_scene_error(vlm: Dict[str, object]) -> RuleHit:
    """场景错误：白名单推导（vlm._normalize），这里只翻译证据。"""
    if vlm.get("status") != "ok":
        return False, 0.0, "VLM 不可用"
    if vlm.get("is_screen"):
        return False, 0.0, "拍屏幕照片，场景判定不适用"
    if not vlm.get("scene_known", False):
        return False, 0.0, "场景不可判（unknown），不判"
    scene = vlm.get("scene", "?")
    if vlm.get("scene_error") is True:
        return True, float(vlm.get("confidence", 0.7) or 0.7), \
            f"场景={scene}，不在陈列白名单（饮料货架/饮料冰柜/割箱堆箱/端架）{_obs(vlm, 'around')}"
    return False, 0.0, f"场景={scene}，属陈列白名单"


def rule_depth(vlm: Dict[str, object], det: Optional[Dict[str, object]] = None) -> RuleHit:
    """纵深不足：后方无产品或只有他牌 -> 命中；后方有元气即合格；unknown 不判。

    路由安全阀同空瓶：合法场景 + CV 多排陈列时 behind 观察实测漂移
    （同一图不同轮报 genki/他牌），置信度封顶转人工。
    """
    if vlm.get("status") != "ok":
        return False, 0.0, "VLM 不可用"
    if vlm.get("is_screen"):
        return False, 0.0, "拍屏幕照片，纵深判定不适用"
    depth = str(vlm.get("depth", "unknown")).lower()
    if depth == "no":
        conf = float(vlm.get("confidence", 0.6) or 0.6) * 0.9
        ev = f"前排元气瓶后方无产品或只有其他品牌，纵深不足{_obs(vlm, 'behind')}"
        if (det or {}).get("status") == "ok" and int((det or {}).get("rows", 0)) > 1 \
                and vlm.get("scene_known") and not vlm.get("scene_error"):
            conf = min(conf, 0.55)
            ev += "（合法场景多排陈列，behind 观察无佐证，低置信转人工）"
        return True, round(conf, 2), ev
    if depth == "unknown":
        return False, 0.0, "后排被遮挡，不可判（按规则丢弃）"
    return False, 0.0, "后排有元气产品，纵深合格"


def rule_reuse(matches: List[dict], threshold: float, store_id: Optional[str]) -> RuleHit:
    if not matches:
        return False, 0.0, "未建索引或无命中"
    top = matches[0]
    sim = float(top.get("similarity", 0))
    if sim < threshold:
        return False, 0.0, f"最高相似度 {sim}，未达阈值 {threshold}"
    same_store = store_id is not None and str(top.get("store_id")) == str(store_id)
    if same_store:
        return True, round(sim, 2), f"与同店历史图相似度 {sim}（同店比对本属正常，仅记录）"
    return True, round(min(0.99, sim), 2), f"与另一门店（{top.get('store_id')}）历史图相似度 {sim}"
"""YOLO 检测层。

设计原则：拿不到权重 / 没有 ultralytics / 没有网络时，不抛错，
返回 status="unavailable" 并在证据链里写明原因，让上游决定降级。
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .logutil import get_logger

log = get_logger(__name__)

# 检测目标语义声明。换微调权重时除了改 weights/bottle_class_names，还必须改这里，
# 否则 rules 的安全阀会继续用旧语义解释 rows（详见 rules._weak_display_context）。
#   bottle = COCO 通用瓶，rows 是"任意瓶子几排"
#   genki  = 微调元气专用，rows 是"元气产品几排"
TARGETS = ("bottle", "genki")


class Detector:
    def __init__(self, cfg: dict, model_root: str):
        self.cfg = cfg
        self.model_root = model_root
        self.model = None
        self.status = "not_loaded"
        self.error = ""
        self.names: Dict[int, str] = {}
        want = str(cfg.get("target", "bottle")).strip().lower()
        if want not in TARGETS:
            log.warning("[detect] 未知 target=%r，按 bottle 处理（可选：%s）", want, "/".join(TARGETS))
            want = "bottle"
        self.target = want

    # ------------------------------------------------------------ 加载

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            from ultralytics import YOLO  # 延迟导入，缺包不影响其他阶段
        except Exception as e:  # pragma: no cover
            self.status, self.error = "unavailable", f"ultralytics 未安装: {e}"
            return False

        weight = self.cfg.get("weights", "yolo26n.pt")
        cand = weight if os.path.isabs(weight) else os.path.join(self.model_root, weight)
        if not os.path.exists(cand):
            cand = weight  # 交给 ultralytics 自动下载
        log.info("[detect] 加载权重：%s（本地未命中则交给 ultralytics 下载）", cand)
        try:
            self.model = YOLO(cand)
            self.names = getattr(self.model, "names", {}) or {}
            self.status = "ok"
            log.info("[detect] 权重就绪，模型含 %d 个类别，target=%s", len(self.names), self.target)
            self._check_target_consistency()
            return True
        except Exception as e:
            self.status, self.error = "unavailable", f"权重加载失败: {e}"
            log.warning("[detect] 权重加载失败，排面结构相关规则与安全阀均不生效：%s", e)
            return False

    def _check_target_consistency(self) -> None:
        """换权重时的两道前置体检，把静默失效提前到加载期暴露。"""
        want = set(self.cfg.get("bottle_class_names", ["bottle"]))
        have = {str(v) for v in self.names.values()}
        if not (have & want):
            log.warning("[detect] 权重类别 %s 与 bottle_class_names %s 无交集，检出必为 0",
                        sorted(have)[:10], sorted(want))
        if self.target == "genki" and want == {"bottle"}:
            log.warning("[detect] target=genki 但 bottle_class_names 仍是通用 ['bottle']，"
                        "疑似只改了语义声明没换权重：安全阀已按元气语义收紧，检测结果却仍是任意瓶")

    # ------------------------------------------------------------ 推理

    def predict(self, image_path: str) -> Dict[str, object]:
        if not self.load():
            return {"status": self.status, "error": self.error, "target": self.target,
                    "bottles": [], "n_bottles": 0, "rows": 0}

        try:
            res = self.model.predict(
                source=image_path,
                imgsz=int(self.cfg.get("imgsz", 1280)),
                conf=float(self.cfg.get("conf", 0.25)),
                device=self.cfg.get("device", "cpu"),
                verbose=False,
            )
        except Exception as e:
            return {"status": "error", "error": str(e), "target": self.target,
                    "bottles": [], "n_bottles": 0, "rows": 0}

        bottles: List[dict] = []
        want_names = set(self.cfg.get("bottle_class_names", ["bottle"]))
        n_raw = 0
        seen_names: Dict[str, int] = {}
        for r in res:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                n_raw += 1
                cls_id = int(b.cls.item())
                name = str(self.names.get(cls_id, cls_id))
                seen_names[name] = seen_names.get(name, 0) + 1
                if name not in want_names:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                bottles.append({
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "cx": round((x1 + x2) / 2, 1),
                    "cy": round((y1 + y2) / 2, 1),
                    "w": round(x2 - x1, 1),
                    "h": round(y2 - y1, 1),
                    "conf": round(float(b.conf.item()), 3),
                })

        top_cls = sorted(seen_names.items(), key=lambda kv: -kv[1])[:8]
        log.debug("[detect] 原始框 %d 个，类别分布：%s", n_raw, dict(top_cls))
        if n_raw and not bottles:
            # 静默失效指纹：检出非零但按类名过滤后为空，几乎总是权重类名与
            # bottle_class_names 不一致（换微调权重时必验此处）。
            log.warning("[detect] 检出 %d 框但按 %s 过滤后为 0，实际类名=%s，"
                        "需同步 config.detector.bottle_class_names",
                        n_raw, sorted(want_names), sorted(seen_names)[:10])

        bottles.sort(key=lambda d: (d["cy"], d["cx"]))
        return {
            "status": "ok",
            "target": self.target,
            "bottles": bottles,
            "n_bottles": len(bottles),
            "rows": self._count_rows(bottles),
        }

    # ------------------------------------------------------------ 排面结构

    @staticmethod
    def _count_rows(bottles: List[dict], tol_ratio: float = 0.35) -> int:
        """按 y 中心聚类估行数。tol 取瓶高的 35%。"""
        if not bottles:
            return 0
        heights = [b["h"] for b in bottles if b["h"] > 0]
        tol = (float(np.median(heights)) if heights else 40.0) * tol_ratio
        rows: List[float] = []
        for b in bottles:
            for i, cy in enumerate(rows):
                if abs(b["cy"] - cy) <= tol:
                    rows[i] = (cy * 0.7 + b["cy"] * 0.3)
                    break
            else:
                rows.append(b["cy"])
        return len(rows)

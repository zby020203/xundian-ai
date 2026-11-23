"""YOLO 检测层。

设计原则：拿不到权重 / 没有 ultralytics / 没有网络时，不抛错，
返回 status="unavailable" 并在证据链里写明原因，让上游决定降级。
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np


class Detector:
    def __init__(self, cfg: dict, model_root: str):
        self.cfg = cfg
        self.model_root = model_root
        self.model = None
        self.status = "not_loaded"
        self.error = ""
        self.names: Dict[int, str] = {}

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
        try:
            self.model = YOLO(cand)
            self.names = getattr(self.model, "names", {}) or {}
            self.status = "ok"
            return True
        except Exception as e:
            self.status, self.error = "unavailable", f"权重加载失败: {e}"
            return False

    # ------------------------------------------------------------ 推理

    def predict(self, image_path: str) -> Dict[str, object]:
        if not self.load():
            return {"status": self.status, "error": self.error, "bottles": [], "n_bottles": 0, "rows": 0}

        try:
            res = self.model.predict(
                source=image_path,
                imgsz=int(self.cfg.get("imgsz", 1280)),
                conf=float(self.cfg.get("conf", 0.25)),
                device=self.cfg.get("device", "cpu"),
                verbose=False,
            )
        except Exception as e:
            return {"status": "error", "error": str(e), "bottles": [], "n_bottles": 0, "rows": 0}

        bottles: List[dict] = []
        target = set(self.cfg.get("bottle_class_names", ["bottle"]))
        for r in res:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls.item())
                name = str(self.names.get(cls_id, cls_id))
                if name not in target:
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

        bottles.sort(key=lambda d: (d["cy"], d["cx"]))
        return {
            "status": "ok",
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

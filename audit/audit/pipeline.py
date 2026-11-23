"""端到端流水线（单图）。

    image
      -> 质量 / 拼接 / 摩尔纹（规则，永远可用）
      -> YOLO 检测瓶体（可选，缺权重则降级）
      -> 1:N 检索（可选，无索引则跳过）
      -> VLM 语义判定（可选，Ollama 不通则降级）
      -> 证据融合 -> 结论 + 是否转人工
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

from . import fusion, imageops, rules
from .config import Config
from .detect import Detector
from .embed import Embedder
from .retrieve import Index
from .vlm import OllamaVLM


class Pipeline:
    def __init__(self, cfg: Optional[Config] = None, use_vlm: bool = True, use_detector: bool = True):
        self.cfg = cfg or Config.load()
        self.use_vlm = use_vlm
        self.use_detector = use_detector
        self.detector = Detector(self.cfg.get("detector", {}), self.cfg.abspath("paths.model_root"))
        self.vlm = OllamaVLM(self.cfg.get("vlm", {}))
        self.embedder = Embedder(self.cfg.get("embedder", {}))
        self.index = Index.load(self.cfg.abspath("paths.index_path"))

    # ------------------------------------------------------------ 路径

    def resolve(self, image: str) -> str:
        if os.path.isabs(image) and os.path.exists(image):
            return image
        root = self.cfg.abspath("paths.dataset_root")
        cand = os.path.join(root, image)
        return cand if os.path.exists(cand) else image

    # ------------------------------------------------------------ 主流程

    def run(self, image: str, store_id: Optional[str] = None, visit_id: Optional[str] = None) -> Dict[str, object]:
        t0 = time.time()
        path = self.resolve(image)
        timings: Dict[str, float] = {}
        stages: Dict[str, object] = {}

        if not os.path.exists(path):
            return {"status": "error", "error": f"图片不存在: {path}", "image": image}

        im = imageops.load_image(path)
        stages["quality"] = imageops.summarize(im, self.cfg.get("quality", {}))
        timings["quality"] = round(time.time() - t0, 3)

        det = {"status": "skipped", "bottles": [], "n_bottles": 0, "rows": 0}
        if self.use_detector:
            t = time.time()
            det = self.detector.predict(path)
            timings["detect"] = round(time.time() - t, 3)
        stages["detect"] = det

        t = time.time()
        matches, emb_status = [], "skipped"
        if self.index.size > 0:
            try:
                v = self.embedder.encode(im)
                matches = self.index.search(v, int(self.cfg.get("retrieval.top_k", 5)))
                # 排除自匹配：索引含查询图自身时 sim=1.0 会让跨图骨架规则恒命中自己。
                rel = os.path.relpath(path, self.cfg.abspath("paths.dataset_root")).replace(os.sep, "/")
                matches = [m for m in matches
                           if str(m.get("image_path", "")).replace("\\", "/") != rel]
                emb_status = f"ok({self.embedder.backend})"
            except Exception as e:
                emb_status = f"error:{e}"
        timings["retrieval"] = round(time.time() - t, 3)
        stages["retrieval"] = {"status": emb_status, "index_size": self.index.size, "matches": matches[:3]}

        vlm_out: Dict[str, object] = {"status": "skipped"}
        if self.use_vlm:
            t = time.time()
            if self.vlm.available():
                vlm_out = self.vlm.judge(im)
            else:
                vlm_out = {"status": "unavailable", "error": "Ollama 未响应"}
            timings["vlm"] = round(time.time() - t, 3)
        stages["vlm"] = vlm_out

        # ---- 规则 ----
        hits = {
            "composite": rules.rule_composite(stages["quality"],
                                              float(self.cfg.get("quality.composite_corr_threshold",
                                                                 rules.COMPOSITE_CORR_THRESHOLD))),
            "single_row": rules.rule_single_row(det),
            "screen": rules.rule_screen(vlm_out),
            "empty_bottle": rules.rule_empty_bottle(vlm_out, det),
            "scene_error": rules.rule_scene_error(vlm_out),
            "depth": rules.rule_depth(vlm_out, det),
            "reuse_other_store": rules.rule_reuse(
                matches,
                float(self.cfg.get("retrieval.reuse_similarity", 0.92)),
                store_id,
            ),
        }
        verdict = fusion.fuse(hits, str(vlm_out.get("status")) == "ok", self.cfg.get("fusion", {}))
        timings["total"] = round(time.time() - t0, 3)

        # ---- 降级声明：任何没真正跑的环节都必须写在这里 ----
        degraded = []
        if not self.use_detector:
            degraded.append("detect:disabled_by_flag")
        elif det.get("status") != "ok":
            degraded.append(f"detect:{det.get('status')}")
        if not self.use_vlm:
            degraded.append("vlm:disabled_by_flag")
        elif vlm_out.get("status") != "ok":
            degraded.append(f"vlm:{vlm_out.get('status')}")
        if self.index.size == 0:
            degraded.append("retrieval:no_index")
        if emb_status != "skipped" and self.embedder.backend == "phash":
            degraded.append("embed:phash_fallback")

        return {
            "status": "ok",
            "image": image,
            "image_path": path,
            "image_id": imageops.md5_file(path),
            "store_id": store_id,
            "visit_id": visit_id,
            "verdict": verdict,
            "degraded": degraded,
            "stages": stages,
            "timings": timings,
        }

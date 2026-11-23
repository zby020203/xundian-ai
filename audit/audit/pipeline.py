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
from .logutil import get_logger
from .retrieve import Index
from .vlm import OllamaVLM

log = get_logger(__name__)


class Pipeline:
    def __init__(self, cfg: Optional[Config] = None, use_vlm: bool = True, use_detector: bool = True):
        self.cfg = cfg or Config.load()
        self.use_vlm = use_vlm
        self.use_detector = use_detector
        self.detector = Detector(self.cfg.get("detector", {}), self.cfg.abspath("paths.model_root"))
        self.vlm = OllamaVLM(self.cfg.get("vlm", {}))
        self.embedder = Embedder(self.cfg.get("embedder", {}))
        self.index = Index.load(self.cfg.abspath("paths.index_path"))
        log.info("初始化：VLM=%s%s | 检测=%s%s | 索引=%d 条%s",
                 "开" if use_vlm else "关",
                 f"({self.cfg.get('vlm.model', '?')})" if use_vlm else "",
                 "开" if use_detector else "关",
                 f"({self.cfg.get('detector.weights', '?')})" if use_detector else "",
                 self.index.size,
                 "" if self.index.size else "（未建，跨图规则不可用）")

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
        log.info("开始处理：%s", image)
        path = self.resolve(image)
        timings: Dict[str, float] = {}
        stages: Dict[str, object] = {}

        if not os.path.exists(path):
            log.error("图片不存在，直接返回错误：%s", path)
            return {"status": "error", "error": f"图片不存在: {path}", "image": image}
        if path != os.path.abspath(image):
            log.debug("路径解析为：%s", path)

        im = imageops.load_image(path)
        stages["quality"] = imageops.summarize(im, self.cfg.get("quality", {}))
        timings["quality"] = round(time.time() - t0, 3)
        q = stages["quality"]
        log.info("[quality] %.2fs -> %sx%s blur=%s%s composite=%s(corr=%s) moire=%s%s",
                 timings["quality"], q["width"], q["height"], q["blur"],
                 " 模糊!" if q["blurry"] else "", q["composite"], q["composite_corr"], q["moire"],
                 " 尺寸偏小!" if q["too_small"] else "")

        det = {"status": "skipped", "bottles": [], "n_bottles": 0, "rows": 0}
        if self.use_detector:
            log.info("[detect] YOLO 推理中（imgsz=%s device=%s，CPU 上约 20~60s）...",
                     self.cfg.get("detector.imgsz", 1280), self.cfg.get("detector.device", "cpu"))
            t = time.time()
            det = self.detector.predict(path)
            timings["detect"] = round(time.time() - t, 3)
            log.info("[detect] %.2fs -> status=%s 瓶=%s 行=%s%s",
                     timings["detect"], det.get("status"), det.get("n_bottles"), det.get("rows"),
                     "" if det.get("status") == "ok" else f"（{det.get('error', '')}）")
        else:
            log.info("[detect] 跳过（--no-detect），单排/多排安全阀将不生效")
        stages["detect"] = det

        t = time.time()
        matches, emb_status = [], "skipped"
        if self.index.size > 0:
            try:
                v = self.embedder.encode(im)
                matches = self.index.search(v, int(self.cfg.get("retrieval.top_k", 5)))
                # 排除自匹配：索引含查询图自身时 sim=1.0 会让跨图骨架规则恒命中自己。
                rel = os.path.relpath(path, self.cfg.abspath("paths.dataset_root")).replace(os.sep, "/")
                n0 = len(matches)
                matches = [m for m in matches
                           if str(m.get("image_path", "")).replace("\\", "/") != rel]
                emb_status = f"ok({self.embedder.backend})"
                log.info("[retrieval] %.2fs -> 后端=%s 索引=%d 命中=%d（剔除自匹配 %d）top1=%s %s",
                         time.time() - t, self.embedder.backend, self.index.size, len(matches),
                         n0 - len(matches),
                         round(float(matches[0]["similarity"]), 4) if matches else "-",
                         matches[0].get("store_id", "") if matches else "")
            except Exception as e:
                emb_status = f"error:{e}"
                log.warning("[retrieval] 失败：%s", e)
        else:
            log.info("[retrieval] 无索引，跳过（跨图类问题不可判）")
        timings["retrieval"] = round(time.time() - t, 3)
        stages["retrieval"] = {"status": emb_status, "index_size": self.index.size, "matches": matches[:3]}

        vlm_out: Dict[str, object] = {"status": "skipped"}
        if self.use_vlm:
            t = time.time()
            if self.vlm.available():
                vlm_out = self.vlm.judge(im)
            else:
                vlm_out = {"status": "unavailable", "error": "Ollama 未响应"}
                log.error("[vlm] Ollama 服务无响应（%s），本图转人工",
                          self.cfg.get("vlm.host", "http://127.0.0.1:11434"))
            timings["vlm"] = round(time.time() - t, 3)
            if vlm_out.get("status") == "ok":
                log.info("[vlm] %.2fs -> scene=%s depth=%s 空瓶=%s is_screen=%s conf=%s",
                         timings["vlm"], vlm_out.get("scene"), vlm_out.get("depth"),
                         vlm_out.get("empty_bottle_count"), vlm_out.get("is_screen"),
                         vlm_out.get("confidence"))
            else:
                log.warning("[vlm] %.2fs -> status=%s %s",
                            timings["vlm"], vlm_out.get("status"), vlm_out.get("error", ""))
        else:
            log.info("[vlm] 跳过（--no-vlm），场景/纵深/空瓶/拍屏均不可判")
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
        for name, (ok, conf, ev) in hits.items():
            if ok:
                log.info("[rule:%s] 命中 conf=%.2f | %s", name, conf, ev)
            else:
                log.debug("[rule:%s] 未命中 | %s", name, ev)
        verdict = fusion.fuse(hits, str(vlm_out.get("status")) == "ok", self.cfg.get("fusion", {}))
        timings["total"] = round(time.time() - t0, 3)
        log.info("[fusion] %s | 多标签=%s | conf=%s | route=%s | total=%.1fs",
                 verdict["label_main"], ",".join(verdict["label_all"]) or "-",
                 verdict["confidence"], verdict["route"], timings["total"])

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

        if degraded:
            for d in degraded:
                log.warning("[degraded] %s —— 该环节未真正跑起来，结论已降级", d)
        else:
            log.info("[degraded] 无（全链路真实跑通）")

        image_id = imageops.md5_file(path)
        log.debug("image_id=%s", image_id)

        return {
            "status": "ok",
            "image": image,
            "image_path": path,
            "image_id": image_id,
            "store_id": store_id,
            "visit_id": visit_id,
            "verdict": verdict,
            "degraded": degraded,
            "stages": stages,
            "timings": timings,
        }

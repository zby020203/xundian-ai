"""图像向量化：用于 1:N 检索（套拍）。

优先 DINOv2/ResNet 语义特征，全部不可用时退化为感知哈希向量，
保证任何环境下都能建索引、都能跑通。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import imageops
from .logutil import get_logger

log = get_logger(__name__)


class Embedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = "none"
        self.model = None
        self._torch = None
        self.status = "not_loaded"

    # ------------------------------------------------------------ 加载

    def load(self) -> str:
        if self.backend != "none":
            return self.backend

        want = str(self.cfg.get("backend", "auto"))
        if want in ("auto", "torchvision"):
            if self._load_resnet():
                self.backend, self.status = "resnet18", "ok"
                log.info("[embed] 向量后端=resnet18（语义特征，512 维）")
                return self.backend

        self.backend, self.status = "phash", "fallback"
        log.warning("[embed] 向量后端退化为 pHash（仅感知哈希 64 维，跨图相似度区分度显著下降）")
        return self.backend

    def _load_resnet(self) -> bool:
        try:
            import torch
            from torchvision import models, transforms
        except Exception as e:
            log.debug("[embed] torch/torchvision 不可用：%s", e)
            return False
        try:
            m = getattr(models, self.cfg.get("torchvision_model", "resnet18"))(weights="DEFAULT")
        except Exception as e:
            log.debug("[embed] resnet18 权重不可用（多为无网下载失败）：%s", e)
            return False  # 没网下不了权重，退化为 phash
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self._torch = torch
        self._tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return True

    # ------------------------------------------------------------ 编码

    def encode(self, im: Image.Image) -> np.ndarray:
        backend = self.load()
        if backend == "resnet18":
            with self._torch.no_grad():
                x = self._tf(im).unsqueeze(0)
                feats = self.model(x)
            v = feats[0].numpy().astype(np.float32)
            n = np.linalg.norm(v) + 1e-8
            return v / n
        # fallback：感知哈希 -> ±1 向量
        h = imageops.phash(im).astype(np.float32)
        v = (h * 2.0 - 1.0)
        return v / (np.linalg.norm(v) + 1e-8)

    @property
    def dim(self) -> int:
        return 512 if self.load() == "resnet18" else 64

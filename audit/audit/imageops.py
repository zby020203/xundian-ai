"""图像基础操作：质量评估、拼接检测、摩尔纹、感知哈希。

全部基于 numpy + PIL，cv2 仅作为加速项（缺失时自动退化为 numpy 实现）。
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
from PIL import Image

try:  # 有 cv2 更好，没有也能跑
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- 基础

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def md5_file(path: str, n: int = 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def gray_array(im: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    g = im.convert("L")
    if size:
        g = g.resize(size, Image.BILINEAR)
    return np.asarray(g, dtype=np.float32)


# ---------------------------------------------------------------- 质量

def blur_score(im: Image.Image) -> float:
    """拉普拉斯方差，越大越清晰。"""
    g = gray_array(im, (512, 512))
    if cv2 is not None:
        return float(cv2.Laplacian(g, cv2.CV_32F).var())
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    )
    return float(out.var())


def is_composite(g: np.ndarray, threshold: float = 0.6) -> Dict[str, object]:
    """左右两半高度相似 => 拼接双图。g 为 128x128 灰度。"""
    l = g[:, : g.shape[1] // 2].ravel()
    r = g[:, g.shape[1] // 2:].ravel()
    if l.std() < 1e-6 or r.std() < 1e-6:
        corr = 0.0
    else:
        corr = float(abs(np.corrcoef(l, r)[0, 1]))
    return {"flag": corr > threshold, "corr": round(corr, 4)}


def moire_score(g: np.ndarray) -> float:
    """频域中高频环带的能量占比，拍屏会出现明显的周期性峰。"""
    x = g - g.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(x)))
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    ring = (r > 0.15) & (r < 0.45)
    if not ring.any():
        return 0.0
    band = spec[ring]
    return float(band.mean() / (np.median(band) + 1e-6))


# ---------------------------------------------------------------- 感知哈希

def _dct_mat(n: int) -> np.ndarray:
    k = np.arange(n)[None, :]
    x = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    m[0, :] *= np.sqrt(1.0 / n)
    m[1:, :] *= np.sqrt(2.0 / n)
    return m.astype(np.float32)


def phash(im: Image.Image, hash_size: int = 8) -> np.ndarray:
    """感知哈希，返回 0/1 数组（64 维）。"""
    n = hash_size * 4
    g = gray_array(im, (n, n))
    d = _dct_mat(n) @ g @ _dct_mat(n).T
    low = d[:hash_size, :hash_size]
    med = np.median(low[1:, :]) if low.size > 1 else np.median(low)
    return (low > med).astype(np.uint8).ravel()


def hash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return 0.0
    return float(1.0 - np.count_nonzero(a != b) / a.size)


# ---------------------------------------------------------------- 汇总

def summarize(im: Image.Image, quality_cfg: dict) -> Dict[str, object]:
    g128 = gray_array(im, (128, 128))
    comp = is_composite(g128, float(quality_cfg.get("composite_corr_threshold", 0.6)))
    b = blur_score(im)
    return {
        "width": im.width,
        "height": im.height,
        "blur": round(b, 2),
        "blurry": b < float(quality_cfg.get("blur_threshold", 60.0)),
        "composite": comp["flag"],
        "composite_corr": comp["corr"],
        "moire": round(moire_score(g128), 3),
        "too_small": min(im.width, im.height) < int(quality_cfg.get("min_side", 480)),
    }

"""1:N 检索索引（套拍）。

用 numpy 存特征矩阵 + 元数据，避免引入 faiss 依赖；
8,727 张图的规模下，一次全量余弦计算约几十毫秒，完全够用。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np


class Index:
    def __init__(self):
        self.vecs: Optional[np.ndarray] = None
        self.meta: List[dict] = []

    # ------------------------------------------------------------ 构建

    def add(self, vec: np.ndarray, meta: dict):
        v = vec.astype(np.float32)[None, :]
        self.vecs = v if self.vecs is None else np.vstack([self.vecs, v])
        self.meta.append(meta)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, vecs=self.vecs if self.vecs is not None else np.zeros((0, 1), np.float32))
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Index":
        idx = cls()
        if not os.path.exists(path):
            return idx
        data = np.load(path)
        idx.vecs = data["vecs"]
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                idx.meta = json.load(f)
        return idx

    @property
    def size(self) -> int:
        return 0 if self.vecs is None else int(self.vecs.shape[0])

    # ------------------------------------------------------------ 查询

    def search(self, vec: np.ndarray, top_k: int = 5) -> List[Dict[str, object]]:
        if self.vecs is None or self.size == 0:
            return []
        v = vec.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        sims = self.vecs @ v
        k = min(top_k, self.size)
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            m = dict(self.meta[i]) if i < len(self.meta) else {}
            m["similarity"] = round(float(sims[i]), 4)
            out.append(m)
        return out

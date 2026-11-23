"""读取业务表 Sheet1.csv。

坑：原表有重名列——`照片url` 和 `匹配照片url` 各出现两次
（列 17/23 是图片编号文本，列 30/31 才是真正的图片相对路径）。
pandas 会重命名成 .1，标准库 csv 则是后者覆盖前者。
这里按“同名取最后一列”的规则处理，正好命中真正的路径列。
"""
from __future__ import annotations

import csv
from typing import Dict, Iterator


def iter_sheet(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        try:
            header = next(rd)
        except StopIteration:
            return
        pos = {}
        for i, h in enumerate(header):
            pos[h] = i  # 同名取最后一列
        n = len(header)
        for raw in rd:
            if len(raw) < n:
                raw = raw + [""] * (n - len(raw))
            d = {h: raw[i] for h, i in pos.items()}
            d["image_path"] = raw[pos.get("照片url", n - 2)]
            d["match_path"] = raw[pos.get("匹配照片url", n - 1)]
            yield d


def find_sheet(root: str) -> str:
    import os
    return os.path.join(root, "Sheet1.csv")

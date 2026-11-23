"""构建 1:N 检索索引（套拍用）。

只依赖标准库 csv + numpy，不依赖 pandas。

    python scripts/build_index.py --limit 300
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit import imageops                        # noqa: E402
from audit.config import Config                   # noqa: E402
from audit.embed import Embedder                  # noqa: E402
from audit.retrieve import Index                  # noqa: E402
from audit.sheet import iter_sheet                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    root = cfg.abspath("paths.dataset_root")
    sheet = args.sheet or os.path.join(root, "Sheet1.csv")
    out = args.out or cfg.abspath("paths.index_path")

    seen, rows = set(), []
    for r in iter_sheet(sheet):
        p = r.get("image_path", "")
        if p and p not in seen:
            seen.add(p)
            rows.append(r)
        if len(rows) >= args.limit:
            break
    print(f"待编码 {len(rows)} 张（按路径去重后取前 {args.limit}）")

    emb = Embedder(cfg.get("embedder", {}))
    idx = Index()
    fail = 0
    for i, r in enumerate(rows, 1):
        p = os.path.join(root, r["image_path"])
        if not os.path.exists(p):
            fail += 1
            continue
        try:
            im = imageops.load_image(p)
            idx.add(emb.encode(im), {
                "image_path": r["image_path"],
                "store_id": r.get("网点编码", ""),
                "store_name": r.get("网点名称", ""),
                "visit_id": r.get("拜访编号", ""),
                "date": r.get("拜访日期", ""),
            })
        except Exception as e:
            fail += 1
            print(f"  ! {p}: {e}")
        if i % 100 == 0:
            print(f"  ... {i}/{len(rows)}")

    idx.save(out)
    print(f"完成：索引 {idx.size} 条，后端 {emb.backend}，失败 {fail} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


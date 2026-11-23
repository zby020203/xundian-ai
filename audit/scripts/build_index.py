"""构建 1:N 检索索引（套拍用）。

只依赖标准库 csv + numpy（+ torch/torchvision 可选），不依赖 pandas。
每张图都实时输出进度，大批量编码时不会一片空白。

    python scripts/build_index.py --limit 300
    python scripts/build_index.py --limit 8727 --log-level debug
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit import imageops                        # noqa: E402
from audit.config import Config                   # noqa: E402
from audit.embed import Embedder                  # noqa: E402
from audit.logutil import get_logger, setup_logging   # noqa: E402
from audit.retrieve import Index                  # noqa: E402
from audit.sheet import iter_sheet                # noqa: E402

log = get_logger("index")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--every", type=int, default=100, help="每几张打一条进度")
    ap.add_argument("--log-level", default="info",
                    choices=["debug", "info", "warning", "error", "quiet"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    setup_logging(args.log_level)

    cfg = Config.load(args.config)
    t0 = time.time()
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
    log.info("待编码 %d 张（按路径去重后取前 %d），数据根=%s", len(rows), args.limit, root)

    emb = Embedder(cfg.get("embedder", {}))
    idx = Index()
    fail = 0
    for i, r in enumerate(rows, 1):
        p = os.path.join(root, r["image_path"])
        if not os.path.exists(p):
            fail += 1
            log.warning("[%d/%d] 图片不存在，跳过：%s", i, len(rows), r["image_path"])
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
            log.debug("[%d/%d] 已编码 %s（%dx%d）", i, len(rows), r["image_path"],
                      im.width, im.height)
        except Exception as e:
            fail += 1
            log.warning("[%d/%d] 编码失败 %s：%s", i, len(rows), p, e)
        if args.every and i % args.every == 0:
            tot = time.time() - t0
            log.info("进度 %d/%d｜已入库 %d 条｜失败 %d｜%.1fs（%.2f 张/秒）",
                     i, len(rows), idx.size, fail, tot, i / max(tot, 1e-6))

    idx.save(out)
    log.info("完成：索引 %d 条，后端 %s，失败 %d，总耗时 %.1fs -> %s",
             idx.size, emb.backend, fail, time.time() - t0, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


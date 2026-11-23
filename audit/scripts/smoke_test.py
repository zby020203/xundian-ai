"""冒烟测试：默认跑满全链路（检测 + 检索 + VLM），并显式打印降级项。

    python scripts/smoke_test.py                # 全链路
    python scripts/smoke_test.py --no-vlm       # 跳过 VLM
    python scripts/smoke_test.py --no-detect    # 跳过检测
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit.config import Config          # noqa: E402
from audit.logutil import get_logger, setup_logging   # noqa: E402
from audit.pipeline import Pipeline      # noqa: E402

log = get_logger("smoke")

SAMPLES = [
    "images/20251004/central/00987_cur.jpg",   # 纵深：洗衣液后排
    "images/20251009/southwest/05025_cur.jpg", # 空瓶：冰柜盖单排
    "images/20251028/northwest/30657_cur.jpg", # 场景错误：影城卖品台
    "images/20251016/southwest/14816_cur.jpg", # 套拍：双联屏
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--log-level", default="info",
                    choices=["debug", "info", "warning", "error", "quiet"])
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    setup_logging(args.log_level)

    cfg = Config.load()
    pipe = Pipeline(cfg, use_vlm=not args.no_vlm, use_detector=not args.no_detect)

    bad = 0
    for i, s in enumerate(SAMPLES, 1):
        log.info("===== %d/%d %s =====", i, len(SAMPLES), s)
        res = pipe.run(s)
        if res.get("status") != "ok":
            bad += 1
            log.error("[FAIL] %s: %s", s, res.get("error"))
            continue
        v, q = res["verdict"], res["stages"]["quality"]
        d = res["stages"]["detect"]
        deg = ",".join(res.get("degraded") or []) or "-"
        log.info("[OK] %s -> %s conf=%s route=%s 瓶=%s 行=%s composite=%s(%s) | 降级: %s",
                 s.split("/")[-1], v["label_main"], v["confidence"], v["route"],
                 d.get("n_bottles"), d.get("rows"), q["composite"], q["composite_corr"], deg)

    log.info("%d/%d 通过", len(SAMPLES) - bad, len(SAMPLES))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

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
from audit.pipeline import Pipeline      # noqa: E402

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
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = Config.load()
    pipe = Pipeline(cfg, use_vlm=not args.no_vlm, use_detector=not args.no_detect)

    bad = 0
    for s in SAMPLES:
        res = pipe.run(s)
        if res.get("status") != "ok":
            bad += 1
            print(f"[FAIL] {s}: {res.get('error')}")
            continue
        v, q = res["verdict"], res["stages"]["quality"]
        d = res["stages"]["detect"]
        deg = ",".join(res.get("degraded") or []) or "-"
        print(f"[OK] {s.split('/')[-1]:>12}  -> {v['label_main']:<10} "
              f"conf={v['confidence']:<5} route={v['route']:<5} "
              f"瓶={d.get('n_bottles')} 行={d.get('rows')} "
              f"composite={q['composite']}({q['composite_corr']}) | 降级: {deg}")

    print(f"\n{len(SAMPLES) - bad}/{len(SAMPLES)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

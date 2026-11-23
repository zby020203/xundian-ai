"""单图端到端测试入口。

用法：
    python scripts/run_single.py --image images/20251004/central/00987_cur.jpg
    python scripts/run_single.py --image <绝对路径> --no-vlm --no-detect
    python scripts/run_single.py --image ... --json-out outputs/one.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit.config import Config          # noqa: E402
from audit.pipeline import Pipeline      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="巡店 AI 单图端到端测试")
    ap.add_argument("--image", required=True, help="图片路径（绝对路径或相对数据集根目录）")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--store-id", default=None)
    ap.add_argument("--visit-id", default=None)
    ap.add_argument("--no-vlm", action="store_true", help="跳过 VLM（离线/省时）")
    ap.add_argument("--no-detect", action="store_true", help="跳过 YOLO 检测")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--full", action="store_true", help="打印完整 stage 详情")
    ap.add_argument("--strict", action="store_true", help="任何环节降级即报错退出，不允许静默降级")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = Config.load(args.config)
    pipe = Pipeline(cfg, use_vlm=not args.no_vlm, use_detector=not args.no_detect)
    res = pipe.run(args.image, store_id=args.store_id, visit_id=args.visit_id)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        v = res.get("verdict", {})
        print("=" * 60)
        print(f"图片      : {res.get('image')}")
        print(f"结论      : {v.get('label_main')}   置信度 {v.get('confidence')}   路由 {v.get('route')}")
        print(f"多标签    : {', '.join(v.get('label_all') or []) or '-'}")
        print(f"理由      : {v.get('reason')}")
        q = res.get("stages", {}).get("quality", {})
        d = res.get("stages", {}).get("detect", {})
        print(f"质量      : {q.get('width')}x{q.get('height')} blur={q.get('blur')} "
              f"composite={q.get('composite')}(corr {q.get('composite_corr')}) moire={q.get('moire')}")
        print(f"检测      : {d.get('status')} 瓶数={d.get('n_bottles')} 行数={d.get('rows')}")
        print(f"降级      : {', '.join(res.get('degraded') or []) or '无'}")
        print(f"耗时(秒)  : {res.get('timings')}")
        if args.full:
            print("-" * 60)
            print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.strict and res.get("degraded"):
        if not args.quiet:
            print("\n[STRICT] 存在降级环节，按 --strict 判定为失败：")
            for d in res["degraded"]:
                print(f"  - {d}")
        return 2
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

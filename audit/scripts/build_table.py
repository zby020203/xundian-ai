"""生成训练表（单表结构）。

只依赖标准库 csv，任何 python 环境都能跑。

输入：Sheet1.csv（业务表）+ 人工标注 CSV
     标注 CSV 至少含：image_path / label_main
输出：train.csv —— 单图任务与图对任务同表，配对列可空。

用法：
    python scripts/build_table.py --labels labels.csv --out outputs/train.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit.config import Config  # noqa: E402
from audit.sheet import iter_sheet  # noqa: E402

COLUMNS = [
    "sample_id", "task", "image_a", "image_b", "label", "label_all",
    "confidence", "reason_cn", "sample_type", "split",
    "image_id", "visit_id", "match_path", "match_similarity",
    "source_row_ids", "n_rows",
    "拜访日期", "3级部门名称", "4级部门名称", "拜访人",
    "网点编码", "网点名称", "照片位置", "摆拍形式", "是否重复",
    "复核", "复核备注",
]


def task_of(label_main: str, has_pair: bool) -> str:
    if has_pair:
        return "套拍-相似图"
    if label_main.startswith("套拍"):
        return "套拍-相似图"
    if label_main in ("正常陈列", "正常"):
        return "正常"
    return f"摆拍-{label_main}"


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def read_csv(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "train.csv"))
    args = ap.parse_args()

    cfg = Config.load(args.config)
    root = cfg.abspath("paths.dataset_root")
    sheet_path = args.sheet or os.path.join(root, "Sheet1.csv")

    # 业务表：路径 -> 首条记录
    meta = {}
    for row in iter_sheet(sheet_path):
        p = row.get("image_path", "")
        if p and p not in meta:
            meta[p] = row
    print(f"业务表 {len(meta)} 个唯一路径")

    out_rows = []
    for i, lab in enumerate(read_csv(args.labels), 1):
        img = lab.get("image_path", "")
        m = meta.get(img, {})
        key = lab.get("image_id") or img
        out_rows.append({
            "sample_id": f"S{i:05d}",
            "task": task_of(lab.get("label_main", ""), bool(lab.get("image_b", ""))),
            "image_a": img,
            "image_b": lab.get("image_b", ""),
            "label": lab.get("label", ""),
            "label_all": lab.get("label_all", ""),
            "confidence": lab.get("confidence", ""),
            "reason_cn": lab.get("reason_cn", ""),
            "sample_type": lab.get("sample_type", "标准"),
            "split": split_of(key),
            "image_id": lab.get("image_id", ""),
            "visit_id": lab.get("visit_id") or m.get("拜访编号", ""),
            "match_path": lab.get("match_path") or m.get("match_path", ""),
            "match_similarity": lab.get("match_similarity", ""),
            "source_row_ids": lab.get("source_row_ids", ""),
            "n_rows": lab.get("n_rows", ""),
            "拜访日期": m.get("拜访日期", ""),
            "3级部门名称": m.get("3级部门名称", ""),
            "4级部门名称": m.get("4级部门名称", ""),
            "拜访人": m.get("拜访人", ""),
            "网点编码": m.get("网点编码", ""),
            "网点名称": m.get("网点名称", ""),
            "照片位置": m.get("照片位置", ""),
            "摆拍形式": m.get("摆拍形式", ""),
            "是否重复": m.get("是否重复", ""),
            "复核": lab.get("复核", ""),
            "复核备注": lab.get("复核备注", ""),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    # 统计
    stat = {}
    for r in out_rows:
        stat[(r["task"], r["sample_type"])] = stat.get((r["task"], r["sample_type"]), 0) + 1
    print(f"写出 {len(out_rows)} 行 -> {args.out}")
    for (t, s), n in sorted(stat.items()):
        print(f"  {t:<14} {s:<4} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


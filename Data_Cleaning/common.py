# -*- coding: utf-8 -*-
"""Data_Cleaning 公共工具：路径契约、主表 schema、读写助手。

设计契约（grill 共识）：
- 原始 data/ 与 audit/ 只读，本目录是唯一写入区；
- 主表一行一图（cur 图），全量弱标签层 + 精标层标记共存；
- 中文列名解析复用 audit/audit/sheet.py 的 iter_sheet（同名取最后一列），
  通过 importlib 按文件路径加载，避免触发 audit 包 __init__ 的重依赖。
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # .../巡店AI/Data_Cleaning
REPO = ROOT.parent                              # .../巡店AI
DATA = REPO / "data"
AUDIT = REPO / "audit"
OUTPUTS = ROOT / "outputs"
CANONICAL = ROOT / "canonical"

MASTER_PATH = OUTPUTS / "master_labels.csv"
PAIRS_PATH = OUTPUTS / "retrieval_pairs.csv"
MANIFEST_PATH = OUTPUTS / "index_manifest.csv"
WORKLIST_PATH = OUTPUTS / "annotation_worklist.csv"
SUMMARY_PATH = OUTPUTS / "run_summary.json"

# 主表列契约：业务元数据 → 五类候选 → 真值层 → 质量/预处理 → 抽样/切分
MASTER_COLUMNS = [
    "image_id", "image_path", "match_path",
    "visit_id", "store_id", "store_name", "visit_date",
    "region", "dept4", "visitor", "photo_pos", "sku_count", "facing_count",
    "match_store_id", "match_visit_id", "match_date",
    "business_form", "business_dup", "business_note", "n_source_rows",
    "cand_empty_bottle", "cand_depth", "cand_scene", "cand_screen",
    "blackbox", "negative_untrusted", "dup_flag",
    "label_empty_bottle", "label_depth", "label_scene", "label_screen",
    "label_all", "label_source", "annotator", "usability",
    "seed_sample_type", "seed_reason",
    "quality_flag", "blur", "width", "height",
    "crop_ratio", "canonical_path", "preprocess_version",
    "stratum", "in_worklist", "split",
]

WORKLIST_COLUMNS = [
    "image_id", "image_path", "canonical_path", "stratum", "quality_flag",
    "cand_empty_bottle", "cand_depth", "cand_scene", "cand_screen",
    "blackbox", "negative_untrusted", "blur", "crop_ratio",
]

LABEL_COLS = ["label_empty_bottle", "label_depth", "label_scene", "label_screen"]
CAND_COLS = ["cand_empty_bottle", "cand_depth", "cand_scene", "cand_screen"]


def load_config() -> dict:
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CANONICAL.mkdir(parents=True, exist_ok=True)


def image_id_of(rel_path: str) -> str:
    """路径字符串 md5 前 16 位作稳定主键（与文件内容无关，重跑不变）。"""
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]


def abs_image(rel_path: str) -> Path:
    return DATA / rel_path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_sheet(path: Path):
    """复用 audit 的原始字段解析缝（重名列坑已处理）。"""
    sheet = _load_module("sheet_adapter", AUDIT / "audit" / "sheet.py")
    return sheet.iter_sheet(str(path))


def read_master() -> list[dict]:
    with open(MASTER_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in MASTER_COLUMNS:
            r.setdefault(c, "")
    return rows


def write_master(rows: list[dict]) -> None:
    with open(MASTER_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def bump_summary(step: str, stats: dict) -> None:
    """各步骤把产出统计追加进 run_summary.json，便于核对流水线。"""
    ensure_dirs()
    summary = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    summary[step] = stats
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def log(msg: str) -> None:
    print(msg, flush=True)
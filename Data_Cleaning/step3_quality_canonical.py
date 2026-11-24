# -*- coding: utf-8 -*-
"""步骤③：质量闸 + canonical 预处理（两个子阶段）。

子阶段 A（默认，全量）：对主表每张 cur 图跑脚本硬指标——
  分辨率/拉普拉斯方差模糊度 → quality_flag ∈ {ok, low_res, blurry, low_res+blurry, missing, corrupt}
  顶部水印横幅自适应检测 → crop_ratio（无水印=0，原图保留不删）。
  水印实测形态=深灰底白字单色矩形（非纯黑）：判据为“与首行基色精确接近(容差2)的像素占比≥0.15”的连续前缀，
  基色需偏暗且低饱和（防白底图误截）。横幅背景是生成器纯色叠加，照片区精确匹配占比≈0；
  密集文字行基色占比可低至0.3，阈值0.5会在文字行提前截断（漏裁多行地址横幅），已否决；
  亮度/饱和度阈值法会把暗货架误当水印或漏检，同样否决。
  读图用 cv2.imdecode(np.fromfile(...))：Windows 下 cv2.imread 不支持中文路径。
   blur 在 512x512 灰度上算，与 audit/imageops.blur_score 口径近似（阈值 60 沿用）。

子阶段 B（--materialize，仅工作清单）：按 crop_ratio 截黑幅，
  干净图副本落盘到 canonical/（本目录内，不碰 data/images），
  回写主表与工作清单的 canonical_path / preprocess_version。
  只物化精标清单（~8500 张）省磁盘；步骤⑥补捞后重跑本子阶段即可增量补齐。

输入：outputs/master_labels.csv（+ --materialize 时读 annotation_worklist.csv）。
输出：主表质量/预处理列；canonical/ 干净图副本。

运行：python step3_quality_canonical.py            # 子阶段 A
      python step3_quality_canonical.py --materialize  # 子阶段 B（步骤④之后）
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp

import cv2
import numpy as np
from PIL import Image

import common as C


def metrics_one(path, qcfg: dict) -> dict:
    """单图硬指标 + 水印带检测。返回写回主表的字段 dict。"""
    wm = qcfg["watermark"]
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        b = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)
    except Exception:
        b = None
    if b is None:
        return {"quality_flag": "corrupt", "blur": "", "width": "", "height": "", "crop_ratio": ""}
    h2, w2 = b.shape[:2]
    width, height = w2 * 2, h2 * 2
    g = b.mean(axis=2)
    small = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    blur = float(cv2.Laplacian(small, cv2.CV_32F).var())

    # 水印带：单色矩形前缀——与首行基色精确接近(容差2)的像素占比≥0.15 的连续行
    # 在 240 宽缩图上算（全分辨率 int16 通道差太慢，实测 ~80ms/图）；比例不受缩放影响
    sb = cv2.resize(b, (240, max(1, int(h2 * 240 / w2))), interpolation=cv2.INTER_AREA)
    sh = sb.shape[0]
    bi = sb.astype(np.int16)
    base = np.median(bi[0], axis=0)
    band = 0
    if base.mean() < float(wm["mean_max"]) and (base.max() - base.min()) < float(wm["spread_threshold"]):
        frac = (np.abs(bi - base).max(axis=2) <= int(wm["base_tol"])).mean(axis=1)
        limit = int(sh * float(wm["max_scan_ratio"]))
        stable = int(wm["stable_rows"])
        is_band = frac >= float(wm["frac_threshold"])
        for i in range(limit):
            if not is_band[i] and np.all(~is_band[i:i + stable]):
                band = i
                break
        if band < sh * float(wm["min_band_ratio"]):
            band = 0
    crop_ratio = round(band / sh, 4) if sh else 0.0

    flags = []
    if min(width, height) < int(qcfg["min_side"]):
        flags.append("low_res")
    if blur < float(qcfg["blur_threshold"]):
        flags.append("blurry")
    return {
        "quality_flag": "+".join(flags) if flags else "ok",
        "blur": round(blur, 2),
        "width": width, "height": height,
        "crop_ratio": crop_ratio,
    }


def run_metrics(workers: int = 1) -> None:
    cfg = C.load_config()
    rows = C.read_master()
    dist = collections.Counter()
    n_band = 0
    results: dict[str, dict] = {}
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, (rel, m) in enumerate(pool.imap_unordered(_work_one, [r["image_path"] for r in rows], chunksize=64)):
                results[rel] = m
                if (i + 1) % 4000 == 0:
                    C.log(f"[step3A] {i + 1}/{len(rows)} ...")
    else:
        for i, r in enumerate(rows):
            p = C.abs_image(r["image_path"])
            m = {"quality_flag": "missing", "blur": "", "width": "", "height": "", "crop_ratio": ""} if not p.exists() else metrics_one(p, cfg["quality"])
            results[r["image_path"]] = m
            if (i + 1) % 2000 == 0:
                C.log(f"[step3A] {i + 1}/{len(rows)} ...")
    for r in rows:
        m = results[r["image_path"]]
        r.update(m)
        dist[m["quality_flag"]] += 1
        if m["crop_ratio"] and float(m["crop_ratio"]) > 0:
            n_band += 1
    C.write_master(rows)
    stats = {
        "images": len(rows),
        "quality_distribution": dict(dist),
        "watermark_band_detected": n_band,
    }
    C.bump_summary("step3_quality_metrics", stats)
    C.log("[step3A] 质量闸完成:")
    for k, v in stats.items():
        C.log(f"  {k}: {v}")


def run_reflag() -> None:
    """阈值校准后重算 quality_flag：只用主表已存指标，不重读图。"""
    cfg = C.load_config()
    q = cfg["quality"]
    rows = C.read_master()
    dist = collections.Counter()
    for r in rows:
        if not r["blur"]:
            flag = r["quality_flag"] or "missing"
        else:
            flags = []
            if min(int(r["width"]), int(r["height"])) < int(q["min_side"]):
                flags.append("low_res")
            if float(r["blur"]) < float(q["blur_threshold"]):
                flags.append("blurry")
            flag = "+".join(flags) if flags else "ok"
        r["quality_flag"] = flag
        dist[flag] += 1
    C.write_master(rows)
    stats = {"reflag_distribution": dict(dist), "blur_threshold": q["blur_threshold"]}
    C.bump_summary("step3_reflag", stats)
    C.log("[step3-reflag] 完成:")
    for k, v in stats.items():
        C.log(f"  {k}: {v}")


_WORKER_CFG = None


def _init_worker(cfg: dict) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _work_one(rel: str):
    p = C.abs_image(rel)
    if not p.exists():
        return rel, {"quality_flag": "missing", "blur": "", "width": "", "height": "", "crop_ratio": ""}
    return rel, metrics_one(p, _WORKER_CFG["quality"])


def run_materialize(workers: int = 1) -> None:
    cfg = C.load_config()
    rows = C.read_master()
    by_path = {r["image_path"]: r for r in rows}
    work = C.read_csv(C.WORKLIST_PATH)
    ver = cfg["preprocess_version"]
    n_done = n_skip = 0
    done_rels: dict[str, str] = {}
    items = []
    for w in work:
        r = by_path.get(w["image_path"])
        if r is None:
            n_skip += 1
            continue
        items.append((w["image_path"], float(r.get("crop_ratio") or 0.0)))
    global _WORKER_CFG
    _WORKER_CFG = cfg
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, (rel, canon) in enumerate(pool.imap_unordered(_mat_one, items, chunksize=32)):
                if canon:
                    done_rels[rel] = canon
                else:
                    n_skip += 1
                if (i + 1) % 1000 == 0:
                    C.log(f"[step3B] {i + 1}/{len(items)} ...")
    else:
        for i, item in enumerate(items):
            rel, canon = _mat_one(item)
            if canon:
                done_rels[rel] = canon
            else:
                n_skip += 1
            if (i + 1) % 1000 == 0:
                C.log(f"[step3B] {i + 1}/{len(items)} ...")
    for rel, canon in done_rels.items():
        by_path[rel]["canonical_path"] = canon
        by_path[rel]["preprocess_version"] = ver
    for w in work:
        if w["image_path"] in done_rels:
            w["canonical_path"] = done_rels[w["image_path"]]
    n_done = len(done_rels)
    C.write_master(rows)
    C.write_csv(C.WORKLIST_PATH, work, C.WORKLIST_COLUMNS)
    stats = {"materialized": n_done, "skipped": n_skip, "preprocess_version": ver}
    C.bump_summary("step3_canonical_materialize", stats)
    C.log("[step3B] canonical 落盘完成:")
    for k, v in stats.items():
        C.log(f"  {k}: {v}")


def _mat_one(item):
    """单图截水印落盘；item=(rel, crop_ratio)。返回 (rel, canonical相对路径) 或 (rel, '')。"""
    rel, ratio = item
    cfg = _WORKER_CFG or C.load_config()
    src = C.abs_image(rel)
    if not src.exists():
        return rel, ""
    q = int(cfg.get("jpeg_quality", 92))
    with Image.open(src) as im:
        im = im.convert("RGB")
        band = int(round(ratio * im.height))
        if band > 0:
            im = im.crop((0, band, im.width, im.height))
        dest = C.CANONICAL / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=q)
    return rel, str(dest.relative_to(C.ROOT)).replace("\\", "/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize", action="store_true", help="子阶段B：按工作清单截水印落盘")
    ap.add_argument("--workers", type=int, default=1, help="子阶段A 并行进程数")
    ap.add_argument("--reflag", action="store_true", help="阈值校准后重算 quality_flag（不重读图）")
    args = ap.parse_args()
    if args.materialize:
        run_materialize(args.workers)
    elif args.reflag:
        run_reflag()
    else:
        run_metrics(args.workers)
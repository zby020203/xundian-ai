# -*- coding: utf-8 -*-
"""产出核对（步骤⑤前半）：校验 1-4 步全部落盘结果的一致性。

检查项：
- 主表/工作清单/配对表/建库清单 行数与关键字段填充率；
- canonical 文件逐一存在性（8165 张）；
- 抽检裁剪几何：canonical 高度 ≈ 原高 × (1 - crop_ratio)；
- 工作清单质量闸全 ok、stratum 分布、种子并入情况。

运行：python verify_outputs.py
"""
from __future__ import annotations

import collections

from PIL import Image

import common as C


def main() -> None:
    rows = C.read_master()
    work = C.read_csv(C.WORKLIST_PATH)
    pairs = C.read_csv(C.PAIRS_PATH)
    manifest = C.read_csv(C.MANIFEST_PATH)
    print(f"master={len(rows)} worklist={len(work)} pairs={len(pairs)} manifest={len(manifest)}")

    # 1) canonical 填充率 + 文件存在性
    n_missing_path = n_missing_file = 0
    for w in work:
        cp = w.get("canonical_path") or ""
        if not cp:
            n_missing_path += 1
        elif not (C.ROOT / cp).exists():
            n_missing_file += 1
    print(f"worklist canonical_path 缺失={n_missing_path} 文件不存在={n_missing_file}")

    n_master_wl = sum(1 for r in rows if r["in_worklist"] == "1")
    n_master_canon = sum(1 for r in rows if r["in_worklist"] == "1" and r["canonical_path"])
    ver = C.load_config()["preprocess_version"]
    n_ver = sum(1 for r in rows if r["preprocess_version"] == ver)
    print(f"master in_worklist={n_master_wl} 其中canonical已填={n_master_canon} {ver}={n_ver}")

    # 2) 质量闸 / 分层 / 种子
    qd = collections.Counter(w["quality_flag"] for w in work)
    sd = collections.Counter(w["stratum"] for w in work)
    print(f"worklist quality: {dict(qd)}")
    print(f"worklist stratum: {dict(sd)}")
    seeds = [r for r in rows if r["label_source"] == "seed_draft"]
    seeds_in = sum(1 for r in seeds if r["in_worklist"] == "1")
    n_human = sum(1 for r in rows if r["label_source"] == "human_verified")
    print(f"种子草稿 seed_draft={len(seeds)} 其中进工作清单={seeds_in}；已签认真值 human_verified={n_human}")

    # 3) 抽检裁剪几何：首/中/尾各 1 张 + 1 张 crop=0
    by_path = {r["image_path"]: r for r in rows}
    samples = [work[0], work[len(work) // 2], work[-1]]
    zero = next((w for w in work if not float(by_path[w["image_path"]]["crop_ratio"] or 0)), None)
    if zero:
        samples.append(zero)
    for w in samples:
        r = by_path[w["image_path"]]
        ratio = float(r["crop_ratio"] or 0)
        with Image.open(C.abs_image(w["image_path"])) as im:
            ow, oh = im.size
        with Image.open(C.ROOT / w["canonical_path"]) as im:
            cw, ch = im.size
        expect = round(oh * (1 - ratio))
        ok = abs(ch - expect) <= 2 and cw == ow
        print(f"  [{'OK' if ok else 'BAD'}] {w['image_id']} orig={ow}x{oh} "
              f"canon={cw}x{ch} ratio={ratio} expect_h={expect}")

    # 4) 配对表标签与 dup_flag 一致性
    bad = sum(1 for p in pairs
              if (p["pair_label"] == "positive") != (p["dup_flag"] == "1"))
    print(f"pairs 标签与dup_flag不一致={bad}")

    print("核对完成")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""步骤①：建主表骨架——把业务表翻译成五类候选表。

输入：data/Sheet1.csv（只读）+ config.json 映射配置 + audit/labels.csv 人工种子。
输出：outputs/master_labels.csv（一行一张 cur 图，全量弱标签层）。

口径（grill 共识）：
- 本步只产"候选"，不定罪；最终真值等步骤⑤⑦；
- 同一 image_path 在多行出现时聚合为一行：候选取并集，n_source_rows 计数；
- 是否重复=是 只置 dup_flag（检索线信号），永不进单图标签；
- 摆拍形式=无 置 negative_untrusted（不可信负样本，待复核）；
- 43 条人工细标按 image_path 并入，label_source=human_verified，是全表唯一定稿真值。

运行：python step1_build_master.py
"""
from __future__ import annotations

import common as C


def main() -> None:
    cfg = C.load_config()
    C.ensure_dirs()
    mapping = cfg["mapping"]
    form_map = mapping["form_to_candidate"]
    scene_pos = set(mapping["scene_positions"])
    blackbox_forms = set(mapping["blackbox_forms"])
    negative_forms = set(mapping["negative_forms"])

    agg: dict[str, dict] = {}
    order: list[str] = []
    n_rows = 0
    n_empty_path = 0
    for d in C.iter_sheet(C.DATA / "Sheet1.csv"):
        rel = (d.get("image_path") or "").strip()
        if not rel:
            n_empty_path += 1
            continue
        n_rows += 1
        form = (d.get("摆拍形式") or "").strip()
        pos = (d.get("照片位置") or "").strip()
        dup = (d.get("是否重复") or "").strip()
        if rel not in agg:
            order.append(rel)
            agg[rel] = {
                "image_id": C.image_id_of(rel),
                "image_path": rel,
                "match_path": (d.get("match_path") or "").strip(),
                "visit_id": (d.get("拜访编号") or "").strip(),
                "store_id": (d.get("网点编码") or "").strip(),
                "store_name": (d.get("网点名称") or "").strip(),
                "visit_date": (d.get("拜访日期") or "").strip(),
                "region": (d.get("3级部门名称") or "").strip(),
                "dept4": (d.get("4级部门名称") or "").strip(),
                "visitor": (d.get("拜访人") or "").strip(),
                "photo_pos": pos,
                "sku_count": (d.get("sku数量") or "").strip(),
                "facing_count": (d.get("排面数") or "").strip(),
                "match_store_id": (d.get("匹配网点编码") or "").strip(),
                "match_visit_id": (d.get("匹配拜访编码") or "").strip(),
                "match_date": (d.get("匹配拜访日期") or "").strip(),
                "business_form": form,
                "business_dup": dup,
                "business_note": (d.get("元气备注") or "").strip(),
                "n_source_rows": 0,
                "cand_empty_bottle": 0, "cand_depth": 0,
                "cand_scene": 0, "cand_screen": 0,
                "blackbox": 0, "negative_untrusted": 1, "dup_flag": 0,
                "label_empty_bottle": "", "label_depth": "",
                "label_scene": "", "label_screen": "",
                "label_all": "", "label_source": "business_weak",
                "annotator": "", "usability": "",
                "seed_sample_type": "", "seed_reason": "",
                "quality_flag": "", "blur": "", "width": "", "height": "",
                "crop_ratio": "", "canonical_path": "", "preprocess_version": "",
                "stratum": "", "in_worklist": 0, "split": "",
            }
        r = agg[rel]
        r["n_source_rows"] += 1
        cand = form_map.get(form)
        if cand:
            r["cand_" + cand] = 1
        if pos in scene_pos:
            r["cand_scene"] = 1
        if form in blackbox_forms:
            r["blackbox"] = 1
        if dup == "是":
            r["dup_flag"] = 1
        # 不可信负样本：所有来源行都说"无"才置位（任一行非"无"即清零）
        if form not in negative_forms:
            r["negative_untrusted"] = 0

    # ---- 并入人工种子真值（audit/labels.csv，43 条）----
    tokens = cfg["label_tokens"]
    seeds = C.read_csv(C.AUDIT / "labels.csv")
    n_seed_merged = 0
    n_seed_missing = 0
    for s in seeds:
        rel = (s.get("image_path") or "").strip()
        if not rel:
            continue
        if rel not in agg:
            n_seed_missing += 1
            C.log(f"[warn] 种子图不在主表: {rel}")
            continue
        r = agg[rel]
        label_all = (s.get("label_all") or "").strip()
        hit = {tokens[t] for t in label_all.split(";") if t in tokens}
        for col in C.LABEL_COLS:
            r[col] = 1 if col.replace("label_", "") in hit else 0
        r["label_all"] = label_all
        r["label_source"] = "human_verified"
        r["annotator"] = "human_seed_v1"
        r["seed_sample_type"] = (s.get("sample_type") or "").strip()
        r["seed_reason"] = (s.get("reason_cn") or "").strip()
        n_seed_merged += 1

    rows = [agg[rel] for rel in order]
    C.write_master(rows)

    cand_dist = {c: sum(int(r[c]) for r in rows) for c in C.CAND_COLS}
    stats = {
        "sheet_rows": n_rows,
        "empty_path_rows": n_empty_path,
        "unique_cur_images": len(rows),
        "candidate_distribution": cand_dist,
        "blackbox": sum(int(r["blackbox"]) for r in rows),
        "negative_untrusted": sum(int(r["negative_untrusted"]) for r in rows),
        "dup_flag": sum(int(r["dup_flag"]) for r in rows),
        "seed_merged": n_seed_merged,
        "seed_missing": n_seed_missing,
    }
    C.bump_summary("step1_build_master", stats)
    C.log("[step1] 主表骨架完成:")
    for k, v in stats.items():
        C.log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""步骤②：检索真值线——给④相册复用/⑤同店多门店造评估尺子。

输入：outputs/master_labels.csv（步骤①产出）。
输出：
- outputs/retrieval_pairs.csv：cur/match 配对评估表。
  是否重复=是 → positive（cur 入历史库应能捞到 match）；否 → negative。
- outputs/index_manifest.csv：建库清单（cur+match 去重，带 store/visit 元数据）。

口径（grill 共识）：
- 所有配对均跨店（match 是业务相似检索产物，不是⑤的同店证据）；
- 本线零标注成本，只搬运整理业务端已有配对；
- ④⑤的验收拿这张表量检索层召回/F1，不进单图微调标签。

运行：python step2_retrieval_pairs.py
"""
from __future__ import annotations

import common as C

PAIR_COLUMNS = [
    "pair_id", "cur_path", "match_path",
    "cur_store_id", "match_store_id", "cur_visit_id", "match_visit_id",
    "cur_date", "match_date", "pair_label", "business_form", "dup_flag",
]
MANIFEST_COLUMNS = [
    "image_path", "role", "store_id", "visit_id", "visit_date", "region",
]


def main() -> None:
    C.ensure_dirs()
    rows = C.read_master()

    pairs = []
    manifest: dict[str, dict] = {}
    n_pos = n_neg = n_no_match = 0
    for r in rows:
        cur = r["image_path"]
        match = (r.get("match_path") or "").strip()
        if cur not in manifest:
            manifest[cur] = {
                "image_path": cur, "role": "cur",
                "store_id": r["store_id"], "visit_id": r["visit_id"],
                "visit_date": r["visit_date"], "region": r["region"],
            }
        if not match:
            n_no_match += 1
            continue
        if match not in manifest:
            manifest[match] = {
                "image_path": match, "role": "match",
                "store_id": r["match_store_id"], "visit_id": r["match_visit_id"],
                "visit_date": r["match_date"], "region": r["region"],
            }
        label = "positive" if r["dup_flag"] == "1" else "negative"
        if label == "positive":
            n_pos += 1
        else:
            n_neg += 1
        pairs.append({
            "pair_id": f"p{len(pairs):06d}",
            "cur_path": cur, "match_path": match,
            "cur_store_id": r["store_id"], "match_store_id": r["match_store_id"],
            "cur_visit_id": r["visit_id"], "match_visit_id": r["match_visit_id"],
            "cur_date": r["visit_date"], "match_date": r["match_date"],
            "pair_label": label, "business_form": r["business_form"],
            "dup_flag": r["dup_flag"],
        })

    C.write_csv(C.PAIRS_PATH, pairs, PAIR_COLUMNS)
    C.write_csv(C.MANIFEST_PATH, list(manifest.values()), MANIFEST_COLUMNS)

    cross = sum(1 for p in pairs if p["cur_store_id"] != p["match_store_id"])
    stats = {
        "pairs_total": len(pairs),
        "positive": n_pos,
        "negative": n_neg,
        "rows_without_match": n_no_match,
        "cross_store_pairs": cross,
        "manifest_images": len(manifest),
    }
    C.bump_summary("step2_retrieval_pairs", stats)
    C.log("[step2] 检索真值线完成:")
    for k, v in stats.items():
        C.log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
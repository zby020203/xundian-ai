# -*- coding: utf-8 -*-
"""步骤④：分层抽样——从全量主表圈出精标工作清单（~8500 张）。

输入：outputs/master_labels.csv（含步骤③A 质量标记）+ config.json quotas。
输出：outputs/annotation_worklist.csv；主表回写 in_worklist / stratum。

口径（grill 共识）：
- 池按优先级互斥：empty_bottle > screen > scene > depth > blackbox > negative，
  多候选图归入最高优先级池（稀缺类优先保）；
- quality_flag != ok 的图剔除（低质量剔出精标层，仍留在弱标签层）；
- empty_bottle / screen 配额 -1 = 全收；depth 按 战区×照片位置 比例分层；
- 边界样本配额在预标后按 VLM 置信度复核（步骤⑥），抽样期以分层多样性保底；
- 固定随机种子，重跑结果一致。

运行：python step4_sample_worklist.py
"""
from __future__ import annotations

import collections
import random

import common as C

POOL_ORDER = ["empty_bottle", "screen", "scene", "depth", "blackbox", "negative"]


def pool_of(r: dict) -> str | None:
    if r["cand_empty_bottle"] == "1":
        return "empty_bottle"
    if r["cand_screen"] == "1":
        return "screen"
    if r["cand_scene"] == "1":
        return "scene"
    if r["cand_depth"] == "1":
        return "depth"
    if r["blackbox"] == "1":
        return "blackbox"
    if r["negative_untrusted"] == "1":
        return "negative"
    return None


def stratified_take(pool: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """按 战区×照片位置 比例分层（最大余数法），层内随机。"""
    if quota < 0 or quota >= len(pool):
        return list(pool)
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in pool:
        groups[(r["region"], r["photo_pos"])].append(r)
    total = len(pool)
    alloc = {k: int(quota * len(v) / total) for k, v in groups.items()}
    remain = quota - sum(alloc.values())
    fracs = sorted(groups, key=lambda k: (quota * len(groups[k]) / total) - alloc[k], reverse=True)
    for k in fracs[:remain]:
        alloc[k] += 1
    picked = []
    for k, v in groups.items():
        rng.shuffle(v)
        picked.extend(v[:alloc[k]])
    return picked


def main() -> None:
    cfg = C.load_config()
    C.ensure_dirs()
    rng = random.Random(int(cfg["seed"]))
    quotas = cfg["quotas"]
    rows = C.read_master()

    pools: dict[str, list[dict]] = {p: [] for p in POOL_ORDER}
    n_excluded_quality = collections.Counter()
    n_no_pool = 0
    for r in rows:
        p = pool_of(r)
        if p is None:
            n_no_pool += 1
            continue
        if r["quality_flag"] != "ok":
            n_excluded_quality[p] += 1
            continue
        pools[p].append(r)

    picked: list[dict] = []
    stats_pools = {}
    for p in POOL_ORDER:
        quota = int(quotas[p])
        pool = pools[p]
        if p == "depth":
            take = stratified_take(pool, quota, rng)
        else:
            take = list(pool) if quota < 0 or quota >= len(pool) else rng.sample(pool, quota)
        for r in take:
            r["in_worklist"] = 1
            r["stratum"] = p
        picked.extend(take)
        stats_pools[p] = {"eligible": len(pool), "picked": len(take),
                          "excluded_quality": n_excluded_quality[p]}

    C.write_master(rows)
    work = [{c: r.get(c, "") for c in C.WORKLIST_COLUMNS} for r in picked]
    C.write_csv(C.WORKLIST_PATH, work, C.WORKLIST_COLUMNS)

    stats = {
        "worklist_total": len(picked),
        "pools": stats_pools,
        "images_without_pool": n_no_pool,
        "seed": int(cfg["seed"]),
    }
    C.bump_summary("step4_sample_worklist", stats)
    C.log("[step4] 精标工作清单完成:")
    C.log(f"  worklist_total: {len(picked)}")
    for p in POOL_ORDER:
        s = stats_pools[p]
        C.log(f"  {p}: eligible={s['eligible']} picked={s['picked']} "
              f"excluded_quality={s['excluded_quality']}")


if __name__ == "__main__":
    main()
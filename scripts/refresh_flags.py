#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh_flags.py — 对数据库存量重跑 hard_filter,刷新 flags + 按变化重算封顶。
                   完全不重抓 Adzuna。

用法:
    py refresh_flags.py            # dry-run:只预览会改多少、改成什么,不写库
    py refresh_flags.py --apply    # 真正写库

逻辑(每条岗):
  - 保留库里 LLM 来源的 flag(label 以 "LLM:" 开头),只用 scan_disqualifiers 刷新正则部分,合并去重。
  - flag 集未变      → 只重写 flags(并按当前 cap 收紧,幂等),llm_score 不动。
  - flag 集变 + 有 knockout → llm_score=5, status=DISQUALIFIED。
  - flag 集变 + 无 knockout → llm_score=NULL, status=待投(交给 backfill 用真 base 重评)。
  - 原本未打分(llm_score IS NULL)的岗:只刷新 flags;若新扫出 knockout 则定 5/DISQUALIFIED,
    否则保持 NULL 等 backfill。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import logging
from collections import Counter

from config import DB_PATH
from infrastructure.database import initialize_database
from pipeline.hard_filter import scan_disqualifiers, apply_flags, dedup_flags

logger = logging.getLogger(__name__)


def codes_of(flags: list) -> tuple:
    return tuple(sorted({f.get("code") for f in flags}))


def plan_row(
    title: str, description: str,
    flags_json: str, llm_score: float | None, status: str | None,
) -> tuple:
    """计算这条岗应有的 (new_flags, new_llm_score, new_status, action)。不写库。"""
    try:
        old_flags = json.loads(flags_json) if flags_json else []
    except Exception:
        old_flags = []

    # 保留 LLM 来源 flag,包括由 llm_review 构造的 discipline 结构化项。
    llm_origin = [
        f for f in old_flags
        if str(f.get("label", "")).startswith("LLM:")
        or f.get("code") == "discipline"
    ]
    new_regex  = scan_disqualifiers(title or "", description or "")
    merged     = dedup_flags(new_regex + llm_origin)

    has_ko = any(f["severity"] == "knockout" for f in merged)
    changed = codes_of(old_flags) != codes_of(merged)

    if not changed:
        if has_ko:
            action = "knockout" if llm_score != 5 or status != "DISQUALIFIED" else "unchanged"
            return merged, 5, "DISQUALIFIED", action
        # flag 集没变:重写 flags(cap 可能微调),分按现有终值收紧(幂等)
        if llm_score is not None and merged:
            new_score, _ = apply_flags(llm_score, merged)
            return merged, new_score, status, "unchanged"
        return merged, llm_score, status, "unchanged"

    # flag 集变了
    if has_ko:
        return merged, 5, "DISQUALIFIED", "knockout"
    if llm_score is None:
        return merged, None, (status or "待投"), "still_null"
    # 有终值但 flag 放宽 → 清 NULL,让 backfill 用真 base 重评
    return merged, None, "待投", "queue_rescore"


def main():
    apply = "--apply" in sys.argv
    con = initialize_database(DB_PATH)
    rows = con.execute(
        "SELECT id, title, company, description, flags, llm_score, status "
        "FROM jobs").fetchall()

    stats = Counter()
    writes = []
    examples = {"knockout": [], "queue_rescore": []}

    for jid, title, company, desc, fl, sc, st in rows:
        new_flags, new_score, new_status, action = plan_row(
            title, desc, fl, sc, st)
        stats[action] += 1
        # 只有真正发生变化的才需要写
        new_fl_json = json.dumps(new_flags, ensure_ascii=False)
        if (new_fl_json != (fl or "[]")) or (new_score != sc) or (new_status != st):
            pipeline_state = (
                "DISQUALIFIED" if new_status == "DISQUALIFIED"
                else "READY_FOR_LLM" if new_score is None
                else "SCORED"
            )
            caps = [f["cap"] for f in new_flags if "cap" in f]
            strictest_cap = min(caps) if caps else None
            if new_status == "DISQUALIFIED":
                score_status, applied_cap = "DISQUALIFIED", strictest_cap
            elif new_score is None:
                score_status, applied_cap = "", None
            elif (strictest_cap is not None and strictest_cap < 100
                  and float(new_score) == float(strictest_cap)):
                score_status, applied_cap = "capped", strictest_cap
            else:
                score_status, applied_cap = "ok", None
            writes.append((new_fl_json, new_score, new_status, pipeline_state,
                           score_status, applied_cap, jid))
            if action in examples and len(examples[action]) < 8:
                examples[action].append(f"{(title or '')[:42]} | {company or ''}")

    print(f"=== refresh 计划(共扫描 {len(rows)} 条)===")
    for k in ("unchanged", "knockout", "queue_rescore", "still_null"):
        print(f"  {k:14s}: {stats[k]}")
    print(f"  需写库       : {len(writes)} 条")
    if examples["knockout"]:
        print("\n  [新增/确认 knockout 示例]")
        for e in examples["knockout"]:
            print("   ⛔", e)
    if examples["queue_rescore"]:
        print("\n  [flag 放宽、清 NULL 待 backfill 重评 示例]")
        for e in examples["queue_rescore"]:
            print("   ↻", e)

    if not apply:
        print("\n[dry-run] 未写库。确认无误后加 --apply 执行。")
        con.close()
        return

    con.executemany(
        """UPDATE jobs
           SET flags=?, llm_score=?, status=?, pipeline_state=?,
               score_status=?, applied_cap=?
           WHERE id=?""", writes)
    con.commit()
    con.close()
    print(f"\n[applied] 已更新 {len(writes)} 条。")
    if stats["queue_rescore"]:
        print(f"提示:{stats['queue_rescore']} 条已清 NULL,跑 backfill 会用真 base 重评。")


if __name__ == "__main__":
    main()

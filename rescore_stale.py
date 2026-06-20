#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rescore_stale.py — 把"用旧 rubric/prompt 评过"的岗清回 NULL,交给 backfill 用新标准重评。

背景:本轮改了 llm_review(加 spam_or_test flag、rubric 五五开、LLM 只给 base 分不自封顶)。
      已打分的岗是旧 prompt 评的,与新标准不一致,需重评。
      但 hard_filter 的 knockout 岗(status=DISQUALIFIED, llm_score=5)与 LLM rubric 无关,
      重评也是 5,纯浪费 —— 故保留不动。

用法:
    py rescore_stale.py            # dry-run:只报会清多少,不写库
    py rescore_stale.py --apply    # 真正清 NULL

清完跑 backfill 重评(它按 WHERE llm_score IS NULL 捞):
    py backfill_scores.py

⚠ 前置顺序:必须先跑过 refresh_flags.py --apply(让库里 flags 已是新词表结果),
   再跑本脚本。否则 backfill 重评时读到的还是旧 flags,封顶会不对。
"""

import sys
import sqlite3

import job_matcher as jm   # 复用 DB_PATH


def main():
    apply = "--apply" in sys.argv
    con = sqlite3.connect(jm.DB_PATH)

    total       = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    disq        = con.execute(
        "SELECT COUNT(*) FROM jobs WHERE status='DISQUALIFIED'").fetchone()[0]
    already_null = con.execute(
        "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]
    to_clear = con.execute("""
        SELECT COUNT(*) FROM jobs
        WHERE llm_score IS NOT NULL AND status != 'DISQUALIFIED'
    """).fetchone()[0]

    print(f"=== rescore_stale 计划(库共 {total} 条)===")
    print(f"  将清 NULL 重评(旧 rubric 评过的非 knockout): {to_clear} 条")
    print(f"  保留不动的 knockout(DISQUALIFIED)         : {disq} 条")
    print(f"  本就未评(已是 NULL,backfill 会顺带评)     : {already_null} 条")
    print(f"  → 重评后 backfill 待评总量将是             : {to_clear + already_null} 条")

    if not apply:
        print("\n[dry-run] 未写库。确认无误后加 --apply。")
        con.close()
        return

    n = con.execute("""
        UPDATE jobs SET llm_score=NULL
        WHERE llm_score IS NOT NULL AND status != 'DISQUALIFIED'
    """).rowcount
    con.commit()
    con.close()
    print(f"\n[applied] 已清 {n} 条为 NULL。")
    print("下一步:py backfill_scores.py  —— 用新 rubric 重评。")


if __name__ == "__main__":
    main()
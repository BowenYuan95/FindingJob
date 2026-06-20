#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_scores.py — 后台持续给"未 LLM 打分"的职位补评分。

在另一个终端单独跑,不影响 Streamlit 面板:
    py backfill_scores.py            # 评完所有未打分的就退出
    py backfill_scores.py --watch    # 守护模式:评完后每 N 秒再扫,自动补评新职位

复用 job_matcher 的 llm_review / PROFILE,打分标准与主程序一致。
逐条评、逐条写库,面板随时刷新可见增量;Ctrl+C 不丢已评结果。

⚠ 与主管线对齐(2026-06 更新):
  - llm_review 现返回四元组 (score, reason, summary, llm_flags)。
  - backfill 与 main() 一样,复评后必须走 apply_flags 统一封顶——
    否则被 wet_lab/clinical_delivery/degree_field 封顶的岗会以未封顶 base 分混入。
"""

import os
import sys
import json
import time
import sqlite3
import subprocess
import datetime as dt
import logging

from config import DB_PATH, PROFILE
from job_matcher import llm_review, llm_flags_to_objs
from hard_filter import apply_flags

logger = logging.getLogger(__name__)

WATCH_INTERVAL = 60                       # 守护模式下,空闲多少秒后再扫一次


def unload_models() -> None:
    """评完后卸载两个模型,释放显存/内存。"""
    for m in ["qwen/qwen3.5-9b", "text-embedding-nomic-embed-text-v1.5"]:
        try:
            subprocess.run(["lms", "unload", m],
                           shell=(os.name == "nt"),
                           capture_output=True, timeout=30)
            logger.info(f"[backfill] 已卸载 {m}")
        except Exception as e:
            logger.warning(f"[backfill] 卸载 {m} 失败: {e}")


def fetch_unscored(con: sqlite3.Connection, limit: int = 100) -> list[tuple]:
    """取还没 LLM 分的职位(embedding 已过阈值、在库里),按 embedding 分优先。
    带上 flags 列:backfill 也要复用入库时 hard_filter 存的正则 flags 来封顶。
    注:knockout 岗入库时 llm_score 已被设为 5(非 NULL),故不会被这里捞出重评。"""
    rows = con.execute("""
        SELECT id, title, company, location, description, flags
        FROM jobs
        WHERE llm_score IS NULL
        ORDER BY sim DESC
        LIMIT ?""", (limit,)).fetchall()
    return rows


def score_one(con: sqlite3.Connection, row: tuple) -> bool:
    jid, title, company, location, description, flags_json = row
    job = {"title": title, "company": company,
           "location": location, "description": description or ""}

    # 接住四元组(新版 llm_review 多返回 llm_flags)
    score, reason, summary, llm_flags = llm_review(PROFILE, job)
    if score is None:
        logger.warning(f"  ✗ {(title or '')[:40]} -> 打分失败,跳过本次")
        return False

    # 复用入库时 hard_filter 存的正则 flags + 本次 LLM 自检 flags,统一封顶
    try:
        flags = json.loads(flags_json) if flags_json else []
    except Exception:
        flags = []
    flags = flags + llm_flags_to_objs(llm_flags)
    if flags:
        score, st = apply_flags(score, flags)
        if st != "ok":
            reason = f"[{st}] " + reason

    # flags 可能新增了 LLM 自检项,回写库保持一致(apply_flags 内部已去重,这里存合并后的)
    con.execute(
        "UPDATE jobs SET llm_score=?, llm_reason=?, summary=?, flags=? WHERE id=?",
        (score, reason, summary, json.dumps(flags, ensure_ascii=False), jid))
    con.commit()           # 逐条提交,面板立即可见
    logger.info(f"  ✓ {(title or '')[:40]} -> {score}")
    return True


def run_once() -> int:
    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]
    if total == 0:
        con.close()
        return 0
    logger.info(f"[backfill] 发现 {total} 条未打分,开始补评… ({dt.datetime.now():%H:%M:%S})")
    done = 0
    while True:
        rows = fetch_unscored(con, limit=50)
        if not rows:
            break
        for row in rows:
            if score_one(con, row):
                done += 1
    con.close()
    logger.info(f"[backfill] 本轮完成,补评 {done} 条。")
    return done


def main() -> None:
    no_unload = "--no-unload" in sys.argv   # 调试用:评完不卸载
    watch     = "--watch" in sys.argv       # 守护模式:评完后定期再扫
    logger.info("[backfill] 启动。确保 LM Studio 开着、模型已加载。")
    try:
        if watch:
            logger.info(f"[backfill] 守护模式:空闲 {WATCH_INTERVAL}s 后再扫,Ctrl+C 退出。")
            while True:
                run_once()
                time.sleep(WATCH_INTERVAL)
        else:
            run_once()
            logger.info("[backfill] 全部职位已打分。")
            if not no_unload:
                logger.info("[backfill] 没有更多待评分,卸载模型释放资源…")
                unload_models()
    except KeyboardInterrupt:
        logger.info("\n[backfill] 已停止(已评的都已存库)。")


if __name__ == "__main__":
    main()

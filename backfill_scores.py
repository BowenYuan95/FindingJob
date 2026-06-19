#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_scores.py — 后台持续给"未 LLM 打分"的职位补评分。

在另一个终端单独跑,不影响 Streamlit 面板:
    py backfill_scores.py            # 评完所有未打分的就退出
    py backfill_scores.py --watch    # 守护模式:评完后每 N 秒再扫,自动补评新职位

复用 job_matcher 的 llm_review / PROFILE,打分标准与主程序一致。
逐条评、逐条写库,面板随时刷新可见增量;Ctrl+C 不丢已评结果。
"""

import argparse
import os
import time
import sqlite3
import datetime as dt
import ctypes
from ctypes import wintypes

import job_matcher as jm   # 复用 PROFILE / llm_review / DB_PATH

WATCH_INTERVAL = 60        # 守护模式下,空闲多少秒后再扫一次
ACTIVE_CHECK_INTERVAL = 10 # 窗口活跃时,多少秒后再检查一次


def _foreground_window_title():
    """Return the current foreground window title on Windows."""
    if os.name != "nt":
        return ""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1)
    return buffer.value


def app_window_is_active(window_title):
    """True when the JobFinder desktop window is currently foreground."""
    if not window_title:
        return False
    return window_title.lower() in _foreground_window_title().lower()


def fetch_unscored(con, limit=100):
    """取还没 LLM 分的职位(embedding 已过阈值、在库里),按 embedding 分优先。"""
    rows = con.execute("""
        SELECT id, title, company, location, description
        FROM jobs
        WHERE llm_score IS NULL
        ORDER BY sim DESC
        LIMIT ?""", (limit,)).fetchall()
    return rows


def score_one(con, row):
    jid, title, company, location, description = row
    job = {"title": title, "company": company,
           "location": location, "description": description or ""}
    score, reason, summary = jm.llm_review(jm.PROFILE, job)
    if score is None:
        print(f"  ✗ {title[:40]} -> 打分失败,跳过本次")
        return False
    con.execute("UPDATE jobs SET llm_score=?, llm_reason=?, summary=? WHERE id=?",
                (score, reason, summary, jid))
    con.commit()           # 逐条提交,面板立即可见
    print(f"  ✓ {title[:40]} -> {score}")
    return True


def run_once(should_pause=None):
    con = sqlite3.connect(jm.DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]
    if total == 0:
        con.close()
        return 0
    print(f"[backfill] 发现 {total} 条未打分,开始补评… ({dt.datetime.now():%H:%M:%S})")
    done = 0
    while True:
        rows = fetch_unscored(con, limit=50)
        if not rows:
            break
        for row in rows:
            if should_pause and should_pause():
                print("[backfill] JobFinder 窗口已激活,暂停本轮补评。")
                con.close()
                return done
            if score_one(con, row):
                done += 1
    con.close()
    print(f"[backfill] 本轮完成,补评 {done} 条。")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="守护模式:定期扫描并补评新职位")
    parser.add_argument("--only-when-inactive", action="store_true",
                        help="仅在指定窗口不是前台活动窗口时运行")
    parser.add_argument("--window-title", default="JobFinder",
                        help="用于判断是否活跃的窗口标题片段")
    args = parser.parse_args()

    watch = args.watch

    def should_pause():
        return args.only_when_inactive and app_window_is_active(args.window_title)

    print("[backfill] 启动。确保 LM Studio 开着、Qwen3.5 已加载。")
    try:
        while True:
            if should_pause():
                print(f"[backfill] JobFinder 窗口活跃,暂停检测,{ACTIVE_CHECK_INTERVAL}s 后重试。")
                if not watch:
                    break
                time.sleep(ACTIVE_CHECK_INTERVAL)
                continue
            run_once(should_pause=should_pause)
            if not watch:
                break
            if should_pause():
                continue
            print(f"[backfill] 空闲,{WATCH_INTERVAL}s 后再扫…(Ctrl+C 退出)")
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        print("\n[backfill] 已停止(已评的都已存库)。")


if __name__ == "__main__":
    main()

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
import socket

from config import DB_PATH, PROFILE, LLM_MODEL
from infrastructure.database import database_session, initialize_database
from infrastructure.job_repository import JobRepository
from infrastructure.lmstudio import LM_CLIENT
from .job_matcher import llm_review, llm_flags_to_objs
from .hard_filter import apply_flags, dedup_flags

logger = logging.getLogger(__name__)

WATCH_INTERVAL = 60                       # 守护模式下,空闲多少秒后再扫一次
LOCK_PORT = 47831                         # 本机单实例锁;进程退出后由 OS 自动释放


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


def ensure_llm_loaded() -> bool:
    """LM Studio 在线但模型已卸载时,自动加载评分模型。"""
    try:
        loaded = LM_CLIENT.loaded_models(timeout=5)
        if LLM_MODEL in loaded:
            return True
    except Exception as e:
        logger.warning(f"[backfill] 无法读取 LM Studio 模型状态: {e}")

    try:
        result = subprocess.run(
            ["lms", "load", LLM_MODEL, "--gpu", "max", "--context-length", "16384"],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            logger.info(f"[backfill] 已加载 {LLM_MODEL}")
            return True
        logger.warning(
            f"[backfill] 模型加载失败: {(result.stderr or result.stdout).strip()}")
    except Exception as e:
        logger.warning(f"[backfill] 模型加载失败: {e}")
    return False


def fetch_unscored(
    con: sqlite3.Connection, limit: int = 100, exclude_ids: set[str] | None = None,
) -> list[tuple]:
    """取还没 LLM 分的职位(embedding 已过阈值、在库里),按 embedding 分优先。
    带上 flags 列:backfill 也要复用入库时 hard_filter 存的正则 flags 来封顶。
    注:knockout 岗入库时 llm_score 已被设为 5(非 NULL),故不会被这里捞出重评。"""
    return JobRepository(con).fetch_scoring_queue(limit, exclude_ids)


def score_one(con: sqlite3.Connection, row: tuple) -> bool:
    jid, title, company, location, description, flags_json = row
    job = {"title": title, "company": company,
           "location": location, "description": description or ""}

    # 接住四元组(新版 llm_review 多返回 llm_flags)
    score, reason, summary, llm_flags = llm_review(PROFILE, job)
    repo = JobRepository(con)
    now = dt.datetime.now().isoformat(timespec="seconds")
    if score is None:
        repo.record_scoring_failure(jid, reason, now)
        con.commit()
        logger.warning(f"  ✗ {(title or '')[:40]} -> 打分失败,跳过本次")
        return False

    # 复用入库时 hard_filter 存的正则 flags + 本次 LLM 自检 flags,统一封顶
    try:
        flags = json.loads(flags_json) if flags_json else []
    except Exception:
        flags = []
    flags = dedup_flags(flags + llm_flags_to_objs(llm_flags))
    st = "ok"
    if flags:
        score, st = apply_flags(score, flags)
        if st != "ok":
            reason = f"[{st}] " + reason

    # flags 可能新增了 LLM 自检项,回写库保持一致(apply_flags 内部已去重,这里存合并后的)
    repo.record_scoring_success(
        job_id=jid,
        score=score,
        reason=reason,
        summary=summary,
        flags=flags,
        disqualified=(st == "DISQUALIFIED"),
        now=now,
    )
    con.commit()           # 逐条提交,面板立即可见
    logger.info(f"  ✓ {(title or '')[:40]} -> {score}")
    return True


def run_once() -> int:
    con = initialize_database(DB_PATH)
    try:
        repo = JobRepository(con)
        total = repo.count_ready_for_scoring()
        if total == 0:
            return 0
        logger.info(f"[backfill] 发现 {total} 条未打分,开始补评… ({dt.datetime.now():%H:%M:%S})")
        done = 0
        failed_ids: set[str] = set()
        while True:
            rows = fetch_unscored(con, limit=50, exclude_ids=failed_ids)
            if not rows:
                break
            for row in rows:
                if score_one(con, row):
                    done += 1
                else:
                    failed_ids.add(row[0])
        if failed_ids:
            logger.warning(
                f"[backfill] 本轮有 {len(failed_ids)} 条在重试后仍失败;下轮再试。")
    finally:
        con.close()
    logger.info(f"[backfill] 本轮完成,补评 {done} 条。")
    return done


def acquire_single_instance() -> socket.socket | None:
    """Use a loopback listener as a crash-safe, cross-platform process lock."""
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
        lock.listen(1)
        return lock
    except OSError:
        lock.close()
        return None


def main() -> None:
    no_unload = "--no-unload" in sys.argv   # 调试用:评完不卸载
    watch     = "--watch" in sys.argv       # 守护模式:评完后定期再扫
    instance_lock = acquire_single_instance()
    if instance_lock is None:
        logger.info("[backfill] 已有补评进程运行,本进程退出。")
        return
    logger.info("[backfill] 启动。确保 LM Studio 开着、模型已加载。")
    try:
        if watch:
            logger.info(f"[backfill] 守护模式:空闲 {WATCH_INTERVAL}s 后再扫,Ctrl+C 退出。")
            while True:
                if ensure_llm_loaded():
                    run_once()
                time.sleep(WATCH_INTERVAL)
        else:
            if ensure_llm_loaded():
                run_once()
            with database_session(DB_PATH) as con:
                remaining = JobRepository(con).count_ready_for_scoring()
            if remaining:
                logger.warning(f"[backfill] 本次结束,仍有 {remaining} 条待后续重试。")
            else:
                logger.info("[backfill] 全部职位已打分。")
            if not no_unload:
                logger.info("[backfill] 卸载模型释放资源…")
                unload_models()
    except KeyboardInterrupt:
        logger.info("\n[backfill] 已停止(已评的都已存库)。")
    finally:
        instance_lock.close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
sources/adzuna_search.py — Adzuna API 采集层

从 Adzuna 官方 API 抓取澳洲职位，按 Adzuna 原生职位 ID 去重后返回标准化列表。
"""

import re
import time
import logging

import requests

from config import (
    ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY, ADZUNA_PAGES,
    WHAT_EXCLUDE, MAX_DAYS_OLD, SEARCHES,
)

logger = logging.getLogger(__name__)


def _get_with_retry(
    url: str, params: dict, tries: int = 4, base_wait: int = 2,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            last_error = e
            if attempt < tries:
                wait = base_wait * attempt
                logger.warning(f"[adzuna] 第 {attempt}/{tries} 次失败({e}),{wait}s 后重试…")
                time.sleep(wait)
    raise RuntimeError(f"Adzuna 请求连续失败: {last_error}") from last_error


def _fmt_salary(job: dict) -> str:
    lo, hi = job.get("salary_min"), job.get("salary_max")
    if lo and hi:
        return f"${int(lo):,}-${int(hi):,}"
    return ""


def _fallback_key(job: dict) -> tuple[str, str, str]:
    """Only used if Adzuna unexpectedly omits its native ID."""
    return tuple(
        re.sub(r"\s+", " ", str(value or "").strip().lower())
        for value in (
            job.get("title"),
            (job.get("company") or {}).get("display_name"),
            (job.get("location") or {}).get("display_name"),
        )
    )


def fetch_adzuna() -> list[dict]:
    jobs: list[dict] = []
    seen: set[tuple] = set()
    raw_count = 0

    for search in SEARCHES:
        logger.info(f"[adzuna] 搜索: {search.get('what', search)}")
        for page in range(1, ADZUNA_PAGES + 1):
            url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}"
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": 50,
                "max_days_old": MAX_DAYS_OLD,
                "sort_by": "date",
                "content-type": "application/json",
            }
            if WHAT_EXCLUDE:
                params["what_exclude"] = WHAT_EXCLUDE
            params.update(search)
            try:
                data = _get_with_retry(url, params).json()
            except Exception as e:
                logger.warning(f"[adzuna] 查询失败(已重试) {search} p{page}: {e}")
                break

            results = data.get("results", [])
            logger.info(f"[adzuna]   第 {page} 页: {len(results)} 条")
            if not results:
                break
            raw_count += len(results)

            for job in results:
                source_id = str(job.get("id") or "").strip()
                key = ("id", source_id) if source_id else ("fallback", *_fallback_key(job))
                if key in seen:
                    continue
                seen.add(key)
                jobs.append({
                    "source_id": source_id,
                    "title": (job.get("title") or "").strip(),
                    "company": (job.get("company") or {}).get("display_name", ""),
                    "location": (job.get("location") or {}).get("display_name", ""),
                    "description": re.sub(r"\s+", " ", job.get("description", "")),
                    "url": job.get("redirect_url", ""),
                    "source": "adzuna",
                    "salary": _fmt_salary(job),
                    "created": job.get("created", ""),
                })

    logger.info(f"[adzuna] 原始 {raw_count} 条,去重后 {len(jobs)} 条")
    return jobs

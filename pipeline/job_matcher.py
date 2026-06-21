#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
job_matcher.py — 个人求职聚合 + 本地匹配排序工具(宽搜窄评版)

数据来源策略(按合规性 / 稳定性排序):
  1. Adzuna API      —— 官方免费 API,聚合澳洲多家招聘网站,首选
  2. 邮件 Job Alert  —— 解析自己收件箱(完全合规)

三层精排(把"严"全部下沉到打分层,搜索层只管"捞够"):
  - hard_filter   确定性硬性淘汰 / 封顶(clearance / AHPRA / 截止已过 / 湿实验室 / allied health …)
  - Nomic embedding 余弦相似度 -> 语义初筛
  - Qwen3.5-9B    对 top-N 复评打分 + 给理由

依赖:  pip install requests numpy python-dotenv
运行:  python job_matcher.py
"""

import re
import json
import time
import sqlite3
import hashlib
import datetime as dt
import logging
from typing import Any

import requests
import numpy as np
from numpy.typing import NDArray

from config import (
    ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY, ADZUNA_PAGES,
    WHAT_EXCLUDE, MAX_DAYS_OLD, SEARCHES,
    LMSTUDIO_BASE, EMBED_MODEL, LLM_MODEL,
    USE_EMBEDDING, USE_LLM_SCORING, TOP_N_FOR_LLM, SCORE_THRESHOLD,
    PROFILE, SENIOR_TERMS, JUNIOR_TERMS, SENIOR_PENALTY, JUNIOR_BONUS,
    DB_PATH, DIGEST_PATH, LLM_FLAG_CAPS,
)

# === [接线 1/3] 顶部导入确定性硬筛层 ===
from .hard_filter import Flag, scan_disqualifiers, apply_flags

# Gmail 解析模块(OAuth + LLM 抽取),见 sources/gmail_alerts.py
try:
    from sources.gmail_alerts import fetch_gmail_alerts
except Exception as _e:
    def fetch_gmail_alerts(_err: str = str(_e)) -> list[dict]:
        logger.warning(f"[gmail] 模块未就绪,跳过: {_err}")
        return []

logger = logging.getLogger(__name__)


# ============================================================
# 2. 数据库(存储 + 去重)
# ============================================================

def db_init() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            title TEXT, company TEXT, location TEXT,
            description TEXT, url TEXT, source TEXT,
            salary TEXT, created TEXT,
            sim REAL, llm_score REAL, llm_reason TEXT,
            first_seen TEXT,
            applied INTEGER DEFAULT 0,
            summary TEXT DEFAULT '',
            status TEXT DEFAULT '待投',
            applied_date TEXT DEFAULT '',
            note TEXT DEFAULT '',
            flags TEXT DEFAULT ''
        )""")
    # 旧库自动迁移:缺列就补
    cols = [r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()]
    for col, ddl in [
        ("applied",      "applied INTEGER DEFAULT 0"),
        ("summary",      "summary TEXT DEFAULT ''"),
        ("status",       "status TEXT DEFAULT '待投'"),
        ("applied_date", "applied_date TEXT DEFAULT ''"),
        ("note",         "note TEXT DEFAULT ''"),
        ("flags",        "flags TEXT DEFAULT ''"),   # === [接线 2/3] 新增 flags 列 ===
    ]:
        if col not in cols:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
    con.commit()
    return con


def job_hash(title: str, company: str) -> str:
    raw = f"{title}|{company}".lower().strip()
    raw = re.sub(r"\s+", " ", raw)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 3. 采集:Adzuna
# ============================================================

def _get_with_retry(url: str, params: dict, tries: int = 4, base_wait: int = 2) -> requests.Response:
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 503:
                raise requests.HTTPError("503", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt < tries:
                wait = base_wait * attempt
                logger.warning(f"[adzuna] 第 {attempt} 次失败({e}),{wait}s 后重试…")
                time.sleep(wait)
    raise last


def fetch_adzuna() -> list[dict]:
    jobs = []
    for s in SEARCHES:
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
            params.update(s)
            try:
                r = _get_with_retry(url, params)
                data = r.json()
            except Exception as e:
                logger.warning(f"[adzuna] 查询失败(已重试) {s} p{page}: {e}")
                break
            results = data.get("results", [])
            if not results:
                break
            for j in results:
                jobs.append({
                    "title":       j.get("title", "").strip(),
                    "company":     (j.get("company") or {}).get("display_name", ""),
                    "location":    (j.get("location") or {}).get("display_name", ""),
                    "description": re.sub(r"\s+", " ", j.get("description", "")),
                    "url":         j.get("redirect_url", ""),
                    "source":      "adzuna",
                    "salary":      _fmt_salary(j),
                    "created":     j.get("created", ""),
                })
    logger.info(f"[adzuna] 抓到 {len(jobs)} 条")
    return jobs


def _fmt_salary(j: dict) -> str:
    lo, hi = j.get("salary_min"), j.get("salary_max")
    if lo and hi:
        return f"${int(lo):,}-${int(hi):,}"
    return ""


# ============================================================
# 5. 匹配层:本地 embedding 相似度
# ============================================================

def embed_batch(texts: list[str]) -> list[NDArray[np.float32]]:
    r = requests.post(f"{LMSTUDIO_BASE}/embeddings",
                      json={"model": EMBED_MODEL, "input": [t[:6000] for t in texts]},
                      timeout=120)
    r.raise_for_status()
    return [np.array(d["embedding"], dtype=np.float32) for d in r.json()["data"]]


def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def seniority_adjust(sim: float, title: str) -> float:
    t = (title or "").lower()
    adj = sim
    if any(term in t for term in SENIOR_TERMS):
        adj -= SENIOR_PENALTY
    if any(term in t for term in JUNIOR_TERMS):
        adj += JUNIOR_BONUS
    return max(0.0, min(1.0, adj))


# ============================================================
# 6. (可选) LLM 复评打分 —— Qwen3.5-9B
# ============================================================

def _extract_json(txt: str) -> str:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"```json|```", "", txt).strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return m.group(0) if m else txt


def llm_review(
    profile: str, job: dict
) -> tuple[float | None, str, str, list[str]]:
    """Call the LLM to score a job. Returns (score, reason, summary, llm_flags)."""
    sys = ("你是一个严格的求职匹配评审,也是信息摘要器。"
           "严格按给定打分细则评分,只输出 JSON,不要任何解释或思考过程。")
    title       = job.get("title", "")
    description = (job.get("description", "") or "")[:2000]
    prompt = f"""候选人画像:
{profile}

待评估职位:
标题:{title}
公司:{job.get('company','')}
地点:{job.get('location','')}
描述:{description}

完成三件事,只输出一个 JSON 对象,不要任何额外文字或思考过程。

== 第一件事:0-100 加权基础分(score)+ 一句话中文理由(reason)==
注意:你只给【加权基础分】,不要自己做任何封顶。封顶由系统按 flags 执行。

【加权基础分(满分 100)】
1. 领域/角色契合(40)—— 是否落在候选人核心领域:XR/AR/VR/MR、空间计算、
   HCI/UX 研究、human-AI / 多模态 / 注视-注意力交互、virtual / embodied agents、
   computer vision(gaze/pose/VLM)、applied ML、研究岗(Research Fellow/Postdoc/
   Research Associate)、Level A-B 教职、digital health 里的 AI/分析。
   完全命中 32-40;相邻可桥接(HCI→HRI、研究→应用 ML、XR→3D/CV)20-31;
   勉强沾边 8-19;基本不沾 0-7。
   注:学术研究岗与产业 R&D / 应用 ML / AI 工程岗,只要领域匹配度相同给分相同;
   不因"是不是学术岗"本身加减分(学术/产业五五开)。
2. 技能/技术栈重叠(25)—— 与画像中"有实证"技能的真实重叠。
   注意:JD 出现关键词 != 重叠,要看职责是否真的需要该技能(反关键词堆砌)。
3. 资历/阶段契合(20)—— 对早期职业者是否友好(学术与产业同等对待):
   学术侧 Level A / Level B / postdoc / 接受应届博士;产业侧 graduate / junior /
   associate / 1-3 年 / early-career R&D。命中任一侧早期信号 16-20;
   未明确但不排斥 10-15;偏资深但未硬卡 4-9。
4. 地点/工作方式(10)。
5. 加成(5)—— 发表文化 / XR-HCI-AI 实验室 / 明确导师 / 博士友好。
【打分纪律】诚实桥接:转移性技能算分但打折,相邻可桥接不得抬到 85+;拿不准往低打。

== 第二件事:硬性淘汰自检(flags,字符串数组)==
仅当 JD 确实命中下列【硬性条件】时,把对应 code 放进 flags(可多选,无则空数组)。
只检测、只列 code,不要因此改 score:
- "clearance"         需安全许可(security clearance / NV1 / NV2 / AGSVA / Baseline)
- "citizenship"       排他性要求澳籍(注意:"citizens, permanent residents…" 是包容性,PR 可投,不算)
- "registration"      需受监管执业注册(AHPRA / registered psychologist / registered nurse 等)
- "degree_field"      essential 要求 PhD in 某非计算/工程领域(如 psychology / neuroscience / biology)
- "wet_lab"           湿实验室/台架技能为核心(stem cell culture / disease modelling / assay 等)
- "clinical_delivery" 临床交付 / allied health / 残障服务为核心(psychological assessment/treatment 等)
- "seniority"         岗位本身是 Senior/Principal/Staff/Lead/Head/Director/Chair/Professor 级,或硬性 8+ 年
- "unrelated_domain"  纯后端/前端/全栈 SWE、devops/SRE、纯数据工程、嵌入式/固件、销售/市场/财务
- "spam_or_test"      疑似测试/占位/无效招聘:标题含 [TEST]/测试字样、JD 空洞无实质职责、
                      明显占位或非真实岗位、内容自相矛盾或像模板填充。命中即视为无效岗。

== 第三件事:要点式摘要(summary,字符串数组)==
中文为主,专业术语保留英文。每条开头标注类别,覆盖 职责/要求/待遇,每类 1-3 条。

== 第四件事:核心学科分类(discipline)==
只看主线职责与选拔标准,忽略通用措辞(innovation/R&D/cutting-edge)和 desirable-only 项。
把该岗位的核心学科分入且仅分入以下三类之一:
- "in_domain"     CS/HCI/XR/HRI/ML-AI/数字健康为核心          -> discipline_multiplier 1.0
- "adjacent"      机器人/传感/人因/通用SWE/comp-design 可桥接   -> discipline_multiplier 0.8
- "out_of_domain" 机械/土木/化工/纯硬件/台架实验/临床心理       -> discipline_multiplier 0.5
判断依据:若岗位核心职责要求候选人拥有其没有且无法诚实桥接的学位或动手经验,则 out_of_domain。

只输出如下 JSON(数组里别放换行):
{{"score": <int 0-100>, "reason": "<一句话中文>", "flags": ["..."], "summary": ["职责:...", "要求:...", "待遇:..."],
  "discipline": {{"core_discipline": "<具体学科名>", "discipline_class": "in_domain|adjacent|out_of_domain",
                  "discipline_multiplier": 1.0, "discipline_reason": "<≤30字中文>"}}}}
"""
    try:
        r = requests.post(f"{LMSTUDIO_BASE}/chat/completions",
                          json={"model": LLM_MODEL,
                                "messages": [{"role": "system", "content": sys},
                                             {"role": "user", "content": prompt}],
                                "temperature": 0.2,
                                "stream": False},
                          timeout=240)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        obj = json.loads(_extract_json(txt))
        sm = obj.get("summary", "")
        if isinstance(sm, list):
            sm = "\n".join(f"- {str(x).strip()}" for x in sm if str(x).strip())
        else:
            sm = str(sm)
        raw_score = obj.get("score", 0)
        try:
            score = float(str(raw_score).split("/")[0].strip())
        except Exception:
            score = 0.0
        llm_flags = obj.get("flags", []) or []
        if not isinstance(llm_flags, list):
            llm_flags = []

        # Discipline multiplier — LLM classifies, Python applies (CLAUDE.md red line)
        disc      = obj.get("discipline") or {}
        disc_cls  = disc.get("discipline_class", "in_domain")
        disc_mult = disc.get("discipline_multiplier", 1.0)
        if disc_mult not in (1.0, 0.8, 0.5):   # clamp to legal values
            disc_mult = 1.0
        disc_reason = disc.get("discipline_reason", "")
        if disc_mult < 1.0 and score is not None:
            score = round(score * disc_mult)
        disc_flag: Flag = {
            "code": "discipline",
            "label": f"学科:{disc_cls}×{disc_mult}",
            "cap": 100,
            "severity": "warn",
            "evidence": disc_reason or f"({disc_cls})",
        }
        return score, str(obj.get("reason", "")), sm, llm_flags + [disc_flag]
    except Exception as e:
        return None, f"(LLM 打分失败: {e})", "", []


def llm_flags_to_objs(codes: list) -> list[Flag]:
    """Convert LLM-returned flag codes (or pre-built Flag dicts) to Flag dicts."""
    out = []
    for c in codes:
        if isinstance(c, dict):          # pre-built Flag (e.g. discipline), pass through
            out.append(c)
        elif c in LLM_FLAG_CAPS:
            cap = LLM_FLAG_CAPS[c]
            out.append({"code": c, "label": f"LLM:{c}", "cap": cap,
                        "severity": "knockout" if cap <= 5 else "warn",
                        "evidence": "(LLM 自检)"})
    return out


# ============================================================
# 7. 主流程
# ============================================================

def main(progress: Any = None) -> None:
    def _p(frac: float, msg: str) -> None:
        logger.info(msg)
        if progress:
            try: progress(frac, msg)
            except Exception: pass

    con = db_init()
    _p(0.05, "正在抓取 Adzuna + Gmail...")
    raw_jobs = fetch_adzuna() + fetch_gmail_alerts()

    # 去重
    new_jobs = []
    seen = set()
    for j in raw_jobs:
        if not j["title"]:
            continue
        jid = job_hash(j["title"], j["company"])
        if jid in seen:
            continue
        if con.execute("SELECT 1 FROM jobs WHERE id=?", (jid,)).fetchone():
            continue
        seen.add(jid)
        j["id"] = jid
        new_jobs.append(j)
    _p(0.12, f"去重后新职位 {len(new_jobs)} 条")
    if not new_jobs:
        logger.info("没有新职位。"); return

    # === [接线 3/3] 确定性硬筛:去重之后、embedding 之前 ===
    # knockout 的不浪费 embedding/LLM 算力,直接定分;非 knockout 的挂上 flags 往下走。
    gated, knocked = [], []
    for j in new_jobs:
        j["flags"] = scan_disqualifiers(j["title"], j["description"])
        j["sim"] = j["sim_raw"] = 0.0
        j["llm_score"], j["llm_reason"], j["summary"] = None, "", ""
        if any(f["severity"] == "knockout" for f in j["flags"]):
            j["llm_score"], _ = apply_flags(0, j["flags"])   # = 5
            j["llm_reason"] = "硬性淘汰:" + ";".join(
                f["label"] for f in j["flags"] if f["severity"] == "knockout")
            knocked.append(j)
        else:
            gated.append(j)
    new_jobs = gated
    _p(0.15, f"硬筛淘汰 {len(knocked)} 条,进入打分 {len(new_jobs)} 条")

    # embedding 初筛(可关)
    if USE_EMBEDDING and new_jobs:
        pvec = embed_batch([PROFILE])[0]
        dim = len(pvec)
        texts = [f"{j['title']} at {j['company']}. {j['description']}" for j in new_jobs]
        BATCH = 32
        vecs = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i+BATCH]
            try:
                vecs.extend(embed_batch(chunk))
            except Exception as e:
                logger.warning(f"[embed] 批次 {i} 失败: {e}")
                vecs.extend([np.zeros(dim, dtype=np.float32)] * len(chunk))
            done = min(i + BATCH, len(texts))
            _p(0.15 + 0.45 * done / len(texts), f"嵌入匹配 {done}/{len(texts)}")
        for j, v in zip(new_jobs, vecs):
            raw = cosine(pvec, v)
            j["sim_raw"] = raw
            j["sim"] = seniority_adjust(raw, j["title"])
        new_jobs = [j for j in new_jobs if j["sim"] >= SCORE_THRESHOLD]
        new_jobs.sort(key=lambda x: x["sim"], reverse=True)
        logger.info(f"[embed] 过阈值({SCORE_THRESHOLD}) {len(new_jobs)} 条")
    elif not USE_EMBEDDING:
        for j in new_jobs:
            j["sim"] = 0.0
        logger.info(f"[skip-embed] 跳过 embedding,保留全部 {len(new_jobs)} 条")

    # LLM 复评(只评 top-N);评完用 flags 统一封顶
    if USE_LLM_SCORING:
        n = min(TOP_N_FOR_LLM, len(new_jobs))
        for idx, j in enumerate(new_jobs[:n], 1):
            score, reason, summary, llm_flags = llm_review(PROFILE, j)
            # 合并:hard_filter 正则 flags + LLM 自检 flags
            j["flags"] = (j.get("flags") or []) + llm_flags_to_objs(llm_flags)
            if score is not None and j["flags"]:
                score, st = apply_flags(score, j["flags"])
                if st != "ok":
                    reason = f"[{st}] " + reason
            j["llm_score"], j["llm_reason"], j["summary"] = score, reason, summary
            logger.info(f"[llm] 复评 {idx}/{n}: {j['title'][:40]} -> {j['llm_score']}")
            _p(0.6 + 0.35 * idx / n, f"LLM 复评 {idx}/{n}: {j['title'][:30]}")
        new_jobs.sort(
            key=lambda x: x["llm_score"] if x["llm_score"] is not None else x["sim"] * 100,
            reverse=True)

    # 落库 + digest(knockout 那批也一并入库,便于去重与审计)
    now = dt.datetime.now().isoformat(timespec="seconds")
    all_jobs = new_jobs + knocked
    for j in all_jobs:
        st = "DISQUALIFIED" if any(
            f["severity"] == "knockout" for f in (j.get("flags") or [])) else "待投"
        con.execute("""INSERT OR REPLACE INTO jobs
            (id,title,company,location,description,url,source,salary,created,
             sim,llm_score,llm_reason,first_seen,summary,flags,
             applied,status,applied_date,note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    COALESCE((SELECT applied      FROM jobs WHERE id=?),0),
                    COALESCE((SELECT status       FROM jobs WHERE id=?),?),
                    COALESCE((SELECT applied_date FROM jobs WHERE id=?),''),
                    COALESCE((SELECT note         FROM jobs WHERE id=?),''))""",
            (j["id"], j["title"], j["company"], j["location"], j["description"],
             j["url"], j["source"], j["salary"], j["created"],
             j["sim"], j["llm_score"], j["llm_reason"], now,
             j.get("summary", ""),
             json.dumps(j.get("flags", []), ensure_ascii=False),
             j["id"], j["id"], st, j["id"], j["id"]))
    con.commit()
    write_digest(new_jobs)
    logger.info(f"[done] 入库 {len(all_jobs)} 条(其中硬淘汰 {len(knocked)}) -> {DIGEST_PATH}")
    _p(1.0, f"完成!打分 {len(new_jobs)} 条,硬淘汰 {len(knocked)} 条")


def write_digest(jobs: list[dict]) -> None:
    lines = [f"# 职位匹配 digest — {dt.date.today()}\n"]
    for i, j in enumerate(jobs, 1):
        score = f"{j['llm_score']:.0f}" if j["llm_score"] is not None else f"~{j['sim']*100:.0f}"
        lines.append(f"## {i}. {j['title']} — {j['company']}  [{score}/100]")
        meta = " · ".join(x for x in [j["location"], j["salary"], j["source"]] if x)
        if meta:            lines.append(f"_{meta}_")
        if j["llm_reason"]: lines.append(f"> {j['llm_reason']}")
        warn = [f for f in (j.get("flags") or []) if f["severity"] == "warn"]
        if warn:
            lines.append("⚠ " + "; ".join(f"{f['label']}(cap{f['cap']})" for f in warn))
        if j["url"]:        lines.append(f"[职位链接]({j['url']})")
        lines.append("")
    with open(DIGEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

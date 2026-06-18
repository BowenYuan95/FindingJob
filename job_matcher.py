#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
job_matcher.py — 个人求职聚合 + 本地匹配排序工具

数据来源策略(按合规性 / 稳定性排序):
  1. Adzuna API      —— 官方免费 API,聚合澳洲多家招聘网站(含大量 Seek/Indeed 转载),首选
  2. 邮件 Job Alert  —— 在 Seek/Indeed/LinkedIn 设置职位提醒,解析自己收件箱(完全合规)
  3. (可选) Playwright 抓取 —— 仅用于无 API 的站点,注意 ToS 风险,本骨架未启用

匹配层(用你本地的 LM Studio):
  - Nomic embeddings 对 [你的画像] vs [职位描述] 做余弦相似度 -> 初筛排序
  - 可选:top-N 交给本地 LLM (Qwen3) 打分 + 给理由

依赖:  pip install requests numpy
运行:  python job_matcher.py
"""

import os
import re
import json
import sqlite3
import hashlib
import datetime as dt

import requests
import numpy as np

# ============================================================
# 1. 配置区
# ============================================================
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY")

# --- Adzuna ---  在 https://developer.adzuna.com/ 免费注册拿 app_id / app_key

ADZUNA_COUNTRY = "au"          # 澳洲
ADZUNA_PAGES   = 3             # 每个查询抓多少页(每页最多 50 条)

# 你的搜索关键词(尽量贴合你的方向)+ 地点
SEARCHES = [
    {"what": "research fellow human computer interaction", "where": "Australia"},
    {"what": "postdoc extended reality",                   "where": "Australia"},
    {"what": "HRI robotics research",                      "where": "Melbourne"},
    {"what": "lecturer HCI",                               "where": "Australia"},
]

# --- LM Studio (OpenAI 兼容本地端点) ---
LMSTUDIO_BASE   = os.environ.get("LMSTUDIO_BASE", "http://localhost:1234/v1")
EMBED_MODEL     = "text-embedding-nomic-embed-text-v1.5"   # 按你 LM Studio 里的实际名字改
LLM_MODEL       = "qwen3-vl-8b"                            # 同上
USE_LLM_SCORING = True        # True = 对 top-N 用 LLM 复评打分;False = 只用 embedding
TOP_N_FOR_LLM   = 15          # embedding 初筛后,送多少条给 LLM 复评

# --- 你的画像:把简历核心 / selection criteria 粘进来,越具体匹配越准 ---
PROFILE = """
PhD in Computer Science (HCI / Extended & Mixed Reality). Research areas:
gaze and spatial tracking, multimodal interaction, virtual agents, adaptive
intelligent systems, user-centred evaluation. Publications at ISMAR, CHI,
IEEE VR, IJHCS, SIGGRAPH Asia. Strong Unity / Meta Quest / VR development,
local RAG and embedding pipelines, teaching experience (data structures,
cloud computing, design thinking). Seeking Research Fellow / Postdoc / Lecturer
roles in HCI, XR, or Human-Robot Interaction, in Australia (Melbourne/Adelaide)
or remote-friendly. Not interested in: pure industry SWE, sales, unrelated fields.
"""

SCORE_THRESHOLD = 0.45        # 低于此余弦相似度的直接丢弃(按效果调)
DB_PATH         = "jobs.db"
DIGEST_PATH     = "digest.md"


# ============================================================
# 2. 数据库(存储 + 去重)
# ============================================================

def db_init():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,      -- 去重哈希
            title TEXT, company TEXT, location TEXT,
            description TEXT, url TEXT, source TEXT,
            salary TEXT, created TEXT,
            sim REAL, llm_score REAL, llm_reason TEXT,
            first_seen TEXT
        )""")
    con.commit()
    return con

def job_hash(title, company, location):
    raw = f"{title}|{company}|{location}".lower().strip()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 3. 采集:Adzuna
# ============================================================

def fetch_adzuna():
    jobs = []
    for s in SEARCHES:
        for page in range(1, ADZUNA_PAGES + 1):
            url = (f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}")
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": 50,
                "what": s["what"],
                "where": s["where"],
                "content-type": "application/json",
            }
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"[adzuna] 查询失败 {s} p{page}: {e}")
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
    print(f"[adzuna] 抓到 {len(jobs)} 条")
    return jobs

def _fmt_salary(j):
    lo, hi = j.get("salary_min"), j.get("salary_max")
    if lo and hi:
        return f"${int(lo):,}–${int(hi):,}"
    return ""


# ============================================================
# 4. (可选) 采集:解析邮箱里的 Job Alert —— 合规覆盖 Seek/Indeed/LinkedIn
# ============================================================
# 用法:在三个平台开职位提醒;邮件进一个专用文件夹/标签;这里用 IMAP 读取。
# Gmail 需用「应用专用密码」(开两步验证后生成),不要用主密码。

def fetch_email_alerts():
    """返回与 Adzuna 同结构的 job dict 列表。默认关闭,配好凭据再启用。"""
    IMAP_HOST = os.environ.get("IMAP_HOST", "imap.gmail.com")
    IMAP_USER = os.environ.get("IMAP_USER")          # 你的邮箱
    IMAP_PASS = os.environ.get("IMAP_PASS")          # 应用专用密码
    IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "JobAlerts")
    if not (IMAP_USER and IMAP_PASS):
        return []

    import imaplib, email
    from email.header import decode_header
    from html.parser import HTMLParser

    class _Strip(HTMLParser):
        def __init__(self): super().__init__(); self.out = []
        def handle_data(self, d): self.out.append(d)
        def text(self): return " ".join(self.out)

    jobs = []
    M = imaplib.IMAP4_SSL(IMAP_HOST)
    M.login(IMAP_USER, IMAP_PASS)
    M.select(IMAP_FOLDER)
    # 只看最近 7 天
    since = (dt.date.today() - dt.timedelta(days=7)).strftime("%d-%b-%Y")
    _, ids = M.search(None, f'(SINCE {since})')
    for num in ids[0].split():
        _, raw = M.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(raw[0][1])
        sender = msg.get("From", "")
        src = ("seek" if "seek" in sender else
               "indeed" if "indeed" in sender else
               "linkedin" if "linkedin" in sender else "email")
        body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    p = _Strip(); p.feed(part.get_payload(decode=True).decode(errors="ignore"))
                    body = p.text(); break
                except Exception:
                    pass
        # 注:每家提醒邮件版式不同,这里只做最粗的标题抓取。
        # 实战中建议对各家写专门的正则/解析,或干脆把整封邮件正文喂给 LLM 抽结构化。
        for m in re.finditer(r"([A-Z][\w &/\-]{6,60})\s+at\s+([\w &/\-,.]{2,50})", body):
            jobs.append({
                "title": m.group(1).strip(), "company": m.group(2).strip(),
                "location": "", "description": body[:1500],
                "url": "", "source": src, "salary": "", "created": "",
            })
    M.logout()
    print(f"[email] 解析到 {len(jobs)} 条")
    return jobs


# ============================================================
# 5. 匹配层:本地 embedding 相似度
# ============================================================

def embed(text):
    r = requests.post(f"{LMSTUDIO_BASE}/embeddings",
                      json={"model": EMBED_MODEL, "input": text[:6000]},
                      timeout=60)
    r.raise_for_status()
    return np.array(r.json()["data"][0]["embedding"], dtype=np.float32)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ============================================================
# 6. (可选) LLM 复评打分
# ============================================================

def llm_score(profile, job):
    prompt = f"""你是一个求职匹配评审。基于候选人画像,给这个职位打 0-100 分,
并用一句话说明理由(中文)。只输出 JSON: {{"score": <int>, "reason": "<str>"}}。

候选人画像:
{profile}

职位:
标题: {job['title']}
公司: {job['company']}
地点: {job['location']}
描述: {job['description'][:2000]}
"""
    try:
        r = requests.post(f"{LMSTUDIO_BASE}/chat/completions",
                          json={"model": LLM_MODEL,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.2},
                          timeout=120)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        txt = re.sub(r"```json|```", "", txt).strip()
        obj = json.loads(txt)
        return float(obj.get("score", 0)), str(obj.get("reason", ""))
    except Exception as e:
        return None, f"(LLM 打分失败: {e})"


# ============================================================
# 7. 主流程
# ============================================================

def main():
    con = db_init()
    raw_jobs = fetch_adzuna() + fetch_email_alerts()

    # 去重:只处理数据库里没见过的
    new_jobs = []
    for j in raw_jobs:
        jid = job_hash(j["title"], j["company"], j["location"])
        exists = con.execute("SELECT 1 FROM jobs WHERE id=?", (jid,)).fetchone()
        if not exists and j["title"]:
            j["id"] = jid
            new_jobs.append(j)
    print(f"[dedup] 新职位 {len(new_jobs)} 条")
    if not new_jobs:
        print("没有新职位。"); return

    # embedding 初筛
    pvec = embed(PROFILE)
    for j in new_jobs:
        text = f"{j['title']} at {j['company']}. {j['description']}"
        try:
            j["sim"] = cosine(pvec, embed(text))
        except Exception as e:
            j["sim"] = 0.0; print(f"[embed] 失败: {e}")
    new_jobs = [j for j in new_jobs if j["sim"] >= SCORE_THRESHOLD]
    new_jobs.sort(key=lambda x: x["sim"], reverse=True)
    print(f"[embed] 过阈值 {len(new_jobs)} 条")

    # LLM 复评(只评 top-N,省算力)
    for j in new_jobs:
        j["llm_score"], j["llm_reason"] = (None, "")
    if USE_LLM_SCORING:
        for j in new_jobs[:TOP_N_FOR_LLM]:
            j["llm_score"], j["llm_reason"] = llm_score(PROFILE, j)
        # 有 LLM 分就按它排,否则退回 embedding 分
        new_jobs.sort(key=lambda x: (x["llm_score"] or x["sim"] * 100), reverse=True)

    # 落库 + 生成 digest
    now = dt.datetime.now().isoformat(timespec="seconds")
    for j in new_jobs:
        con.execute("""INSERT OR REPLACE INTO jobs VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (j["id"], j["title"], j["company"], j["location"], j["description"],
             j["url"], j["source"], j["salary"], j["created"],
             j["sim"], j["llm_score"], j["llm_reason"], now))
    con.commit()
    write_digest(new_jobs)
    print(f"[done] 写入 {len(new_jobs)} 条 -> {DIGEST_PATH}")


def write_digest(jobs):
    lines = [f"# 职位匹配 digest — {dt.date.today()}\n"]
    for i, j in enumerate(jobs, 1):
        score = f"{j['llm_score']:.0f}" if j["llm_score"] is not None else f"~{j['sim']*100:.0f}"
        lines.append(f"## {i}. {j['title']} — {j['company']}  [{score}/100]")
        meta = " · ".join(x for x in [j["location"], j["salary"], j["source"]] if x)
        if meta:   lines.append(f"_{meta}_")
        if j["llm_reason"]: lines.append(f"> {j['llm_reason']}")
        if j["url"]:        lines.append(f"[职位链接]({j['url']})")
        lines.append("")
    with open(DIGEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

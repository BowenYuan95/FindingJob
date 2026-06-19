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
  - Qwen3.5-9B 对 top-N 复评打分 + 给理由(踢掉"形似神不似"的假阳性)

依赖:  pip install requests numpy python-dotenv
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
from dotenv import load_dotenv

# Gmail 解析模块(OAuth + LLM 抽取),见 gmail_alerts.py
try:
    from gmail_alerts import fetch_gmail_alerts
except Exception as _e:
    def fetch_gmail_alerts():
        print(f'[gmail] 模块未就绪,跳过: {_e}')
        return []

# ============================================================
# 1. 配置区
# ============================================================
load_dotenv()

# --- Adzuna ---  在 https://developer.adzuna.com/ 免费注册拿 app_id / app_key
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = "au"          # 澳洲
ADZUNA_PAGES   = 3             # 每个查询抓多少页(每页最多 50 条)

# 每条搜索可单独带 what / what_phrase / what_exclude / where / title_only 等
SEARCHES = [
    # ---- 学术线(量少,求全;phrase / title 优先)----
    {"what_phrase": "research fellow"},
    {"what_phrase": "research associate"},
    {"what": "postdoctoral"},
    {"what": "postdoc"},                          # 不少岗写 postdoc 不写 postdoctoral
    {"what": "lecturer", "title_only": 1},        # title_only 砍掉正文顺带提一句的噪音

    # ---- XR / 空间计算线(拼写全称,别只靠缩写)----
    {"what_phrase": "augmented reality"},
    {"what_phrase": "virtual reality"},
    {"what_phrase": "mixed reality"},
    {"what_phrase": "spatial computing"},
    {"what": "XR developer"},
    {"what": "unity developer"},

    # ---- AI / ML / CV 线 ----
    {"what": "AI engineer"},                      # 你近期最强匹配类型,原列表没有
    {"what": "machine learning engineer"},
    {"what": "computer vision engineer"},
    {"what": "applied scientist"},
    {"what": "research scientist"},
    {"what": "research engineer"},                # 工业界常见 title,介于 scientist 和 dev 之间

    # ---- HCI / UX / 交互线 ----
    {"what_phrase": "human computer interaction"},
    {"what": "UX researcher"},
    {"what_phrase": "human robot interaction"},   # 你的 HCI→HRI 桥接目标
    {"what_phrase": "virtual agent"},             # 可选:命中虚拟人 / 对话 Agent 岗
]

MAX_DAYS_OLD = 90
# 去掉 clinical(误伤数字健康);marketing 补上,和你的 anti-target 一致
WHAT_EXCLUDE = "sales marketing recruitment nursing finance accounting"

# --- LM Studio (OpenAI 兼容本地端点) ---
LMSTUDIO_BASE   = os.environ.get("LMSTUDIO_BASE", "http://localhost:1234/v1")
EMBED_MODEL     = "text-embedding-nomic-embed-text-v1.5"   # /v1/models 里的 embedding id
LLM_MODEL       = "qwen/qwen3.5-9b"                        # /v1/models 里的 chat id
USE_EMBEDDING   = True        # False = 只跑采集 + 存库 + 出全量清单(不打分)
USE_LLM_SCORING = True        # True = 对 top-N 用 Qwen3.5 复评打分
TOP_N_FOR_LLM   = 50          # embedding 初筛后,送多少条给 LLM 复评

# --- 你的画像:把简历核心 / selection criteria 粘进来,越具体匹配越准 ---
PROFILE = """
# Identity & career stage
Early-career researcher/engineer. PhD in Computer Science (HCI / Extended &
Mixed Reality), thesis on dependency-aware mixed reality guidance, expected
mid-2026. Australian Permanent Resident. Targeting Level A–B equivalents:
Research Fellow, Postdoc, Research Associate, (Associate/Assistant) Lecturer,
and early-career industry R&D / applied-research / AI-ML engineering roles.
NOT seeking Senior/Principal/Lead/Staff/Director/Head/Professorial positions
or roles requiring extensive (8+ yrs) industry seniority.

# Core research & technical strengths (evidence-backed)
- Extended/Augmented/Mixed/Virtual Reality (AR/MR/VR/XR) systems; Diminished
  Reality (visiting researcher, City University of Hong Kong).
- Multimodal interaction: gaze/eye-tracking, head & hand motion, spatial &
  location tracking, speech interaction, biosignals (HRV/PPG, EEG /
  neurophysiological).
- Adaptive, attention-aware & context-aware interactive systems; spatial
  computing; non-linear task guidance via dependency / DAG modelling (STAGE).
- AI-driven interaction: LLM API integration, LLM-powered virtual & embodied
  agents, text embeddings, vision-language models, speech-to-text,
  real-time pose detection / applied ML.
- User-centred study design & evaluation: controlled user studies, usability,
  presence, cognitive load (quantitative + qualitative).
- Engineering: Unity3D, C#, MRTK, Meta Quest / HMD development; Python
  (PyTorch), C++, Java, JavaScript/React/React Native; SQL databases.
- Publications: ACM CHI, IEEE VR, IEEE ISMAR, ACM SIGGRAPH Asia / VRCAI, IJHCS.
- Teaching: data structures, cloud & concurrent programming, intro programming
  (Python), design thinking (incl. transnational delivery in Xi'an).

# Good-fit role types (include)
Research Fellow / Postdoctoral Researcher / Research Associate; Lecturer
(teaching or teaching-research); industry research scientist / applied
research engineer / R&D engineer / AI-ML engineer in: XR/AR/VR/MR, spatial
computing, HCI / UX research, human-AI interaction, virtual & embodied agents,
multimodal & gaze/attention systems, computer vision (gaze, pose, VLM),
human-robot interaction (as an HCI-to-HRI extension), applied/interaction ML,
digital health & immersive technologies, advanced analytics in research or
health contexts.

# Location
Australia — Melbourne (home base), Adelaide, Sydney, Brisbane, Canberra — or
remote / hybrid / relocation-supported within Australia.

# Poor-fit / exclude
- Seniority mismatch: Senior/Principal/Lead/Staff/Director/Head/Chair/Professor
  or postings demanding extensive industry seniority.
- HARD FILTER (PR holder): roles requiring Australian citizenship, security
  clearance (NV1/NV2), AGSVA/Baseline, or defence-sector eligibility.
- Field mismatch: pure backend/web/full-stack SWE, devops/SRE, pure data
  engineering pipelines, embedded/firmware, sales/marketing/finance,
  management consulting, and roles unrelated to HCI/XR/AI/ML/research.
"""

# --- 级别降权:高级岗压后、初级岗轻微加分(在 embedding 分基础上调整)---
SENIOR_TERMS = [          # 命中任一即降权(超出 early-career 范围)
    "senior", "principal", "lead", "head of", "chair",
    "professor", "associate professor", "director", "manager",
]
JUNIOR_TERMS = [          # 命中即轻微加分(贴合 early-career)
    "associate", "assistant", "junior", "graduate", "level a", "early career",
]
SENIOR_PENALTY = 0.15     # 每命中高级词,从相似度里扣多少(0~1 尺度)
JUNIOR_BONUS   = 0.03     # 命中初级词加多少
# 注:澳洲 "Senior Lecturer"/"Senior Research Fellow" 实为中级,这里按你的选择也一并压低

SCORE_THRESHOLD = 0.6         # 低于此余弦相似度的直接丢弃(砍长尾,精筛交给 LLM)
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
            first_seen TEXT,
            applied INTEGER DEFAULT 0,    -- 兼容旧字段(status!=待投 即视为已处理)
            summary TEXT DEFAULT '',      -- LLM 生成的要点式摘要
            status TEXT DEFAULT '待投',    -- 待投 / 已投 / 面试 / 拒 / offer
            applied_date TEXT DEFAULT '', -- 状态变更日期
            note TEXT DEFAULT ''          -- 备注
        )""")
    # 旧库自动迁移:缺列就补
    cols = [r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()]
    for col, ddl in [
        ("applied",      "applied INTEGER DEFAULT 0"),
        ("summary",      "summary TEXT DEFAULT ''"),
        ("status",       "status TEXT DEFAULT '待投'"),
        ("applied_date", "applied_date TEXT DEFAULT ''"),
        ("note",         "note TEXT DEFAULT ''"),
    ]:
        if col not in cols:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
    con.commit()
    return con

def job_hash(title, company, location=None):   # location 不参与,避免 Adzuna 多地点标签导致漏去重
    raw = f"{title}|{company}".lower().strip()
    raw = re.sub(r"\s+", " ", raw)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# 3. 采集:Adzuna
# ============================================================

def _get_with_retry(url, params, tries=4, base_wait=2):
    """带重试的 GET:遇 503/网络抖动退避重试。"""
    import time
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 503:        # 服务临时不可用,值得重试
                raise requests.HTTPError("503", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt < tries:
                wait = base_wait * attempt
                print(f"[adzuna] 第 {attempt} 次失败({e}),{wait}s 后重试…")
                time.sleep(wait)
    raise last


def fetch_adzuna():
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
            params.update(s)   # 合并本条搜索的 what / what_phrase / where 等
            try:
                r = _get_with_retry(url, params)
                data = r.json()
            except Exception as e:
                print(f"[adzuna] 查询失败(已重试) {s} p{page}: {e}")
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
# 5. 匹配层:本地 embedding 相似度
# ============================================================

def embed_batch(texts):
    """一次给多条文本,返回向量列表。"""
    r = requests.post(f"{LMSTUDIO_BASE}/embeddings",
                      json={"model": EMBED_MODEL, "input": [t[:6000] for t in texts]},
                      timeout=120)
    r.raise_for_status()
    return [np.array(d["embedding"], dtype=np.float32) for d in r.json()["data"]]

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def seniority_adjust(sim, title):
    """根据标题里的级别词,对 embedding 相似度做加减(高级压低、初级加分)。"""
    t = (title or "").lower()
    adj = sim
    if any(term in t for term in SENIOR_TERMS):
        adj -= SENIOR_PENALTY
    if any(term in t for term in JUNIOR_TERMS):
        adj += JUNIOR_BONUS
    return max(0.0, min(1.0, adj))   # 夹在 0~1


# ============================================================
# 6. (可选) LLM 复评打分 —— Qwen3.5-9B
# ============================================================

def _extract_json(txt):
    """从模型输出里稳健地抠出 JSON:去掉 <think> 块、代码围栏,取第一个 {...}。"""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"```json|```", "", txt).strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return m.group(0) if m else txt

def llm_review(profile, job):
    """一次调用返回 (score, reason, summary)。加权基础分 + 硬性 caps。"""
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

完成两件事,只输出一个 JSON 对象,不要任何额外文字或思考过程。

== 第一件事:0-100 匹配分(score)+ 一句话中文理由(reason)==

先算【加权基础分】,再套用【硬性上限 caps】,最终 score = min(基础分, 命中的最小 cap)。

【硬性上限 caps —— 命中即封顶,reason 必须点明原因】
- 要求 Australian citizenship / security clearance (NV1/NV2) / AGSVA / Baseline /
  defence eligibility → cap = 5(候选人为 PR,硬性不符)。
- 资历错配:岗位本身是 Senior / Principal / Staff / Lead / Head / Director /
  Chair / Professor 级,或硬性要求 8+ 年业界经验、或以带团队/管理为主
  → cap = 35。判定看「岗位级别与硬性年限」,不要被职责里顺带的
  "lead the project""senior stakeholders" 等措辞误触发。
- 领域完全无关:纯后端/前端/全栈 SWE、devops/SRE、纯数据工程、嵌入式/固件、
  销售/市场/财务/会计、纯临床护理岗 → cap = 30。

【加权基础分(满分 100)】
1. 领域/角色契合(40)—— 工作内容是否落在候选人核心领域:XR/AR/VR/MR、
   空间计算、HCI/UX 研究、human-AI / 多模态 / 注视-注意力交互、virtual /
   embodied agents、computer vision(gaze/pose/VLM)、applied ML、研究岗
   (Research Fellow/Postdoc/Research Associate)、Level A-B 教职、digital
   health 里的 AI/分析。
   完全命中 32-40;相邻可桥接(HCI→HRI、研究→应用 ML、XR→3D/CV)20-31;
   勉强沾边 8-19;基本不沾 0-7。
2. 技能/技术栈重叠(25)—— 与画像中"有实证"技能的真实重叠:Unity/C#/MRTK/
   HMD、Python/PyTorch、gaze/眼动/多模态、LLM API/embeddings/VLM/STT、
   用户研究与评估、SQL。按命中数量与核心程度给分。
   注意:JD 出现关键词 != 重叠,要看职责是否真的需要该技能(反关键词堆砌)。
3. 资历/阶段契合(20)—— 是否对早期研究者/Level A-B 友好:
   明确 early-career / Level A / Level B / 0-3 年 / 接受应届博士 16-20;
   未明确但不排斥 10-15;偏资深但未硬性卡死 4-9。
4. 地点/工作方式(10)—— 墨尔本/阿德莱德/悉尼/布里斯班/堪培拉、全澳、
   remote/hybrid/支持搬迁 8-10;澳洲其他地区 5-7;需海外或不利通勤且无
   remote 0-4。
5. 加成(5)—— 鼓励发表/研究文化、XR/HCI/AI 实验室、明确导师或团队、
   博士友好:酌情 0-5。

【打分纪律】
- 诚实桥接:转移性技能算分但打折,"相邻可桥接"的岗不得硬抬到 85+。
- 拿不准往低打,reason 里点明最主要的 gap。
- 数据标注 / AI trainer / 众包打标 / freelance 计件 / 纯 PhD 奖学金,score <= 30。

== 第二件事:要点式摘要(summary,字符串数组)==
中文为主,专业术语保留英文。每条要点开头标注类别,覆盖 职责 / 要求 / 待遇
三类,每类 1-3 条,没信息可省略该类。

只输出如下 JSON(summary 必须是字符串数组,不要在字符串里放换行):
{{"score": <int 0-100>, "reason": "<一句话中文>", "summary": ["职责:...", "要求:...", "待遇:..."]}}
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
        return score, str(obj.get("reason", "")), sm
    except Exception as e:
        return None, f"(LLM 打分失败: {e})", ""


# ============================================================
# 7. 主流程
# ============================================================

def main(progress=None):
    """progress(frac, msg): frac=0~1 进度, msg=状态文字。供 UI 显示进度条用。"""
    def _p(frac, msg):
        print(msg)
        if progress:
            try: progress(frac, msg)
            except Exception: pass

    con = db_init()
    _p(0.05, "正在抓取 Adzuna + Gmail...")
    raw_jobs = fetch_adzuna() + fetch_gmail_alerts()

    # 去重:既要跟数据库比对,也要挡住"本次批内"的重复
    new_jobs = []
    seen = set()                       # 本次运行内已收录的 hash
    for j in raw_jobs:
        if not j["title"]:
            continue
        jid = job_hash(j["title"], j["company"], j["location"])
        if jid in seen:                # 批内重复(同岗多地点标签等)
            continue
        if con.execute("SELECT 1 FROM jobs WHERE id=?", (jid,)).fetchone():
            continue                   # 数据库里已存在
        seen.add(jid)
        j["id"] = jid
        new_jobs.append(j)
    _p(0.15, f"去重后新职位 {len(new_jobs)} 条")
    if not new_jobs:
        print("没有新职位。"); return

    # embedding 初筛(可关)
    if USE_EMBEDDING:
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
                print(f"[embed] 批次 {i} 失败: {e}")
                vecs.extend([np.zeros(dim, dtype=np.float32)] * len(chunk))
            done = min(i + BATCH, len(texts))
            _p(0.15 + 0.45 * done / len(texts), f"嵌入匹配 {done}/{len(texts)}")
        for j, v in zip(new_jobs, vecs):
            raw = cosine(pvec, v)
            j["sim_raw"] = raw                              # 保留原始分,便于排查
            j["sim"] = seniority_adjust(raw, j["title"])    # 级别降权后的分
        new_jobs = [j for j in new_jobs if j["sim"] >= SCORE_THRESHOLD]
        new_jobs.sort(key=lambda x: x["sim"], reverse=True)
        print(f"[embed] 过阈值({SCORE_THRESHOLD}) {len(new_jobs)} 条")
    else:
        for j in new_jobs:
            j["sim"] = 0.0
        print(f"[skip-embed] 跳过 embedding,保留全部 {len(new_jobs)} 条")

    # LLM 复评(只评 top-N,省算力)
    for j in new_jobs:
        j["llm_score"], j["llm_reason"], j["summary"] = None, "", ""
    if USE_LLM_SCORING:
        n = min(TOP_N_FOR_LLM, len(new_jobs))
        for idx, j in enumerate(new_jobs[:n], 1):
            j["llm_score"], j["llm_reason"], j["summary"] = llm_review(PROFILE, j)
            print(f"[llm] 复评 {idx}/{n}: {j['title'][:40]} -> {j['llm_score']}")
            _p(0.6 + 0.35 * idx / n, f"LLM 复评 {idx}/{n}: {j['title'][:30]}")
        # 有 LLM 分按它排,没有的退回 embedding 分
        new_jobs.sort(
            key=lambda x: x["llm_score"] if x["llm_score"] is not None else x["sim"] * 100,
            reverse=True)

    # 落库 + 生成 digest
    now = dt.datetime.now().isoformat(timespec="seconds")
    for j in new_jobs:
        # 只插入新职位(旧职位在去重阶段已被跳过),applied 默认 0
        con.execute("""INSERT OR REPLACE INTO jobs
            (id,title,company,location,description,url,source,salary,created,
             sim,llm_score,llm_reason,first_seen,summary,
             applied,status,applied_date,note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    COALESCE((SELECT applied      FROM jobs WHERE id=?),0),
                    COALESCE((SELECT status       FROM jobs WHERE id=?),'待投'),
                    COALESCE((SELECT applied_date FROM jobs WHERE id=?),''),
                    COALESCE((SELECT note         FROM jobs WHERE id=?),''))""",
            (j["id"], j["title"], j["company"], j["location"], j["description"],
             j["url"], j["source"], j["salary"], j["created"],
             j["sim"], j["llm_score"], j["llm_reason"], now,
             j.get("summary", ""),
             j["id"], j["id"], j["id"], j["id"]))
    con.commit()
    write_digest(new_jobs)
    print(f"[done] 写入 {len(new_jobs)} 条 -> {DIGEST_PATH}")
    _p(1.0, f"完成!写入 {len(new_jobs)} 条")


def write_digest(jobs):
    lines = [f"# 职位匹配 digest — {dt.date.today()}\n"]
    for i, j in enumerate(jobs, 1):
        score = f"{j['llm_score']:.0f}" if j["llm_score"] is not None else f"~{j['sim']*100:.0f}"
        lines.append(f"## {i}. {j['title']} — {j['company']}  [{score}/100]")
        meta = " · ".join(x for x in [j["location"], j["salary"], j["source"]] if x)
        if meta:            lines.append(f"_{meta}_")
        if j["llm_reason"]: lines.append(f"> {j['llm_reason']}")
        if j["url"]:        lines.append(f"[职位链接]({j['url']})")
        lines.append("")
    with open(DIGEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
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
import os
import json
import time
import sqlite3
import hashlib
import datetime as dt
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from infrastructure.database import initialize_database
from infrastructure.job_repository import JobRepository
from infrastructure.lmstudio import LM_CLIENT
from .job_urls import normalize_job_url

from config import (
    ADZUNA_APP_ID, ADZUNA_APP_KEY,
    LMSTUDIO_BASE, EMBED_MODEL, LLM_MODEL,
    USE_EMBEDDING, USE_LLM_SCORING, TOP_N_FOR_LLM, SCORE_THRESHOLD,
    PROFILE, SENIOR_TERMS, JUNIOR_TERMS, SENIOR_PENALTY, JUNIOR_BONUS,
    DB_PATH, DIGEST_PATH, LLM_FLAG_CAPS,
)

# === [接线 1/3] 顶部导入确定性硬筛层 ===
from .hard_filter import Flag, scan_disqualifiers, apply_flags

# Gmail 解析模块(OAuth + LLM 抽取),见 sources/gmail_alerts.py
GMAIL_IMPORT_ERROR: str | None = None
try:
    from sources.gmail_alerts import fetch_gmail_alerts
except Exception as _e:
    GMAIL_IMPORT_ERROR = str(_e)
    def fetch_gmail_alerts(_err: str = str(_e)) -> list[dict]:
        logger.warning(f"[gmail] 模块未就绪,跳过: {_err}")
        return []

logger = logging.getLogger(__name__)


# ============================================================
# 2. 数据库(存储 + 去重)
# ============================================================

def db_init() -> sqlite3.Connection:
    return initialize_database(DB_PATH)


def job_hash(
    title: str, company: str, source: str = "", source_id: str = "",
) -> str:
    if source and source_id:
        raw = f"{source}|{source_id}".lower().strip()
    else:
        raw = f"{title}|{company}".lower().strip()
    raw = re.sub(r"\s+", " ", raw)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _semantic_key(job: dict) -> tuple[str, str, str]:
    return tuple(
        re.sub(r"\s+", " ", str(job.get(field, "") or "").strip().lower())
        for field in ("title", "company", "location")
    )


def find_existing_job(repo: JobRepository, job: dict) -> str | None:
    """Find native-ID or legacy title/company records without duplicating old DB rows."""
    source = str(job.get("source") or "")
    source_id = str(job.get("source_id") or "")
    canonical_id = job_hash(
        job.get("title", ""), job.get("company", ""), source, source_id
    )
    legacy_id = job_hash(job.get("title", ""), job.get("company", ""))
    return repo.find_existing(
        source=source,
        source_id=source_id,
        canonical_id=canonical_id,
        legacy_id=legacy_id,
    )


# ============================================================
# 3. 采集:Adzuna
# ============================================================

ADZUNA_IMPORT_ERROR: str | None = None
try:
    from sources.adzuna_search import fetch_adzuna
except Exception as _e:
    ADZUNA_IMPORT_ERROR = str(_e)
    fetch_adzuna = None


def validate_runtime() -> tuple[list[str], list[str]]:
    """Return (blocking errors, non-blocking warnings) before starting the pipeline."""
    errors: list[str] = []
    warnings: list[str] = []

    if ADZUNA_IMPORT_ERROR or not callable(fetch_adzuna):
        errors.append(f"Adzuna 模块未就绪: {ADZUNA_IMPORT_ERROR or '不可调用'}")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        errors.append("缺少 ADZUNA_APP_ID / ADZUNA_APP_KEY")
    if GMAIL_IMPORT_ERROR:
        warnings.append(f"Gmail 模块未就绪,本轮将跳过: {GMAIL_IMPORT_ERROR}")

    db_parent = os.path.dirname(os.path.abspath(DB_PATH)) or os.getcwd()
    if not os.path.isdir(db_parent) or not os.access(db_parent, os.W_OK):
        errors.append(f"数据库目录不可写: {db_parent}")

    required_models = set()
    if USE_EMBEDDING:
        required_models.add(EMBED_MODEL)
    if USE_LLM_SCORING:
        required_models.add(LLM_MODEL)
    if required_models:
        try:
            loaded = LM_CLIENT.loaded_models(timeout=5)
            missing = sorted(required_models - loaded)
            if missing:
                errors.append("LM Studio 模型未加载: " + ", ".join(missing))
        except Exception as e:
            errors.append(f"LM Studio 未就绪({LMSTUDIO_BASE}): {e}")

    return errors, warnings


# ============================================================
# 5. 匹配层:本地 embedding 相似度
# ============================================================

def embed_batch(texts: list[str]) -> list[NDArray[np.float32]]:
    vectors = LM_CLIENT.embeddings(texts, EMBED_MODEL)
    return [np.array(vector, dtype=np.float32) for vector in vectors]


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
   未明确但不排斥 10-15;偏资深但未硬卡或博士奖学金项目：4-9。
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
- "phd_scholarship"   该岗位本质是【攻读博士学位的奖学金/招生名额】,而非受聘研究岗:
                      PhD scholarship / (funded) PhD position / doctoral candidate /
                      博士研究生招生 / 提供 stipend 资助去攻读 PhD。
                      关键区分:候选人是【入学去读】博士,而不是【已持博士受聘】做研究。
                      Postdoc / Research Fellow / Research Associate 这类"已获博士后受聘"
                      的岗位不算,不要打此 flag。
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
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = LM_CLIENT.chat_completion(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=240,
            )
            msg = response["choices"][0]["message"]
            txt = msg.get("content") or msg.get("reasoning_content") or ""
            obj = json.loads(_extract_json(txt))
            sm = obj.get("summary", "")
            if isinstance(sm, list):
                sm = "\n".join(f"- {str(x).strip()}" for x in sm if str(x).strip())
            else:
                sm = str(sm)

            raw_score = obj.get("score")
            if raw_score is None:
                raise ValueError("response missing score")
            score = float(str(raw_score).split("/")[0].strip())
            if not 0 <= score <= 100:
                raise ValueError(f"score out of range: {score}")

            llm_flags = obj.get("flags", []) or []
            if not isinstance(llm_flags, list):
                raise ValueError("flags must be an array")

            # LLM only classifies. Python owns the class -> multiplier mapping.
            disc = obj.get("discipline") or {}
            disc_cls = disc.get("discipline_class")
            disc_multipliers = {
                "in_domain": 1.0,
                "adjacent": 0.8,
                "out_of_domain": 0.5,
            }
            if disc_cls not in disc_multipliers:
                raise ValueError(f"invalid discipline_class: {disc_cls!r}")
            disc_mult = disc_multipliers[disc_cls]
            disc_reason = disc.get("discipline_reason", "")
            score = round(score * disc_mult)

            # PhD scholarship(攻读博士的奖学金/招生名额,非受聘研究岗):硬性封顶 40。
            # 与 discipline 同理:LLM 只识别 flag,封顶逻辑由 Python 拥有。
            # 用 min() 而非强制赋值:已经低于 40 的(如 out_of_domain 折算后)不被抬高。
            PHD_SCHOLARSHIP_CAP = 40
            extra_flags: list[Flag] = []
            if "phd_scholarship" in llm_flags:
                llm_flags = [f for f in llm_flags if f != "phd_scholarship"]
                score = min(score, PHD_SCHOLARSHIP_CAP)
                extra_flags.append({
                    "code": "phd_scholarship",
                    "label": f"PhD scholarship 封顶 {PHD_SCHOLARSHIP_CAP}",
                    "cap": PHD_SCHOLARSHIP_CAP,
                    "severity": "warn",
                    "evidence": "攻读博士的奖学金/招生名额,非受聘研究岗",
                })

            disc_flag: Flag = {
                "code": "discipline",
                "label": f"学科:{disc_cls}×{disc_mult}",
                "cap": 100,
                "severity": "warn",
                "evidence": disc_reason or f"({disc_cls})",
            }
            return score, str(obj.get("reason", "")), sm, llm_flags + extra_flags + [disc_flag]
        except Exception as e:
            last_error = e
            logger.warning(f"[llm] 第 {attempt}/3 次调用失败: {e}")
            if isinstance(e, RuntimeError) and "连续失败" in str(e):
                break
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    return None, f"(LLM 打分失败,已重试 3 次: {last_error})", "", []    profile: str, job: dict
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
   未明确但不排斥 10-15;偏资深但未硬卡或博士奖学金项目：4-9。
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
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = LM_CLIENT.chat_completion(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=240,
            )
            msg = response["choices"][0]["message"]
            txt = msg.get("content") or msg.get("reasoning_content") or ""
            obj = json.loads(_extract_json(txt))
            sm = obj.get("summary", "")
            if isinstance(sm, list):
                sm = "\n".join(f"- {str(x).strip()}" for x in sm if str(x).strip())
            else:
                sm = str(sm)

            raw_score = obj.get("score")
            if raw_score is None:
                raise ValueError("response missing score")
            score = float(str(raw_score).split("/")[0].strip())
            if not 0 <= score <= 100:
                raise ValueError(f"score out of range: {score}")

            llm_flags = obj.get("flags", []) or []
            if not isinstance(llm_flags, list):
                raise ValueError("flags must be an array")

            # LLM only classifies. Python owns the class -> multiplier mapping.
            disc = obj.get("discipline") or {}
            disc_cls = disc.get("discipline_class")
            disc_multipliers = {
                "in_domain": 1.0,
                "adjacent": 0.8,
                "out_of_domain": 0.5,
            }
            if disc_cls not in disc_multipliers:
                raise ValueError(f"invalid discipline_class: {disc_cls!r}")
            disc_mult = disc_multipliers[disc_cls]
            disc_reason = disc.get("discipline_reason", "")
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
            last_error = e
            logger.warning(f"[llm] 第 {attempt}/3 次调用失败: {e}")
            if isinstance(e, RuntimeError) and "连续失败" in str(e):
                break
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    return None, f"(LLM 打分失败,已重试 3 次: {last_error})", "", []


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

    errors, warnings = validate_runtime()
    for warning in warnings:
        logger.warning(f"[ready] {warning}")
    if errors:
        raise RuntimeError("运行环境未就绪:\n- " + "\n- ".join(errors))

    con = db_init()
    repo = JobRepository(con)
    _p(0.05, "正在抓取 Adzuna + Gmail...")
    raw_jobs = fetch_adzuna() + fetch_gmail_alerts()

    # 去重:Adzuna 优先 native source_id;无 native ID 的来源回退 title+company。
    new_jobs = []
    seen_ids: set[str] = set()
    seen_semantic: set[tuple[str, str, str]] = set()
    for j in raw_jobs:
        if not j["title"]:
            continue
        source = str(j.get("source") or "")
        source_id = str(j.get("source_id") or "")
        jid = job_hash(j["title"], j["company"], source, source_id)
        semantic = _semantic_key(j)
        if jid in seen_ids or (not source_id and semantic in seen_semantic):
            continue
        existing_id = find_existing_job(repo, j)
        if existing_id:
            normalized_url = normalize_job_url(j.get("url"))
            if normalized_url:
                j["url"] = normalized_url
                repo.refresh_source_metadata(
                    existing_id, j, dt.datetime.now().isoformat(timespec="seconds")
                )
            state = repo.pipeline_state(existing_id)
            # Transient embedding failures are eligible for a later ingestion retry.
            if state != "EMBEDDING_FAILED":
                continue
            jid = existing_id
        seen_ids.add(jid)
        if not source_id:
            seen_semantic.add(semantic)
        j["id"] = jid
        new_jobs.append(j)
    _p(0.12, f"去重后新职位 {len(new_jobs)} 条")
    if not new_jobs:
        logger.info("没有新职位。")
        con.commit()  # persist legacy source_id bridges created during dedup
        con.close()
        return

    ingested_jobs = list(new_jobs)

    # === [接线 3/3] 确定性硬筛:去重之后、embedding 之前 ===
    # knockout 的不浪费 embedding/LLM 算力,直接定分;非 knockout 的挂上 flags 往下走。
    gated, knocked = [], []
    for j in new_jobs:
        j["flags"] = scan_disqualifiers(j["title"], j["description"])
        j["sim"] = j["sim_raw"] = 0.0
        j["llm_score"], j["llm_reason"], j["summary"] = None, "", ""
        j["pipeline_state"], j["score_attempts"], j["last_error"] = "INGESTED", 0, ""
        if any(f["severity"] == "knockout" for f in j["flags"]):
            # Knockout uses a visible sentinel score of 5. min(base, cap) with
            # base=0 would incorrectly store 0 instead.
            j["llm_score"], _ = apply_flags(5, j["flags"])
            j["llm_reason"] = "硬性淘汰:" + ";".join(
                f["label"] for f in j["flags"] if f["severity"] == "knockout")
            j["pipeline_state"] = "DISQUALIFIED"
            knocked.append(j)
        else:
            gated.append(j)
    new_jobs = gated
    _p(0.15, f"硬筛淘汰 {len(knocked)} 条,进入打分 {len(new_jobs)} 条")

    # embedding 初筛(可关)
    if USE_EMBEDDING and new_jobs:
        try:
            pvec = embed_batch([PROFILE])[0]
            texts = [f"{j['title']} at {j['company']}. {j['description']}" for j in new_jobs]
            BATCH = 32
            results: list[tuple[NDArray[np.float32] | None, str]] = []
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i+BATCH]
                try:
                    results.extend((vector, "") for vector in embed_batch(chunk))
                except Exception as e:
                    error = str(e)
                    logger.warning(f"[embed] 批次 {i} 失败: {error}")
                    results.extend((None, error) for _ in chunk)
                done = min(i + BATCH, len(texts))
                _p(0.15 + 0.45 * done / len(texts), f"嵌入匹配 {done}/{len(texts)}")

            accepted = []
            for j, (vector, error) in zip(new_jobs, results):
                if vector is None:
                    j["pipeline_state"] = "EMBEDDING_FAILED"
                    j["last_error"] = error
                    continue
                raw = cosine(pvec, vector)
                j["sim_raw"] = raw
                j["sim"] = seniority_adjust(raw, j["title"])
                if j["sim"] >= SCORE_THRESHOLD:
                    j["pipeline_state"] = "READY_FOR_LLM"
                    accepted.append(j)
                else:
                    j["pipeline_state"] = "EMBEDDING_REJECTED"
            new_jobs = accepted
        except Exception as e:
            error = str(e)
            logger.warning(f"[embed] 画像 embedding 失败,本轮全部延后: {error}")
            for j in new_jobs:
                j["pipeline_state"] = "EMBEDDING_FAILED"
                j["last_error"] = error
            new_jobs = []
        new_jobs.sort(key=lambda x: x["sim"], reverse=True)
        logger.info(f"[embed] 过阈值({SCORE_THRESHOLD}) {len(new_jobs)} 条")
    elif not USE_EMBEDDING:
        for j in new_jobs:
            j["sim"] = 0.0
            j["pipeline_state"] = "READY_FOR_LLM"
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
            j["score_attempts"] += 1
            if score is None:
                j["pipeline_state"] = "READY_FOR_LLM"
                j["last_error"] = reason
            elif any(f["severity"] == "knockout" for f in j["flags"]):
                j["pipeline_state"] = "DISQUALIFIED"
            else:
                j["pipeline_state"] = "SCORED"
            logger.info(f"[llm] 复评 {idx}/{n}: {j['title'][:40]} -> {j['llm_score']}")
            _p(0.6 + 0.35 * idx / n, f"LLM 复评 {idx}/{n}: {j['title'][:30]}")
        new_jobs.sort(
            key=lambda x: x["llm_score"] if x["llm_score"] is not None else x["sim"] * 100,
            reverse=True)

    # 所有采集结果都落库;pipeline_state 记录被淘汰、拒绝或等待评分的原因。
    now = dt.datetime.now().isoformat(timespec="seconds")
    all_jobs = ingested_jobs
    for j in all_jobs:
        st = "DISQUALIFIED" if any(
            f["severity"] == "knockout" for f in (j.get("flags") or [])) else "待投"
        repo.upsert(j, now, st)
    con.commit()
    con.close()
    write_digest(new_jobs)
    logger.info(f"[done] 入库 {len(all_jobs)} 条(可评分 {len(new_jobs)},硬淘汰 {len(knocked)}) -> {DIGEST_PATH}")
    _p(1.0, f"完成!采集入库 {len(all_jobs)} 条,可评分 {len(new_jobs)},硬淘汰 {len(knocked)} 条")


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

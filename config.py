#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — 单一配置源。所有模块从这里导入常量,不重复定义。
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

# --- Adzuna ---
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = "au"
ADZUNA_PAGES   = 4    # 宽搜下 4 页通常够;注意 API 调用数 = 搜索条数 × 页数,留意配额

# 宽搜窄评:搜索层只负责"捞够",所有"严"的判断交给
# embedding(语义精排) + LLM 复评 + hard_filter(确定性淘汰)三层。
SEARCHES = [
    # ---- 学术线 ----
    {"what": "research fellow"},
    {"what": "research associate"},
    {"what": "postdoctoral"},
    {"what": "postdoc"},
    {"what": "lecturer"},                 # 级别压制交给 seniority_adjust + LLM cap,不用 title_only
    {"what": "research scientist"},
    {"what": "research engineer"},

    # ---- XR / 空间计算线 ----
    {"what": "augmented reality"},        # what 下是 augmented AND reality,比精确短语宽
    {"what": "virtual reality"},
    {"what": "mixed reality"},
    {"what": "extended reality"},
    {"what": "spatial computing"},
    {"what": "immersive"},
    {"what": "unity developer"},
    {"what": "3D interaction"},

    # ---- AI / ML / CV 线 ----
    {"what": "AI engineer"},
    {"what": "machine learning engineer"},
    {"what": "machine learning"},
    {"what": "applied scientist"},
    {"what": "large language model"},
    {"what": "generative AI"},

    # ---- HCI / UX / 交互线 ----
    {"what": "human computer interaction"},
    {"what": "UX researcher"},
    {"what": "interaction designer"},

    # ---- HRI / 具身&对话 Agent(HCI→HRI 桥接)----
    {"what": "human robot interaction"},
    {"what": "social robot"},
    {"what": "embodied agent"},
    {"what": "conversational agent"},
    {"what": "virtual agent"},

    # ---- digital health(原 exclude 误杀,现下放打分层)----
    {"what": "digital health"},
    {"what": "health technology"},

    # ---- 多模态信号 ----
    {"what": "eye tracking"},
]

MAX_DAYS_OLD = 90
# exclude 收到"绝对噪音":这几个词出现基本就是销售/财务岗,且不会误伤目标域。
WHAT_EXCLUDE = "sales marketing recruitment finance accounting"

# --- LM Studio (OpenAI 兼容本地端点) ---
LMSTUDIO_BASE   = os.environ.get("LMSTUDIO_BASE", "http://localhost:1234/v1")
EMBED_MODEL     = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL       = "qwen/qwen3.5-9b"
USE_EMBEDDING   = True
USE_LLM_SCORING = True
TOP_N_FOR_LLM   = 50

# --- 你的画像 ---
PROFILE = """
# Identity & career stage
Early-career researcher/engineer. PhD in Computer Science (HCI / Extended &
Mixed Reality), thesis on dependency-aware mixed reality guidance, expected
mid-2026. Australian Permanent Resident. Targeting Level A-B equivalents:
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

# IMPORTANT robotics boundary: my "robot/agent" work is VIRTUAL LLM-driven
# embodied agents / avatars in Unity, NOT physical robotics. I do NOT have
# experience in physical robot control, sensor fusion, autonomous-operation
# algorithms, ROS, motion planning, or field robotics. Roles centred on
# physical robotic control / autonomy are at best adjacent-bridgeable, not a
# core match — score their domain fit lower accordingly.

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

# --- 级别降权 ---
SENIOR_TERMS = [
    "senior", "principal", "lead", "head of", "chair",
    "professor", "associate professor", "director", "manager",
]
JUNIOR_TERMS = [
    "associate", "assistant", "junior", "graduate", "level a", "early career",
]
SENIOR_PENALTY = 0.15
JUNIOR_BONUS   = 0.03

SCORE_THRESHOLD = 0.6    # 语义层的"严"闸;宽搜后若过闸太多,先在此收紧(0.62-0.65)再说
DB_PATH         = "jobs.db"
DIGEST_PATH     = "digest.md"

# LLM 自检出的 flag code -> cap。作为 hard_filter 正则的"第二张网":
# 兜确定性正则漏掉的新型淘汰条件。最终封顶仍由 Python(apply_flags)统一执行,
# 不让 LLM 自己算 min。
LLM_FLAG_CAPS = {
    "clearance": 5, "citizenship": 5, "registration": 5, "deadline_passed": 5,
    "spam_or_test": 5,
    "degree_field": 20, "wet_lab": 25, "clinical_delivery": 30, "seniority": 35,
    "unrelated_domain": 30,
}

# --- 统一日志格式 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

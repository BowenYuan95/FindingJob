# -*- coding: utf-8 -*-
"""
hard_filter.py — 确定性硬性淘汰 / 封顶层(在 embedding + LLM 打分之前跑)

目的:把"关键词形似、实质淘汰"的岗在打分前就拦下,避免它们顶着一堆
research 关键词混进 85-95 高分段。检测与执行分离——本模块只做确定性检测,
返回结构化 flags;最终封顶/淘汰由调用方按 flags 强制执行。

设计原则:
- 宁可漏拦、不可错杀。能确定性高置信判定的才设 knockout(直接出局);
  领域类(湿实验室 / allied health / 学历领域)设 cap + warn,沉到底部但仍
  可见可审计,由 Bowen 自己决定是否升级为硬淘汰。
- 公民身份要区分【排他性要求】(must be a citizen → 淘汰)和【包容性表述】
  (citizens, permanent residents, ... → PR 可投,不拦)。

每个 flag:{code, label, cap, severity, evidence}
  severity: "knockout"(出局) | "warn"(封顶但保留可见)
  cap:      该 flag 对最终分的封顶值(0-100)
"""

import re
import datetime as dt
from typing import Literal, TypedDict


class Flag(TypedDict):
    code: str
    label: str
    cap: int
    severity: Literal["knockout", "warn"]
    evidence: str

# ------------------------------------------------------------
# 日期解析(用于 deadline 检测,无外部依赖)
# ------------------------------------------------------------
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]
_MONTHS = {}
for _i, _n in enumerate(_MONTH_NAMES, 1):
    _MONTHS[_n] = _i
    _MONTHS[_n[:3]] = _i

# "15 June 2026" / "24th June" / "June 15, 2026"
_DATE_DM  = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\b(?:\s+(\d{4}))?", re.I)
_DATE_MD  = re.compile(r"\b([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", re.I)
# "12/07/2026" / "12-07-2026"  —— 澳洲日/月/年
_DATE_DMY = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

_CLOSE_KW = re.compile(
    r"(applications?[^.]{0,40}(?:close|closing|submitted|received)"
    r"|closing date|apply (?:by|before)|submit(?:ted)? before|deadline)",
    re.I)


def _mk_date(day: str, mon: str | int, year: str | None, today: dt.date) -> dt.date | None:
    try:
        day = int(day); mon = int(mon)
        year = int(year) if year else today.year
        if not (1 <= mon <= 12 and 1 <= day <= 31):
            return None
        return dt.date(year, mon, day)
    except Exception:
        return None


def _parse_dates_in(window: str, today: dt.date) -> list[dt.date]:
    out = []
    for m in _DATE_DM.finditer(window):
        mon = _MONTHS.get(m.group(2).lower())
        if mon:
            d = _mk_date(m.group(1), mon, m.group(3), today)
            if d:
                out.append(d)
    for m in _DATE_MD.finditer(window):
        mon = _MONTHS.get(m.group(1).lower())
        if mon:
            d = _mk_date(m.group(2), mon, m.group(3), today)
            if d:
                out.append(d)
    for m in _DATE_DMY.finditer(window):
        d = _mk_date(m.group(1), m.group(2), m.group(3), today)
        if d:
            out.append(d)
    return out


def find_deadline(text: str, today: dt.date | None = None) -> dt.date | None:
    """在 closing 关键词附近 120 字窗口内找日期,返回最早的一个(截止日)。"""
    today = today or dt.date.today()
    cands = []
    for kw in _CLOSE_KW.finditer(text):
        window = text[kw.start(): kw.start() + 120]
        cands += _parse_dates_in(window, today)
    if not cands:
        return None
    # 取离今天最近、且看起来像申请截止的那个(简单取最小合理值)
    plausible = [d for d in cands if abs((d - today).days) <= 400]
    return min(plausible) if plausible else None


# ------------------------------------------------------------
# 词表
# ------------------------------------------------------------
_CLEARANCE = re.compile(
    r"\b(nv1|nv2|negative vetting|positive vetting|agsva|baseline clearance"
    r"|security clearance|defence clearance|pv clearance"
    r"|tssc|nv-?1|nv-?2)\b", re.I)

# AHPRA / 受监管执业注册 —— 两路触发,降低对"顺带提及"的误杀:
#  (a) 岗位标题本身就是受监管职业(该岗 = 这个职业)
#  (b) 正文出现【要求候选人持注册】的义务语气(而非只是出现 "registered nurse" 字样)
_REG_PROFESSION_TITLE = re.compile(
    r"\b(registered nurse|enrolled nurse|registered midwife|nurse practitioner"
    r"|psychologist|physiotherapist|physiotherapy|occupational therapist"
    r"|speech patholog|exercise physiolog|osteopath|chiropractor|dietitian"
    r"|podiatrist|pharmacist|paramedic|audiologist|orthopt|prosthet|sonographer"
    r"|radiographer|clinician|general practitioner|\bgp\b|medical officer"
    r"|anaesthet|registrar|resident medical|dentist|optometrist"
    r"|social worker|rehabilitation consultant)", re.I)
_REG_OBLIGATION = re.compile(
    r"(ahpra|australian health practitioner regulation"
    r"|registration required|must be registered|must hold (?:current |general )?registration"
    r"|(?:ahpra|general|current|professional) registration"
    r"|current registration (?:as|with)|registration as a |registered (?:psychologist|nurse) with"
    r"|eligib\w+ for registration|be registered with)", re.I)

# 学历领域:essential 的 "PhD in <非CS领域>"
_PHD_IN = re.compile(r"phd\s+in\s+(?:a\s+)?([a-z][a-z /&\-]{2,40})", re.I)
_BLOCK_FIELDS = {
    "psychology", "psychological science", "neuroscience", "biology", "biological science",
    "biochemistry", "chemistry", "molecular biology", "cell biology", "genetics",
    "nursing", "medicine", "pharmacy", "pharmacology", "physiotherapy",
    "occupational therapy", "speech pathology", "allied health", "public health",
    "epidemiology", "social work", "nutrition", "dietetics", "immunology",
    "microbiology", "physiology", "anatomy", "clinical psychology",
}
_OK_FIELDS = (
    "computer science", "computing", "information technology", "software",
    "engineering", "human-computer interaction", "hci", "machine learning",
    "artificial intelligence", "data science", "electrical", "robotics",
    "relevant field", "relevant discipline", "related field", "related discipline",
    "a relevant", "a related",
)

# 湿实验室 / 台架
_WETLAB = re.compile(
    r"\b(stem cell|cell culture|tissue culture|disease model|wet[\s-]?lab"
    r"|in vitro|in vivo|western blot|immunohisto|histolog|\bpcr\b|qpcr"
    r"|pipett|organoid|high[\s-]?throughput screen|assay development|cell line"
    r"|protein structure|protein purification|structural biology|crystallograph"
    r"|bioanalytic|chromatograph|mass spec|flow cytometr|electrophoresis|\belisa\b"
    r"|preclinical|murine|knock[\s-]?out mouse|drug discovery|medicinal chemistr)\b", re.I)

# 临床交付 / allied health / 残障
_CLINICAL_DELIVERY = re.compile(
    r"\b(allied health|disability service|cerebral palsy|paediatric|pediatric"
    r"|occupational therap|physiotherap|speech patholog"
    r"|psychological (?:assessment|treatment|therapy)"
    r"|clinical placement|patient[\s-]?facing|clinical caseload|caseload)\b", re.I)


def _sentences(text):
    return re.split(r"(?<=[.!?;:])\s+|\n+", text)


def _exclusive_citizenship(text):
    """只在【排他性】公民身份要求时返回证据;包容性表述(含 PR)不触发。"""
    for s in _sentences(text):
        sl = s.lower()
        if "australian citizen" not in sl and "citizenship" not in sl:
            continue
        # 包容性:同句出现 PR / 居留 / working rights → PR 可投,跳过
        if any(k in sl for k in ("permanent resident", "permanent residency",
                                 "working rights", "residing in australia",
                                 "right to work")):
            continue
        # 排他性触发词
        if any(k in sl for k in ("must be an australian citizen",
                                 "must be australian citizen",
                                 "must hold australian citizen",
                                 "australian citizenship is required",
                                 "australian citizenship is mandatory",
                                 "restricted to australian citizen",
                                 "only australian citizen",
                                 "australian citizens only",
                                 "required to be an australian citizen",
                                 "eligibility for a security",
                                 "able to obtain a security")):
            return s.strip()[:160]
    return None


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def scan_disqualifiers(title: str, description: str, today: dt.date | None = None) -> list[Flag]:
    """Return a list of disqualifier flags for a job. Caller applies caps/knockouts."""
    today = today or dt.date.today()
    text = f"{title}\n{description}"
    tl = text.lower()
    flags = []

    def add(code, label, cap, severity, evidence):
        flags.append({"code": code, "label": label, "cap": cap,
                      "severity": severity, "evidence": (evidence or "")[:160]})

    # 1) 安全许可 / clearance —— 硬淘汰
    m = _CLEARANCE.search(text)
    if m:
        add("clearance", "需安全许可(PR 不符)", 5, "knockout", _ctx(text, m))

    # 2) 排他性公民身份 —— 硬淘汰
    ev = _exclusive_citizenship(text)
    if ev:
        add("citizenship", "排他性要求澳籍(PR 不符)", 5, "knockout", ev)

    # 3) 受监管执业注册(AHPRA 等)—— 硬淘汰。标题即职业,或正文有义务语气
    mt = _REG_PROFESSION_TITLE.search(title or "")
    mo = _REG_OBLIGATION.search(text)
    if mt:
        add("registration", "需受监管执业注册(岗位即受监管职业)", 5, "knockout",
            f"title: {title}")
    elif mo:
        add("registration", "需受监管执业注册(如 AHPRA)", 5, "knockout", _ctx(text, mo))

    # 4) 截止日期已过 —— 硬淘汰
    dl = find_deadline(text, today)
    if dl and dl < today:
        add("deadline_passed", f"截止已过({dl.isoformat()})", 5, "knockout",
            f"detected closing date {dl.isoformat()} < today {today.isoformat()}")

    # 5) 学历领域硬性错配(essential PhD in <非CS领域>)—— 封顶 + warn
    for m in _PHD_IN.finditer(text):
        field = m.group(1).strip().lower()
        field = re.split(r"\b(or|and)\b", field)[0].strip()  # "psychology or a related" → psychology
        if any(ok in field for ok in _OK_FIELDS):
            continue
        if any(field.startswith(bf) or bf in field for bf in _BLOCK_FIELDS):
            add("degree_field", f"要求 PhD in {field}(非 CS/HCI)", 20, "warn", _ctx(text, m))
            break

    # 6) 湿实验室 / 台架为核心 —— 封顶 + warn
    wl = _WETLAB.findall(tl)
    if len(set(x if isinstance(x, str) else x[0] for x in wl)) >= 1 and _WETLAB.search(text):
        add("wet_lab", "湿实验室/台架技能为核心(不可迁移)", 25, "warn",
            _ctx(text, _WETLAB.search(text)))

    # 7) 临床交付 / allied health / 残障为核心 —— 封顶 + warn
    m = _CLINICAL_DELIVERY.search(text)
    if m:
        add("clinical_delivery", "临床交付/allied health/残障为核心", 30, "warn", _ctx(text, m))

    return flags


def _ctx(text: str, m: re.Match, span: int = 70) -> str:
    a = max(0, m.start() - span)
    b = min(len(text), m.end() + span)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def dedup_flags(flags: list[Flag]) -> list[Flag]:
    """Deduplicate flags by code, keeping the strictest cap per code."""
    best = {}
    for f in flags or []:
        c = f.get("code")
        if c not in best or f["cap"] < best[c]["cap"]:
            best[c] = f
    return list(best.values())


def apply_flags(
    base_score: float, flags: list[Flag]
) -> tuple[float, Literal["ok", "capped", "DISQUALIFIED"]]:
    """Apply flag caps to base_score. Returns (final_score, status)."""
    flags = dedup_flags(flags)
    if not flags:
        return base_score, "ok"
    cap = min(f["cap"] for f in flags)
    disq = any(f["severity"] == "knockout" for f in flags)
    final = min(base_score, cap)
    if disq:
        return final, "DISQUALIFIED"
    return final, ("capped" if final < base_score else "ok")


# ------------------------------------------------------------
# 自测
# ------------------------------------------------------------
if __name__ == "__main__":
    today = dt.date(2026, 6, 20)

    CASES = {
        "Flinders MND (wet-lab)": (
            "Postdoctoral Research Associate/Research Fellow",
            "Join a world-leading research program at the MND Centre for Drug Discovery. "
            "Practical expertise in techniques such as stem cell culture, disease modelling "
            "or imaging-based analysis. Working with high-throughput robotics. A completed "
            "(or near-completed) PhD in a relevant field. Contribute to publications and "
            "presentations. Applications to be submitted before 10.00pm: 15 June 2026."),

        "Deakin (allied health)": (
            "Associate Research Fellow",
            "Design and conduct high quality collaborative research with industry partners "
            "to improve outcomes for people with complex physical needs. Build partnerships "
            "with Kids+ and Cerebral Palsy Education Centre. PhD in a relevant discipline "
            "and/or demonstrated research in allied health, disability, health sciences. "
            "Experience in allied health, disability, and or paediatric settings is highly "
            "desirable. Applications close on Wednesday 24th June at 11:55pm."),

        "Macquarie eCentreClinic (psych+AHPRA)": (
            "Postdoctoral Research Fellow",
            "Contribute to clinical trials advancing digital mental health interventions. "
            "Provide psychological assessment and therapy, report writing. Essential: PhD in "
            "psychology or a related discipline. Registered Psychologist with the Australian "
            "Health Practitioner Regulation Agency (AHPRA). Proficiency in SPSS, R. "
            "Note: Applicants must be Australian citizens, permanent residents, or currently "
            "residing in Australia with full working rights. Applications Close 12/07/2026."),

        "POS-CTRL: XR Research Fellow (好岗,不应触发)": (
            "Research Fellow in Mixed Reality",
            "Join our HCI lab to develop adaptive augmented reality guidance systems using "
            "Unity, gaze tracking and LLM-driven agents. PhD in computer science or a related "
            "field. Level A/B, early-career researchers encouraged. Melbourne, hybrid. "
            "Applications close 30 July 2026."),
    }

    for name, (title, desc) in CASES.items():
        flags = scan_disqualifiers(title, desc, today)
        final, status = apply_flags(90, flags)  # 假设 base=90,看封顶后
        print(f"\n=== {name} ===")
        print(f"  base 90 → final {final}  [{status}]")
        for f in flags:
            print(f"   • [{f['severity']:8s} cap{f['cap']:>3}] {f['code']:18s} {f['label']}")
            print(f"     evidence: {f['evidence']}")
        if not flags:
            print("   (无 flag —— 正常通过)")

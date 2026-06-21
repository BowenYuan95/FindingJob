#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — 求职匹配 + 申请追踪 Streamlit 面板

两个视图:
  📋 待投递   —— 匹配排序后的职位,可筛选/搜索/一键更新/改状态
  📊 申请追踪 —— 所有非"待投"状态的职位,看状态/日期/备注,可编辑

依赖:  pip install streamlit pandas
运行:  streamlit run app.py
"""

import os
import sys
import json
import logging
import webbrowser
import datetime as dt
from collections import deque

import streamlit as st

from config import DB_PATH
from infrastructure.database import database_session
from infrastructure.job_repository import JobRepository
from infrastructure.lmstudio import LM_CLIENT
from pipeline.job_urls import normalize_job_url

STATUSES = ["待投", "已投", "面试", "拒", "offer"]
TRACKED_STATUSES = [s for s in STATUSES if s != "待投"]
PAGE_SIZE = 20

st.set_page_config(page_title="求职匹配 + 申请追踪", page_icon="🎯", layout="wide")


class _StreamlitLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__(logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.format(record))
        except Exception:
            self.handleError(record)


# ----------------------------------------------------------------
# 数据
# ----------------------------------------------------------------

def dashboard_snapshot() -> dict[str, object]:
    try:
        with database_session(DB_PATH, initialize=True) as con:
            repo = JobRepository(con)
            return {
                "total": repo.total_jobs(),
                "pipeline": repo.pipeline_counts(),
            }
    except Exception as e:
        st.error(f"读取 {DB_PATH} 失败:{e}。先跑一次 pipeline/job_matcher.py。")
        return {"total": 0, "pipeline": {}}


def page_window(
    state_key: str, filter_signature: tuple[object, ...], total: int,
) -> tuple[int, int]:
    """Reset on filter changes and clamp after rows move off the current page."""
    signature_key = f"{state_key}_filters"
    if st.session_state.get(signature_key) != filter_signature:
        st.session_state[signature_key] = filter_signature
        st.session_state[state_key] = 1
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(int(st.session_state.get(state_key, 1)), 1), total_pages)
    st.session_state[state_key] = page
    return page, total_pages


def render_pagination(state_key: str, page: int, total_pages: int) -> None:
    previous, label, following = st.columns([0.25, 0.5, 0.25])
    if previous.button(
        "← 上一页", key=f"{state_key}_previous", disabled=page <= 1,
        use_container_width=True,
    ):
        st.session_state[state_key] = page - 1
        st.rerun()
    label.markdown(
        f"<div style='text-align:center'>第 {page} / {total_pages} 页</div>",
        unsafe_allow_html=True,
    )
    if following.button(
        "下一页 →", key=f"{state_key}_next", disabled=page >= total_pages,
        use_container_width=True,
    ):
        st.session_state[state_key] = page + 1
        st.rerun()


def purge_expired_deadlines() -> int:
    """重扫待投岗位的截止日期,并同步 status / score / flags。"""
    from pipeline.hard_filter import dedup_flags, scan_disqualifiers
    try:
        with database_session(DB_PATH) as con:
            repo = JobRepository(con)
            updated = 0
            after_id = ""
            while True:
                rows = repo.pending_deadline_candidates(after_id, limit=100)
                if not rows:
                    break
                for jid, title, desc, flags_json in rows:
                    rescanned = scan_disqualifiers(title or "", desc or "")
                    dl = next(
                        (f for f in rescanned if f["code"] == "deadline_passed"),
                        None,
                    )
                    if not dl:
                        continue
                    try:
                        old_flags = json.loads(flags_json) if flags_json else []
                    except (TypeError, json.JSONDecodeError):
                        old_flags = []
                    flags = dedup_flags(old_flags + [dl])
                    repo.mark_deadline_disqualified(
                        jid,
                        f"硬性淘汰:截止已过 {dl['evidence'][:80]}",
                        json.dumps(flags, ensure_ascii=False),
                        dt.datetime.now().isoformat(timespec="seconds"),
                    )
                    updated += 1
                after_id = rows[-1][0]
        return updated
    except Exception as e:
        st.warning(f"截止日期清理失败:{e}")
        return 0


def update_job(job_id: str, **fields: object) -> None:
    """更新某职位的 status / note 等字段。"""
    if not fields:
        return
    try:
        with database_session(DB_PATH) as con:
            JobRepository(con).update_user_fields(job_id, fields)
    except Exception as e:
        st.error(f"更新失败:{e}")


def manually_disqualify(job_id: str) -> None:
    try:
        with database_session(DB_PATH) as con:
            JobRepository(con).mark_manually_disqualified(
                job_id, dt.datetime.now().isoformat(timespec="seconds")
            )
    except Exception as e:
        st.error(f"淘汰失败：{e}")


def set_status(job_id: str, status: str) -> None:
    """改状态;首次离开待投时记录投递日期,后续状态变化保留该日期。"""
    today = dt.date.today().isoformat()
    applied_date = ""
    if status != "待投":
        try:
            with database_session(DB_PATH) as con:
                previous = JobRepository(con).applied_date(job_id)
            applied_date = previous or today
        except Exception:
            applied_date = today
    update_job(job_id, status=status,
               applied=(0 if status == "待投" else 1),
               applied_date=applied_date)


def lmstudio_alive() -> bool:
    try:
        LM_CLIENT.loaded_models(timeout=3)
        return True
    except Exception:
        return False


def ensure_models_loaded():
    """点更新前只加载当前确实缺失的模型。"""
    import subprocess
    loaded = LM_CLIENT.loaded_models(timeout=5)
    cmds = {
        "text-embedding-nomic-embed-text-v1.5":
            ["lms", "load", "text-embedding-nomic-embed-text-v1.5",
             "--identifier", "text-embedding-nomic-embed-text-v1.5", "--gpu", "max"],
        "qwen/qwen3.5-9b":
            ["lms", "load", "qwen/qwen3.5-9b", "--identifier", "qwen/qwen3.5-9b",
             "--gpu", "max", "--context-length", "16384"],
    }
    for model, c in cmds.items():
        if model in loaded:
            continue
        try:
            result = subprocess.run(c, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        except Exception as e:
            raise RuntimeError(f"模型加载失败 ({model}): {e}") from e


def trigger_backfill():
    """后台启动 backfill:补评剩余职位(不阻塞面板)。"""
    import subprocess
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        subprocess.Popen([sys.executable, "-m", "pipeline.backfill_scores"],
                         cwd=os.path.dirname(os.path.abspath(__file__)),
                         creationflags=flags)
    except Exception as e:
        st.warning(f"后台补评启动失败:{e}")


def last_update_time() -> str:
    try:
        with database_session(DB_PATH) as con:
            value = JobRepository(con).last_seen()
        return value or "—"
    except Exception:
        return "—"


def backfill_status() -> tuple[int, int]:
    """Returns (unscored_count, total_pending) — live query, no cache."""
    try:
        with database_session(DB_PATH) as con:
            return JobRepository(con).pending_counts()
    except Exception:
        return 0, 0


@st.fragment(run_every=8)
def _backfill_progress_panel() -> None:
    init = st.session_state.get("backfill_initial", 0)
    unscored, _ = backfill_status()
    if init == 0 and unscored == 0:
        return
    if unscored == 0:
        st.success("✅ 补评全部完成")
        if "backfill_initial" in st.session_state:
            del st.session_state["backfill_initial"]
    elif init > 0:
        done = max(0, init - unscored)
        frac = done / init
        st.progress(frac, text=f"后台补评 {done}/{init} ({frac:.0%})")
    else:
        st.caption(f"🔄 待补评: {unscored} 条")


# ----------------------------------------------------------------
# 侧栏:一键更新 + 视图切换
# ----------------------------------------------------------------

run_surface = st.empty()
snapshot = dashboard_snapshot()
_expired = purge_expired_deadlines()
if _expired:
    snapshot = dashboard_snapshot()

st.sidebar.title("🎯 求职面板")
st.sidebar.caption(f"上次更新:{last_update_time()}")

if st.sidebar.button("🔄 一键更新(抓取 + 匹配)", use_container_width=True, type="primary"):
    if not lmstudio_alive():
        st.sidebar.error("LM Studio 没连上(localhost:1234)。先 Start Server 再更新。")
    else:
        before = int(snapshot["total"])
        bar = st.sidebar.progress(0.0, text="加载模型中…")
        llm_label = st.sidebar.empty()
        llm_bar2  = st.sidebar.empty()
        log_lines: deque[str] = deque(maxlen=250)
        with run_surface.container():
            st.subheader("🔄 正在抓取和匹配")
            st.caption("运行期间暂时隐藏数据库结果；完成后自动显示最新结果。")
            main_progress = st.progress(0.0, text="准备运行…")
            log_box = st.empty()

        def append_log(message: str) -> None:
            log_lines.append(message)
            log_box.code("\n".join(log_lines), language=None)

        handler = _StreamlitLogHandler(append_log)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        failed = False
        try:
            append_log("正在加载 embedding 与 LLM 模型…")
            ensure_models_loaded()          # 确保模型在(上轮可能已卸载)
            import importlib
            from pipeline import job_matcher
            importlib.reload(job_matcher)
            def on_progress(frac: float, msg: str) -> None:
                bar.progress(min(max(frac, 0.0), 1.0), text=msg)
                main_progress.progress(min(max(frac, 0.0), 1.0), text=msg)
                if msg.startswith("LLM 复评"):
                    try:
                        parts = msg[len("LLM 复评 "):].split(": ", 1)
                        cur, total = map(int, parts[0].split("/"))
                        title = parts[1] if len(parts) > 1 else ""
                        llm_label.markdown(f"**🧠 LLM 打分** · {cur}/{total} ({cur / total:.0%})")
                        llm_bar2.progress(cur / total, text=f"正在评:{title[:25]}")
                    except Exception:
                        pass
                else:
                    llm_label.empty()
                    llm_bar2.empty()
            job_matcher.main(progress=on_progress)
        except Exception as e:
            failed = True
            append_log(f"运行失败: {e}")
            run_surface.error(f"运行失败: {e}")
            st.sidebar.error(f"运行失败:{e}")
        else:
            latest = dashboard_snapshot()
            after = int(latest["total"])
            bar.empty()
            llm_label.empty()
            llm_bar2.empty()
            trigger_backfill()              # 后台评剩余,评完自动卸载模型
            unscored_now, _ = backfill_status()
            st.session_state["backfill_initial"] = unscored_now
            states_value = latest["pipeline"]
            states = states_value if isinstance(states_value, dict) else {}
            st.session_state["last_run_summary"] = (
                f"本轮新增 {max(0, after - before)} 条；"
                f"待 LLM 补评 {unscored_now} 条；"
                f"已评分 {int(states.get('SCORED', 0))} 条；"
                f"embedding 淘汰 {int(states.get('EMBEDDING_REJECTED', 0))} 条。"
            )
            st.sidebar.success(f"完成!新增 {max(0, after - before)} 条。"
                               f"后台正在补评剩余,完成后会自动卸载模型。")
            st.rerun()
        finally:
            root_logger.removeHandler(handler)
        if failed:
            bar.empty()
            llm_label.empty()
            llm_bar2.empty()
            st.stop()

with st.sidebar:
    _backfill_progress_panel()
st.sidebar.divider()
view = st.sidebar.radio("视图", ["📋 待投递", "📊 申请追踪"], label_visibility="collapsed")

# ================================================================
# 视图一:待投递
# ================================================================

def render_todo() -> None:
    try:
        with database_session(DB_PATH) as con:
            sources = JobRepository(con).sources()
    except Exception as e:
        st.error(f"读取职位来源失败:{e}")
        return
    pick_src   = st.sidebar.multiselect("来源", sources, default=sources)
    min_score  = st.sidebar.slider("最低分数", 0, 100, 0, step=5)
    kw         = st.sidebar.text_input("关键词(标题/公司)", "").strip().lower()
    only_llm   = st.sidebar.checkbox("只看 LLM 复评过的", value=False)

    filters = (tuple(pick_src), min_score, kw, only_llm)
    try:
        with database_session(DB_PATH) as con:
            repo = JobRepository(con)
            total, average = repo.todo_stats(pick_src, min_score, kw, only_llm)
            page, total_pages = page_window("todo_page", filters, total)
            rows = repo.fetch_todo_page(
                pick_src, min_score, kw, only_llm,
                PAGE_SIZE, (page - 1) * PAGE_SIZE,
            )
    except Exception as e:
        st.error(f"读取待投职位失败:{e}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("待投职位", total)
    c2.metric("数据库总数", int(snapshot["total"]))
    c3.metric("平均分", f"{average:.0f}" if average is not None else "—")
    st.divider()

    if not rows:
        st.warning("当前筛选下没有待投职位。"); return

    for r in rows:
        tag = "🧠" if r["has_llm"] else "📐"
        with st.container(border=True):
            top = st.columns([0.8, 0.2])
            top[0].markdown(f"### {r['title']}")
            top[1].markdown(f"## `{int(r['score'])}` {tag}")
            meta = " · ".join(str(x) for x in [r["company"], r["location"], r["salary"], r["source"]]
                              if x and str(x).strip())
            st.caption(meta)
            if isinstance(r["llm_reason"], str) and r["llm_reason"].strip():
                st.markdown(f"> {r['llm_reason']}")
            if r.get("score_status") == "capped":
                cap = r.get("applied_cap")
                st.caption(f"⚠ 系统封顶: {cap:.0f}" if cap is not None else "⚠ 系统封顶")

            bcol = st.columns([0.3, 0.38, 0.15, 0.17])
            job_url = normalize_job_url(r.get("url"))
            if job_url:
                if bcol[0].button("🔗 查看职位", key=f"url_{r['id']}"):
                    if not webbrowser.open_new_tab(job_url):
                        bcol[0].warning("浏览器未能打开链接")
            elif isinstance(r.get("url"), str) and r["url"].strip():
                bcol[0].caption("⚠ 无效链接")
            # 状态选择(选了非待投即进追踪表)
            new_status = bcol[1].selectbox(
                "状态", STATUSES, index=0, key=f"st_{r['id']}", label_visibility="collapsed")
            if new_status != "待投":
                set_status(r["id"], new_status); st.rerun()
            if bcol[2].button("❌ 淘汰", key=f"dq_{r['id']}", help="手动标为不投"):
                manually_disqualify(r["id"]); st.rerun()

            summary = r.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                with st.expander("📋 职位摘要(LLM 整理)"):
                    st.markdown(summary)
            elif isinstance(r["description"], str) and r["description"].strip():
                with st.expander("职位描述"):
                    st.write(r["description"][:1500])

    render_pagination("todo_page", page, total_pages)
    st.caption("🧠 = Qwen 复评分 · 📐 = embedding 相似度分")


# ================================================================
# 视图二:申请追踪
# ================================================================

def render_tracker() -> None:
    try:
        with database_session(DB_PATH) as con:
            overall_counts = JobRepository(con).tracker_status_counts(TRACKED_STATUSES)
    except Exception as e:
        st.error(f"读取申请追踪统计失败:{e}")
        return
    overall_total = sum(overall_counts.values())

    # 顶部:各状态统计
    cols = st.columns(len(TRACKED_STATUSES) + 1)
    cols[0].metric("追踪总数", overall_total)
    for i, sname in enumerate(TRACKED_STATUSES, 1):
        cols[i].metric(sname, overall_counts.get(sname, 0))
    st.divider()

    if overall_total == 0:
        st.info("还没有已处理的职位。去『待投递』里把投过的职位改个状态,就会出现在这里。")
        return

    # 状态筛选
    pick = st.multiselect("筛选状态", TRACKED_STATUSES, default=TRACKED_STATUSES)
    try:
        with database_session(DB_PATH) as con:
            repo = JobRepository(con)
            total = repo.count_tracker(pick)
            page, total_pages = page_window("tracker_page", (tuple(pick),), total)
            rows = repo.fetch_tracker_page(
                pick, PAGE_SIZE, (page - 1) * PAGE_SIZE,
            )
    except Exception as e:
        st.error(f"读取申请追踪职位失败:{e}")
        return

    if not rows:
        st.warning("当前筛选下没有申请记录。")
        return

    for r in rows:
        with st.container(border=True):
            head = st.columns([0.6, 0.4])
            head[0].markdown(f"### {r['title']}")
            badge = {"已投": "🟦", "面试": "🟨", "拒": "🟥", "offer": "🟩"}.get(r["status"], "")
            head[1].markdown(f"### {badge} {r['status']}")
            meta = " · ".join(str(x) for x in [r["company"], r["location"], r["source"]]
                              if x and str(x).strip())
            date = f" · 📅 {r['applied_date']}" if r["applied_date"] else ""
            st.caption(meta + date)
            job_url = normalize_job_url(r.get("url"))
            if job_url:
                if st.button("🔗 查看职位", key=f"url_{r['id']}"):
                    if not webbrowser.open_new_tab(job_url):
                        st.warning("浏览器未能打开链接")
            elif isinstance(r.get("url"), str) and r["url"].strip():
                st.caption("⚠ 无效职位链接")

            edit = st.columns([0.3, 0.7])
            cur_idx = STATUSES.index(r["status"]) if r["status"] in STATUSES else 0
            ns = edit[0].selectbox("状态", STATUSES, index=cur_idx,
                                   key=f"tst_{r['id']}")
            if ns != r["status"]:
                set_status(r["id"], ns); st.rerun()
            note = edit[1].text_input("备注", value=r.get("note", "") or "",
                                      key=f"note_{r['id']}",
                                      placeholder="面试时间、联系人、跟进事项…")
            if note != (r.get("note", "") or ""):
                update_job(r["id"], note=note); st.rerun()

    render_pagination("tracker_page", page, total_pages)


# ----------------------------------------------------------------
with run_surface.container():
    if st.session_state.get("last_run_summary"):
        st.success("✅ 更新完成 · " + st.session_state["last_run_summary"])
    if int(snapshot["total"]) == 0:
        st.info("数据库还没数据。点击左侧『一键更新』开始抓取。")
    elif view == "📋 待投递":
        render_todo()
    else:
        render_tracker()

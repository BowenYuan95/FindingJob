"""Central SQLite connection policy and schema migrations."""

import os
import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator

from config import DB_PATH

BUSY_TIMEOUT_MS = 30_000


def connect_db(path: str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    con.execute("PRAGMA foreign_keys=ON")
    if path != ":memory:":
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    return con


@contextmanager
def database_session(
    path: str = DB_PATH, *, initialize: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Transaction boundary that always commits/rolls back and closes."""
    con = initialize_database(path) if initialize else connect_db(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize_database(path: str = DB_PATH) -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    con = connect_db(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            title TEXT, company TEXT, location TEXT,
            description TEXT, url TEXT, source TEXT,
            source_id TEXT DEFAULT '',
            salary TEXT, created TEXT,
            sim REAL, llm_score REAL, llm_reason TEXT,
            first_seen TEXT,
            applied INTEGER DEFAULT 0,
            summary TEXT DEFAULT '',
            status TEXT DEFAULT '待投',
            applied_date TEXT DEFAULT '',
            note TEXT DEFAULT '',
            flags TEXT DEFAULT '',
            pipeline_state TEXT DEFAULT 'INGESTED',
            score_attempts INTEGER DEFAULT 0,
            last_error TEXT DEFAULT '',
            updated_at TEXT DEFAULT '',
            score_status TEXT DEFAULT '',
            applied_cap REAL
        )""")

    columns = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
    migrations = [
        ("applied", "applied INTEGER DEFAULT 0"),
        ("summary", "summary TEXT DEFAULT ''"),
        ("status", "status TEXT DEFAULT '待投'"),
        ("applied_date", "applied_date TEXT DEFAULT ''"),
        ("note", "note TEXT DEFAULT ''"),
        ("flags", "flags TEXT DEFAULT ''"),
        ("source_id", "source_id TEXT DEFAULT ''"),
        # Existing NULL-score rows are already eligible for backfill.
        ("pipeline_state", "pipeline_state TEXT DEFAULT 'READY_FOR_LLM'"),
        ("score_attempts", "score_attempts INTEGER DEFAULT 0"),
        ("last_error", "last_error TEXT DEFAULT ''"),
        ("updated_at", "updated_at TEXT DEFAULT ''"),
        ("score_status", "score_status TEXT DEFAULT ''"),
        ("applied_cap", "applied_cap REAL"),
    ]
    normalize_score_metadata = False
    for name, ddl in migrations:
        if name not in columns:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")
            if name in {"score_status", "applied_cap"}:
                normalize_score_metadata = True

    if normalize_score_metadata:
        rows = con.execute(
            "SELECT id,llm_score,llm_reason,status,flags FROM jobs"
        ).fetchall()
    else:
        # Covers a still-running pre-migration worker that scores a row without
        # populating the new metadata columns.
        rows = con.execute("""
            SELECT id,llm_score,llm_reason,status,flags FROM jobs
            WHERE score_status='' AND llm_score IS NOT NULL
        """).fetchall()
    if rows:
        for job_id, score, reason, status, flags_json in rows:
            try:
                flags = json.loads(flags_json) if flags_json else []
            except (TypeError, json.JSONDecodeError):
                flags = []
            caps = [float(flag["cap"]) for flag in flags if "cap" in flag]
            strictest_cap = min(caps) if caps else None
            clean_reason = str(reason or "")
            if clean_reason.startswith("[capped] "):
                clean_reason = clean_reason[len("[capped] "):]

            if status == "DISQUALIFIED":
                score_status = "DISQUALIFIED"
                applied_cap = strictest_cap
            elif score is None:
                score_status = ""
                applied_cap = None
            elif (strictest_cap is not None and strictest_cap < 100
                  and abs(float(score) - strictest_cap) < 1e-9):
                score_status = "capped"
                applied_cap = strictest_cap
                clean_reason = "[capped] " + clean_reason
            else:
                score_status = "ok"
                applied_cap = None

            con.execute("""
                UPDATE jobs
                SET score_status=?, applied_cap=?, llm_reason=?
                WHERE id=?
            """, (score_status, applied_cap, clean_reason, job_id))

    # A knockout flag is authoritative even if an older worker forgot to sync status.
    knockout_rows = con.execute("""
        SELECT id,llm_score,llm_reason,flags FROM jobs
        WHERE status!='DISQUALIFIED' AND flags LIKE '%"severity": "knockout"%'
    """).fetchall()
    for job_id, score, reason, flags_json in knockout_rows:
        try:
            flags = json.loads(flags_json) if flags_json else []
        except (TypeError, json.JSONDecodeError):
            continue
        knockout_caps = [
            float(flag["cap"]) for flag in flags
            if flag.get("severity") == "knockout" and "cap" in flag
        ]
        if not knockout_caps:
            continue
        cap = min(knockout_caps)
        clean_reason = str(reason or "")
        if clean_reason.startswith("[capped] "):
            clean_reason = clean_reason[len("[capped] "):]
        if not clean_reason.startswith("[DISQUALIFIED] "):
            clean_reason = "[DISQUALIFIED] " + clean_reason
        con.execute("""
            UPDATE jobs
            SET llm_score=?, llm_reason=?, status='DISQUALIFIED',
                pipeline_state='DISQUALIFIED', score_status='DISQUALIFIED',
                applied_cap=?
            WHERE id=?
        """, (min(float(score), cap) if score is not None else cap,
              clean_reason, cap, job_id))

    # Normalize legacy rows after adding pipeline_state without overwriting
    # explicit rejection/failure states created by the new pipeline.
    con.execute("""
        UPDATE jobs SET pipeline_state='DISQUALIFIED'
        WHERE status='DISQUALIFIED' AND pipeline_state!='DISQUALIFIED'
    """)
    con.execute("""
        UPDATE jobs SET pipeline_state='SCORED'
        WHERE llm_score IS NOT NULL AND status!='DISQUALIFIED'
          AND pipeline_state='READY_FOR_LLM'
    """)

    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_source_identity
        ON jobs(source, source_id)
        WHERE source_id IS NOT NULL AND source_id != ''
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_pipeline_queue
        ON jobs(pipeline_state, status, llm_score, sim DESC)
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_todo_score
        ON jobs(
            status,
            pipeline_state,
            COALESCE(llm_score, COALESCE(sim, 0) * 100.0) DESC,
            id
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_tracker_date
        ON jobs(status, applied_date DESC, id)
    """)
    con.commit()
    return con

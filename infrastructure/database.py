"""Central SQLite connection policy and schema migrations."""

import os
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
            updated_at TEXT DEFAULT ''
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
    ]
    for name, ddl in migrations:
        if name not in columns:
            con.execute(f"ALTER TABLE jobs ADD COLUMN {ddl}")

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
    con.commit()
    return con

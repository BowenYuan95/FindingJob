"""SQLite repository for job persistence and scoring queue operations."""

import json
import sqlite3
from typing import Any


class JobRepository:
    USER_EDITABLE_FIELDS = {"status", "applied", "applied_date", "note"}

    def __init__(self, connection: sqlite3.Connection):
        self.con = connection

    def find_existing(
        self,
        *,
        source: str,
        source_id: str,
        canonical_id: str,
        legacy_id: str,
    ) -> str | None:
        if source and source_id:
            row = self.con.execute(
                "SELECT id FROM jobs WHERE source=? AND source_id=?",
                (source, source_id),
            ).fetchone()
            if row:
                return row[0]
            row = self.con.execute(
                """SELECT id FROM jobs
                   WHERE id=? AND source=? AND COALESCE(source_id,'')=''""",
                (legacy_id, source),
            ).fetchone()
            if row:
                self.con.execute(
                    "UPDATE jobs SET source_id=? WHERE id=?", (source_id, row[0])
                )
                return row[0]
        row = self.con.execute("SELECT id FROM jobs WHERE id=?", (canonical_id,)).fetchone()
        return row[0] if row else None

    def pipeline_state(self, job_id: str) -> str | None:
        row = self.con.execute(
            "SELECT pipeline_state FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        return row[0] if row else None

    def refresh_source_metadata(
        self, job_id: str, job: dict[str, Any], now: str,
    ) -> None:
        """Refresh mutable source fields without touching scores or user workflow."""
        self.con.execute("""
            UPDATE jobs
            SET url=?, location=?, description=?, salary=?, created=?, updated_at=?
            WHERE id=?
        """, (job.get("url", ""), job.get("location", ""),
              job.get("description", ""), job.get("salary", ""),
              job.get("created", ""), now, job_id))

    def fetch_all(self) -> list[dict[str, Any]]:
        cursor = self.con.execute("SELECT * FROM jobs")
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def update_user_fields(self, job_id: str, fields: dict[str, Any]) -> None:
        unknown = set(fields) - self.USER_EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"unsupported user-editable fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{field}=?" for field in fields)
        self.con.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?",
            (*fields.values(), job_id),
        )

    def applied_date(self, job_id: str) -> str:
        row = self.con.execute(
            "SELECT applied_date FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        return row[0] if row and row[0] else ""

    def pending_deadline_candidates(self) -> list[tuple]:
        return self.con.execute(
            "SELECT id, title, description, flags FROM jobs WHERE status='待投'"
        ).fetchall()

    def mark_deadline_disqualified(
        self, job_id: str, reason: str, flags_json: str, now: str,
    ) -> None:
        self.con.execute("""
            UPDATE jobs
            SET status='DISQUALIFIED', llm_score=5, llm_reason=?, flags=?,
                pipeline_state='DISQUALIFIED', updated_at=?,
                score_status='DISQUALIFIED', applied_cap=5
            WHERE id=?
        """, (reason, flags_json, now, job_id))

    def mark_manually_disqualified(self, job_id: str, now: str) -> None:
        self.con.execute("""
            UPDATE jobs
            SET status='DISQUALIFIED',
                llm_score=COALESCE(llm_score, 5),
                applied_cap=COALESCE(applied_cap, 5),
                llm_reason='用户手动淘汰',
                pipeline_state='DISQUALIFIED',
                score_status='DISQUALIFIED',
                updated_at=?
            WHERE id=?
        """, (now, job_id))

    def last_seen(self) -> str:
        row = self.con.execute("SELECT MAX(first_seen) FROM jobs").fetchone()
        return row[0] if row and row[0] else ""

    def pending_counts(self) -> tuple[int, int]:
        unscored = self.count_ready_for_scoring()
        total = self.con.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='待投'"
        ).fetchone()[0]
        return unscored, total

    def upsert(self, job: dict[str, Any], now: str, status: str) -> None:
        self.con.execute("""INSERT INTO jobs
            (id,title,company,location,description,url,source,source_id,salary,created,
             sim,llm_score,llm_reason,first_seen,summary,flags,
             pipeline_state,score_attempts,last_error,updated_at,
             score_status,applied_cap,applied,status,applied_date,note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title, company=excluded.company, location=excluded.location,
              description=excluded.description, url=excluded.url, source=excluded.source,
              source_id=excluded.source_id, salary=excluded.salary, created=excluded.created,
              sim=excluded.sim, llm_score=excluded.llm_score,
              llm_reason=excluded.llm_reason, summary=excluded.summary, flags=excluded.flags,
              pipeline_state=excluded.pipeline_state,
              score_attempts=jobs.score_attempts + excluded.score_attempts,
              last_error=excluded.last_error, updated_at=excluded.updated_at,
              score_status=excluded.score_status, applied_cap=excluded.applied_cap,
              status=CASE WHEN excluded.status='DISQUALIFIED'
                          THEN 'DISQUALIFIED' ELSE jobs.status END""",
            (job["id"], job["title"], job["company"], job["location"],
             job["description"], job["url"], job["source"], job.get("source_id", ""),
             job["salary"], job["created"], job["sim"], job["llm_score"],
             job["llm_reason"], now, job.get("summary", ""),
             json.dumps(job.get("flags", []), ensure_ascii=False),
             job.get("pipeline_state", "INGESTED"), job.get("score_attempts", 0),
             job.get("last_error", ""), now, job.get("score_status", ""),
             job.get("applied_cap"), 0, status, "", ""))

    def count_ready_for_scoring(self) -> int:
        return self.con.execute("""
            SELECT COUNT(*) FROM jobs
            WHERE llm_score IS NULL AND status='待投'
              AND pipeline_state='READY_FOR_LLM'
        """).fetchone()[0]

    def fetch_scoring_queue(
        self, limit: int, exclude_ids: set[str] | None = None,
    ) -> list[tuple]:
        exclude_ids = exclude_ids or set()
        excluded_sql = ""
        params: list[object] = []
        if exclude_ids:
            excluded_sql = f" AND id NOT IN ({','.join('?' for _ in exclude_ids)})"
            params.extend(sorted(exclude_ids))
        params.append(limit)
        return self.con.execute(f"""
            SELECT id, title, company, location, description, flags
            FROM jobs
            WHERE llm_score IS NULL AND status='待投'
              AND pipeline_state='READY_FOR_LLM'{excluded_sql}
            ORDER BY sim DESC
            LIMIT ?
        """, params).fetchall()

    def record_scoring_failure(self, job_id: str, reason: str, now: str) -> None:
        self.con.execute("""
            UPDATE jobs
            SET score_attempts=score_attempts+1, last_error=?, updated_at=?
            WHERE id=?
        """, (reason, now, job_id))

    def record_scoring_success(
        self,
        *,
        job_id: str,
        score: float,
        reason: str,
        summary: str,
        flags: list[dict],
        score_status: str,
        applied_cap: float | None,
        now: str,
    ) -> None:
        self.con.execute("""
            UPDATE jobs
            SET llm_score=?, llm_reason=?, summary=?, flags=?,
                status=CASE WHEN ? THEN 'DISQUALIFIED' ELSE status END,
                pipeline_state=CASE WHEN ? THEN 'DISQUALIFIED' ELSE 'SCORED' END,
                score_attempts=score_attempts+1, last_error='', updated_at=?,
                score_status=?, applied_cap=?
            WHERE id=?
        """, (score, reason, summary, json.dumps(flags, ensure_ascii=False),
              score_status == "DISQUALIFIED", score_status == "DISQUALIFIED",
              now, score_status, applied_cap, job_id))

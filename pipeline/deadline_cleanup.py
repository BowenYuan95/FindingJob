"""Database-wide cleanup of pending jobs whose application deadline has passed."""

import datetime as dt
import json

from config import DB_PATH
from infrastructure.database import database_session
from infrastructure.job_repository import JobRepository
from pipeline.hard_filter import dedup_flags, scan_disqualifiers


def purge_expired_deadlines(
    db_path: str = DB_PATH,
    today: dt.date | None = None,
) -> int:
    """Scan every pending DB row and remove expired jobs from active results."""
    today = today or dt.date.today()
    updated = 0
    with database_session(db_path, initialize=True) as con:
        repo = JobRepository(con)
        after_id = ""
        while True:
            rows = repo.pending_deadline_candidates(after_id, limit=100)
            if not rows:
                break
            for job_id, title, description, flags_json in rows:
                rescanned = scan_disqualifiers(
                    title or "", description or "", today=today
                )
                deadline_flag = next(
                    (flag for flag in rescanned if flag["code"] == "deadline_passed"),
                    None,
                )
                if not deadline_flag:
                    continue
                try:
                    old_flags = json.loads(flags_json) if flags_json else []
                except (TypeError, json.JSONDecodeError):
                    old_flags = []
                flags = dedup_flags(old_flags + [deadline_flag])
                repo.mark_deadline_disqualified(
                    job_id,
                    f"硬性淘汰:截止已过 {deadline_flag['evidence'][:80]}",
                    json.dumps(flags, ensure_ascii=False),
                    dt.datetime.now().isoformat(timespec="seconds"),
                )
                updated += 1
            after_id = rows[-1][0]
    return updated

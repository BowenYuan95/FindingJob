import json
import datetime as dt
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline import backfill_scores, job_matcher
from pipeline.hard_filter import apply_flags, find_deadline
from scripts.refresh_flags import plan_row
from sources import adzuna_search
from sources.gmail_alerts import _denoise, _restore_url
from infrastructure.database import initialize_database
from infrastructure.job_repository import JobRepository
from infrastructure.lmstudio import LMStudioClient
from pipeline.deadline_cleanup import purge_expired_deadlines
from pipeline.job_urls import normalize_job_url


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class PipelineLogicTests(unittest.TestCase):
    def test_find_deadline_returns_closing_date_when_available(self):
        self.assertEqual(
            find_deadline(
                "Applications close 30 July 2026.",
                dt.date(2026, 6, 24),
            ),
            dt.date(2026, 7, 30),
        )
        self.assertIsNone(
            find_deadline("This role starts in July.", dt.date(2026, 6, 24))
        )

    def test_apply_flags_marks_capped_only_when_score_is_reduced(self):
        discipline = [{
            "code": "discipline", "label": "in domain", "cap": 100,
            "severity": "warn", "evidence": "test",
        }]
        self.assertEqual(apply_flags(85, discipline), (85, "ok"))

        scholarship = [{
            "code": "phd_scholarship", "label": "scholarship", "cap": 40,
            "severity": "warn", "evidence": "test",
        }]
        self.assertEqual(apply_flags(85, scholarship), (40, "capped"))
        self.assertEqual(apply_flags(35, scholarship), (35, "ok"))

    def test_gmail_long_url_placeholder_is_reversible(self):
        url = "https://example.test/redirect?token=" + "a" * 120
        body, url_map = _denoise(f"Researcher at Example {url}")
        self.assertNotIn(url, body)
        token = next(iter(url_map))
        self.assertEqual(_restore_url(token, url_map), url)
        self.assertEqual(_restore_url("[URL_1]", url_map), url)

    def test_job_url_validation_rejects_placeholders(self):
        self.assertIsNone(normalize_job_url("[URL]"))
        self.assertEqual(
            normalize_job_url("www.example.com/job"),
            "https://www.example.com/job",
        )

    @patch("sources.adzuna_search._get_with_retry")
    def test_adzuna_deduplicates_native_job_ids(self, get):
        duplicate = {
            "id": "adz-1", "title": "Researcher",
            "company": {"display_name": "Example"},
            "location": {"display_name": "Sydney"},
            "description": "Role", "redirect_url": "https://example.test",
        }
        separate_opening = {**duplicate, "id": "adz-2"}
        get.return_value = FakeResponse(
            {"results": [duplicate, duplicate, separate_opening]}
        )
        with patch.object(adzuna_search, "SEARCHES", [{"what": "researcher"}]), \
             patch.object(adzuna_search, "ADZUNA_PAGES", 1):
            jobs = adzuna_search.fetch_adzuna()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["source_id"], "adz-1")
        self.assertEqual(jobs[1]["source_id"], "adz-2")

    def test_native_source_ids_produce_distinct_job_ids(self):
        first = job_matcher.job_hash("Researcher", "Example", "adzuna", "1")
        second = job_matcher.job_hash("Researcher", "Example", "adzuna", "2")
        self.assertNotEqual(first, second)

    def test_readiness_reports_missing_adzuna_credentials(self):
        with patch.object(job_matcher, "ADZUNA_APP_ID", None), \
             patch.object(job_matcher, "ADZUNA_APP_KEY", None), \
             patch.object(job_matcher, "USE_EMBEDDING", False), \
             patch.object(job_matcher, "USE_LLM_SCORING", False):
            errors, _ = job_matcher.validate_runtime()
        self.assertTrue(any("ADZUNA_APP_ID" in error for error in errors))

    def test_main_persists_adzuna_source_id(self):
        job = {
            "source_id": "adz-42", "title": "XR Researcher",
            "company": "Example", "location": "Melbourne",
            "description": "Build mixed reality systems.",
            "url": "https://example.test/42", "source": "adzuna",
            "salary": "", "created": "2026-06-21",
        }
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            digest = Path(tmp) / "digest.md"
            with patch.object(job_matcher, "DB_PATH", str(db)), \
                 patch.object(job_matcher, "DIGEST_PATH", str(digest)), \
                 patch.object(job_matcher, "validate_runtime", return_value=([], [])), \
                 patch.object(job_matcher, "fetch_adzuna", return_value=[job]), \
                 patch.object(job_matcher, "fetch_gmail_alerts", return_value=[]), \
                 patch.object(job_matcher, "USE_EMBEDDING", False), \
                 patch.object(job_matcher, "USE_LLM_SCORING", False):
                job_matcher.main()
            con = sqlite3.connect(db)
            try:
                row = con.execute("SELECT source, source_id FROM jobs").fetchone()
            finally:
                con.close()
            self.assertEqual(row, ("adzuna", "adz-42"))

    def test_embedding_rejected_job_is_persisted_for_audit(self):
        job = {
            "source_id": "adz-low", "title": "Unrelated Role",
            "company": "Example", "location": "Melbourne",
            "description": "Unrelated work.", "url": "https://example.test/low",
            "source": "adzuna", "salary": "", "created": "2026-06-21",
        }
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            digest = Path(tmp) / "digest.md"
            with patch.object(job_matcher, "DB_PATH", str(db)), \
                 patch.object(job_matcher, "DIGEST_PATH", str(digest)), \
                 patch.object(job_matcher, "validate_runtime", return_value=([], [])), \
                 patch.object(job_matcher, "fetch_adzuna", return_value=[job]), \
                 patch.object(job_matcher, "fetch_gmail_alerts", return_value=[]), \
                 patch.object(job_matcher, "USE_EMBEDDING", True), \
                 patch.object(job_matcher, "USE_LLM_SCORING", False), \
                 patch.object(job_matcher, "embed_batch", side_effect=[
                     [np.array([1.0, 0.0], dtype=np.float32)],
                     [np.array([0.0, 1.0], dtype=np.float32)],
                 ]):
                job_matcher.main()
            con = sqlite3.connect(db)
            try:
                row = con.execute(
                    "SELECT pipeline_state, llm_score FROM jobs"
                ).fetchone()
            finally:
                con.close()
            self.assertEqual(row, ("EMBEDDING_REJECTED", None))

    def test_database_enables_wal_and_identity_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            con = initialize_database(str(db))
            try:
                mode = con.execute("PRAGMA journal_mode").fetchone()[0]
                indexes = {
                    row[1] for row in con.execute("PRAGMA index_list(jobs)").fetchall()
                }
            finally:
                con.close()
            self.assertEqual(mode.lower(), "wal")
            self.assertIn("idx_jobs_source_identity", indexes)
            self.assertIn("idx_jobs_todo_score", indexes)
        self.assertIn("idx_jobs_tracker_date", indexes)

    def test_deadline_cleanup_scans_all_pending_jobs_before_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            con = initialize_database(str(db))
            con.executemany(
                """INSERT INTO jobs(id,title,description,status,pipeline_state,flags)
                   VALUES (?,?,?,?,?,?)""",
                [
                    ("expired", "Old role", "Applications close 1 June 2026.",
                     "待投", "SCORED", "[]"),
                    ("future", "Open role", "Applications close 30 July 2026.",
                     "待投", "SCORED", "[]"),
                    ("applied", "Applied role", "Applications close 1 June 2026.",
                     "已投", "SCORED", "[]"),
                ],
            )
            con.commit()
            con.close()

            count = purge_expired_deadlines(str(db), dt.date(2026, 6, 24))

            con = sqlite3.connect(db)
            try:
                statuses = dict(con.execute("SELECT id, status FROM jobs").fetchall())
            finally:
                con.close()
            self.assertEqual(count, 1)
            self.assertEqual(statuses["expired"], "DISQUALIFIED")
            self.assertEqual(statuses["future"], "待投")
            self.assertEqual(statuses["applied"], "已投")

    def test_todo_pagination_is_stable_and_bounded(self):
        con = initialize_database(":memory:")
        con.executemany(
            """INSERT INTO jobs(
                   id,title,company,source,sim,llm_score,status,pipeline_state)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (f"job-{i:02}", f"Role {i}", "Example", "adzuna", 0.8, None,
                 "待投", "READY_FOR_LLM")
                for i in range(45)
            ],
        )
        repo = JobRepository(con)

        total, average = repo.todo_stats(["adzuna"], 0, "", False)
        first = repo.fetch_todo_page(["adzuna"], 0, "", False, 20, 0)
        second = repo.fetch_todo_page(["adzuna"], 0, "", False, 20, 20)
        last = repo.fetch_todo_page(["adzuna"], 0, "", False, 20, 40)

        self.assertEqual((total, average), (45, 80.0))
        self.assertEqual(len(first), 20)
        self.assertEqual(len(second), 20)
        self.assertEqual(len(last), 5)
        self.assertEqual(first[0]["id"], "job-00")
        self.assertEqual(second[0]["id"], "job-20")
        self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})
        con.close()

    def test_todo_pagination_applies_all_filters(self):
        con = initialize_database(":memory:")
        con.executemany(
            """INSERT INTO jobs(
                   id,title,company,source,sim,llm_score,status,pipeline_state)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                ("1", "ML Researcher", "Alpha", "adzuna", 0.7, 90,
                 "待投", "SCORED"),
                ("2", "Data Scientist", "Alpha", "adzuna", 0.8, None,
                 "待投", "READY_FOR_LLM"),
                ("3", "ML Engineer", "Beta", "gmail", 0.95, 95,
                 "待投", "SCORED"),
                ("4", "ML Researcher", "Alpha", "adzuna", 0.99, 99,
                 "DISQUALIFIED", "DISQUALIFIED"),
            ],
        )
        repo = JobRepository(con)

        total, average = repo.todo_stats(["adzuna"], 85, "research", True)
        rows = repo.fetch_todo_page(["adzuna"], 85, "research", True, 20, 0)

        self.assertEqual((total, average), (1, 90.0))
        self.assertEqual([row["id"] for row in rows], ["1"])
        self.assertEqual(repo.todo_stats([], 0, "", False)[0], 0)
        con.close()

    def test_tracker_pagination_and_counts(self):
        con = initialize_database(":memory:")
        con.executemany(
            """INSERT INTO jobs(id,title,status,applied_date)
               VALUES (?,?,?,?)""",
            [
                (f"applied-{i:02}", f"Role {i}", "已投", f"2026-06-{i + 1:02}")
                for i in range(25)
            ] + [("interview", "Interview", "面试", "2026-07-01")],
        )
        repo = JobRepository(con)

        counts = repo.tracker_status_counts(["已投", "面试", "拒", "offer"])
        second = repo.fetch_tracker_page(["已投"], 20, 20)

        self.assertEqual(counts, {"已投": 25, "面试": 1})
        self.assertEqual(repo.count_tracker(["已投"]), 25)
        self.assertEqual(len(second), 5)
        self.assertEqual(second[-1]["id"], "applied-00")
        self.assertEqual(repo.count_tracker([]), 0)
        self.assertEqual(repo.fetch_tracker_page([], 20, 0), [])
        con.close()

    @patch("infrastructure.lmstudio.time.sleep")
    @patch("infrastructure.lmstudio.requests.post")
    def test_embedding_retries_transient_failure(self, post, _sleep):
        post.side_effect = [
            OSError("temporary model failure"),
            FakeResponse({"data": [{"embedding": [1.0, 2.0]}]}),
        ]
        client = LMStudioClient("http://test/v1")
        vectors = client.embeddings(["text"], "embed-model")
        np.testing.assert_array_equal(vectors[0], np.array([1.0, 2.0]))
        self.assertEqual(post.call_count, 2)

    @patch("pipeline.job_matcher.time.sleep")
    @patch.object(job_matcher.LM_CLIENT, "chat_completion")
    def test_llm_review_retries_and_maps_discipline_in_python(self, chat, _sleep):
        content = json.dumps({
            "score": 90,
            "reason": "test",
            "flags": [],
            "summary": [],
            "discipline": {
                "discipline_class": "out_of_domain",
                "discipline_multiplier": 1.0,
                "discipline_reason": "test",
            },
        }, ensure_ascii=False)
        chat.side_effect = [
            OSError("temporary model failure"),
            {
                "choices": [{"message": {"content": content}}],
            },
        ]

        score, _, _, flags = job_matcher.llm_review("profile", {"title": "role"})

        self.assertEqual(score, 45)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(flags[-1]["label"], "学科:out_of_domain×0.5")

    def test_backfill_knockout_updates_status(self):
        con = initialize_database(":memory:")
        con.execute(
            """INSERT INTO jobs(id,llm_score,llm_reason,summary,flags,status,pipeline_state)
               VALUES (?,?,?,?,?,?,?)""",
            ("1", None, "", "", "[]", "待投", "READY_FOR_LLM"),
        )
        row = ("1", "role", "company", "location", "description", "[]")
        with patch.object(
            backfill_scores,
            "llm_review",
            return_value=(80, "reason", "summary", ["clearance"]),
        ):
            self.assertTrue(backfill_scores.score_one(con, row))

        score, status = con.execute(
            "SELECT llm_score, status FROM jobs WHERE id='1'"
        ).fetchone()
        self.assertEqual(score, 5)
        self.assertEqual(status, "DISQUALIFIED")
        con.close()

    @patch("pipeline.backfill_scores.subprocess.run")
    @patch.object(backfill_scores.LM_CLIENT, "loaded_models")
    def test_backfill_does_not_load_when_api_is_unreachable(self, loaded_models, run):
        loaded_models.side_effect = OSError("server unavailable")

        self.assertEqual(backfill_scores.ensure_llm_loaded(), (False, False))
        run.assert_not_called()

    @patch("pipeline.backfill_scores.ensure_llm_loaded")
    @patch("pipeline.backfill_scores.pending_count", return_value=0)
    @patch("pipeline.backfill_scores.acquire_single_instance")
    def test_backfill_with_empty_queue_does_not_wake_model(
        self, acquire_lock, _pending_count, ensure_loaded,
    ):
        lock = MagicMock()
        acquire_lock.return_value = lock
        with patch.object(sys, "argv", ["backfill_scores"]):
            backfill_scores.main()

        ensure_loaded.assert_not_called()
        lock.close.assert_called_once()

    def test_failed_job_is_skipped_for_current_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "jobs.db"
            con = initialize_database(str(db))
            con.execute(
                """INSERT INTO jobs(
                       id,title,company,location,description,flags,sim,llm_score,status,
                       pipeline_state)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                ("1", "role", "company", "location", "desc", "[]", 0.8, None,
                 "待投", "READY_FOR_LLM"),
            )
            con.commit()
            con.close()
            with patch.object(backfill_scores, "DB_PATH", str(db)), patch.object(
                backfill_scores, "llm_review", return_value=(None, "failed", "", [])
            ):
                self.assertEqual(backfill_scores.run_once(), 0)

    def test_refresh_flags_preserves_discipline(self):
        discipline = [{
            "code": "discipline", "label": "学科:adjacent×0.8", "cap": 100,
            "severity": "warn", "evidence": "test",
        }]
        flags, score, status, action = plan_row(
            "AI Engineer", "Build ML systems", json.dumps(discipline), 72, "待投"
        )
        self.assertEqual(action, "unchanged")
        self.assertEqual(score, 72)
        self.assertEqual(status, "待投")
        self.assertEqual(flags[0]["code"], "discipline")

    def test_refresh_flags_repairs_unchanged_knockout_status(self):
        knockout = [{
            "code": "clearance", "label": "LLM:clearance", "cap": 5,
            "severity": "knockout", "evidence": "test",
        }]
        _, score, status, action = plan_row(
            "Researcher", "Research role", json.dumps(knockout), 5, "待投"
        )
        self.assertEqual(action, "knockout")
        self.assertEqual(score, 5)
        self.assertEqual(status, "DISQUALIFIED")

        _, score, status, action = plan_row(
            "Researcher", "Research role", json.dumps(knockout), 5, "DISQUALIFIED"
        )
        self.assertEqual(action, "unchanged")


if __name__ == "__main__":
    unittest.main()

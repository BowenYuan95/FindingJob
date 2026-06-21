import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline import backfill_scores, job_matcher
from scripts.refresh_flags import plan_row
from sources import adzuna_search
from infrastructure.database import initialize_database


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class PipelineLogicTests(unittest.TestCase):
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

    @patch("pipeline.job_matcher.time.sleep")
    @patch("pipeline.job_matcher.requests.post")
    def test_embedding_retries_transient_failure(self, post, _sleep):
        post.side_effect = [
            OSError("temporary model failure"),
            FakeResponse({"data": [{"embedding": [1.0, 2.0]}]}),
        ]
        vectors = job_matcher.embed_batch(["text"])
        np.testing.assert_array_equal(vectors[0], np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(post.call_count, 2)

    @patch("pipeline.job_matcher.time.sleep")
    @patch("pipeline.job_matcher.requests.post")
    def test_llm_review_retries_and_maps_discipline_in_python(self, post, _sleep):
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
        post.side_effect = [
            OSError("temporary model failure"),
            FakeResponse({
                "choices": [{"message": {"content": content}}],
            }),
        ]

        score, _, _, flags = job_matcher.llm_review("profile", {"title": "role"})

        self.assertEqual(score, 45)
        self.assertEqual(post.call_count, 2)
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

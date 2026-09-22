from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase8.app.repository import LearnerNotFoundError, Repository, warning_messages
from phase8.tests.helpers import build_fixture_database


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "fixture.sqlite3"
        build_fixture_database(self.database_path)
        self.repository = Repository(self.database_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_recommendations_are_ranked_and_plain_by_default(self):
        rows = self.repository.recommendations(
            "learner-a", candidate_policy="all_supported",
            relevance_definition="attempted", limit=10,
        )
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertNotIn("audit_explanation", rows[0])
        self.assertEqual(len(rows[0]["warning_messages"]), 1)

    def test_audit_fields_are_opt_in(self):
        row = self.repository.recommendations(
            "learner-a", candidate_policy="all_supported",
            relevance_definition="attempted", include_audit=True,
        )[0]
        self.assertEqual(row["evidence_warning"], "limited_supported_history")
        self.assertIn("audit_explanation", row)

    def test_unknown_learner_is_explicit(self):
        with self.assertRaises(LearnerNotFoundError):
            self.repository.profile("missing")

    def test_teacher_filter_does_not_create_risk_labels(self):
        rows = self.repository.teacher_learners(warning="limited_supported_history")
        self.assertEqual([row["learner_id"] for row in rows], ["learner-a"])
        self.assertNotIn("risk", rows[0])

    def test_route_filter_and_overview_use_saved_evidence(self):
        rows = self.repository.teacher_learners(dominant_route="content")
        self.assertEqual([row["learner_id"] for row in rows], ["learner-a"])
        overview = self.repository.teacher_overview()
        self.assertEqual(overview["learner_count"], 1)
        self.assertEqual(overview["history_groups"][0]["learner_count"], 1)
        self.assertTrue(any(
            row["source"] == "phase6_decision.csv"
            for row in overview["saved_metrics"]
        ))

    def test_public_profile_excludes_validation_future_counts(self):
        profile = self.repository.profile("learner-a")
        self.assertNotIn("future_interactions", profile)
        self.assertNotIn("future_problem_items", profile)

    def test_warning_translation_handles_known_and_unknown_codes(self):
        self.assertIn("small amount", warning_messages("limited_supported_history")[0])
        self.assertEqual(
            warning_messages("unrecognised_code"),
            ["Additional evidence limitations apply."],
        )


if __name__ == "__main__":
    unittest.main()

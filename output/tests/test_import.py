from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase8.app.config import Settings
from phase8.app.database import connect_database
from phase8.app.import_data import (
    ImportContractError, _validate_config, audit_database, validate_handoff,
)
from phase8.tests.helpers import build_fixture_database


class ImportContractTests(unittest.TestCase):
    def settings(self, root: Path) -> Settings:
        return Settings(
            phase2_root=root / "phase2",
            phase2_manifest=root / "artifact_manifest.csv",
            phase6_root=root / "phase6",
            phase7_root=root / "phase7",
            database_path=root / "phase8.sqlite3",
            api_url="http://127.0.0.1:8000",
            dashboard_port=8501,
        )

    def test_selected_model_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(Path(directory))
            config = {
                "phase": 6,
                "selected_project_model": "different_model",
                "selected_target_success": 0.55,
                "selected_dkt_weight": 0.02,
            }
            with self.assertRaisesRegex(ImportContractError, "selected_project_model mismatch"):
                _validate_config(config, settings, 6)

    def test_missing_phase2_handoff_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ImportContractError, "Missing Phase 2 learner split"):
                validate_handoff(self.settings(Path(directory)))

    def test_database_audit_checks_primary_serving_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "fixture.sqlite3"
            build_fixture_database(database_path)
            connection = connect_database(database_path, read_only=True)
            try:
                audit = audit_database(
                    connection, self.settings(root), enforce_expected_count=False
                )
            finally:
                connection.close()
            self.assertEqual(audit["learner_count"], 1)
            self.assertEqual(audit["maximum_recommendations_per_learner"], 2)
            self.assertEqual(audit["empty_explanations"], 0)
            self.assertEqual(audit["serving_future_columns"], [])


if __name__ == "__main__":
    unittest.main()

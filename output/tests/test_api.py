from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from phase8.app.repository import Repository
from phase8.tests.helpers import build_fixture_database


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("pydantic"),
    "FastAPI dependencies are not installed",
)
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from phase8.app.api import app, get_repository

        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "fixture.sqlite3"
        build_fixture_database(database_path)
        app.dependency_overrides[get_repository] = lambda: Repository(database_path)
        self.app = app
        self.client = TestClient(app)

    def tearDown(self):
        self.app.dependency_overrides.clear()
        self.temporary_directory.cleanup()

    def test_health_and_metadata(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["selected_model"], "gated_hybrid_with_lstm_dkt")

    def test_recommendation_limit_and_unknown_learner(self):
        response = self.client.get("/learners/learner-a/recommendations?limit=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["recommendations"]), 1)
        self.assertEqual(self.client.get("/learners/missing/profile").status_code, 404)

    def test_invalid_limit_is_rejected(self):
        self.assertEqual(
            self.client.get("/learners/learner-a/recommendations?limit=21").status_code,
            422,
        )

    def test_all_public_endpoints_and_privacy_headers(self):
        paths = [
            "/learners",
            "/learners/learner-a/profile",
            "/learners/learner-a/skills",
            "/teacher/overview",
            "/teacher/learners?dominant_route=cf",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_profile_does_not_expose_validation_future_fields(self):
        body = self.client.get("/learners/learner-a/profile").json()
        self.assertNotIn("future_interactions", body)
        self.assertNotIn("future_problem_items", body)

    def test_policy_isolation_and_enumeration(self):
        response = self.client.get(
            "/learners/learner-a/recommendations?candidate_policy=novel_only"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        self.assertEqual(
            self.client.get(
                "/learners/learner-a/recommendations?candidate_policy=invalid"
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/teacher/learners?dominant_route=invalid").status_code,
            422,
        )

    def test_openapi_contains_documented_contract(self):
        schema = self.client.get("/openapi.json").json()
        expected = {
            "/health", "/metadata", "/learners",
            "/learners/{learner_id}/profile",
            "/learners/{learner_id}/skills",
            "/learners/{learner_id}/recommendations",
            "/teacher/overview", "/teacher/learners",
        }
        self.assertTrue(expected.issubset(schema["paths"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import httpx

from phase8.app.api_client import APIClient, APIClientError


class ApiClientTests(unittest.TestCase):
    def test_client_uses_primary_slice_and_returns_recommendations(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/learners/learner-a/recommendations")
            self.assertEqual(request.url.params["candidate_policy"], "all_supported")
            self.assertEqual(request.url.params["relevance_definition"], "attempted")
            return httpx.Response(200, json={"recommendations": [{"rank": 1}]})

        client = APIClient("http://test", transport=httpx.MockTransport(handler))
        self.assertEqual(client.recommendations("learner-a"), [{"rank": 1}])

    def test_server_error_becomes_readable_dashboard_error(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(503, json={"detail": "Database not ready"})
        )
        with self.assertRaisesRegex(APIClientError, "Database not ready"):
            APIClient("http://test", transport=transport).health()


if __name__ == "__main__":
    unittest.main()

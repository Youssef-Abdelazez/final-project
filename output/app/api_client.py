"""Small HTTP client used by the Streamlit dashboard."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class APIClientError(RuntimeError):
    """Readable dashboard error for connection and API failures."""


class APIClient:
    def __init__(
        self, base_url: str, *, timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        clean_params = {
            key: value for key, value in (params or {}).items() if value is not None
        }
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=self.timeout, transport=self.transport
            ) as client:
                response = client.get(path, params=clean_params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as error:
            try:
                detail = error.response.json().get("detail", error.response.text)
            except ValueError:
                detail = error.response.text
            raise APIClientError(f"The recommendation service returned an error: {detail}") from error
        except (httpx.RequestError, ValueError) as error:
            raise APIClientError(
                "The recommendation service is unavailable. Start the API and confirm "
                "PHASE8_API_URL before retrying."
            ) from error

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def metadata(self) -> dict[str, Any]:
        return self._get("/metadata")

    def learners(self, *, query: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self._get("/learners", params={"q": query, "limit": limit})

    def profile(self, learner_id: str) -> dict[str, Any]:
        return self._get(f"/learners/{quote(learner_id, safe='')}/profile")

    def skills(self, learner_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        return self._get(
            f"/learners/{quote(learner_id, safe='')}/skills", params={"limit": limit}
        )

    def recommendations(self, learner_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        response = self._get(
            f"/learners/{quote(learner_id, safe='')}/recommendations",
            params={
                "candidate_policy": "all_supported",
                "relevance_definition": "attempted",
                "limit": limit,
            },
        )
        return response["recommendations"]

    def teacher_overview(self) -> dict[str, Any]:
        return self._get("/teacher/overview")

    def teacher_learners(
        self, *, cold_start: bool | None = None, state_source: str | None = None,
        warning: str | None = None, dominant_route: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._get(
            "/teacher/learners",
            params={
                "cold_start": cold_start,
                "state_source": state_source,
                "warning": warning,
                "dominant_route": dominant_route,
                "limit": limit,
            },
        )

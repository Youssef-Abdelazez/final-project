"""FastAPI contract for the frozen Phase 8 recommendation demonstrator."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response

from .config import get_settings
from .repository import DatabaseUnavailableError, LearnerNotFoundError, Repository
from .schemas import (
    HealthResponse, LearnerProfile, LearnerSummary, MetadataResponse,
    RecommendationResponse, SkillState, TeacherOverview,
)


app = FastAPI(
    title="Personalised Educational Recommendation Demonstrator",
    version="0.1.0",
    description=(
        "Read-only access to frozen ASSISTments recommendations."
    ),
)


@app.middleware("http")
async def privacy_headers(request, call_next) -> Response:
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def get_repository() -> Repository:
    return Repository(get_settings().database_path)


def _not_ready(error: DatabaseUnavailableError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


def _not_found(learner_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Unknown learner: {learner_id}")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(repository: Repository = Depends(get_repository)):
    return repository.health()


@app.get("/metadata", response_model=MetadataResponse, tags=["system"])
def metadata(repository: Repository = Depends(get_repository)):
    try:
        return repository.metadata()
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error


@app.get("/learners", response_model=list[LearnerSummary], tags=["learners"])
def learners(
    q: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=100),
    repository: Repository = Depends(get_repository),
):
    try:
        return repository.list_learners(query=q, limit=limit)
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error


@app.get("/learners/{learner_id}/profile", response_model=LearnerProfile, tags=["learners"])
def learner_profile(learner_id: str, repository: Repository = Depends(get_repository)):
    try:
        return repository.profile(learner_id)
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error
    except LearnerNotFoundError as error:
        raise _not_found(learner_id) from error


@app.get("/learners/{learner_id}/skills", response_model=list[SkillState], tags=["learners"])
def learner_skills(
    learner_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    repository: Repository = Depends(get_repository),
):
    try:
        return repository.skill_states(learner_id, limit=limit)
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error
    except LearnerNotFoundError as error:
        raise _not_found(learner_id) from error


@app.get(
    "/learners/{learner_id}/recommendations",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def learner_recommendations(
    learner_id: str,
    candidate_policy: Literal["all_supported", "novel_only"] = "all_supported",
    relevance_definition: Literal["attempted", "successful"] = "attempted",
    limit: int = Query(default=10, ge=1, le=20),
    include_audit: bool = False,
    repository: Repository = Depends(get_repository),
):
    try:
        rows = repository.recommendations(
            learner_id,
            candidate_policy=candidate_policy,
            relevance_definition=relevance_definition,
            limit=limit,
            include_audit=include_audit,
        )
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error
    except LearnerNotFoundError as error:
        raise _not_found(learner_id) from error
    return {
        "learner_id": learner_id,
        "candidate_policy": candidate_policy,
        "relevance_definition": relevance_definition,
        "recommendations": rows,
    }


@app.get("/teacher/overview", response_model=TeacherOverview, tags=["educator"])
def teacher_overview(repository: Repository = Depends(get_repository)):
    try:
        return repository.teacher_overview()
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error


@app.get("/teacher/learners", response_model=list[LearnerSummary], tags=["educator"])
def teacher_learners(
    cold_start: bool | None = None,
    state_source: str | None = Query(default=None, max_length=80),
    warning: str | None = Query(default=None, max_length=80),
    dominant_route: Literal[
        "cf", "sequential", "content", "popularity", "dkt", "unresolved_base"
    ] | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    repository: Repository = Depends(get_repository),
):
    try:
        return repository.teacher_learners(
            cold_start=cold_start, state_source=state_source,
            warning=warning, dominant_route=dominant_route, limit=limit,
        )
    except DatabaseUnavailableError as error:
        raise _not_ready(error) from error

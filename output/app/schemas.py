"""Typed public response models for the Phase 8 API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database_ready: bool
    artifact_version: str | None = None
    detail: str | None = None


class MetadataResponse(BaseModel):
    artifact_version: str | None = None
    imported_at: str | None = None
    selected_model: str
    selected_target_success: float
    selected_dkt_weight: float
    primary_candidate_policy: str
    primary_relevance_definition: str
    primary_k: int
    row_counts: dict[str, int] = Field(default_factory=dict)
    schema_version: str | int
    database_audit: dict[str, Any] = Field(default_factory=dict)
    import_duration_seconds: float | None = None
    database_size_bytes: int | None = None


class LearnerSummary(BaseModel):
    learner_id: str
    cold_start_problem_history: bool
    state_source: str | None = None
    supported_early_interactions: int | None = None
    history_evidence_strength: float | None = None


class LearnerProfile(LearnerSummary):
    cohort: str
    early_interactions: int | None = None
    early_problem_items: int | None = None
    imported_at: str


class SkillState(BaseModel):
    skill_id: str
    predicted_correctness: float
    state_source: str | None = None
    supported_early_interactions: int | None = None
    history_evidence_strength: float | None = None


class Recommendation(BaseModel):
    learner_id: str
    candidate_policy: str
    relevance_definition: str
    rank: int
    item_id: str
    problem_label: str | None = None
    score: float | None = None
    dominant_component: str | None = None
    route_label: str
    primary_skill_id: str | None = None
    skill_label: str | None = None
    predicted_correctness: float | None = None
    state_source: str | None = None
    supported_early_interactions: int | None = None
    dashboard_explanation: str
    warning_messages: list[str] = Field(default_factory=list)
    audit_explanation: str | None = None
    evidence_warning: str | None = None
    has_shap_evidence: bool | None = None
    imported_at: str | None = None


class RecommendationResponse(BaseModel):
    learner_id: str
    candidate_policy: str
    relevance_definition: str
    recommendations: list[Recommendation]


class TeacherOverview(BaseModel):
    learner_count: int
    recommendation_count: int
    routes: list[dict[str, Any]]
    evidence_sources: list[dict[str, Any]]
    history_groups: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    saved_metrics: list[dict[str, Any]]

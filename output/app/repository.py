"""Read-only service layer shared by the API and dashboard."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

from .database import connect_database


WARNING_MESSAGES = {
    "missing_supported_skill_mapping": "No supported skill label was available for this problem.",
    "limited_supported_history": "This explanation is based on only a small amount of personal history.",
    "prior_state_not_lstm_evidence": "A general learner pattern was used because there was not enough personal history.",
    "popularity_fallback": "Overall problem popularity was used as fallback evidence.",
    "unresolved_nested_base_provenance": "Part of the ranking evidence could not be separated into individual components.",
}

ROUTE_LABELS = {
    "cf": "similar learners",
    "sequential": "recent learning order",
    "content": "related skills",
    "popularity": "overall popularity",
    "dkt": "estimated learning level",
    "unresolved_base": "available learning information",
}


class DatabaseUnavailableError(RuntimeError):
    pass


class LearnerNotFoundError(LookupError):
    pass


def warning_messages(codes: str | None) -> list[str]:
    if not codes or codes == "none":
        return []
    return [WARNING_MESSAGES.get(code, "Additional evidence limitations apply.") for code in codes.split(";")]


class Repository:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    def _connect(self):
        if not self.database_path.is_file():
            raise DatabaseUnavailableError(
                f"Database not found at {self.database_path}. Run the Phase 8 import first."
            )
        return connect_database(self.database_path, read_only=True)

    def health(self) -> dict[str, Any]:
        if not self.database_path.is_file():
            return {"status": "not_ready", "database_ready": False, "artifact_version": None}
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value FROM artifact_metadata WHERE key = 'artifact_version'"
                ).fetchone()
                connection.execute("SELECT 1 FROM recommendations LIMIT 1").fetchone()
            return {
                "status": "ready",
                "database_ready": True,
                "artifact_version": json.loads(row["value"]) if row else None,
            }
        except Exception as error:
            return {
                "status": "not_ready", "database_ready": False,
                "artifact_version": None, "detail": str(error),
            }

    def metadata(self) -> dict[str, Any]:
        keys = (
            "artifact_version", "imported_at", "selected_model",
            "selected_target_success", "selected_dkt_weight",
            "primary_candidate_policy", "primary_relevance_definition", "primary_k",
            "row_counts", "schema_version", "database_audit",
            "import_duration_seconds", "database_size_bytes",
        )
        placeholders = ",".join("?" for _ in keys)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT key, value FROM artifact_metadata WHERE key IN ({placeholders})", keys
            ).fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    def list_learners(self, *, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = """SELECT learner_id, cold_start_problem_history, state_source,
                 supported_early_interactions, history_evidence_strength
                 FROM learners"""
        parameters: list[Any] = []
        if query:
            sql += " WHERE learner_id LIKE ?"
            parameters.append(f"%{query}%")
        sql += " ORDER BY learner_id LIMIT ?"
        parameters.append(limit)
        with closing(self._connect()) as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def profile(self, learner_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT learner_id, cohort, cold_start_problem_history,
                   early_interactions, early_problem_items, state_source,
                   supported_early_interactions, history_evidence_strength, imported_at
                   FROM learners WHERE learner_id = ?""",
                (learner_id,),
            ).fetchone()
        if row is None:
            raise LearnerNotFoundError(learner_id)
        result = dict(row)
        result["cold_start_problem_history"] = bool(result["cold_start_problem_history"])
        return result

    def skill_states(self, learner_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        self.profile(learner_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT skill_id, predicted_correctness, state_source,
                   supported_early_interactions, history_evidence_strength
                   FROM skill_states WHERE learner_id = ?
                   ORDER BY predicted_correctness, skill_id LIMIT ?""",
                (learner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recommendations(
        self, learner_id: str, *, candidate_policy: str,
        relevance_definition: str, limit: int = 10, include_audit: bool = False,
    ) -> list[dict[str, Any]]:
        self.profile(learner_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM recommendations
                   WHERE learner_id = ? AND candidate_policy = ? AND relevance_definition = ?
                   ORDER BY rank LIMIT ?""",
                (learner_id, candidate_policy, relevance_definition, limit),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["has_shap_evidence"] = bool(item["has_shap_evidence"])
            item["route_label"] = ROUTE_LABELS.get(
                item.get("dominant_component"), "available learning information"
            )
            item["warning_messages"] = warning_messages(item.get("evidence_warning"))
            if not include_audit:
                item.pop("audit_explanation", None)
                item.pop("evidence_warning", None)
                item.pop("has_shap_evidence", None)
                item.pop("imported_at", None)
            results.append(item)
        return results

    def teacher_overview(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            learner_count = connection.execute("SELECT COUNT(*) FROM learners").fetchone()[0]
            recommendation_count = connection.execute(
                "SELECT COUNT(*) FROM recommendations"
            ).fetchone()[0]
            routes = [dict(row) for row in connection.execute(
                """SELECT dominant_component AS route, COUNT(*) AS recommendation_count
                   FROM recommendations GROUP BY dominant_component ORDER BY recommendation_count DESC"""
            ).fetchall()]
            for route in routes:
                route["route_label"] = ROUTE_LABELS.get(
                    route.get("route"), "available learning information"
                )
            evidence_sources = [dict(row) for row in connection.execute(
                """SELECT COALESCE(state_source, 'unavailable') AS state_source,
                   COUNT(*) AS learner_count FROM learners GROUP BY state_source
                   ORDER BY learner_count DESC"""
            ).fetchall()]
            history_groups = [dict(row) for row in connection.execute(
                """SELECT cold_start_problem_history,
                   COUNT(*) AS learner_count FROM learners
                   GROUP BY cold_start_problem_history ORDER BY cold_start_problem_history"""
            ).fetchall()]
            warning_rows = connection.execute(
                "SELECT evidence_warning FROM recommendations"
            ).fetchall()
            coverage = [dict(row) for row in connection.execute(
                "SELECT source, row_number, payload_json FROM system_metrics ORDER BY source, row_number"
            ).fetchall()]
        warning_counts: dict[str, int] = {}
        for row in warning_rows:
            for code in str(row["evidence_warning"]).split(";"):
                warning_counts[code] = warning_counts.get(code, 0) + 1
        return {
            "learner_count": learner_count,
            "recommendation_count": recommendation_count,
            "routes": routes,
            "evidence_sources": evidence_sources,
            "history_groups": history_groups,
            "warnings": [
                {
                    "code": code,
                    "message": (
                        warning_messages(code)[0]
                        if warning_messages(code)
                        else "No recorded explanation limitation."
                    ),
                    "recommendation_count": count,
                }
                for code, count in sorted(warning_counts.items(), key=lambda value: -value[1])
            ],
            "saved_metrics": [
                {"source": row["source"], "row_number": row["row_number"],
                 "values": json.loads(row["payload_json"])} for row in coverage
            ],
        }

    def teacher_learners(
        self, *, cold_start: bool | None = None, state_source: str | None = None,
        warning: str | None = None, dominant_route: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = """SELECT DISTINCT l.learner_id, l.cold_start_problem_history,
                 l.state_source, l.supported_early_interactions,
                 l.history_evidence_strength FROM learners l"""
        clauses: list[str] = []
        parameters: list[Any] = []
        if warning or dominant_route:
            sql += " JOIN recommendations r ON r.learner_id = l.learner_id"
        if warning:
            clauses.append("(';' || r.evidence_warning || ';') LIKE ?")
            parameters.append(f"%;{warning};%")
        if dominant_route:
            clauses.append("r.dominant_component = ?")
            parameters.append(dominant_route)
        if cold_start is not None:
            clauses.append("l.cold_start_problem_history = ?")
            parameters.append(int(cold_start))
        if state_source:
            clauses.append("l.state_source = ?")
            parameters.append(state_source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY l.learner_id LIMIT ?"
        parameters.append(limit)
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

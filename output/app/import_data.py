"""Validate the frozen Phase 2/6/7 handoff and build the Phase 8 database."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import Settings, get_settings
from .database import connect_database, initialise_database


class ImportContractError(RuntimeError):
    """Raised when frozen artifacts do not satisfy the application contract."""


PHASE6_FILES = (
    "artifact_manifest.csv",
    "phase6_config.json",
    "validation_state_summary.csv",
    "validation_skill_states.parquet",
    "comparison_metrics.csv",
    "paired_uncertainty.csv",
    "phase6_decision.csv",
    "prediction_metrics.csv",
)
PHASE7_FILES = (
    "artifact_manifest.csv",
    "phase7_config.json",
    "recommendation_explanations.parquet",
    "global_component_summary.csv",
    "route_summary.csv",
    "explanation_coverage.csv",
    "warning_summary.csv",
    "explanation_quality_metrics.csv",
)
PHASE6_METRIC_FILES = (
    "comparison_metrics.csv",
    "paired_uncertainty.csv",
    "phase6_decision.csv",
    "prediction_metrics.csv",
)
PHASE7_METRIC_FILES = PHASE7_FILES[3:]
EXPECTED_VALIDATION_LEARNERS = 6_667


def _require_files(root: Path, filenames: Iterable[str], label: str) -> None:
    missing = [str(root / name) for name in filenames if not (root / name).is_file()]
    if missing:
        raise ImportContractError(f"{label} handoff is incomplete; missing: {', '.join(missing)}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ImportContractError(f"Cannot read valid JSON from {path}: {error}") from error


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise ImportContractError(f"Cannot read manifest {path}: {error}") from error


def _validate_manifest(
    root: Path, manifest_path: Path, required: Iterable[str], *, dataset_name: str | None = None
) -> None:
    rows = _manifest_rows(manifest_path)
    for filename in required:
        if filename == "artifact_manifest.csv":
            continue
        matches = [row for row in rows if Path(row.get("File", "")).name == filename]
        if dataset_name is not None:
            matches = [
                row for row in matches
                if row.get("Dataset", "").casefold() == dataset_name.casefold()
            ]
        if len(matches) != 1:
            raise ImportContractError(f"{filename} is not declared in {manifest_path}")
        row = matches[0]
        path = root / row["File"]
        if not path.is_file():
            raise ImportContractError(f"Manifest entry does not exist: {path}")
        expected_bytes = row.get("Bytes", "").strip()
        if expected_bytes and path.stat().st_size != int(float(expected_bytes)):
            raise ImportContractError(
                f"Byte-size mismatch for {path}: expected {expected_bytes}, found {path.stat().st_size}"
            )
        expected_rows = row.get("Rows", "").strip()
        if expected_rows:
            if path.suffix == ".parquet":
                observed_rows = pq.ParquetFile(path).metadata.num_rows
            elif path.suffix == ".csv":
                observed_rows = len(_manifest_rows(path))
            else:
                observed_rows = 1
            if observed_rows != int(float(expected_rows)):
                raise ImportContractError(
                    f"Row-count mismatch for {path}: expected {expected_rows}, found {observed_rows}"
                )


def _validate_config(config: dict[str, Any], settings: Settings, phase: int) -> None:
    expected = {
        "selected_project_model": settings.selected_model,
        "selected_target_success": settings.selected_target_success,
        "selected_dkt_weight": settings.selected_dkt_weight,
    }
    if int(config.get("phase", -1)) != phase:
        raise ImportContractError(f"Expected Phase {phase} config, found {config.get('phase')!r}")
    for key, value in expected.items():
        observed = config.get(key)
        if isinstance(value, float):
            matches = observed is not None and abs(float(observed) - value) < 1e-12
        else:
            matches = observed == value
        if not matches:
            raise ImportContractError(f"{key} mismatch: expected {value!r}, found {observed!r}")
    if phase == 7:
        expected_slice = {
            "primary_candidate_policy": settings.primary_candidate_policy,
            "primary_relevance_definition": settings.primary_relevance_definition,
            "primary_k": settings.primary_k,
        }
        for key, value in expected_slice.items():
            if config.get(key) != value:
                raise ImportContractError(
                    f"Phase 7 {key} mismatch: expected {value!r}, found {config.get(key)!r}"
                )


def _parquet_batches(path: Path, columns: list[str]):
    arrow_dataset = ds.dataset(path, format="parquet")
    missing = sorted(set(columns) - set(arrow_dataset.schema.names))
    if missing:
        raise ImportContractError(f"{path.name} is missing required columns: {missing}")
    yield from arrow_dataset.to_batches(columns=columns, batch_size=25_000)


def _value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if value is None else value


def _optional(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _import_learners(connection: sqlite3.Connection, settings: Settings, imported_at: str) -> int:
    path = settings.phase2_root / "learner_splits.parquet"
    columns = [
        "learner_id", "cohort", "cold_start_problem_history",
        "early_interactions", "early_catalog_problems",
    ]
    count = 0
    for batch in _parquet_batches(path, columns):
        rows = []
        for row in batch.to_pylist():
            if row["cohort"] != "validation":
                continue
            rows.append((
                str(row["learner_id"]), "validation",
                int(bool(row["cold_start_problem_history"])),
                _optional(row, "early_interactions"),
                _optional(row, "early_catalog_problems"),
                imported_at,
            ))
        connection.executemany(
            """INSERT INTO learners(
                learner_id, cohort, cold_start_problem_history,
                early_interactions, early_problem_items, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        count += len(rows)
    if count == 0:
        raise ImportContractError("Phase 2 contains no validation learners")
    return count


def _import_state_summary(connection: sqlite3.Connection, settings: Settings) -> int:
    path = settings.phase6_root / "validation_state_summary.csv"
    rows = _manifest_rows(path)
    updates = [(
        _optional(row, "StateSource"),
        int(float(_value(row, "SupportedEarlyInteractionsUsed", 0))),
        _optional(row, "HistoryEvidenceStrength"),
        str(row["learner_id"]),
    ) for row in rows]
    connection.executemany(
        """UPDATE learners SET state_source = ?, supported_early_interactions = ?,
           history_evidence_strength = ? WHERE learner_id = ?""",
        updates,
    )
    if connection.execute("SELECT COUNT(*) FROM learners WHERE state_source IS NULL").fetchone()[0]:
        raise ImportContractError("Phase 6 state summary does not cover every validation learner")
    return len(updates)


def _import_skill_states(connection: sqlite3.Connection, settings: Settings) -> int:
    path = settings.phase6_root / "validation_skill_states.parquet"
    columns = [
        "learner_id", "skill_item_id", "PredictedCorrectness",
        "SupportedEarlyInteractionsUsed", "StateSource", "HistoryEvidenceStrength",
    ]
    count = 0
    for batch in _parquet_batches(path, columns):
        rows = [(
            str(row["learner_id"]), str(row["skill_item_id"]),
            float(row["PredictedCorrectness"]), _optional(row, "StateSource"),
            int(_value(row, "SupportedEarlyInteractionsUsed", 0)),
            _optional(row, "HistoryEvidenceStrength"),
        ) for row in batch.to_pylist()]
        connection.executemany(
            """INSERT INTO skill_states(
                learner_id, skill_id, predicted_correctness, state_source,
                supported_early_interactions, history_evidence_strength
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        count += len(rows)
    return count


def _import_recommendations(
    connection: sqlite3.Connection, settings: Settings, imported_at: str
) -> int:
    path = settings.phase7_root / "recommendation_explanations.parquet"
    arrow_dataset = ds.dataset(path, format="parquet")
    required = {
        "learner_id", "item_id", "rank", "CandidatePolicy", "RelevanceDefinition",
        "dominant_component", "DashboardExplanation", "AuditExplanation",
        "EvidenceWarning", "HasShapEvidence",
    }
    missing = sorted(required - set(arrow_dataset.schema.names))
    if missing:
        raise ImportContractError(f"{path.name} is missing required columns: {missing}")
    optional = [
        "problem_label", "score", "final_score", "primary_skill_id", "skill_label",
        "explanation_predicted_correctness", "state_source_for_primary_skill",
        "dkt_state_source", "supported_early_interactions_used",
    ]
    columns = sorted(required) + [c for c in optional if c in arrow_dataset.schema.names]
    filter_expression = (
        (ds.field("CandidatePolicy") == settings.primary_candidate_policy)
        & (ds.field("RelevanceDefinition") == settings.primary_relevance_definition)
        & (ds.field("rank") <= settings.primary_k)
    )
    count = 0
    for batch in arrow_dataset.to_batches(
        columns=columns, filter=filter_expression, batch_size=25_000
    ):
        rows = []
        for row in batch.to_pylist():
            explanation = str(_value(row, "DashboardExplanation", "")).strip()
            if not explanation:
                raise ImportContractError("A primary recommendation has no dashboard explanation")
            rows.append((
                str(row["learner_id"]), row["CandidatePolicy"], row["RelevanceDefinition"],
                int(row["rank"]), str(row["item_id"]),
                _optional(row, "problem_label"),
                _optional(row, "score", "final_score"),
                _optional(row, "dominant_component"),
                _optional(row, "primary_skill_id"), _optional(row, "skill_label"),
                _optional(row, "explanation_predicted_correctness"),
                _optional(row, "state_source_for_primary_skill", "dkt_state_source"),
                _optional(row, "supported_early_interactions_used"),
                explanation, _optional(row, "AuditExplanation"),
                str(_value(row, "EvidenceWarning", "none")),
                int(bool(_value(row, "HasShapEvidence", False))), imported_at,
            ))
        connection.executemany(
            """INSERT INTO recommendations(
                learner_id, candidate_policy, relevance_definition, rank, item_id,
                problem_label, score, dominant_component, primary_skill_id, skill_label,
                predicted_correctness, state_source, supported_early_interactions,
                dashboard_explanation, audit_explanation, evidence_warning,
                has_shap_evidence, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        count += len(rows)
    if count == 0:
        raise ImportContractError("No recommendations matched the frozen primary slice")
    return count


def _import_metrics(connection: sqlite3.Connection, settings: Settings) -> int:
    count = 0
    sources = [
        (settings.phase6_root, filename) for filename in PHASE6_METRIC_FILES
    ] + [
        (settings.phase7_root, filename) for filename in PHASE7_METRIC_FILES
    ]
    for root, filename in sources:
        for row_number, row in enumerate(_manifest_rows(root / filename), start=1):
            connection.execute(
                "INSERT INTO system_metrics(source, row_number, payload_json) VALUES (?, ?, ?)",
                (filename, row_number, json.dumps(row, ensure_ascii=False)),
            )
            count += 1
    return count


def audit_database(
    connection: sqlite3.Connection, settings: Settings, *, enforce_expected_count: bool = True
) -> dict[str, Any]:
    learner_count = connection.execute("SELECT COUNT(*) FROM learners").fetchone()[0]
    recommendation_count = connection.execute(
        "SELECT COUNT(*) FROM recommendations"
    ).fetchone()[0]
    maximum_per_learner = connection.execute(
        """SELECT COALESCE(MAX(item_count), 0) FROM (
               SELECT COUNT(*) AS item_count FROM recommendations GROUP BY learner_id
           )"""
    ).fetchone()[0]
    empty_explanations = connection.execute(
        """SELECT COUNT(*) FROM recommendations
           WHERE TRIM(COALESCE(dashboard_explanation, '')) = ''"""
    ).fetchone()[0]
    slice_rows = connection.execute(
        """SELECT candidate_policy, relevance_definition, COUNT(*) AS row_count
           FROM recommendations GROUP BY candidate_policy, relevance_definition"""
    ).fetchall()
    imported_slices = [dict(row) for row in slice_rows]
    exposed_tables = ("learners", "skill_states", "recommendations")
    prohibited_columns = []
    for table in exposed_tables:
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall():
            if "future" in str(row["name"]).casefold():
                prohibited_columns.append(f"{table}.{row['name']}")

    if enforce_expected_count and learner_count != EXPECTED_VALIDATION_LEARNERS:
        raise ImportContractError(
            f"Expected {EXPECTED_VALIDATION_LEARNERS} validation learners, found {learner_count}"
        )
    if maximum_per_learner > settings.primary_k:
        raise ImportContractError(
            f"A learner has {maximum_per_learner} recommendations; maximum is {settings.primary_k}"
        )
    if empty_explanations:
        raise ImportContractError(
            f"{empty_explanations} imported recommendations have no dashboard explanation"
        )
    expected_slice = [{
        "candidate_policy": settings.primary_candidate_policy,
        "relevance_definition": settings.primary_relevance_definition,
        "row_count": recommendation_count,
    }]
    if imported_slices != expected_slice:
        raise ImportContractError(
            f"Imported recommendation slices do not match the primary contract: {imported_slices}"
        )
    if prohibited_columns:
        raise ImportContractError(
            "Validation-future fields reached serving tables: " + ", ".join(prohibited_columns)
        )
    return {
        "expected_validation_learners": EXPECTED_VALIDATION_LEARNERS,
        "learner_count": learner_count,
        "recommendation_count": recommendation_count,
        "maximum_recommendations_per_learner": maximum_per_learner,
        "empty_explanations": empty_explanations,
        "imported_slices": imported_slices,
        "serving_future_columns": prohibited_columns,
    }


def _write_build_manifest(path: Path, payload: dict[str, Any]) -> None:
    target = path.with_suffix(".build.json")
    temporary = target.with_suffix(".build.json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def validate_handoff(settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    phase2_file = settings.phase2_root / "learner_splits.parquet"
    if not phase2_file.is_file():
        raise ImportContractError(f"Missing Phase 2 learner split: {phase2_file}")
    if not settings.phase2_manifest.is_file():
        raise ImportContractError(f"Missing Phase 2 manifest: {settings.phase2_manifest}")
    _require_files(settings.phase6_root, PHASE6_FILES, "Phase 6")
    _require_files(settings.phase7_root, PHASE7_FILES, "Phase 7")
    _validate_manifest(
        settings.phase2_root, settings.phase2_manifest,
        ("learner_splits.parquet",), dataset_name="ASSISTments",
    )
    _validate_manifest(settings.phase6_root, settings.phase6_root / "artifact_manifest.csv", PHASE6_FILES)
    _validate_manifest(settings.phase7_root, settings.phase7_root / "artifact_manifest.csv", PHASE7_FILES)
    phase6_config = _read_json(settings.phase6_root / "phase6_config.json")
    phase7_config = _read_json(settings.phase7_root / "phase7_config.json")
    _validate_config(phase6_config, settings, 6)
    _validate_config(phase7_config, settings, 7)
    return phase6_config, phase7_config


def build_database(settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    started = time.perf_counter()
    phase6_config, phase7_config = validate_handoff(settings)
    imported_at = datetime.now(timezone.utc).isoformat()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="phase8-", suffix=".sqlite3", dir=settings.database_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        connection = connect_database(temporary_path)
        try:
            initialise_database(connection)
            counts = {
                "learners": _import_learners(connection, settings, imported_at),
                "state_summaries": _import_state_summary(connection, settings),
                "skill_states": _import_skill_states(connection, settings),
                "recommendations": _import_recommendations(connection, settings, imported_at),
                "system_metrics": _import_metrics(connection, settings),
            }
            audit = audit_database(connection, settings)
            import_duration_seconds = round(time.perf_counter() - started, 3)
            metadata = {
                "artifact_version": imported_at,
                "imported_at": imported_at,
                "selected_model": settings.selected_model,
                "selected_target_success": settings.selected_target_success,
                "selected_dkt_weight": settings.selected_dkt_weight,
                "primary_candidate_policy": settings.primary_candidate_policy,
                "primary_relevance_definition": settings.primary_relevance_definition,
                "primary_k": settings.primary_k,
                "phase2_manifest": str(settings.phase2_manifest),
                "phase6_root": str(settings.phase6_root),
                "phase7_root": str(settings.phase7_root),
                "phase6_config": phase6_config,
                "phase7_config": phase7_config,
                "row_counts": counts,
                "database_audit": audit,
                "import_duration_seconds": import_duration_seconds,
            }
            connection.executemany(
                "INSERT OR REPLACE INTO artifact_metadata(key, value) VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
            )
            connection.commit()
            # Store the allocated SQLite size after metadata has been written
            # A second pass makes the recorded value stable if the first size row itself causes allocation of another page
            for _ in range(2):
                database_size_bytes = (
                    connection.execute("PRAGMA page_count").fetchone()[0]
                    * connection.execute("PRAGMA page_size").fetchone()[0]
                )
                metadata["database_size_bytes"] = database_size_bytes
                connection.execute(
                    "INSERT OR REPLACE INTO artifact_metadata(key, value) VALUES (?, ?)",
                    ("database_size_bytes", json.dumps(database_size_bytes)),
                )
                connection.commit()
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ImportContractError("Imported database contains invalid learner references")
        finally:
            connection.close()
        os.replace(temporary_path, settings.database_path)
        _write_build_manifest(settings.database_path, metadata)
        return counts
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        counts = build_database()
    except ImportContractError as error:
        raise SystemExit(f"Phase 8 import failed: {error}") from None
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

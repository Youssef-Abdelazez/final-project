"""SQLite connection and schema helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = "2"


def connect_database(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path).resolve()
    if read_only:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialise_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS learners (
            learner_id TEXT PRIMARY KEY,
            cohort TEXT NOT NULL,
            cold_start_problem_history INTEGER NOT NULL CHECK (cold_start_problem_history IN (0, 1)),
            early_interactions INTEGER,
            early_problem_items INTEGER,
            state_source TEXT,
            supported_early_interactions INTEGER,
            history_evidence_strength REAL,
            imported_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS skill_states (
            learner_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            predicted_correctness REAL NOT NULL,
            state_source TEXT,
            supported_early_interactions INTEGER,
            history_evidence_strength REAL,
            PRIMARY KEY (learner_id, skill_id),
            FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            learner_id TEXT NOT NULL,
            candidate_policy TEXT NOT NULL,
            relevance_definition TEXT NOT NULL,
            rank INTEGER NOT NULL CHECK (rank > 0),
            item_id TEXT NOT NULL,
            problem_label TEXT,
            score REAL,
            dominant_component TEXT,
            primary_skill_id TEXT,
            skill_label TEXT,
            predicted_correctness REAL,
            state_source TEXT,
            supported_early_interactions INTEGER,
            dashboard_explanation TEXT NOT NULL,
            audit_explanation TEXT,
            evidence_warning TEXT NOT NULL,
            has_shap_evidence INTEGER NOT NULL CHECK (has_shap_evidence IN (0, 1)),
            imported_at TEXT NOT NULL,
            PRIMARY KEY (learner_id, candidate_policy, relevance_definition, rank),
            UNIQUE (learner_id, candidate_policy, relevance_definition, item_id),
            FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
        );

        CREATE TABLE IF NOT EXISTS system_metrics (
            source TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (source, row_number)
        );

        CREATE TABLE IF NOT EXISTS artifact_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_skill_states_learner
            ON skill_states(learner_id);
        CREATE INDEX IF NOT EXISTS idx_skill_states_source
            ON skill_states(state_source);
        CREATE INDEX IF NOT EXISTS idx_recommendations_learner_rank
            ON recommendations(learner_id, rank);
        CREATE INDEX IF NOT EXISTS idx_recommendations_slice_rank
            ON recommendations(candidate_policy, relevance_definition, rank);
        CREATE INDEX IF NOT EXISTS idx_recommendations_component
            ON recommendations(dominant_component);
        CREATE INDEX IF NOT EXISTS idx_recommendations_warning
            ON recommendations(evidence_warning);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO artifact_metadata(key, value) VALUES (?, ?)",
        ("schema_version", SCHEMA_VERSION),
    )
    connection.commit()

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from phase8.app.database import connect_database, initialise_database


def build_fixture_database(path: Path) -> None:
    with closing(connect_database(path)) as connection:
        initialise_database(connection)
        connection.execute(
            """INSERT INTO learners VALUES
            ('learner-a', 'validation', 0, 20, 9,
             'lstm_dkt_early_history', 20, 0.5, '2026-09-18T00:00:00+00:00')"""
        )
        connection.execute(
            """INSERT INTO skill_states VALUES
            ('learner-a', 'skill:1', 0.42, 'lstm_dkt_early_history', 20, 0.5)"""
        )
        connection.executemany(
            """INSERT INTO recommendations VALUES
            (?, 'all_supported', 'attempted', ?, ?, ?, ?, ?, 'skill:1', 'Fractions',
             0.42, 'lstm_dkt_early_history', 20, ?, ?, ?, 0, '2026-09-18T00:00:00+00:00')""",
            [
                (
                    "learner-a", 1, "problem:1", "Problem 1", 0.9, "cf",
                    "Problem 1 is recommended because similar learners worked on it.",
                    "Technical audit text.", "limited_supported_history",
                ),
                (
                    "learner-a", 2, "problem:2", "Problem 2", 0.8, "content",
                    "Problem 2 is recommended because it practises related skills.",
                    "Technical audit text.", "none",
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO system_metrics(source, row_number, payload_json) VALUES (?, ?, ?)",
            [
                (
                    "phase6_decision.csv", 1,
                    json.dumps({
                        "SelectedProjectModel": "gated_hybrid_with_lstm_dkt",
                        "RecallDeltaVsCurrentHybrid": "0.0001",
                        "NDCGDeltaVsCurrentHybrid": "0.0001",
                    }),
                ),
                (
                    "paired_uncertainty.csv", 1,
                    json.dumps({"Metric": "NDCGAt10", "Lower95": "-0.0001", "Upper95": "0.0002"}),
                ),
            ],
        )
        metadata = {
            "artifact_version": "2026-09-18T00:00:00+00:00",
            "imported_at": "2026-09-18T00:00:00+00:00",
            "selected_model": "gated_hybrid_with_lstm_dkt",
            "selected_target_success": 0.55,
            "selected_dkt_weight": 0.02,
            "primary_candidate_policy": "all_supported",
            "primary_relevance_definition": "attempted",
            "primary_k": 10,
            "row_counts": {"learners": 1, "recommendations": 2},
        }
        connection.executemany(
            "INSERT OR REPLACE INTO artifact_metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.commit()

"""Environment-driven configuration for the frozen Phase 8 handoff."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PHASE8_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    phase2_root: Path
    phase2_manifest: Path
    phase6_root: Path
    phase7_root: Path
    database_path: Path
    api_url: str
    dashboard_port: int
    selected_model: str = "gated_hybrid_with_lstm_dkt"
    selected_target_success: float = 0.55
    selected_dkt_weight: float = 0.02
    primary_candidate_policy: str = "all_supported"
    primary_relevance_definition: str = "attempted"
    primary_k: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        phase2_root = _path_from_env(
            "PHASE2_ROOT", PHASE8_ROOT / "data" / "source" / "phase2"
        )
        return cls(
            phase2_root=phase2_root,
            phase2_manifest=_path_from_env(
                "PHASE2_MANIFEST", phase2_root.parent / "artifact_manifest.csv"
            ),
            phase6_root=_path_from_env(
                "PHASE6_ROOT", PHASE8_ROOT / "data" / "source" / "phase6"
            ),
            phase7_root=_path_from_env(
                "PHASE7_ROOT", PHASE8_ROOT / "data" / "source" / "phase7"
            ),
            database_path=_path_from_env(
                "PHASE8_DATABASE", PHASE8_ROOT / "data" / "phase8.sqlite3"
            ),
            api_url=os.getenv("PHASE8_API_URL", "http://127.0.0.1:8000"),
            dashboard_port=int(os.getenv("PHASE8_DASHBOARD_PORT", "8501")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()

"""SQLite cache so a feature is never explained twice."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class ExplanationCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
                key TEXT PRIMARY KEY,
                feature_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                detection_score REAL,
                fuzzing_score REAL,
                explained_by TEXT,
                cost_usd REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def make_key(feature_id: int, model: str, fingerprint: str) -> str:
        return f"{feature_id}:{model}:{fingerprint}"

    def get(self, key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT feature_id, text, detection_score, fuzzing_score, explained_by,"
            " cost_usd, created_at FROM explanations WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "feature_id": row[0], "text": row[1], "detection_score": row[2],
            "fuzzing_score": row[3], "explained_by": row[4], "cost_usd": row[5],
            "created_at": row[6],
        }

    def put(
        self,
        key: str,
        *,
        feature_id: int,
        text: str,
        detection_score: float | None = None,
        fuzzing_score: float | None = None,
        explained_by: str = "",
        cost_usd: float = 0.0,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO explanations"
            " (key, feature_id, text, detection_score, fuzzing_score, explained_by, cost_usd)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, feature_id, text, detection_score, fuzzing_score, explained_by, cost_usd),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def fingerprint_examples(examples) -> str:
    """Stable hash of the example set — same examples => cache hit."""
    import hashlib

    blob = json.dumps(examples.all_positive(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

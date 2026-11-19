"""Persistent AutoInterp results keyed by checkpoint, feature, model and prompt."""

import hashlib
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from probekit.autointerp.scorer import DetectionScore


def checkpoint_id(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class Label:
    description: str
    score: DetectionScore | None


class LabelCache:
    def __init__(self, path: Path, sae_id: str, model: str, prompt_version: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.scope = (sae_id, model, prompt_version)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    sae_id TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
                    feature_id INTEGER NOT NULL, description TEXT NOT NULL,
                    accuracy REAL, auroc REAL, examples INTEGER,
                    PRIMARY KEY (sae_id, feature_id, model, prompt_version)
                )
            """)

    def _get(self, db: sqlite3.Connection, feature_id: int) -> Label | None:
        row = db.execute(
            """
            SELECT description, accuracy, auroc, examples FROM labels
            WHERE sae_id = ? AND model = ? AND prompt_version = ? AND feature_id = ?
        """,
            (*self.scope, feature_id),
        ).fetchone()
        if row is None:
            return None
        score = DetectionScore(*row[1:]) if row[3] is not None else None
        return Label(row[0], score)

    def label(self, feature_id: int, explain: Callable[[], str]) -> str:
        with closing(sqlite3.connect(self.path, timeout=120)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            cached = self._get(db, feature_id)
            if cached is not None:
                return cached.description
            description = explain().strip()
            if not description:
                raise ValueError("explainer returned an empty label")
            db.execute(
                """
                INSERT INTO labels (sae_id, model, prompt_version, feature_id, description)
                VALUES (?, ?, ?, ?, ?)
            """,
                (*self.scope, feature_id, description),
            )
            return description

    def score(self, feature_id: int, detect: Callable[[str], DetectionScore]) -> DetectionScore:
        with closing(sqlite3.connect(self.path, timeout=120)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            cached = self._get(db, feature_id)
            if cached is None:
                raise ValueError("cache the explanation before scoring")
            if cached.score is not None:
                return cached.score
            score = detect(cached.description)
            db.execute(
                """
                UPDATE labels SET accuracy = ?, auroc = ?, examples = ?
                WHERE sae_id = ? AND model = ? AND prompt_version = ? AND feature_id = ?
            """,
                (score.accuracy, score.auroc, score.examples, *self.scope, feature_id),
            )
            return score

    def descriptions(self) -> dict[int, str]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                """
                SELECT feature_id, description FROM labels
                WHERE sae_id = ? AND model = ? AND prompt_version = ?
            """,
                self.scope,
            )
            return dict(rows)

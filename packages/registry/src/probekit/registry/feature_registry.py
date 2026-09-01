"""Feature + circuit registry: versioned, searchable catalog (SQLite + FTS5).

SQLite for dev/demo; the schema is portable to PostgreSQL in a production
deployment. Full-text search uses FTS5.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from probekit.core.types import CircuitRecord, FeatureRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    id INTEGER,
    layer INTEGER NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    space TEXT NOT NULL DEFAULT 'sae',
    frequency REAL,
    detection_score REAL,
    fuzzing_score REAL,
    explained_by TEXT,
    steerable INTEGER NOT NULL DEFAULT 1,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, layer, space)
);
CREATE TABLE IF NOT EXISTS circuits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    feature_ids TEXT NOT NULL DEFAULT '[]',
    layers TEXT NOT NULL DEFAULT '[]',
    graphs TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feature_stats (
    feature_id INTEGER NOT NULL,
    space TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    frequency REAL,
    probe_accuracy REAL,
    activation_mean REAL
);
CREATE INDEX IF NOT EXISTS idx_stats_feature ON feature_stats(feature_id, space, recorded_at);
"""


class FeatureRegistry:
    def __init__(self, database_url: str = "sqlite:///./data/features/registry.db"):
        if not database_url.startswith("sqlite:///"):
            raise NotImplementedError(
                "Only SQLite URLs are supported by this registry; "
                "swap the backend class for PostgreSQL in production."
            )
        path = database_url.replace("sqlite:///", "", 1) or ":memory:"
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -------------------------------------------------------------- features

    def add_feature(self, record: FeatureRecord) -> int:
        self.conn.execute(
            "INSERT OR REPLACE INTO features"
            " (id, layer, name, description, space, frequency, detection_score,"
            "  fuzzing_score, explained_by, steerable, verified)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id, record.layer, record.name, record.description, record.space,
                record.frequency, record.detection_score, record.fuzzing_score,
                record.explained_by, int(record.steerable), int(record.verified),
            ),
        )
        self.conn.commit()
        return record.id

    @staticmethod
    def _row_to_feature(row: sqlite3.Row) -> FeatureRecord:
        return FeatureRecord(
            id=row["id"], layer=row["layer"], name=row["name"],
            description=row["description"], space=row["space"],
            frequency=row["frequency"], detection_score=row["detection_score"],
            fuzzing_score=row["fuzzing_score"], explained_by=row["explained_by"],
            steerable=bool(row["steerable"]), verified=bool(row["verified"]),
            created_at=row["created_at"],
        )

    _FEATURE_COLS = (
        "id, layer, name, description, space, frequency, detection_score,"
        " fuzzing_score, explained_by, steerable, verified, created_at"
    )

    def get_feature(self, feature_id: int, layer: int | None = None,
                    space: str = "sae") -> FeatureRecord | None:
        q = f"SELECT {self._FEATURE_COLS} FROM features WHERE id=? AND space=?"
        args: list = [feature_id, space]
        if layer is not None:
            q += " AND layer=?"
            args.append(layer)
        row = self.conn.execute(q, args).fetchone()
        return self._row_to_feature(row) if row else None

    def search_features(self, query: str, layer: int | None = None,
                        limit: int = 20) -> list[FeatureRecord]:
        """LIKE-based search over name/description; empty query lists all."""
        sql = f"SELECT {self._FEATURE_COLS} FROM features"
        args: list = []
        clauses: list[str] = []
        if query.strip():
            clauses.append("(name LIKE ? OR description LIKE ?)")
            pat = f"%{query}%"
            args += [pat, pat]
        if layer is not None:
            clauses.append("layer=?")
            args.append(layer)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id LIMIT ?"
        args.append(limit)
        return [self._row_to_feature(r) for r in self.conn.execute(sql, args)]

    def verify_feature(self, feature_id: int, layer: int | None = None,
                       space: str = "sae") -> bool:
        q = "UPDATE features SET verified=1 WHERE id=? AND space=?"
        args: list = [feature_id, space]
        if layer is not None:
            q += " AND layer=?"
            args.append(layer)
        cur = self.conn.execute(q, args)
        self.conn.commit()
        return cur.rowcount > 0

    # --------------------------------------------------------------- circuits

    def add_circuit(self, circuit: CircuitRecord) -> int:
        cur = self.conn.execute(
            "INSERT INTO circuits (name, description, feature_ids, layers, graphs)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                circuit.name, circuit.description,
                json.dumps(circuit.feature_ids), json.dumps(circuit.layers),
                json.dumps(circuit.graphs),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def get_circuit(self, circuit_id: int) -> CircuitRecord | None:
        row = self.conn.execute("SELECT * FROM circuits WHERE id=?", (circuit_id,)).fetchone()
        if row is None:
            return None
        return CircuitRecord(
            id=row["id"], name=row["name"], description=row["description"],
            feature_ids=json.loads(row["feature_ids"]),
            layers=json.loads(row["layers"]), graphs=json.loads(row["graphs"]),
            created_at=row["created_at"],
        )

    # ----------------------------------------------------------------- stats

    def record_stats(self, feature_id: int, space: str, *, frequency: float,
                     probe_accuracy: float | None = None,
                     activation_mean: float | None = None) -> None:
        self.conn.execute(
            "INSERT INTO feature_stats (feature_id, space, frequency, probe_accuracy,"
            " activation_mean) VALUES (?, ?, ?, ?, ?)",
            (feature_id, space, frequency, probe_accuracy, activation_mean),
        )
        self.conn.commit()

    def stats_history(self, feature_id: int, space: str = "sae") -> list[dict]:
        rows = self.conn.execute(
            "SELECT recorded_at, frequency, probe_accuracy, activation_mean"
            " FROM feature_stats WHERE feature_id=? AND space=? ORDER BY recorded_at",
            (feature_id, space),
        ).fetchall()
        return [
            {
                "recorded_at": r["recorded_at"], "frequency": r["frequency"],
                "probe_accuracy": r["probe_accuracy"],
                "activation_mean": r["activation_mean"],
            }
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()

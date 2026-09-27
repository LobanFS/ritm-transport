"""Durable, optional collection of causally valid production examples."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from backend.arrivals import CurrentDeviation
from common.contracts import Prediction, PredictionRequest


SCHEMA_VERSION = "production-delay-example-v1"
RECENT_TARGET_CACHE = 100_000


def _json(value) -> str:
    return json.dumps(value.model_dump(mode="json"), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))


def sampling_horizon_s(tr_id: int, target_id: str) -> int:
    """Stable, approximately uniform target horizon in the allowed window."""
    digest = hashlib.sha256(f"{tr_id}:{target_id}".encode()).digest()
    return 601 + int.from_bytes(digest[:8], "big") % 300


class LearningStore:
    """SQLite journal kept outside in-memory dispatcher state.

    Predictions and labels are written independently.  The export joins them
    only when the request existed before the operational arrival became known.
    Opening a fresh connection per operation keeps the class safe if FastAPI
    moves an operation to another thread later.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    tr_id INTEGER NOT NULL,
                    target_id TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    target_scheduled_at TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    method TEXT NOT NULL CHECK (method = 'learned'),
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (tr_id, target_id)
                );
                CREATE INDEX IF NOT EXISTS predictions_target
                    ON predictions(tr_id, target_id, issued_at);
                CREATE UNIQUE INDEX IF NOT EXISTS predictions_one_per_target
                    ON predictions(tr_id, target_id);
                CREATE TABLE IF NOT EXISTS arrival_labels (
                    tr_id INTEGER NOT NULL,
                    target_id TEXT NOT NULL,
                    arrived_at TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    target_delay_s REAL NOT NULL,
                    source TEXT NOT NULL CHECK (source = 'arrival'),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tr_id, target_id)
                );
            """)
            connection.commit()
            self._recorded_targets = set(connection.execute(
                "SELECT tr_id,target_id FROM predictions ORDER BY created_at DESC LIMIT ?",
                (RECENT_TARGET_CACHE,),
            ).fetchall())

    def record_prediction(self, request: PredictionRequest, prediction: Prediction,
                          *, published_at: datetime) -> bool:
        if prediction.method != "learned":
            return False
        if prediction.request_id != request.request_id or prediction.target != request.target:
            raise ValueError("Prediction does not match learning request")
        if published_at < request.issued_at:
            raise ValueError("Prediction cannot be published before it was issued")
        lead_s = (request.target.scheduled_at-request.issued_at).total_seconds()
        if lead_s > sampling_horizon_s(request.tr_id, request.target.id):
            return False
        target_key = (request.tr_id, request.target.id)
        if target_key in self._recorded_targets:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            cursor = connection.execute("""
                INSERT OR IGNORE INTO predictions
                (request_id,tr_id,target_id,issued_at,target_scheduled_at,published_at,
                 model_version,method,request_json,response_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (request.request_id, request.tr_id, request.target.id,
                  request.issued_at.isoformat(), request.target.scheduled_at.isoformat(),
                  published_at.isoformat(), prediction.model_version, prediction.method,
                  _json(request), _json(prediction), now))
            if cursor.rowcount == 0:
                by_id = connection.execute(
                    "SELECT tr_id,target_id,request_json,response_json FROM predictions WHERE request_id=?",
                    (request.request_id,),
                ).fetchone()
                expected = (request.tr_id, request.target.id, _json(request), _json(prediction))
                if by_id is not None and by_id != expected:
                    raise ValueError("request_id already contains different learning data")
            connection.commit()
            if len(self._recorded_targets) >= RECENT_TARGET_CACHE:
                self._recorded_targets.pop()
            self._recorded_targets.add(target_key)
            return cursor.rowcount == 1

    def record_arrival(self, tr_id: int, deviation: CurrentDeviation) -> bool:
        if deviation.source != "arrival" or deviation.planned_stop_id is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            cursor = connection.execute("""
                INSERT OR IGNORE INTO arrival_labels
                (tr_id,target_id,arrived_at,received_at,target_delay_s,source,created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (tr_id, deviation.planned_stop_id,
                  deviation.observed_at.isoformat(), deviation.received_at.isoformat(),
                  deviation.delay_s, deviation.source, now))
            if cursor.rowcount == 0:
                stored = connection.execute("""
                    SELECT arrived_at,received_at,target_delay_s,source
                    FROM arrival_labels WHERE tr_id=? AND target_id=?
                """, (tr_id, deviation.planned_stop_id)).fetchone()
                expected = (deviation.observed_at.isoformat(), deviation.received_at.isoformat(),
                            deviation.delay_s, deviation.source)
                if stored != expected:
                    raise ValueError("arrival label already contains a different fact")
            connection.commit()
            return cursor.rowcount == 1

    def examples(self):
        """Return deterministic rows whose label was unavailable at request time."""
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("""
                SELECT p.*, l.arrived_at, l.received_at AS label_received_at,
                       l.target_delay_s
                FROM predictions p
                JOIN arrival_labels l
                  ON l.tr_id=p.tr_id AND l.target_id=p.target_id
                WHERE p.issued_at < l.received_at
                ORDER BY p.issued_at, p.request_id
            """).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            predictions = connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
            labels = connection.execute("SELECT count(*) FROM arrival_labels").fetchone()[0]
            examples = connection.execute("""
                SELECT count(*) FROM predictions p JOIN arrival_labels l
                ON l.tr_id=p.tr_id AND l.target_id=p.target_id
                WHERE p.issued_at < l.received_at
            """).fetchone()[0]
        return {"predictions": predictions, "labels": labels, "examples": examples}

# SPDX-License-Identifier: Apache-2.0
"""Persistent SQLite-backed job store for the RUNE API server."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rune_bench.metrics import MetricsEvent


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    tenant_id: str
    kind: str
    status: str
    request_payload: dict
    result_payload: dict | None
    error: str | None
    message: str | None
    created_at: float
    updated_at: float


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    tenant_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, operation, idempotency_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    event TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL,
                    error_type TEXT,
                    labels_json TEXT,
                    recorded_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_events_job_id ON workflow_events(job_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_events_event ON workflow_events(event)"
            )

    def mark_incomplete_jobs_failed(self, message: str = "server restarted before job completion") -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?, message = ?, updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                ("failed", message, message, now),
            )

    def create_job(
        self,
        *,
        tenant_id: str,
        kind: str,
        request_payload: dict,
        idempotency_key: str | None = None,
    ) -> tuple[str, bool]:
        now = time.time()
        with self._connect() as conn:
            if idempotency_key:
                existing = conn.execute(
                    """
                    SELECT job_id
                    FROM idempotency_keys
                    WHERE tenant_id = ? AND operation = ? AND idempotency_key = ?
                    """,
                    (tenant_id, kind, idempotency_key),
                ).fetchone()
                if existing is not None:
                    return str(existing["job_id"]), False

            job_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO jobs(job_id, tenant_id, kind, status, request_json, result_json, error, message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    tenant_id,
                    kind,
                    "queued",
                    json.dumps(request_payload, sort_keys=True),
                    None,
                    None,
                    "accepted",
                    now,
                    now,
                ),
            )
            if idempotency_key:
                conn.execute(
                    """
                    INSERT INTO idempotency_keys(tenant_id, operation, idempotency_key, job_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (tenant_id, kind, idempotency_key, job_id, now),
                )
            return job_id, True

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result_payload: dict | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []

        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if result_payload is not None:
            fields.append("result_json = ?")
            values.append(json.dumps(result_payload, sort_keys=True))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if message is not None:
            fields.append("message = ?")
            values.append(message)

        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(job_id)

        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values)  # nosec B608 — fields are hardcoded column names, values are parameterized

    def record_workflow_event(self, event: "MetricsEvent") -> None:
        """Persist a single workflow lifecycle event."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_events(job_id, event, status, duration_ms, error_type, labels_json, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.job_id,
                    event.event,
                    event.status,
                    event.duration_ms,
                    event.error_type,
                    json.dumps(event.labels, sort_keys=True) if event.labels else None,
                    event.recorded_at,
                ),
            )

    def get_events_summary(self, *, job_id: str | None = None) -> list[dict]:
        """Return per-event aggregate statistics.

        When *job_id* is given, only events for that job are included.
        Rows are ordered by event name and include count, ok/error split,
        and avg/min/max duration in milliseconds — suitable for cross-run comparison.
        """
        query = """
            SELECT
                event,
                COUNT(*)                                              AS total,
                SUM(CASE WHEN status = 'ok'    THEN 1 ELSE 0 END)   AS ok_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)    AS error_count,
                ROUND(AVG(duration_ms), 1)                           AS avg_ms,
                ROUND(MIN(duration_ms), 1)                           AS min_ms,
                ROUND(MAX(duration_ms), 1)                           AS max_ms
            FROM workflow_events
        """
        params: list[object] = []
        if job_id is not None:
            query += " WHERE job_id = ?"
            params.append(job_id)
        query += " GROUP BY event ORDER BY event"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "event": str(row["event"]),
                "total": int(row["total"]),
                "ok": int(row["ok_count"]),
                "error": int(row["error_count"]),
                "avg_ms": float(row["avg_ms"] or 0.0),
                "min_ms": float(row["min_ms"] or 0.0),
                "max_ms": float(row["max_ms"] or 0.0),
            }
            for row in rows
        ]

    def get_events_for_job(self, job_id: str) -> list[dict]:
        """Return all raw workflow events for a single job, ordered by recorded_at."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event, status, duration_ms, error_type, labels_json, recorded_at
                FROM workflow_events
                WHERE job_id = ?
                ORDER BY recorded_at
                """,
                (job_id,),
            ).fetchall()

        return [
            {
                "event": str(row["event"]),
                "status": str(row["status"]),
                "duration_ms": float(row["duration_ms"] or 0.0),
                "error_type": str(row["error_type"]) if row["error_type"] else None,
                "labels": json.loads(row["labels_json"]) if row["labels_json"] else {},
                "recorded_at": float(row["recorded_at"]),
            }
            for row in rows
        ]

    def get_job(self, job_id: str, *, tenant_id: str | None = None) -> JobRecord | None:
        query = "SELECT * FROM jobs WHERE job_id = ?"
        params: list[object] = [job_id]
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()

        if row is None:
            return None

        return JobRecord(
            job_id=str(row["job_id"]),
            tenant_id=str(row["tenant_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            request_payload=json.loads(row["request_json"]),
            result_payload=json.loads(row["result_json"]) if row["result_json"] else None,
            error=str(row["error"]) if row["error"] is not None else None,
            message=str(row["message"]) if row["message"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

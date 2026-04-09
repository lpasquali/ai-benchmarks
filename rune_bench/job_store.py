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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chain_state (
                    job_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    # ── Chain state ────────────────────────────────────────────────────────
    #
    # Persists the live execution state of a multi-agent chain (DAG) keyed by
    # job_id. The full state is stored as a single JSON blob (one row per job)
    # to keep node-status updates atomic and avoid per-node row management.
    #
    # Schema of state_json:
    #   {
    #     "nodes": [
    #       {"id": "step-name", "agent_name": "...", "status": "pending|running|success|failed|skipped",
    #        "started_at": <float|null>, "finished_at": <float|null>, "error": <str|null>}
    #     ],
    #     "edges": [{"from": "dep-step-name", "to": "step-name"}]
    #   }

    _CHAIN_STATUS_PRIORITY = ("failed", "running", "pending", "success", "skipped")

    @classmethod
    def _compute_overall_chain_status(cls, nodes: list[dict]) -> str:
        if not nodes:
            return "pending"
        statuses = {n.get("status", "pending") for n in nodes}
        if "failed" in statuses:
            return "failed"
        if "running" in statuses:
            return "running"
        if "pending" in statuses:
            return "pending"
        # all nodes terminal-non-failed: success unless every node was skipped
        if statuses == {"skipped"}:
            return "skipped"
        return "success"

    def record_chain_initialized(
        self,
        *,
        job_id: str,
        nodes: list[dict],
        edges: list[dict],
    ) -> None:
        """Initialize chain state for a job. Idempotent — overwrites any prior state."""
        normalized_nodes: list[dict] = []
        for node in nodes:
            normalized_nodes.append(
                {
                    "id": node["id"],
                    "agent_name": node.get("agent_name", ""),
                    "status": node.get("status", "pending"),
                    "started_at": node.get("started_at"),
                    "finished_at": node.get("finished_at"),
                    "error": node.get("error"),
                }
            )
        normalized_edges = [{"from": e["from"], "to": e["to"]} for e in edges]
        state = {"nodes": normalized_nodes, "edges": normalized_edges}
        overall = self._compute_overall_chain_status(normalized_nodes)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chain_state(job_id, state_json, overall_status, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    overall_status = excluded.overall_status,
                    updated_at = excluded.updated_at
                """,
                (job_id, json.dumps(state), overall, now),
            )

    def record_chain_node_transition(
        self,
        *,
        job_id: str,
        node_id: str,
        status: str,
        started_at: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
    ) -> None:
        """Update one node's fields and recompute overall status. Raises if chain not initialized."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM chain_state WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"chain state not initialized for job_id={job_id}")
            state = json.loads(row["state_json"])
            for node in state["nodes"]:
                if node["id"] == node_id:
                    node["status"] = status
                    if started_at is not None:
                        node["started_at"] = started_at
                    if finished_at is not None:
                        node["finished_at"] = finished_at
                    if error is not None:
                        node["error"] = error
                    break
            else:
                raise RuntimeError(
                    f"chain node {node_id!r} not found in state for job_id={job_id}"
                )
            overall = self._compute_overall_chain_status(state["nodes"])
            conn.execute(
                """
                UPDATE chain_state
                SET state_json = ?, overall_status = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(state), overall, time.time(), job_id),
            )

    def get_chain_state(self, job_id: str) -> dict | None:
        """Return {nodes, edges, overall_status} for the chain, or None if no state recorded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json, overall_status FROM chain_state WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            state = json.loads(row["state_json"])
            return {
                "nodes": state["nodes"],
                "edges": state["edges"],
                "overall_status": row["overall_status"],
            }

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

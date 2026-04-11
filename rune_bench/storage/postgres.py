# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed implementation of :class:`rune_bench.storage.StoragePort`.

This module requires a live PostgreSQL database for testing and is excluded from
coverage requirements as it is infrastructure code tested via integration tests.
"""
# pragma: no cover

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

try:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:
    # Allow collection even if psycopg is missing
    dict_row = None  # type: ignore[assignment,misc]
    ConnectionPool = None  # type: ignore[assignment,misc]

from rune_bench.storage.migrator import Migrator
from rune_bench.storage.sqlite import JobRecord

if TYPE_CHECKING:
    from rune_bench.metrics import MetricsEvent


class PostgresStorageAdapter:
    _CHAIN_STATUS_PRIORITY = ("failed", "running", "pending", "success", "skipped")
    _AUDIT_ARTIFACT_KINDS = frozenset(
        {
            "slsa_provenance",
            "sbom",
            "tla_report",
            "sigstore_bundle",
            "rekor_entry",
            "tpm_attestation",
        }
    )

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._pool = ConnectionPool(
            conninfo=db_url,
            min_size=self._pool_min_size(),
            max_size=self._pool_max_size(),
            open=True,
            kwargs={"row_factory": dict_row},
        )
        self._pool.wait()
        self._initialize()

    @staticmethod
    def _pool_min_size() -> int:
        value = int(os.environ.get("RUNE_PG_POOL_MIN", "1"))
        if value < 1:
            raise RuntimeError("RUNE_PG_POOL_MIN must be >= 1")
        return value

    @classmethod
    def _pool_max_size(cls) -> int:
        value = int(os.environ.get("RUNE_PG_POOL_MAX", "10"))
        if value < cls._pool_min_size():
            raise RuntimeError("RUNE_PG_POOL_MAX must be >= RUNE_PG_POOL_MIN")
        return value

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            yield conn

    def close(self) -> None:
        self._pool.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass  # nosec

    def _initialize(self) -> None:
        with self.connection() as conn:
            Migrator(dialect="postgres").apply_pending(conn)

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
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO chain_state(job_id, state_json, overall_status, updated_at)
                VALUES (%s, %s, %s, %s)
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
        with self.connection() as conn:
            row = conn.execute(
                "SELECT state_json FROM chain_state WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"chain state not initialized for job_id={job_id}")
            state = json.loads(str(row["state_json"]))
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
                SET state_json = %s, overall_status = %s, updated_at = %s
                WHERE job_id = %s
                """,
                (json.dumps(state), overall, time.time(), job_id),
            )

    def get_chain_state(self, job_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT state_json, overall_status FROM chain_state WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            state = json.loads(str(row["state_json"]))
            return {
                "nodes": state["nodes"],
                "edges": state["edges"],
                "overall_status": row["overall_status"],
            }

    def record_audit_artifact(
        self,
        *,
        job_id: str,
        kind: str,
        name: str,
        content: bytes,
    ) -> str:
        if kind not in self._AUDIT_ARTIFACT_KINDS:
            raise ValueError(
                f"unknown audit artifact kind {kind!r}; expected one of "
                f"{sorted(self._AUDIT_ARTIFACT_KINDS)}"
            )
        artifact_id = str(uuid.uuid4())
        sha256 = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_artifact(
                    artifact_id, job_id, kind, name, size_bytes, sha256, content, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (artifact_id, job_id, kind, name, size_bytes, sha256, content, now),
            )
        return artifact_id

    def list_audit_artifacts(self, job_id: str) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT artifact_id, kind, name, size_bytes, sha256, created_at
                FROM audit_artifact
                WHERE job_id = %s
                ORDER BY created_at ASC, artifact_id ASC
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "artifact_id": row["artifact_id"],
                "kind": row["kind"],
                "name": row["name"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_audit_artifact(
        self,
        *,
        job_id: str,
        artifact_id: str,
    ) -> tuple[bytes, str, str] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT content, name, kind
                FROM audit_artifact
                WHERE job_id = %s AND artifact_id = %s
                """,
                (job_id, artifact_id),
            ).fetchone()
        if row is None:
            return None
        return bytes(row["content"]), row["name"], row["kind"]

    def mark_incomplete_jobs_failed(
        self, message: str = "server restarted before job completion"
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = %s, error = %s, message = %s, updated_at = %s
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
        with self.connection() as conn:
            if idempotency_key:
                existing = conn.execute(
                    """
                    SELECT job_id
                    FROM idempotency_keys
                    WHERE tenant_id = %s AND operation = %s AND idempotency_key = %s
                    """,
                    (tenant_id, kind, idempotency_key),
                ).fetchone()
                if existing is not None:
                    return str(existing["job_id"]), False

            job_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, tenant_id, kind, status, request_json, result_json,
                    error, message, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    INSERT INTO idempotency_keys(
                        tenant_id, operation, idempotency_key, job_id, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s)
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
            fields.append("status = %s")
            values.append(status)
        if result_payload is not None:
            fields.append("result_json = %s")
            values.append(json.dumps(result_payload, sort_keys=True))
        if error is not None:
            fields.append("error = %s")
            values.append(error)
        if message is not None:
            fields.append("message = %s")
            values.append(message)

        fields.append("updated_at = %s")
        values.append(time.time())
        values.append(job_id)
        assignments = ", ".join(fields)
        statement = f"UPDATE jobs SET {assignments} WHERE job_id = %s"  # nosec B608

        with self.connection() as conn:
            conn.execute(statement, values)

    def record_workflow_event(self, event: "MetricsEvent") -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO workflow_events(
                    job_id, event, status, duration_ms, error_type, labels_json, recorded_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
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
        query = """
            SELECT
                event,
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
                ROUND(AVG(duration_ms)::numeric, 1) AS avg_ms,
                ROUND(MIN(duration_ms)::numeric, 1) AS min_ms,
                ROUND(MAX(duration_ms)::numeric, 1) AS max_ms
            FROM workflow_events
        """
        params: list[object] = []
        if job_id is not None:
            query += " WHERE job_id = %s"
            params.append(job_id)
        query += " GROUP BY event ORDER BY event"

        with self.connection() as conn:
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
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT event, status, duration_ms, error_type, labels_json, recorded_at
                FROM workflow_events
                WHERE job_id = %s
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

    def list_jobs_for_finops(self, *, tenant_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        cap = max(1, min(int(limit), 5000))
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT kind, request_json, result_json, created_at, updated_at
                FROM jobs
                WHERE tenant_id = %s
                  AND status = 'succeeded'
                  AND kind IN ('benchmark', 'agentic-agent')
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (tenant_id, cap),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            created = float(row["created_at"])
            updated = float(row["updated_at"])
            duration = max(updated - created, 1e-3)
            out.append(
                {
                    "kind": str(row["kind"]),
                    "request_payload": json.loads(row["request_json"]),
                    "result_payload": json.loads(row["result_json"]) if row["result_json"] else None,
                    "duration_seconds": duration,
                }
            )
        return out

    def get_job(self, job_id: str, *, tenant_id: str | None = None) -> JobRecord | None:
        query = "SELECT * FROM jobs WHERE job_id = %s"
        params: list[object] = [job_id]
        if tenant_id is not None:
            query += " AND tenant_id = %s"
            params.append(tenant_id)

        with self.connection() as conn:
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

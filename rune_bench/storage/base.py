# SPDX-License-Identifier: Apache-2.0
"""Storage port protocol for RUNE job persistence.

Defines :class:`StoragePort`, the interface every storage adapter must
implement. Today only :class:`rune_bench.storage.sqlite.SQLiteStorageAdapter`
implements it, but future adapters (Postgres, etc.) will plug in here
without touching the API server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rune_bench.metrics import MetricsEvent
    from rune_bench.storage.sqlite import JobRecord


@runtime_checkable
class StoragePort(Protocol):
    """Protocol defining every public method of a RUNE job store.

    Implementations must be safe to call from the HTTP request-handling
    threads of :class:`rune_bench.api_server.RuneApiApplication`.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def mark_incomplete_jobs_failed(
        self, message: str = "server restarted before job completion"
    ) -> None:
        """Mark any job stuck in ``queued`` or ``running`` as ``failed``."""
        ...

    # ── Jobs ───────────────────────────────────────────────────────────────

    def create_job(
        self,
        *,
        tenant_id: str,
        kind: str,
        request_payload: dict,
        idempotency_key: str | None = None,
    ) -> tuple[str, bool]:
        """Create a new job and return ``(job_id, created)``.

        When ``idempotency_key`` matches an existing record for the same
        ``(tenant_id, kind)``, the existing ``job_id`` is returned and
        ``created`` is ``False``.
        """
        ...

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result_payload: dict | None = None,
        error: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update a subset of a job's mutable fields."""
        ...

    def get_job(
        self, job_id: str, *, tenant_id: str | None = None
    ) -> "JobRecord | None":
        """Return a :class:`JobRecord` or ``None`` if not found.

        When ``tenant_id`` is given, the lookup is scoped to that tenant —
        cross-tenant fetches return ``None`` even if the ``job_id`` exists.
        """
        ...

    # ── Workflow events ────────────────────────────────────────────────────

    def record_workflow_event(self, event: "MetricsEvent") -> None:
        """Persist a single workflow lifecycle event."""
        ...

    def get_events_summary(self, *, job_id: str | None = None) -> list[dict]:
        """Return per-event aggregate statistics (count, ok/error, avg/min/max ms)."""
        ...

    def get_events_for_job(self, job_id: str) -> list[dict]:
        """Return all raw workflow events for a single job, ordered by time."""
        ...

    # ── Chain state ────────────────────────────────────────────────────────

    def record_chain_initialized(
        self,
        *,
        job_id: str,
        nodes: list[dict],
        edges: list[dict],
    ) -> None:
        """Initialize chain state for a job (idempotent; overwrites prior state)."""
        ...

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
        """Update one node's fields and recompute overall chain status."""
        ...

    def get_chain_state(self, job_id: str) -> dict | None:
        """Return ``{nodes, edges, overall_status}`` or ``None`` if no state."""
        ...

    # ── Audit artifacts ────────────────────────────────────────────────────

    def record_audit_artifact(
        self,
        *,
        job_id: str,
        kind: str,
        name: str,
        content: bytes,
    ) -> str:
        """Insert a new audit artifact and return its generated ``artifact_id``."""
        ...

    def list_audit_artifacts(self, job_id: str) -> list[dict]:
        """Return metadata for all artifacts of a job (no bytes), oldest first."""
        ...

    def get_audit_artifact(
        self,
        *,
        job_id: str,
        artifact_id: str,
    ) -> tuple[bytes, str, str] | None:
        """Return ``(content_bytes, name, kind)`` or ``None`` if missing.

        The ``job_id`` is required and matched in the WHERE clause so callers
        cannot fetch an artifact that does not belong to the job they are
        querying — this is the building block for tenant scoping.
        """
        ...

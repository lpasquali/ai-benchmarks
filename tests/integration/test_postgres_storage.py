# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

import pytest

from rune_bench.metrics import MetricsEvent
from rune_bench.storage import make_storage
from rune_bench.storage.postgres import PostgresStorageAdapter

pytestmark = pytest.mark.integration_postgres


def _pg_test_url() -> str:
    url = os.environ.get("RUNE_PG_TEST_URL", "").strip()
    if not url:
        pytest.skip("set RUNE_PG_TEST_URL to run postgres integration tests")
    return url


def _reset_store(store: PostgresStorageAdapter) -> None:
    with store.connection() as conn:
        conn.execute(
            """
            TRUNCATE TABLE
                audit_artifact,
                chain_state,
                workflow_events,
                idempotency_keys,
                jobs
            RESTART IDENTITY
            """
        )


@pytest.fixture
def pg_store():
    store = make_storage(_pg_test_url())
    assert isinstance(store, PostgresStorageAdapter)
    _reset_store(store)
    try:
        yield store
    finally:
        _reset_store(store)
        store.close()


def test_postgres_storage_round_trips_jobs_and_idempotency(pg_store) -> None:
    job_id_1, created_1 = pg_store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )
    job_id_2, created_2 = pg_store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )
    job_id_3, created_3 = pg_store.create_job(
        tenant_id="tenant-b",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )

    assert created_1 is True
    assert created_2 is False
    assert created_3 is True
    assert job_id_1 == job_id_2
    assert job_id_3 != job_id_1

    pg_store.update_job(
        job_id_1,
        status="running",
        result_payload={"phase": "start"},
        message="working",
    )
    job = pg_store.get_job(job_id_1, tenant_id="tenant-a")
    assert job is not None
    assert job.status == "running"
    assert job.result_payload == {"phase": "start"}
    assert job.message == "working"
    assert pg_store.get_job(job_id_1, tenant_id="tenant-b") is None


def test_postgres_storage_metrics_chain_audit_and_recovery(pg_store) -> None:
    job_id, _ = pg_store.create_job(
        tenant_id="tenant-a",
        kind="agentic-agent",
        request_payload={"question": "q1"},
    )
    pg_store.update_job(job_id, status="running", message="still running")

    pg_store.record_workflow_event(
        MetricsEvent(
            event="phase.a",
            status="ok",
            duration_ms=12.5,
            labels={"backend": "pg"},
            recorded_at=time.time(),
            job_id=job_id,
        )
    )
    summary = {row["event"]: row for row in pg_store.get_events_summary(job_id=job_id)}
    assert summary["phase.a"]["total"] == 1
    rows = pg_store.get_events_for_job(job_id)
    assert rows[0]["labels"]["backend"] == "pg"

    pg_store.record_chain_initialized(
        job_id=job_id,
        nodes=[{"id": "draft", "agent_name": "DraftAgent"}],
        edges=[],
    )
    pg_store.record_chain_node_transition(
        job_id=job_id,
        node_id="draft",
        status="success",
        finished_at=42.0,
    )
    state = pg_store.get_chain_state(job_id)
    assert state is not None
    assert state["overall_status"] == "success"
    assert state["nodes"][0]["finished_at"] == 42.0

    artifact_id = pg_store.record_audit_artifact(
        job_id=job_id,
        kind="sbom",
        name="sbom.json",
        content=b"{}",
    )
    artifacts = pg_store.list_audit_artifacts(job_id)
    assert artifacts[0]["artifact_id"] == artifact_id
    content, name, kind = pg_store.get_audit_artifact(
        job_id=job_id,
        artifact_id=artifact_id,
    ) or (b"", "", "")
    assert content == b"{}"
    assert name == "sbom.json"
    assert kind == "sbom"

    pg_store.mark_incomplete_jobs_failed("restarted")
    job = pg_store.get_job(job_id, tenant_id="tenant-a")
    assert job is not None
    assert job.status == "failed"
    assert job.error == "restarted"

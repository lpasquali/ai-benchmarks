# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

try:
    import psycopg  # noqa: F401
    import psycopg_pool  # noqa: F401
    from psycopg.rows import dict_row  # noqa: F401
except ImportError:
    pytest.skip("psycopg or psycopg_pool not installed", allow_module_level=True)

from rune_bench.storage.postgres import PostgresStorageAdapter
from rune_bench.metrics import MetricsEvent


@pytest.fixture
def mock_pool():
    with patch("rune_bench.storage.postgres.ConnectionPool") as m:
        pool = m.return_value
        conn = MagicMock()
        pool.connection.return_value.__enter__.return_value = conn
        yield pool, conn


def test_postgres_storage_init(mock_pool):
    storage = PostgresStorageAdapter("postgresql://user:pass@host/db")
    assert storage._db_url == "postgresql://user:pass@host/db"


def test_pool_size_config(monkeypatch):
    monkeypatch.setenv("RUNE_PG_POOL_MIN", "5")
    monkeypatch.setenv("RUNE_PG_POOL_MAX", "20")
    with patch("rune_bench.storage.postgres.ConnectionPool") as m:
        PostgresStorageAdapter("postgresql://")
        args, kwargs = m.call_args
        assert kwargs["min_size"] == 5
        assert kwargs["max_size"] == 20


def test_pool_size_validation(monkeypatch):
    monkeypatch.setenv("RUNE_PG_POOL_MIN", "0")
    with patch("rune_bench.storage.postgres.ConnectionPool"):
        with pytest.raises(RuntimeError, match="RUNE_PG_POOL_MIN must be >= 1"):
            PostgresStorageAdapter("postgresql://")

    monkeypatch.setenv("RUNE_PG_POOL_MIN", "10")
    monkeypatch.setenv("RUNE_PG_POOL_MAX", "5")
    with patch("rune_bench.storage.postgres.ConnectionPool"):
        with pytest.raises(
            RuntimeError, match="RUNE_PG_POOL_MAX must be >= RUNE_PG_POOL_MIN"
        ):
            PostgresStorageAdapter("postgresql://")


def test_record_chain_initialized(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    storage.record_chain_initialized(
        job_id="j1",
        nodes=[{"id": "n1", "agent_name": "a1"}],
        edges=[{"from": "n1", "to": "n2"}],
    )
    assert conn.execute.called


def test_record_chain_node_transition(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")

    # Mock get_chain_state part inside transition
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [{"id": "n1", "status": "pending"}]})
    }

    storage.record_chain_node_transition(job_id="j1", node_id="n1", status="running")
    assert conn.execute.called


def test_record_chain_node_transition_not_found(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [{"id": "n1"}]})
    }
    with pytest.raises(RuntimeError, match="not found"):
        storage.record_chain_node_transition(
            job_id="j1", node_id="n2", status="running"
        )


def test_record_chain_node_transition_uninitialized(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="not initialized"):
        storage.record_chain_node_transition(
            job_id="j1", node_id="n1", status="running"
        )


def test_get_chain_state(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [], "edges": []}),
        "overall_status": "success",
    }
    res = storage.get_chain_state("j1")
    assert res["overall_status"] == "success"


def test_get_chain_state_none(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    assert storage.get_chain_state("j1") is None


def test_record_audit_artifact(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    aid = storage.record_audit_artifact(
        job_id="j1", kind="sbom", name="n", content=b"{}"
    )
    assert aid is not None
    assert conn.execute.called


def test_record_audit_artifact_invalid_kind(mock_pool):
    storage = PostgresStorageAdapter("postgresql://")
    with pytest.raises(ValueError, match="unknown audit artifact kind"):
        storage.record_audit_artifact(
            job_id="j1", kind="unknown", name="n", content=b""
        )


def test_list_audit_artifacts(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "artifact_id": "a1",
            "kind": "sbom",
            "name": "n",
            "size_bytes": 10,
            "sha256": "s",
            "created_at": 1.0,
        }
    ]
    res = storage.list_audit_artifacts("j1")
    assert len(res) == 1
    assert res[0]["artifact_id"] == "a1"


def test_get_audit_artifact(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "content": b"{}",
        "name": "n",
        "kind": "sbom",
    }
    res = storage.get_audit_artifact(job_id="j1", artifact_id="a1")
    assert res[0] == b"{}"


def test_get_audit_artifact_none(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    assert storage.get_audit_artifact(job_id="j1", artifact_id="a1") is None


def test_mark_incomplete_jobs_failed(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    storage.mark_incomplete_jobs_failed("fail")
    assert conn.execute.called


def test_create_job_idempotent(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {"job_id": "existing"}
    jid, created = storage.create_job(
        tenant_id="t", kind="k", request_payload={}, idempotency_key="i"
    )
    assert jid == "existing"
    assert created is False


def test_create_job_new(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    jid, created = storage.create_job(tenant_id="t", kind="k", request_payload={})
    assert created is True
    assert jid is not None


def test_update_job(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    storage.update_job(
        "j1", status="s", result_payload={"r": 1}, error="e", message="m"
    )
    assert conn.execute.called


def test_record_workflow_event(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    ev = MetricsEvent(
        job_id="j1",
        event="e",
        status="ok",
        duration_ms=1.0,
        labels={"l": 1},
        recorded_at=1.0,
    )
    storage.record_workflow_event(ev)
    assert conn.execute.called


def test_get_events_summary(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "event": "e",
            "total": 1,
            "ok_count": 1,
            "error_count": 0,
            "avg_ms": 1.0,
            "min_ms": 1.0,
            "max_ms": 1.0,
        }
    ]
    res = storage.get_events_summary(job_id="j1")
    assert len(res) == 1
    assert res[0]["event"] == "e"


def test_get_events_for_job(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "event": "e",
            "status": "ok",
            "duration_ms": 1.0,
            "error_type": None,
            "labels_json": "{}",
            "recorded_at": 1.0,
        }
    ]
    res = storage.get_events_for_job("j1")
    assert len(res) == 1


def test_list_jobs_for_finops(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "kind": "benchmark",
            "request_json": "{}",
            "result_json": "{}",
            "created_at": 1.0,
            "updated_at": 2.0,
        }
    ]
    res = storage.list_jobs_for_finops(tenant_id="t1")
    assert len(res) == 1
    assert res[0]["duration_seconds"] == 1.0


def test_get_job(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "job_id": "j1",
        "tenant_id": "t",
        "kind": "k",
        "status": "s",
        "request_json": "{}",
        "result_json": None,
        "error": None,
        "message": None,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    res = storage.get_job("j1")
    assert res.job_id == "j1"


def test_compute_overall_chain_status_edge_cases():
    S = PostgresStorageAdapter
    assert S._compute_overall_chain_status([]) == "pending"
    assert S._compute_overall_chain_status([{"status": "failed"}]) == "failed"
    assert S._compute_overall_chain_status([{"status": "skipped"}]) == "skipped"
    assert (
        S._compute_overall_chain_status([{"status": "success"}, {"status": "skipped"}])
        == "success"
    )


def test_record_chain_node_transition_optional_params(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [{"id": "n1", "status": "pending"}]})
    }
    storage.record_chain_node_transition(
        job_id="j1",
        node_id="n1",
        status="running",
        started_at=1.0,
        finished_at=2.0,
        error="e",
    )
    assert conn.execute.called


def test_create_job_with_idempotency_key(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.reset_mock()
    conn.execute.return_value.fetchone.return_value = None
    jid, created = storage.create_job(
        tenant_id="t", kind="k", request_payload={}, idempotency_key="i"
    )
    assert created is True
    # Verify that idempotency key was inserted:
    # 1. SELECT idempotency_keys
    # 2. INSERT jobs
    # 3. INSERT idempotency_keys
    assert conn.execute.call_count == 3


def test_get_job_with_tenant_id(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "job_id": "j1",
        "tenant_id": "t",
        "kind": "k",
        "status": "s",
        "request_json": "{}",
        "result_json": None,
        "error": None,
        "message": None,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    res = storage.get_job("j1", tenant_id="t")
    assert res.job_id == "j1"
    # Verify that query was modified (line 475-476)
    args, kwargs = conn.execute.call_args
    assert "AND tenant_id = %s" in args[0]


def test_get_job_none(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    assert storage.get_job("j1") is None


# ─── Additional coverage for edge cases ──────────────────────────────────
def test_compute_overall_chain_status_empty():
    """Test _compute_overall_chain_status with empty nodes."""
    result = PostgresStorageAdapter._compute_overall_chain_status([])
    assert result == "pending"


def test_compute_overall_chain_status_all_failed():
    """Test when any node failed."""
    nodes = [
        {"id": "n1", "status": "success"},
        {"id": "n2", "status": "failed"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "failed"


def test_compute_overall_chain_status_running():
    """Test when any node is running."""
    nodes = [
        {"id": "n1", "status": "success"},
        {"id": "n2", "status": "running"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "running"


def test_compute_overall_chain_status_pending():
    """Test when any node is pending."""
    nodes = [
        {"id": "n1", "status": "success"},
        {"id": "n2", "status": "pending"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "pending"


def test_compute_overall_chain_status_all_skipped():
    """Test when all nodes are skipped."""
    nodes = [
        {"id": "n1", "status": "skipped"},
        {"id": "n2", "status": "skipped"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "skipped"


def test_compute_overall_chain_status_all_success():
    """Test when all nodes are success."""
    nodes = [
        {"id": "n1", "status": "success"},
        {"id": "n2", "status": "success"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "success"


def test_compute_overall_chain_status_mixed_with_skipped():
    """Test when mix of skipped and other statuses."""
    nodes = [
        {"id": "n1", "status": "skipped"},
        {"id": "n2", "status": "success"},
    ]
    result = PostgresStorageAdapter._compute_overall_chain_status(nodes)
    assert result == "success"


def test_record_chain_node_transition_with_timestamps(mock_pool):
    """Test recording transition with all optional parameters."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [{"id": "n1", "status": "pending"}]})
    }

    storage.record_chain_node_transition(
        job_id="j1",
        node_id="n1",
        status="success",
        started_at=100.0,
        finished_at=110.0,
        error="test error",
    )
    assert conn.execute.called


def test_record_chain_initialized_normalizes_nodes(mock_pool):
    """Test that record_chain_initialized normalizes node data."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")

    # Nodes without optional fields
    nodes = [{"id": "n1"}]
    edges = []

    storage.record_chain_initialized(job_id="j1", nodes=nodes, edges=edges)
    assert conn.execute.called

    # INSERT uses positional args: (sql, (job_id, state_json, overall, now))
    _stmt, params = conn.execute.call_args[0]
    assert params[0] == "j1"


def test_get_job(mock_pool):
    """Test get_job retrieves job record."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "job_id": "j1",
        "tenant_id": "t1",
        "kind": "benchmark",
        "status": "success",
        "request_json": "{}",
        "result_json": "{}",
        "error": None,
        "message": None,
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    result = storage.get_job("j1")
    assert result is not None
    assert result.job_id == "j1"


def test_get_job_not_found(mock_pool):
    """Test get_job returns None when not found."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    result = storage.get_job("j1")
    assert result is None


def test_get_job_with_tenant_filter(mock_pool):
    """Test get_job with tenant_id filter."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "job_id": "j1",
        "tenant_id": "t1",
        "kind": "benchmark",
        "status": "success",
        "request_json": "{}",
        "result_json": "{}",
        "error": None,
        "message": None,
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    result = storage.get_job("j1", tenant_id="t1")
    assert result is not None
    assert result.tenant_id == "t1"


def test_list_jobs_for_finops_with_limit(mock_pool):
    """Test list_jobs_for_finops with custom limit."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "kind": "benchmark",
            "request_json": "{}",
            "result_json": "{}",
            "created_at": 1.0,
            "updated_at": 2.0,
        }
    ]
    result = storage.list_jobs_for_finops(tenant_id="t1", limit=100)
    assert len(result) == 1
    assert conn.execute.called
    _stmt, params = conn.execute.call_args[0]
    assert params == ("t1", 100)


def test_get_events_summary_without_job_filter(mock_pool):
    """Test get_events_summary without job_id filter."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {
            "event": "test_event",
            "total": 5,
            "ok_count": 4,
            "error_count": 1,
            "avg_ms": 1.5,
            "min_ms": 1.0,
            "max_ms": 2.0,
        }
    ]
    result = storage.get_events_summary()
    assert len(result) == 1
    assert result[0]["event"] == "test_event"
    assert result[0]["total"] == 5


def test_pool_context_manager(mock_pool):
    """Test connection context manager."""
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    with storage.connection() as c:
        assert c is not None

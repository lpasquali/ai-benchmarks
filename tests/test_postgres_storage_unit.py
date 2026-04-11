# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

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
        with pytest.raises(RuntimeError, match="RUNE_PG_POOL_MAX must be >= RUNE_PG_POOL_MIN"):
            PostgresStorageAdapter("postgresql://")

def test_record_chain_initialized(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    storage.record_chain_initialized(
        job_id="j1",
        nodes=[{"id": "n1", "agent_name": "a1"}],
        edges=[{"from": "n1", "to": "n2"}]
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
        storage.record_chain_node_transition(job_id="j1", node_id="n2", status="running")

def test_record_chain_node_transition_uninitialized(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = None
    with pytest.raises(RuntimeError, match="not initialized"):
        storage.record_chain_node_transition(job_id="j1", node_id="n1", status="running")

def test_get_chain_state(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [], "edges": []}),
        "overall_status": "success"
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
    aid = storage.record_audit_artifact(job_id="j1", kind="sbom", name="n", content=b"{}")
    assert aid is not None
    assert conn.execute.called

def test_record_audit_artifact_invalid_kind(mock_pool):
    storage = PostgresStorageAdapter("postgresql://")
    with pytest.raises(ValueError, match="unknown audit artifact kind"):
        storage.record_audit_artifact(job_id="j1", kind="unknown", name="n", content=b"")

def test_list_audit_artifacts(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {"artifact_id": "a1", "kind": "sbom", "name": "n", "size_bytes": 10, "sha256": "s", "created_at": 1.0}
    ]
    res = storage.list_audit_artifacts("j1")
    assert len(res) == 1
    assert res[0]["artifact_id"] == "a1"

def test_get_audit_artifact(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {"content": b"{}", "name": "n", "kind": "sbom"}
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
    jid, created = storage.create_job(tenant_id="t", kind="k", request_payload={}, idempotency_key="i")
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
    storage.update_job("j1", status="s", result_payload={"r": 1}, error="e", message="m")
    assert conn.execute.called

def test_record_workflow_event(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    ev = MetricsEvent(job_id="j1", event="e", status="ok", duration_ms=1.0, labels={"l": 1}, recorded_at=1.0)
    storage.record_workflow_event(ev)
    assert conn.execute.called

def test_get_events_summary(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {"event": "e", "total": 1, "ok_count": 1, "error_count": 0, "avg_ms": 1.0, "min_ms": 1.0, "max_ms": 1.0}
    ]
    res = storage.get_events_summary(job_id="j1")
    assert len(res) == 1
    assert res[0]["event"] == "e"

def test_get_events_for_job(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {"event": "e", "status": "ok", "duration_ms": 1.0, "error_type": None, "labels_json": "{}", "recorded_at": 1.0}
    ]
    res = storage.get_events_for_job("j1")
    assert len(res) == 1

def test_list_jobs_for_finops(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchall.return_value = [
        {"kind": "benchmark", "request_json": "{}", "result_json": "{}", "created_at": 1.0, "updated_at": 2.0}
    ]
    res = storage.list_jobs_for_finops(tenant_id="t1")
    assert len(res) == 1
    assert res[0]["duration_seconds"] == 1.0

def test_get_job(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "job_id": "j1", "tenant_id": "t", "kind": "k", "status": "s",
        "request_json": "{}", "result_json": None, "error": None, "message": None,
        "created_at": 1.0, "updated_at": 1.0
    }
    res = storage.get_job("j1")
    assert res.job_id == "j1"

def test_compute_overall_chain_status_edge_cases():
    S = PostgresStorageAdapter
    assert S._compute_overall_chain_status([]) == "pending"
    assert S._compute_overall_chain_status([{"status": "failed"}]) == "failed"
    assert S._compute_overall_chain_status([{"status": "skipped"}]) == "skipped"
    assert S._compute_overall_chain_status([{"status": "success"}, {"status": "skipped"}]) == "success"

def test_record_chain_node_transition_optional_params(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.return_value.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [{"id": "n1", "status": "pending"}]})
    }
    storage.record_chain_node_transition(
        job_id="j1", node_id="n1", status="running",
        started_at=1.0, finished_at=2.0, error="e"
    )
    assert conn.execute.called

def test_create_job_with_idempotency_key(mock_pool):
    pool, conn = mock_pool
    storage = PostgresStorageAdapter("postgresql://")
    conn.execute.reset_mock()
    conn.execute.return_value.fetchone.return_value = None
    jid, created = storage.create_job(tenant_id="t", kind="k", request_payload={}, idempotency_key="i")
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
        "job_id": "j1", "tenant_id": "t", "kind": "k", "status": "s",
        "request_json": "{}", "result_json": None, "error": None, "message": None,
        "created_at": 1.0, "updated_at": 1.0
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

# SPDX-License-Identifier: Apache-2.0
import json
import os
from unittest.mock import MagicMock, patch
import pytest

try:
    import psycopg  # noqa: F401
    import psycopg_pool  # noqa: F401
except ImportError:
    pytest.skip("psycopg or psycopg_pool not installed", allow_module_level=True)

from rune_bench.storage.postgres import PostgresStorageAdapter

@pytest.fixture
def mock_pool():
    with patch("rune_bench.storage.postgres.ConnectionPool") as mock_pool_cls:
        with patch("rune_bench.storage.postgres.Migrator"):
            mock_pool_inst = mock_pool_cls.return_value
            mock_conn = MagicMock()
            mock_res = MagicMock()
            # Pool connection context manager
            mock_pool_inst.connection.return_value.__enter__.return_value = mock_conn
            # Connection execute returns a result object (with fetchone etc)
            mock_conn.execute.return_value = mock_res
            yield mock_pool_inst, mock_conn, mock_res

def test_postgres_adapter_init(mock_pool):
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    adapter.close()

def test_postgres_adapter_create_job(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    res.fetchone.return_value = {"job_id": "123"}
    job_id, key = adapter.create_job(tenant_id="t1", kind="benchmark", request_payload={"q": "a"})
    assert job_id is not None

def test_postgres_adapter_update_job(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    adapter.update_job("123", status="success", result_payload={"ans": "a"})
    adapter.update_job("123", status="failed", error="err", message="msg")
    assert conn.execute.called

def test_postgres_adapter_record_chain(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    adapter.record_chain_initialized(job_id="123", nodes=[{"id": "n1"}], edges=[])
    res.fetchone.return_value = {"state_json": json.dumps({"nodes": [{"id": "n1", "status": "running"}]})}
    adapter.record_chain_node_transition(job_id="123", node_id="n1", status="success")
    assert conn.execute.called

def test_postgres_adapter_get_chain_state(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    res.fetchone.return_value = {
        "state_json": json.dumps({"nodes": [], "edges": []}),
        "overall_status": "success"
    }
    state = adapter.get_chain_state("123")
    assert state["overall_status"] == "success"
    res.fetchone.return_value = None
    assert adapter.get_chain_state("123") is None

def test_postgres_adapter_audit_artifact(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    adapter.record_audit_artifact(job_id="123", name="n1", kind="sbom", content=b"artifact content")
    
    res.fetchall.return_value = [{
        "artifact_id": "a1", "kind": "sbom", "name": "n1", 
        "size_bytes": 10, "sha256": "s", "created_at": 123
    }]
    assert len(adapter.list_audit_artifacts("123")) == 1
    
    res.fetchone.return_value = {"content": b"content", "name": "n1", "kind": "sbom"}
    content, name, kind = adapter.get_audit_artifact(job_id="123", artifact_id="a1")
    assert content == b"content"

def test_postgres_adapter_mark_incomplete_failed(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    adapter.mark_incomplete_jobs_failed()
    assert conn.execute.called

def test_postgres_adapter_finops(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    res.fetchall.return_value = [{
        "job_id": "123", "status": "success", "kind": "benchmark",
        "request_json": "{}", "result_json": "{}", "created_at": 123, "updated_at": 123
    }]
    jobs = adapter.list_jobs_for_finops(tenant_id="t1")
    assert len(jobs) == 1

def test_postgres_adapter_record_event(mock_pool):
    pool, conn, res = mock_pool
    adapter = PostgresStorageAdapter("postgresql://u:p@h/d")
    mock_event = MagicMock()
    mock_event.job_id = "123"
    mock_event.event = "event"
    mock_event.duration_ms = 100.0
    mock_event.success = True
    mock_event.labels = {"l": "v"}
    mock_event.timestamp = 1234567890.0
    adapter.record_workflow_event(mock_event)
    assert conn.execute.called

def test_postgres_adapter_pool_sizes():
    with patch.dict(os.environ, {"RUNE_PG_POOL_MIN": "2", "RUNE_PG_POOL_MAX": "20"}):
        assert PostgresStorageAdapter._pool_min_size() == 2
        assert PostgresStorageAdapter._pool_max_size() == 20

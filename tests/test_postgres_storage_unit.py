# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from rune_bench.metrics import MetricsEvent
from rune_bench.storage import postgres as postgres_module
from rune_bench.storage.postgres import PostgresStorageAdapter


class _Result:
    def __init__(self, *, one=None, many=None):
        self._one = one
        self._many = list(many or [])

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._many)


class _FakeConnection:
    def __init__(self, *, results=None):
        self.results = list(results or [])
        self.executed: list[tuple[str, object]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, query: str, params=None):
        normalized = " ".join(str(query).split())
        self.executed.append((normalized, params))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return _Result()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakePool:
    def __init__(self, conn: _FakeConnection):
        self.conn = conn
        self.closed = False
        self.wait_called = False

    @contextmanager
    def connection(self):
        yield self.conn

    def close(self) -> None:
        self.closed = True

    def wait(self) -> None:
        self.wait_called = True


def _make_adapter(conn: _FakeConnection | None = None) -> PostgresStorageAdapter:
    adapter = PostgresStorageAdapter.__new__(PostgresStorageAdapter)
    adapter._db_url = "postgresql://user:pass@localhost:5432/rune"
    adapter._pool = _FakePool(conn or _FakeConnection())
    return adapter


def test_postgres_adapter_constructor_uses_pool_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeConnectionPool(_FakePool):
        def __init__(self, **kwargs):
            super().__init__(_FakeConnection())
            captured.update(kwargs)

    initialize_calls: list[str] = []

    def fake_initialize(self) -> None:
        initialize_calls.append(self._db_url)

    monkeypatch.setenv("RUNE_PG_POOL_MIN", "2")
    monkeypatch.setenv("RUNE_PG_POOL_MAX", "5")
    monkeypatch.setattr(postgres_module, "ConnectionPool", FakeConnectionPool)
    monkeypatch.setattr(PostgresStorageAdapter, "_initialize", fake_initialize)

    adapter = PostgresStorageAdapter("postgresql://user:pass@localhost:5432/rune")

    assert captured == {
        "conninfo": "postgresql://user:pass@localhost:5432/rune",
        "min_size": 2,
        "max_size": 5,
        "open": True,
        "kwargs": {"row_factory": postgres_module.dict_row},
    }
    assert isinstance(adapter._pool, FakeConnectionPool)
    assert adapter._pool.wait_called is True
    assert initialize_calls == ["postgresql://user:pass@localhost:5432/rune"]


def test_postgres_pool_size_validation(monkeypatch) -> None:
    monkeypatch.setenv("RUNE_PG_POOL_MIN", "3")
    monkeypatch.setenv("RUNE_PG_POOL_MAX", "7")
    assert PostgresStorageAdapter._pool_min_size() == 3
    assert PostgresStorageAdapter._pool_max_size() == 7

    monkeypatch.setenv("RUNE_PG_POOL_MIN", "0")
    with pytest.raises(RuntimeError, match="RUNE_PG_POOL_MIN must be >= 1"):
        PostgresStorageAdapter._pool_min_size()

    monkeypatch.setenv("RUNE_PG_POOL_MIN", "4")
    monkeypatch.setenv("RUNE_PG_POOL_MAX", "2")
    with pytest.raises(RuntimeError, match="RUNE_PG_POOL_MAX must be >= RUNE_PG_POOL_MIN"):
        PostgresStorageAdapter._pool_max_size()


def test_postgres_connection_and_initialize_use_pool(monkeypatch) -> None:
    conn = _FakeConnection()
    adapter = _make_adapter(conn)
    calls: list[object] = []

    class FakeMigrator:
        def __init__(self, *, dialect: str):
            calls.append(dialect)

        def apply_pending(self, received_conn) -> None:
            calls.append(received_conn)

    monkeypatch.setattr(postgres_module, "Migrator", FakeMigrator)

    with adapter.connection() as raw_conn:
        assert raw_conn is conn
    adapter._initialize()

    assert calls == ["postgres", conn]


@pytest.mark.parametrize(
    ("nodes", "expected"),
    [
        ([], "pending"),
        ([{"id": "a"}], "pending"),
        ([{"id": "a", "status": "running"}], "running"),
        ([{"id": "a", "status": "failed"}], "failed"),
        ([{"id": "a", "status": "skipped"}], "skipped"),
        ([{"id": "a", "status": "success"}], "success"),
    ],
)
def test_compute_overall_chain_status(nodes: list[dict], expected: str) -> None:
    assert PostgresStorageAdapter._compute_overall_chain_status(nodes) == expected


def test_record_chain_initialized_normalizes_state() -> None:
    conn = _FakeConnection()
    adapter = _make_adapter(conn)

    adapter.record_chain_initialized(
        job_id="job-1",
        nodes=[{"id": "draft", "agent_name": "DraftAgent"}],
        edges=[{"from": "draft", "to": "review"}],
    )

    query, params = conn.executed[0]
    state = json.loads(params[1])
    assert "INSERT INTO chain_state(job_id, state_json, overall_status, updated_at)" in query
    assert params[0] == "job-1"
    assert params[2] == "pending"
    assert state == {
        "nodes": [
            {
                "id": "draft",
                "agent_name": "DraftAgent",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
        ],
        "edges": [{"from": "draft", "to": "review"}],
    }


def test_record_chain_node_transition_updates_state() -> None:
    conn = _FakeConnection(
        results=[
            _Result(
                one={
                    "state_json": json.dumps(
                        {
                            "nodes": [
                                {
                                    "id": "draft",
                                    "agent_name": "DraftAgent",
                                    "status": "pending",
                                    "started_at": None,
                                    "finished_at": None,
                                    "error": None,
                                }
                            ],
                            "edges": [],
                        }
                    )
                }
            )
        ]
    )
    adapter = _make_adapter(conn)

    adapter.record_chain_node_transition(
        job_id="job-1",
        node_id="draft",
        status="success",
        started_at=1.0,
        finished_at=2.0,
    )

    _, params = conn.executed[1]
    state = json.loads(params[0])
    assert state["nodes"][0]["status"] == "success"
    assert state["nodes"][0]["started_at"] == 1.0
    assert state["nodes"][0]["finished_at"] == 2.0
    assert params[1] == "success"
    assert params[3] == "job-1"


def test_record_chain_node_transition_sets_node_error() -> None:
    conn = _FakeConnection(
        results=[
            _Result(
                one={
                    "state_json": json.dumps(
                        {
                            "nodes": [
                                {
                                    "id": "draft",
                                    "agent_name": "DraftAgent",
                                    "status": "running",
                                    "started_at": None,
                                    "finished_at": None,
                                    "error": None,
                                }
                            ],
                            "edges": [],
                        }
                    )
                }
            )
        ]
    )
    adapter = _make_adapter(conn)

    adapter.record_chain_node_transition(
        job_id="job-1",
        node_id="draft",
        status="failed",
        error="node blew up",
    )

    _, params = conn.executed[1]
    state = json.loads(params[0])
    assert state["nodes"][0]["status"] == "failed"
    assert state["nodes"][0]["error"] == "node blew up"


def test_record_chain_node_transition_rejects_missing_state_or_node() -> None:
    missing_state = _make_adapter(_FakeConnection(results=[_Result(one=None)]))
    with pytest.raises(RuntimeError, match="not initialized"):
        missing_state.record_chain_node_transition(
            job_id="job-1",
            node_id="draft",
            status="running",
        )

    missing_node = _make_adapter(
        _FakeConnection(
            results=[_Result(one={"state_json": json.dumps({"nodes": [], "edges": []})})]
        )
    )
    with pytest.raises(RuntimeError, match="not found"):
        missing_node.record_chain_node_transition(
            job_id="job-1",
            node_id="draft",
            status="running",
        )


def test_get_chain_state_handles_missing_and_present_rows() -> None:
    missing = _make_adapter(_FakeConnection(results=[_Result(one=None)]))
    assert missing.get_chain_state("job-1") is None

    present = _make_adapter(
        _FakeConnection(
            results=[
                _Result(
                    one={
                        "state_json": json.dumps({"nodes": [{"id": "draft"}], "edges": []}),
                        "overall_status": "running",
                    }
                )
            ]
        )
    )
    assert present.get_chain_state("job-1") == {
        "nodes": [{"id": "draft"}],
        "edges": [],
        "overall_status": "running",
    }


def test_audit_artifact_methods_validate_and_round_trip(monkeypatch) -> None:
    conn = _FakeConnection(
        results=[
            _Result(),
            _Result(
                many=[
                    {
                        "artifact_id": "artifact-1",
                        "kind": "sbom",
                        "name": "sbom.json",
                        "size_bytes": 2,
                        "sha256": "ab" * 32,
                        "created_at": 10.0,
                    }
                ]
            ),
            _Result(one={"content": b"{}", "name": "sbom.json", "kind": "sbom"}),
            _Result(one=None),
        ]
    )
    adapter = _make_adapter(conn)
    monkeypatch.setattr(postgres_module.uuid, "uuid4", lambda: "artifact-1")
    monkeypatch.setattr(postgres_module.time, "time", lambda: 10.0)

    with pytest.raises(ValueError, match="unknown audit artifact kind"):
        adapter.record_audit_artifact(
            job_id="job-1",
            kind="unknown",
            name="bad.bin",
            content=b"payload",
        )

    artifact_id = adapter.record_audit_artifact(
        job_id="job-1",
        kind="sbom",
        name="sbom.json",
        content=b"{}",
    )

    insert_query, insert_params = conn.executed[0]
    assert artifact_id == "artifact-1"
    assert "INSERT INTO audit_artifact(" in insert_query
    assert insert_params == (
        "artifact-1",
        "job-1",
        "sbom",
        "sbom.json",
        2,
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        b"{}",
        10.0,
    )

    assert adapter.list_audit_artifacts("job-1") == [
        {
            "artifact_id": "artifact-1",
            "kind": "sbom",
            "name": "sbom.json",
            "size_bytes": 2,
            "sha256": "ab" * 32,
            "created_at": 10.0,
        }
    ]
    assert adapter.get_audit_artifact(job_id="job-1", artifact_id="artifact-1") == (
        b"{}",
        "sbom.json",
        "sbom",
    )
    assert adapter.get_audit_artifact(job_id="job-1", artifact_id="missing") is None


def test_mark_incomplete_jobs_failed_updates_rows(monkeypatch) -> None:
    conn = _FakeConnection()
    adapter = _make_adapter(conn)
    monkeypatch.setattr(postgres_module.time, "time", lambda: 42.0)

    adapter.mark_incomplete_jobs_failed("restarted")

    query, params = conn.executed[0]
    assert "UPDATE jobs SET status = %s, error = %s, message = %s, updated_at = %s" in query
    assert params == ("failed", "restarted", "restarted", 42.0)


def test_create_job_handles_new_and_existing_idempotency_keys(monkeypatch) -> None:
    existing_conn = _FakeConnection(results=[_Result(one={"job_id": "job-existing"})])
    existing_adapter = _make_adapter(existing_conn)

    assert existing_adapter.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q"},
        idempotency_key="idem-1",
    ) == ("job-existing", False)
    assert len(existing_conn.executed) == 1

    new_conn = _FakeConnection(results=[_Result(one=None)])
    new_adapter = _make_adapter(new_conn)
    monkeypatch.setattr(postgres_module.uuid, "uuid4", lambda: "job-new")
    monkeypatch.setattr(postgres_module.time, "time", lambda: 11.5)

    assert new_adapter.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q"},
        idempotency_key="idem-2",
    ) == ("job-new", True)

    assert "SELECT job_id FROM idempotency_keys" in new_conn.executed[0][0]
    assert "INSERT INTO jobs(" in new_conn.executed[1][0]
    assert new_conn.executed[1][1][0] == "job-new"
    assert new_conn.executed[1][1][4] == json.dumps({"question": "q"}, sort_keys=True)
    assert "INSERT INTO idempotency_keys(" in new_conn.executed[2][0]


def test_update_job_and_event_queries_are_parameterized(monkeypatch) -> None:
    conn = _FakeConnection()
    adapter = _make_adapter(conn)
    monkeypatch.setattr(postgres_module.time, "time", lambda: 33.0)

    adapter.update_job(
        "job-1",
        status="running",
        result_payload={"phase": "start"},
        error="none",
        message="working",
    )
    adapter.record_workflow_event(
        MetricsEvent(
            event="phase.a",
            status="ok",
            duration_ms=12.5,
            labels={"backend": "pg"},
            recorded_at=44.0,
            job_id="job-1",
        )
    )

    update_query, update_params = conn.executed[0]
    assert update_query == (
        "UPDATE jobs SET status = %s, result_json = %s, error = %s, message = %s, "
        "updated_at = %s WHERE job_id = %s"
    )
    assert update_params == [
        "running",
        json.dumps({"phase": "start"}, sort_keys=True),
        "none",
        "working",
        33.0,
        "job-1",
    ]

    event_query, event_params = conn.executed[1]
    assert "INSERT INTO workflow_events(" in event_query
    assert event_params == (
        "job-1",
        "phase.a",
        "ok",
        12.5,
        None,
        json.dumps({"backend": "pg"}, sort_keys=True),
        44.0,
    )


def test_event_summary_and_job_history_parse_rows() -> None:
    conn = _FakeConnection(
        results=[
            _Result(
                many=[
                    {
                        "event": "phase.a",
                        "total": 2,
                        "ok_count": 1,
                        "error_count": 1,
                        "avg_ms": 12.5,
                        "min_ms": 10.0,
                        "max_ms": 15.0,
                    }
                ]
            ),
            _Result(
                many=[
                    {
                        "event": "phase.b",
                        "total": 1,
                        "ok_count": 1,
                        "error_count": 0,
                        "avg_ms": None,
                        "min_ms": None,
                        "max_ms": None,
                    }
                ]
            ),
            _Result(
                many=[
                    {
                        "event": "phase.a",
                        "status": "ok",
                        "duration_ms": 12.5,
                        "error_type": None,
                        "labels_json": '{"backend": "pg"}',
                        "recorded_at": 55.0,
                    }
                ]
            ),
        ]
    )
    adapter = _make_adapter(conn)

    assert adapter.get_events_summary(job_id="job-1") == [
        {
            "event": "phase.a",
            "total": 2,
            "ok": 1,
            "error": 1,
            "avg_ms": 12.5,
            "min_ms": 10.0,
            "max_ms": 15.0,
        }
    ]
    assert adapter.get_events_summary() == [
        {
            "event": "phase.b",
            "total": 1,
            "ok": 1,
            "error": 0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    ]
    assert adapter.get_events_for_job("job-1") == [
        {
            "event": "phase.a",
            "status": "ok",
            "duration_ms": 12.5,
            "error_type": None,
            "labels": {"backend": "pg"},
            "recorded_at": 55.0,
        }
    ]
    assert "WHERE job_id = %s GROUP BY event ORDER BY event" in conn.executed[0][0]
    assert "GROUP BY event ORDER BY event" in conn.executed[1][0]
    assert "SELECT event, status, duration_ms, error_type, labels_json, recorded_at" in conn.executed[2][0]


def test_get_job_returns_records_and_filters_tenant() -> None:
    job_row = {
        "job_id": "job-1",
        "tenant_id": "tenant-a",
        "kind": "benchmark",
        "status": "running",
        "request_json": '{"question": "q"}',
        "result_json": '{"answer": "a"}',
        "error": None,
        "message": "working",
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    conn = _FakeConnection(results=[_Result(one=job_row), _Result(one=None)])
    adapter = _make_adapter(conn)

    record = adapter.get_job("job-1", tenant_id="tenant-a")
    assert record is not None
    assert record.job_id == "job-1"
    assert record.request_payload == {"question": "q"}
    assert record.result_payload == {"answer": "a"}
    assert record.message == "working"

    assert adapter.get_job("job-1") is None
    assert conn.executed[0][0] == "SELECT * FROM jobs WHERE job_id = %s AND tenant_id = %s"
    assert conn.executed[1][0] == "SELECT * FROM jobs WHERE job_id = %s"


def test_list_jobs_for_finops_maps_rows() -> None:
    row = {
        "kind": "benchmark",
        "request_json": json.dumps({"model": "m"}),
        "result_json": None,
        "created_at": 10.0,
        "updated_at": 13.0,
    }
    conn = _FakeConnection(results=[_Result(many=[row])])
    adapter = _make_adapter(conn)
    out = adapter.list_jobs_for_finops(tenant_id="t1", limit=100)
    assert len(out) == 1
    assert out[0]["kind"] == "benchmark"
    assert out[0]["request_payload"] == {"model": "m"}
    assert out[0]["result_payload"] is None
    assert out[0]["duration_seconds"] == pytest.approx(3.0)
    assert conn.executed[0][1] == ("t1", 100)


def test_close_and_del_are_best_effort() -> None:
    adapter = _make_adapter(_FakeConnection())
    adapter.close()
    assert adapter._pool.closed is True

    class BrokenPool(_FakePool):
        def close(self) -> None:
            raise RuntimeError("boom")

    broken = PostgresStorageAdapter.__new__(PostgresStorageAdapter)
    broken._pool = BrokenPool(_FakeConnection())
    PostgresStorageAdapter.__del__(broken)

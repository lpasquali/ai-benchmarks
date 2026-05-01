# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock
from rune_bench.storage.sqlite import SQLiteStorageAdapter


@pytest.fixture
def adapter(tmp_path):
    db_path = tmp_path / "test.db"
    storage = SQLiteStorageAdapter(str(db_path))
    yield storage
    storage.close()


def test_sqlite_adapter_init_memory():
    storage = SQLiteStorageAdapter(":memory:")
    assert storage._db_path == ":memory:"
    storage.close()


def test_sqlite_adapter_create_job(adapter):
    job_id, created = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )
    assert job_id is not None
    assert created is True

    # Idempotent (not found)
    job_id2, created2 = adapter.create_job(
        tenant_id="t1",
        kind="benchmark",
        request_payload={"q": "a"},
        idempotency_key="k1",
    )
    assert created2 is True

    # Idempotent (found)
    job_id3, created3 = adapter.create_job(
        tenant_id="t1",
        kind="benchmark",
        request_payload={"q": "a"},
        idempotency_key="k1",
    )
    assert job_id3 == job_id2
    assert created3 is False


def test_sqlite_adapter_get_job(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )
    job = adapter.get_job(job_id)
    assert job.job_id == job_id
    assert job.request_payload == {"q": "a"}

    assert adapter.get_job("non-existent") is None
    assert adapter.get_job(job_id, tenant_id="t2") is None


def test_sqlite_adapter_update_job(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )
    # Hit various branches
    adapter.update_job(
        job_id,
        status="success",
        result_payload={"ans": "a"},
        message="done",
        error="err",
    )
    job = adapter.get_job(job_id)
    assert job.status == "success"
    assert job.result_payload == {"ans": "a"}
    assert job.message == "done"
    assert job.error == "err"


def test_sqlite_adapter_mark_incomplete_failed(adapter):
    adapter.create_job(tenant_id="t1", kind="benchmark", request_payload={"q": "a"})
    adapter.mark_incomplete_jobs_failed("crash")
    # All queued/running should be failed
    with adapter.connection() as conn:
        row = conn.execute("SELECT status FROM jobs").fetchone()
        assert row["status"] == "failed"


def test_sqlite_adapter_audit_artifacts(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )

    # record
    art_id = adapter.record_audit_artifact(
        job_id=job_id, kind="sbom", name="n1", content=b"data"
    )
    assert art_id is not None

    # invalid kind
    with pytest.raises(ValueError, match="unknown audit artifact kind"):
        adapter.record_audit_artifact(
            job_id=job_id, kind="invalid", name="n1", content=b"d"
        )

    # list
    arts = adapter.list_audit_artifacts(job_id)
    assert len(arts) == 1
    assert arts[0]["name"] == "n1"

    # get
    content, name, kind = adapter.get_audit_artifact(job_id=job_id, artifact_id=art_id)
    assert content == b"data"
    assert name == "n1"
    assert kind == "sbom"

    assert adapter.get_audit_artifact(job_id=job_id, artifact_id="missing") is None


def test_sqlite_adapter_chain_state(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )

    # initialized - multiple nodes
    # Using specific order to ensure we hit the loop logic correctly
    nodes = [{"id": "n1", "agent_name": "a1"}, {"id": "n2", "agent_name": "a2"}]
    edges = [{"from": "n1", "to": "n2"}]
    adapter.record_chain_initialized(job_id=job_id, nodes=nodes, edges=edges)

    state = adapter.get_chain_state(job_id)
    assert state["overall_status"] == "pending"

    # transition n2 (hits line 196 in loop if n1 is first)
    adapter.record_chain_node_transition(
        job_id=job_id,
        node_id="n2",
        status="success",
        started_at=1.0,
        finished_at=2.0,
        error="err",
    )
    state = adapter.get_chain_state(job_id)
    assert state["nodes"][1]["status"] == "success"
    assert state["nodes"][1]["error"] == "err"

    # transition n1
    adapter.record_chain_node_transition(job_id=job_id, node_id="n1", status="success")
    state = adapter.get_chain_state(job_id)
    assert state["overall_status"] == "success"

    # transition - not found
    with pytest.raises(RuntimeError, match="not initialized"):
        adapter.record_chain_node_transition(
            job_id="missing", node_id="n1", status="success"
        )

    with pytest.raises(RuntimeError, match="not found"):
        adapter.record_chain_node_transition(
            job_id=job_id, node_id="missing", status="success"
        )

    # get_chain_state - None
    assert adapter.get_chain_state("missing") is None


def test_sqlite_adapter_workflow_events(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )

    event = MagicMock()
    event.job_id = job_id
    event.event = "e1"
    event.status = "ok"
    event.duration_ms = 100
    event.error_type = None
    event.labels = {"l": "v"}
    event.recorded_at = 1.0

    adapter.record_workflow_event(event)

    # event without labels
    event.labels = None
    adapter.record_workflow_event(event)

    # summary
    summary = adapter.get_events_summary(job_id=job_id)
    assert len(summary) == 1
    assert summary[0]["event"] == "e1"

    summary_all = adapter.get_events_summary()
    assert len(summary_all) >= 1

    # events for job
    events = adapter.get_events_for_job(job_id)
    assert len(events) == 2
    assert events[0]["event"] == "e1"

    # events for job with after_id
    events2 = adapter.get_events_for_job(job_id, after_id=events[0]["id"])
    assert len(events2) == 1


def test_sqlite_adapter_list_jobs_finops(adapter):
    job_id, _ = adapter.create_job(
        tenant_id="t1", kind="benchmark", request_payload={"q": "a"}
    )
    adapter.update_job(job_id, status="succeeded")

    jobs = adapter.list_jobs_for_finops(tenant_id="t1")
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "benchmark"


def test_compute_overall_chain_status():
    from rune_bench.storage.sqlite import SQLiteStorageAdapter as SS

    assert SS._compute_overall_chain_status([]) == "pending"
    assert SS._compute_overall_chain_status([{"status": "failed"}]) == "failed"
    assert SS._compute_overall_chain_status([{"status": "running"}]) == "running"
    assert SS._compute_overall_chain_status([{"status": "pending"}]) == "pending"
    assert SS._compute_overall_chain_status([{"status": "skipped"}]) == "skipped"
    assert (
        SS._compute_overall_chain_status([{"status": "success"}, {"status": "skipped"}])
        == "success"
    )

def test_sqlite_adapter_settings(adapter):
    # test get empty
    assert adapter.get_setting("foo") is None
    
    # test set
    adapter.set_setting("foo", {"bar": "baz"})
    assert adapter.get_setting("foo") == {"bar": "baz"}
    
    # test update
    adapter.set_setting("foo", {"bar": "qux"})
    assert adapter.get_setting("foo") == {"bar": "qux"}
    
    # test list
    adapter.set_setting("foo.1", {"x": 1})
    adapter.set_setting("bar.1", {"y": 2})
    all_settings = adapter.list_settings()
    assert len(all_settings) == 3
    
    prefix_settings = adapter.list_settings(prefix="foo")
    assert len(prefix_settings) == 2
    assert "foo" in prefix_settings
    assert "foo.1" in prefix_settings
    assert "bar.1" not in prefix_settings
    
    # test delete
    adapter.delete_setting("foo")
    assert adapter.get_setting("foo") is None


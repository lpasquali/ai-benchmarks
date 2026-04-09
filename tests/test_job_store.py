# SPDX-License-Identifier: Apache-2.0
import pytest

from rune_bench.job_store import JobStore


def test_job_store_idempotency_is_tenant_scoped(tmp_path):
    store = JobStore(tmp_path / "jobs.db")

    job_id_1, created_1 = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )
    job_id_2, created_2 = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )
    job_id_3, created_3 = store.create_job(
        tenant_id="tenant-b",
        kind="benchmark",
        request_payload={"question": "q1"},
        idempotency_key="same-key",
    )

    assert created_1 is True
    assert created_2 is False
    assert job_id_1 == job_id_2
    assert created_3 is True
    assert job_id_3 != job_id_1


def test_job_store_marks_incomplete_jobs_failed(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id, _ = store.create_job(
        tenant_id="tenant-a",
        kind="agentic-agent",
        request_payload={"question": "q1"},
    )
    store.update_job(job_id, status="running", message="still running")

    store.mark_incomplete_jobs_failed("restarted")

    job = store.get_job(job_id, tenant_id="tenant-a")
    assert job is not None
    assert job.status == "failed"
    assert job.error == "restarted"
    assert job.message == "restarted"


# ── Chain state ─────────────────────────────────────────────────────────────


def _two_node_chain():
    nodes = [
        {"id": "draft", "agent_name": "DraftAgent"},
        {"id": "review", "agent_name": "ReviewAgent"},
    ]
    edges = [{"from": "draft", "to": "review"}]
    return nodes, edges


def test_chain_state_returns_none_when_uninitialized(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    assert store.get_chain_state("missing") is None


def test_chain_state_initialize_normalizes_and_round_trips(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)

    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "pending"
    assert [n["id"] for n in state["nodes"]] == ["draft", "review"]
    # Defaults for status, started_at, finished_at, error are populated.
    for node in state["nodes"]:
        assert node["status"] == "pending"
        assert node["started_at"] is None
        assert node["finished_at"] is None
        assert node["error"] is None
    assert state["edges"] == [{"from": "draft", "to": "review"}]


def test_chain_state_initialize_is_idempotent(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    # Re-initialize with a different node set; latest write wins.
    new_nodes = [{"id": "only", "agent_name": "Only"}]
    store.record_chain_initialized(job_id="job-1", nodes=new_nodes, edges=[])
    state = store.get_chain_state("job-1")
    assert state is not None
    assert [n["id"] for n in state["nodes"]] == ["only"]
    assert state["edges"] == []


def test_chain_state_transition_marks_running_then_success(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)

    store.record_chain_node_transition(
        job_id="job-1", node_id="draft", status="running", started_at=10.0
    )
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "running"
    draft = next(n for n in state["nodes"] if n["id"] == "draft")
    assert draft["status"] == "running"
    assert draft["started_at"] == 10.0

    store.record_chain_node_transition(
        job_id="job-1", node_id="draft", status="success", finished_at=11.5
    )
    state = store.get_chain_state("job-1")
    assert state is not None
    # One node still pending → overall stays pending.
    assert state["overall_status"] == "pending"
    draft = next(n for n in state["nodes"] if n["id"] == "draft")
    assert draft["status"] == "success"
    assert draft["finished_at"] == 11.5


def test_chain_state_transition_propagates_failure_to_overall(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    store.record_chain_node_transition(
        job_id="job-1", node_id="review", status="failed", error="boom"
    )
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "failed"
    review = next(n for n in state["nodes"] if n["id"] == "review")
    assert review["status"] == "failed"
    assert review["error"] == "boom"


def test_chain_state_transition_all_success_overall_success(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    store.record_chain_node_transition(job_id="job-1", node_id="draft", status="success")
    store.record_chain_node_transition(job_id="job-1", node_id="review", status="success")
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "success"


def test_chain_state_transition_all_skipped_overall_skipped(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    store.record_chain_node_transition(job_id="job-1", node_id="draft", status="skipped")
    store.record_chain_node_transition(job_id="job-1", node_id="review", status="skipped")
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "skipped"


def test_chain_state_transition_unknown_job_raises(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    with pytest.raises(RuntimeError, match="not initialized"):
        store.record_chain_node_transition(
            job_id="missing", node_id="x", status="running"
        )


def test_chain_state_transition_unknown_node_raises(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    with pytest.raises(RuntimeError, match="not found"):
        store.record_chain_node_transition(
            job_id="job-1", node_id="ghost", status="running"
        )


def test_chain_state_initialize_empty_nodes_overall_pending(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.record_chain_initialized(job_id="job-empty", nodes=[], edges=[])
    state = store.get_chain_state("job-empty")
    assert state is not None
    assert state["nodes"] == []
    assert state["edges"] == []
    assert state["overall_status"] == "pending"

# SPDX-License-Identifier: Apache-2.0
import pytest
from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore


@pytest.fixture
def store(tmp_path):
    storage = JobStore(tmp_path / "jobs.db")
    yield storage
    storage.close()


def test_job_store_idempotency_is_tenant_scoped(store):
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


def test_job_store_marks_incomplete_jobs_failed(store):
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


def test_chain_state_returns_none_when_uninitialized(store):
    assert store.get_chain_state("missing") is None


def test_chain_state_initialize_normalizes_and_round_trips(store):
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


def test_chain_state_initialize_is_idempotent(store):
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    # Re-initialize with a different node set; latest write wins.
    new_nodes = [{"id": "only", "agent_name": "Only"}]
    store.record_chain_initialized(job_id="job-1", nodes=new_nodes, edges=[])
    state = store.get_chain_state("job-1")
    assert state is not None
    assert [n["id"] for n in state["nodes"]] == ["only"]
    assert state["edges"] == []


def test_chain_state_transition_marks_running_then_success(store):
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


def test_chain_state_transition_propagates_failure_to_overall(store):
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


def test_chain_state_transition_all_success_overall_success(store):
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    store.record_chain_node_transition(
        job_id="job-1", node_id="draft", status="success"
    )
    store.record_chain_node_transition(
        job_id="job-1", node_id="review", status="success"
    )
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "success"


def test_chain_state_transition_all_skipped_overall_skipped(store):
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    store.record_chain_node_transition(
        job_id="job-1", node_id="draft", status="skipped"
    )
    store.record_chain_node_transition(
        job_id="job-1", node_id="review", status="skipped"
    )
    state = store.get_chain_state("job-1")
    assert state is not None
    assert state["overall_status"] == "skipped"


def test_chain_state_transition_unknown_job_raises(store):
    with pytest.raises(RuntimeError, match="not initialized"):
        store.record_chain_node_transition(
            job_id="missing", node_id="x", status="running"
        )


def test_chain_state_transition_unknown_node_raises(store):
    nodes, edges = _two_node_chain()
    store.record_chain_initialized(job_id="job-1", nodes=nodes, edges=edges)
    with pytest.raises(RuntimeError, match="not found"):
        store.record_chain_node_transition(
            job_id="job-1", node_id="ghost", status="running"
        )


def test_chain_state_initialize_empty_nodes_overall_pending(store):
    store.record_chain_initialized(job_id="job-empty", nodes=[], edges=[])
    state = store.get_chain_state("job-empty")
    assert state is not None
    assert state["nodes"] == []
    assert state["edges"] == []
    assert state["overall_status"] == "pending"


# ── Audit artifacts ─────────────────────────────────────────────────────────


import hashlib  # noqa: E402 — grouped with audit-artifact tests


def test_audit_artifact_record_and_list_returns_metadata_only(store):
    payload = b'{"_type": "slsa.provenance"}'
    artifact_id = store.record_audit_artifact(
        job_id="job-1", kind="slsa_provenance", name="provenance.json", content=payload
    )
    assert artifact_id  # uuid

    artifacts = store.list_audit_artifacts("job-1")
    assert len(artifacts) == 1
    a = artifacts[0]
    assert a["artifact_id"] == artifact_id
    assert a["kind"] == "slsa_provenance"
    assert a["name"] == "provenance.json"
    assert a["size_bytes"] == len(payload)
    assert a["sha256"] == hashlib.sha256(payload).hexdigest()
    assert a["created_at"] > 0
    # Bytes are intentionally NOT in the list response
    assert "content" not in a


def test_audit_artifact_list_orders_by_created_at(store):
    a1 = store.record_audit_artifact(
        job_id="job-1", kind="sbom", name="first.json", content=b"{}"
    )
    a2 = store.record_audit_artifact(
        job_id="job-1", kind="tla_report", name="second.txt", content=b"PASS"
    )
    artifacts = store.list_audit_artifacts("job-1")
    assert [a["artifact_id"] for a in artifacts] == [a1, a2]


def test_audit_artifact_list_returns_empty_for_unknown_job(store):
    assert store.list_audit_artifacts("nope") == []


def test_audit_artifact_list_is_job_scoped(store):
    store.record_audit_artifact(
        job_id="job-a", kind="sbom", name="a.json", content=b"{}"
    )
    store.record_audit_artifact(
        job_id="job-b", kind="sbom", name="b.json", content=b"{}"
    )
    a = store.list_audit_artifacts("job-a")
    b = store.list_audit_artifacts("job-b")
    assert len(a) == 1 and a[0]["name"] == "a.json"
    assert len(b) == 1 and b[0]["name"] == "b.json"


def test_audit_artifact_get_returns_bytes_name_kind(store):
    payload = b"binary\x00data"
    artifact_id = store.record_audit_artifact(
        job_id="job-1", kind="sigstore_bundle", name="bundle.sig", content=payload
    )
    fetched = store.get_audit_artifact(job_id="job-1", artifact_id=artifact_id)
    assert fetched is not None
    content, name, kind = fetched
    assert content == payload
    assert name == "bundle.sig"
    assert kind == "sigstore_bundle"


def test_audit_artifact_get_returns_none_for_unknown_artifact(store):
    assert store.get_audit_artifact(job_id="job-1", artifact_id="missing") is None


def test_audit_artifact_get_is_job_scoped(store):
    """Cross-job access by artifact_id alone must fail (returns None)."""
    aid = store.record_audit_artifact(
        job_id="job-a", kind="sbom", name="x.json", content=b"{}"
    )
    # Same artifact_id, wrong job_id → not found
    assert store.get_audit_artifact(job_id="job-b", artifact_id=aid) is None


def test_audit_artifact_record_rejects_unknown_kind(store):
    with pytest.raises(ValueError, match="unknown audit artifact kind"):
        store.record_audit_artifact(
            job_id="job-1", kind="invented", name="x", content=b""
        )


def test_audit_artifact_record_accepts_all_known_kinds(store):
    for kind in [
        "slsa_provenance",
        "sbom",
        "tla_report",
        "sigstore_bundle",
        "rekor_entry",
        "tpm_attestation",
    ]:
        store.record_audit_artifact(
            job_id="job-1", kind=kind, name=f"{kind}.bin", content=b"x"
        )
    assert len(store.list_audit_artifacts("job-1")) == 6

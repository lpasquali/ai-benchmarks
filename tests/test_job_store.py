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

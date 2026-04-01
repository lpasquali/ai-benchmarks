import pytest

from rune_bench.api_client import RuneApiClient


def test_normalize_url_accepts_host_without_scheme():
    assert RuneApiClient._normalize_url("localhost:8080") == "http://localhost:8080"


def test_normalize_url_rejects_invalid():
    with pytest.raises(RuntimeError):
        RuneApiClient._normalize_url(None)


def test_get_vastai_models_validates_payload(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"models": [{"name": "x"}]})

    models = client.get_vastai_models()
    assert models == [{"name": "x"}]


def test_get_ollama_models_validates_payload(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "ollama_url": "http://localhost:11434",
            "models": ["a", "b"],
            "running_models": ["a"],
        },
    )

    payload = client.get_ollama_models("http://localhost:11434")
    assert payload["ollama_url"] == "http://localhost:11434"
    assert payload["models"] == ["a", "b"]
    assert payload["running_models"] == ["a"]


def test_submit_agentic_agent_job_returns_job_id(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"job_id": "job-123"})

    job_id = client.submit_agentic_agent_job({"question": "q"})
    assert job_id == "job-123"


def test_submit_benchmark_job_returns_job_id(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"job_id": "job-987"})

    job_id = client.submit_benchmark_job({"question": "q"})
    assert job_id == "job-987"


def test_wait_for_job_returns_on_success(monkeypatch):
    client = RuneApiClient("http://api:8080")
    statuses = iter([
        {"status": "queued"},
        {"status": "running", "message": "phase 1"},
        {"status": "succeeded", "result": {"answer": "ok"}},
    ])
    monkeypatch.setattr(client, "get_job_status", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr("rune_bench.api_client.time.sleep", lambda *_args, **_kwargs: None)

    payload = client.wait_for_job("job-1", timeout_seconds=5, poll_interval_seconds=0)
    assert payload["status"] == "succeeded"
    assert payload["result"]["answer"] == "ok"


def test_wait_for_job_raises_on_failure(monkeypatch):
    client = RuneApiClient("http://api:8080")
    statuses = iter([
        {"status": "running"},
        {"status": "failed", "error": "boom"},
    ])
    monkeypatch.setattr(client, "get_job_status", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr("rune_bench.api_client.time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="failed"):
        client.wait_for_job("job-2", timeout_seconds=5, poll_interval_seconds=0)

import pytest

from rune_bench.api_client import RuneApiClient
from rune_bench.common import normalize_url


def test_normalize_url_accepts_host_without_scheme():
    assert normalize_url("localhost:8080", service_name="RUNE API") == "http://localhost:8080"


def test_normalize_url_rejects_invalid():
    with pytest.raises(RuntimeError):
        normalize_url(None, service_name="RUNE API")


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


def test_submit_ollama_instance_job_returns_job_id(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"job_id": "job-654"})

    job_id = client.submit_ollama_instance_job({"vastai": False})
    assert job_id == "job-654"


def test_request_adds_auth_tenant_and_idempotency_headers(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout, context=None):
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", fake_urlopen)

    client = RuneApiClient("http://api:8080", api_token="secret", tenant_id="tenant-a")
    payload = client._request("POST", "/v1/jobs/benchmark", body={"x": 1}, idempotency_key="idem-1")

    assert payload == {"ok": True}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["headers"]["X-api-key"] == "secret"
    assert captured["headers"]["X-tenant-id"] == "tenant-a"
    assert captured["headers"]["Idempotency-key"] == "idem-1"


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

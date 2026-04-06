import hashlib
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rune_bench.api_client import RuneApiClient
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.job_store import JobStore


@pytest.fixture
def rune_api_server(tmp_path):
    state = {"agentic_calls": 0}

    def run_agentic(request):
        state["agentic_calls"] += 1
        return {"answer": f"ok:{request.question}"}

    app = RuneApiApplication(
        store=JobStore(tmp_path / "jobs.db"),
        security=ApiSecurityConfig(
            auth_disabled=False, 
            tenant_tokens={
                "tenant-a": hashlib.sha256(b"token-a").hexdigest(), 
                "tenant-b": hashlib.sha256(b"token-b").hexdigest()
            }
        ),
        backend_functions={
            "agentic-agent": run_agentic,
            "benchmark": lambda request: {"answer": "bench"},
            "ollama-instance": lambda request: {"mode": "existing", "backend_url": request.backend_url},
        },
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url, state
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_healthz_is_public(rune_api_server):
    base_url, _state = rune_api_server
    with urlopen(f"{base_url}/healthz") as response:  # nosec  # test request mock/local execution
        payload = json.loads(response.read().decode("utf-8"))

    assert payload == {"status": "ok"}


def test_api_server_requires_auth(rune_api_server):
    base_url, _state = rune_api_server
    request = Request(f"{base_url}/v1/catalog/vastai-models")

    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec  # test request mock/local execution

    assert exc.value.code == 401


def test_api_server_enforces_tenant_scoping_and_idempotency(rune_api_server):
    base_url, state = rune_api_server
    client_a = RuneApiClient(base_url, api_token="token-a", tenant_id="tenant-a")  # nosec  # test credentials
    client_b = RuneApiClient(base_url, api_token="token-b", tenant_id="tenant-b")  # nosec  # test credentials
    request_payload = {
        "question": "What is unhealthy?",
        "model": "llama3.1:8b",
        "backend_url": None,
        "backend_warmup": False,
        "backend_warmup_timeout": 1,
        "kubeconfig": "/tmp/config",  # nosec  # test artifact paths
    }

    job_id_1 = client_a.submit_agentic_agent_job(request_payload, idempotency_key="idem-1")
    job_id_2 = client_a.submit_agentic_agent_job(request_payload, idempotency_key="idem-1")

    assert job_id_1 == job_id_2

    payload = client_a.wait_for_job(job_id_1, timeout_seconds=5, poll_interval_seconds=0.01)
    assert payload["result"]["answer"] == "ok:What is unhealthy?"
    assert state["agentic_calls"] == 1

    with pytest.raises(RuntimeError, match="job not found"):
        client_b.get_job_status(job_id_1)


def test_api_server_rate_limiting(rune_api_server):
    base_url, _state = rune_api_server
    
    # Attempt 10 failed logins
    for i in range(10):
        request = Request(f"{base_url}/v1/catalog/vastai-models")
        request.add_header("Authorization", "Bearer invalid-token")
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 401
    
    # 11th attempt should trigger rate limit
    request = Request(f"{base_url}/v1/catalog/vastai-models")
    request.add_header("Authorization", "Bearer invalid-token")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)
    assert exc.value.code == 401
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert payload["error"] == "rate limit exceeded"

# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from argon2 import PasswordHasher

from rune_bench.api_client import RuneApiClient
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.job_store import JobStore

_ph = PasswordHasher()


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
            "llm-instance": lambda request: {"mode": "existing", "backend_url": request.backend_url},
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


# ── /v1/chains/{run_id}/state ───────────────────────────────────────────────


def _auth_headers(token: str, tenant: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-Tenant-ID": tenant,
    }


def _get_json(url: str, headers: dict) -> dict:
    request = Request(url)
    for k, v in headers.items():
        request.add_header(k, v)
    with urlopen(request) as response:  # nosec  # local test request
        return json.loads(response.read().decode("utf-8"))


def test_chain_state_returns_404_for_unknown_run(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/chains/does-not-exist/state")
    request.add_header("Authorization", "Bearer token-a")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "chain run not found" in payload["error"]


def test_chain_state_returns_404_for_other_tenants_run(rune_api_server):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token="token-a", tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="t1",
    )

    request = Request(f"{base_url}/v1/chains/{job_id}/state")
    request.add_header("Authorization", "Bearer token-b")  # nosec
    request.add_header("X-Tenant-ID", "tenant-b")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404


def test_chain_state_returns_empty_shell_when_job_exists_but_no_chain_state(rune_api_server, tmp_path):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token="token-a", tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="empty-shell",
    )

    payload = _get_json(
        f"{base_url}/v1/chains/{job_id}/state",
        _auth_headers("token-a", "tenant-a"),
    )
    assert payload == {
        "run_id": job_id,
        "nodes": [],
        "edges": [],
        "overall_status": "pending",
    }


def test_chain_state_returns_full_state_shape(rune_api_server):
    """Verify the API endpoint returns the documented JSON shape.

    Full populated-state behavior is exercised at the JobStore level in
    test_job_store.py (where we control the DB directly). This test confirms
    the wire format the dashboard will consume.
    """
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token="token-a", tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="full-state",
    )

    payload = _get_json(
        f"{base_url}/v1/chains/{job_id}/state",
        _auth_headers("token-a", "tenant-a"),
    )
    # Empty-shell shape (no chain state initialized for this job)
    assert payload["run_id"] == job_id
    assert payload["overall_status"] == "pending"
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_chain_state_endpoint_requires_auth(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/chains/anything/state")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 401


def test_chain_state_endpoint_rejects_empty_run_id(rune_api_server):
    base_url, _ = rune_api_server
    # /v1/chains//state has empty run_id between the two slashes
    request = Request(f"{base_url}/v1/chains//state")
    request.add_header("Authorization", "Bearer token-a")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 400
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "missing run_id" in payload["error"]

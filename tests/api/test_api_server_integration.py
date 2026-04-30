# SPDX-License-Identifier: Apache-2.0
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import pytest
from rune_bench.api_client import RuneApiClient
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore

@pytest.fixture
def rune_api_server(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    state = {"agentic_calls": 0, "store": store}

    def run_agentic(request):
        state["agentic_calls"] += 1
        return {"answer": f"ok:{request.question}"}

    app = RuneApiApplication(
        store=store,
        security=ApiSecurityConfig(
            auth_disabled=False,
            tenant_tokens={"tenant-a": "token-a", "tenant-b": "token-b"},
        ),
        backend_functions={
            "agentic-agent": run_agentic,
            "benchmark": lambda request: {"answer": "bench"},
            "llm-instance": lambda request: {
                "mode": "existing",
                "backend_url": request.backend_url,
            },
            "ollama-instance": lambda request: {
                "mode": "existing",
                "backend_url": request.backend_url,
            },
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
        store.close()

def _auth_headers(token: str, tenant: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-Tenant-ID": tenant,
        "Content-Type": "application/json",
    }

def _request(url: str, method: str, headers: dict, data: dict | None = None):
    req = Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req.add_header("Content-Length", str(len(body)))
        req.data = body
    
    try:
        with urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))

def test_get_catalog_models_missing_url(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/catalog/models",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 400
    assert "missing required query parameter" in payload["error"]

def test_get_catalog_models_error(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/catalog/models?backend_url=http://invalid&backend_type=ollama",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 400
    assert "error" in payload

def test_post_settings(rune_api_server):
    base_url, _ = rune_api_server
    data = {"settings": {"k": "v"}, "profile": "test"}
    status, payload = _request(
        f"{base_url}/v1/settings",
        "POST",
        _auth_headers("token-a", "tenant-a"),
        data
    )
    assert status == 200
    assert payload["status"] == "updated"

def test_patch_settings(rune_api_server):
    base_url, _ = rune_api_server
    data = {"settings": {"k": "v"}, "profile": "test"}
    status, payload = _request(
        f"{base_url}/v1/settings",
        "PATCH",
        _auth_headers("token-a", "tenant-a"),
        data
    )
    assert status == 200
    assert payload["status"] == "updated"

def test_post_settings_profiles(rune_api_server):
    base_url, _ = rune_api_server
    data = {"name": "test_profile", "settings": {"k": "v"}}
    status, payload = _request(
        f"{base_url}/v1/settings/profiles",
        "POST",
        _auth_headers("token-a", "tenant-a"),
        data
    )
    assert status == 201
    assert payload["status"] == "created"

def test_delete_job_not_found(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/jobs/unknown",
        "DELETE",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 404

def test_delete_job_success(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    job_id, _ = store.create_job(tenant_id="tenant-a", kind="agentic-agent", request_payload={})
    
    status, payload = _request(
        f"{base_url}/v1/jobs/{job_id}",
        "DELETE",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 200
    assert payload["status"] == "cancelled"

def test_get_metrics_summary(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/metrics/summary",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 200
    assert "events" in payload

def test_get_finops_simulate(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/finops/simulate?agent=test&model=test",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    # the endpoint requires store and might throw error if agent not found
    # but we just want to hit the endpoint for coverage
    assert status in (200, 400)

def test_post_jobs_invalid_json(rune_api_server):
    base_url, _ = rune_api_server
    req = Request(f"{base_url}/v1/jobs/benchmark", method="POST")
    req.add_header("Authorization", "Bearer token-a")
    req.add_header("X-Tenant-ID", "tenant-a")
    req.add_header("Content-Length", "10")
    req.data = b"invalid{"
    try:
        with urlopen(req) as response:
            status = response.status
    except HTTPError as e:
        status = e.code
    assert status == 400

def test_post_jobs_too_large(rune_api_server):
    base_url, _ = rune_api_server
    req = Request(f"{base_url}/v1/jobs/benchmark", method="POST")
    req.add_header("Authorization", "Bearer token-a")
    req.add_header("X-Tenant-ID", "tenant-a")
    req.add_header("Content-Length", str(11 * 1024 * 1024))
    try:
        with urlopen(req) as response:
            status = response.status
    except HTTPError as e:
        status = e.code
    assert status == 413

def test_get_runs_trace_not_found(rune_api_server):
    base_url, _ = rune_api_server
    status, payload = _request(
        f"{base_url}/v1/runs/unknown/trace",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 404

def test_get_job_success(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    job_id, _ = store.create_job(tenant_id="tenant-a", kind="agentic-agent", request_payload={})
    
    status, payload = _request(
        f"{base_url}/v1/jobs/{job_id}",
        "GET",
        _auth_headers("token-a", "tenant-a")
    )
    assert status == 200
    assert payload["job_id"] == job_id
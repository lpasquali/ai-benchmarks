# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from rune_bench.resources.vastai.instance import InstanceManager
from rune_bench.resources.vastai.contracts import ConnectionDetails
import rune_bench.api_server as api_server
from rune_bench.job_store import JobStore

def test_instance_manager_details_parsing():
    details = InstanceManager.build_connection_details(
        1,
        {"state": "running", "ports": {"svc": [{"HostIp": "1.2.3.4", "HostPort": "8080"}]}, "ssh_host": "h1", "ssh_port": 2222},
    )
    assert details.status == "running"
    assert details.ssh_host == "h1"
    assert details.ssh_port == 2222
    assert len(details.service_urls) == 1
    assert details.service_urls[0]["direct"] == "http://1.2.3.4:8080"

    details = InstanceManager.build_connection_details(
        1,
        {"state": "created", "ports": {"svc": [{"HostIp": "1.2.3.4", "HostPort": "8080"}]}, "ssh_host": None, "ssh_port": None},
    )
    assert details.status == "created"
    assert details.service_urls[0]["proxy"] is None


@pytest.fixture
def misc_server(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "1")
    monkeypatch.delenv("RUNE_DB_URL", raising=False)
    # The from_env uses RUNE_DATABASE_URL
    monkeypatch.setenv("RUNE_DATABASE_URL", f"sqlite:///{tmp_path}/jobs.db")
    app = api_server.RuneApiApplication.from_env()
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", app
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        app.store.close()


def test_api_security_from_env(monkeypatch):
    monkeypatch.delenv("RUNE_API_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("RUNE_API_TOKENS", raising=False)
    with pytest.raises(RuntimeError, match="no tenants are configured"):
        api_server.ApiSecurityConfig.from_env()

    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "1")
    assert api_server.ApiSecurityConfig.from_env().auth_disabled is True

    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "0")
    long_a, long_b = "a" * 32, "b" * 32
    monkeypatch.setenv("RUNE_API_TOKENS", f"tenant-a:{long_a},tenant-b:{long_b}")
    cfg = api_server.ApiSecurityConfig.from_env()
    assert cfg.tenant_tokens["tenant-b"] == hashlib.sha256(long_b.encode("utf-8")).hexdigest()

def test_api_server_misc_paths(misc_server):
    base_url, app = misc_server

    with urlopen(f"{base_url}/v1/catalog/vastai-models") as response:  # nosec
        payload = json.loads(response.read().decode("utf-8"))
    assert isinstance(payload, list)
    assert any(m["name"] == "llama3.1:8b" for m in payload)


    req = Request(f"{base_url}/v1/catalog/models")
    with pytest.raises(HTTPError) as exc:
        urlopen(req)  # nosec
    assert exc.value.code == 400

    bad_req = Request(
        f"{base_url}/v1/jobs/agentic-agent",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc:
        urlopen(bad_req)  # nosec
    assert exc.value.code == 400

    unknown_req = Request(f"{base_url}/nope")
    with pytest.raises(HTTPError) as exc:
        urlopen(unknown_req)  # nosec
    assert exc.value.code == 404

    job = app.store.create_job(tenant_id="default", kind="agentic-agent", request_payload={})[0]

    with urlopen(f"{base_url}/v1/jobs/{job}") as response:  # nosec  # test request mock/local execution
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["job_id"] == job


@pytest.mark.asyncio
async def test_api_server_internal_dispatch_and_failures(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        app = api_server.RuneApiApplication(
            store=store,
            security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
            backend_functions={"agentic-agent": lambda request: {"ok": True}},
        )

        with pytest.raises(RuntimeError, match="unsupported job kind"):
            await app._dispatch("nope", {})

        with pytest.raises(RuntimeError, match="no backend function registered"):
            await app._dispatch(
                "benchmark",
                {
                    "vastai": False,
                    "template_hash": "t",
                    "min_dph": 1,
                    "max_dph": 2,
                    "reliability": 0.9,
                    "backend_url": None,
                    "question": "q",
                    "model": "m",
                    "backend_warmup": False,
                    "backend_warmup_timeout": 1,
                    "kubeconfig": "/tmp/k",  # nosec
                    "vastai_stop_instance": False,
                },
            )

        handler = app.backend_functions["agentic-agent"]
        await app._execute_job("missing", handler, "agentic-agent", {"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"})  # nosec
        assert store.get_job("missing") is None
    finally:
        store.close()


def test_job_to_payload_fields(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        job_id, _ = store.create_job(tenant_id="t", kind="agentic-agent", request_payload={"x": 1})
        store.update_job(job_id, status="failed", error="boom", message="bad")
        job = store.get_job(job_id)
        payload = api_server._job_to_payload(job)
        assert payload["error"] == "boom"
        assert payload["kind"] == "agentic-agent"
    finally:
        store.close()

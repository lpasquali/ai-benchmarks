# SPDX-License-Identifier: Apache-2.0
import hashlib
from unittest.mock import MagicMock
from urllib.request import Request, urlopen

import pytest
from argon2 import PasswordHasher

import rune_bench.api_backend as api_backend
import rune_bench.api_client as api_client_module
import rune_bench.api_server as api_server
import rune_bench.workflows as workflows
from rune_bench.agents.sre.holmes import HolmesRunner
from rune_bench.api_client import RuneApiClient
from rune_bench.resources.vastai import InstanceManager
from rune_bench.resources.vastai import OfferFinder
from rune_bench.resources.vastai import TemplateLoader

_ph = PasswordHasher()


def test_holmes_runner_remaining_paths(monkeypatch, tmp_path):
    """Test transport delegation and error propagation in HolmesDriverClient."""
    import rune_bench.drivers.holmes as holmes_driver_module

    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")

    # transport error is propagated as RuntimeError
    failing_transport = MagicMock()
    failing_transport.call.side_effect = RuntimeError("transport failed")
    runner = HolmesRunner(kubeconfig, transport=failing_transport)
    with pytest.raises(RuntimeError, match="transport failed"):
        runner.ask("q", "m")

    # transport returns answer correctly
    ok_transport = MagicMock()
    ok_transport.call.return_value = {"answer": "great"}
    runner2 = HolmesRunner(kubeconfig, transport=ok_transport)
    assert runner2.ask("q", "m") == "great"

    # _fetch_model_limits returns {} when get_backend raises
    bad_backend = MagicMock()
    bad_backend.normalize_model_name.side_effect = RuntimeError("norm failed")
    monkeypatch.setattr(holmes_driver_module, "get_backend", lambda *_args, **_kw: bad_backend)
    limits = runner2._fetch_model_limits(model="m", backend_url="http://x")
    assert limits == {}

    # _fetch_model_limits returns {} when no backend_url
    limits2 = runner2._fetch_model_limits(model="m", backend_url=None)
    assert limits2 == {}


def test_api_client_remaining_paths(monkeypatch):
    monkeypatch.setenv("RUNE_API_TOKEN", "env-token")
    monkeypatch.setenv("RUNE_API_TENANT", "env-tenant")
    client = RuneApiClient("api:8080")
    assert client.api_token == "env-token"  # nosec  # test credentials
    assert client.tenant_id == "env-tenant"

    from urllib.error import HTTPError, URLError

    class NoDetailHTTPError(HTTPError):
        def __init__(self):
            super().__init__("http://api", 500, "bad", hdrs=None, fp=None)

        def read(self):
            return b""

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(NoDetailHTTPError()))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client._request("GET", "/x")

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        client._request("GET", "/x")

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("late")))
    with pytest.raises(RuntimeError, match="late"):
        client._request("GET", "/x")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __init__(self, raw):
            self._raw = raw

        def read(self):
            return self._raw

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *args, **kwargs: Response(b"[]"))
    with pytest.raises(RuntimeError, match="Unexpected JSON payload"):
        client._request("GET", "/x")

    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="missing 'models' list"):
        client.get_vastai_models()
    with pytest.raises(RuntimeError, match="missing 'models' list"):
        client.get_ollama_models("http://x")

    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {"models": [], "running_models": "bad"})
    with pytest.raises(RuntimeError, match="missing 'running_models' list"):
        client.get_ollama_models("http://x")

    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    for submit in (client.submit_agentic_agent_job, client.submit_benchmark_job, client.submit_ollama_instance_job):
        with pytest.raises(RuntimeError, match="missing 'job_id'"):
            submit({})
    with pytest.raises(RuntimeError, match="missing 'status'"):
        client.get_job_status("job")

    statuses = iter([{"status": "running"}, {"status": "cancelled", "message": "bye"}])
    monkeypatch.setattr(client, "get_job_status", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr(api_client_module.time, "sleep", lambda *_args, **_kwargs: None)
    updates = []
    with pytest.raises(RuntimeError, match="bye"):
        client.wait_for_job("job", timeout_seconds=5, poll_interval_seconds=0, on_update=lambda status, message: updates.append((status, message)))
    assert updates[0] == ("running", None)

    values = iter([0.0, 10.0])
    monkeypatch.setattr(api_client_module.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(client, "get_job_status", lambda *_args, **_kwargs: {"status": "running"})
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        client.wait_for_job("job", timeout_seconds=1, poll_interval_seconds=0)


def test_api_server_remaining_paths(monkeypatch, tmp_path):
    store = api_server.JobStore(tmp_path / "jobs.db")
    created = []

    def backend_ollama(request):
        created.append(request.backend_url)
        return {"mode": "existing"}

    def backend_bench(request):
        return {"answer": request.question}

    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=False, tenant_tokens={"tenant": hashlib.sha256(b"token").hexdigest()}),
        backend_functions={"llm-instance": backend_ollama, "ollama-instance": backend_ollama, "benchmark": backend_bench, "agentic-agent": lambda request: (_ for _ in ()).throw(RuntimeError("bad-run"))},
    )
    monkeypatch.setattr(
        api_server,
        "list_backend_models",
        lambda backend_url, **kw: {"backend_url": backend_url, "backend_type": kw.get("backend_type", "ollama"), "models": [], "running_models": []},
    )
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        req = Request(f"{base}/v1/ollama/models?backend_url=http://x", headers={"Authorization": "Bearer token", "X-Tenant-ID": "tenant"})
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            assert response.status == 200

        bad_auth = Request(f"{base}/v1/jobs/whatever", method="POST", data=b"{}", headers={"Authorization": "Bearer wrong", "X-Tenant-ID": "tenant", "Content-Type": "application/json"})
        with pytest.raises(Exception):
            urlopen(bad_auth)  # nosec  # test request mock/local execution

        bad_kind = Request(f"{base}/v1/jobs/whatever", method="POST", data=b"{}", headers={"Authorization": "Bearer token", "X-Tenant-ID": "tenant", "Content-Type": "application/json"})
        with pytest.raises(Exception):
            urlopen(bad_kind)  # nosec  # test request mock/local execution

        req = Request(f"{base}/v1/jobs/ollama-instance", method="POST", data=b'{"vastai": false, "template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9, "backend_url": "http://x"}', headers={"Authorization": "Bearer token", "X-Tenant-ID": "tenant", "Content-Type": "application/json"})
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            payload = response.read().decode("utf-8")
            assert "job_id" in payload

        req = Request(f"{base}/v1/jobs/benchmark", method="POST", data=b'{"vastai": false, "template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9, "backend_url": null, "question": "q", "model": "m", "backend_warmup": false, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k", "vastai_stop_instance": false}', headers={"Authorization": "Bearer token", "X-Tenant-ID": "tenant", "Content-Type": "application/json", "Idempotency-Key": "id1"})
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            assert response.status == 202
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    jobs = [store.get_job(job_id) for job_id, _ in [store.create_job(tenant_id="tenant", kind="agentic-agent", request_payload={})]]
    assert jobs[0] is not None

    job_id, _ = store.create_job(tenant_id="tenant", kind="agentic-agent", request_payload={"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"})  # nosec  # test artifact paths
    app._execute_job(job_id, "agentic-agent", {"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"})  # nosec  # test artifact paths
    assert store.get_job(job_id).status == "failed"

    assert app._dispatch("llm-instance", {"vastai": False, "template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9, "backend_url": "http://x"}) == {"mode": "existing"}
    assert app._dispatch("benchmark", {"vastai": False, "template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9, "backend_url": None, "question": "qq", "model": "m", "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k", "vastai_stop_instance": False}) == {"answer": "qq"}  # nosec  # test artifact paths

    monkeypatch.setattr(api_server, "ThreadingHTTPServer", lambda *args, **kwargs: type("S", (), {"serve_forever": lambda self: None, "server_close": lambda self: None})())
    api_server.RuneApiApplication(store=store, security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={})).serve("127.0.0.1", 0)


def test_api_server_error_paths(monkeypatch, tmp_path):
    class ExplodingStore(api_server.JobStore):
        def create_job(self, **kwargs):
            raise RuntimeError("db-down")

    store = ExplodingStore(tmp_path / "jobs.db")
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=False, tenant_tokens={"tenant": hashlib.sha256(b"token").hexdigest()}),
        backend_functions={"agentic-agent": lambda request: {"answer": "ok"}},
    )
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        req = Request(
            f"{base}/v1/jobs/agentic-agent",
            method="POST",
            data=b"{",
            headers={"X-API-Key": "token", "X-Tenant-ID": "tenant", "Content-Type": "application/json"},
        )
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution

        req = Request(
            f"{base}/v1/jobs/agentic-agent",
            method="POST",
            data=b"{}",
            headers={"X-API-Key": "token", "X-Tenant-ID": "tenant", "Content-Type": "application/json"},
        )
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution

        req = Request(f"{base}/v1/jobs/missing", headers={"X-API-Key": "token", "X-Tenant-ID": "tenant"})
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    real_store = api_server.JobStore(tmp_path / "ok.db")
    app = api_server.RuneApiApplication(
        store=real_store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
        backend_functions={"agentic-agent": lambda request: {"answer": request.question}},
    )
    job_id, _ = real_store.create_job(
        tenant_id="default",
        kind="agentic-agent",
        request_payload={"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"},  # nosec  # test artifact paths
    )
    app._execute_job(job_id, "agentic-agent", {"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"})  # nosec  # test artifact paths
    assert real_store.get_job(job_id).status == "succeeded"


def test_offer_template_backend_instance_and_workflow_remaining(monkeypatch, tmp_path):
    sdk = MagicMock()
    sdk.search_offers.side_effect = Exception("down")
    with pytest.raises(RuntimeError, match="offer search failed"):
        OfferFinder(sdk).find_best(1, 2, 0.9)

    sdk = MagicMock()
    sdk.search_offers.return_value = [{"id": None, "gpu_total_ram": 0}]
    with pytest.raises(RuntimeError, match="missing id or gpu_total_ram"):
        OfferFinder(sdk).find_best(1, 2, 0.9)

    sdk = MagicMock()
    sdk.show_templates.side_effect = Exception("down")
    with pytest.raises(RuntimeError, match="Failed to fetch Vast.ai templates"):
        TemplateLoader(sdk).load("x")

    sdk = MagicMock()
    sdk.show_templates.return_value = {"not": "a-list"}
    with pytest.raises(RuntimeError, match="Template 'x' not found"):
        TemplateLoader(sdk).load("x")

    assert TemplateLoader._find([{"hash_id": "x"}], "x") == {"hash_id": "x"}

    from rune_bench.resources.base import ProvisioningResult

    stopped = []
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda req: type("P", (), {
            "provision": lambda self: ProvisioningResult(backend_url="http://x", model="m", provider_handle=8),
            "teardown": lambda self, r: stopped.append(True),
        })(),
    )
    monkeypatch.setattr(
        api_backend,
        "_make_agent_runner",
        lambda path: type("R", (), {"ask": lambda self, **_: "a"})(),
    )
    result = api_backend.run_benchmark(api_backend.RunBenchmarkRequest(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url=None, question="q", model="m", backend_warmup=False, backend_warmup_timeout=1, kubeconfig=str(kubeconfig), vastai_stop_instance=True))
    assert result["contract_id"] == 8
    assert stopped == [True]

    sdk = MagicMock()
    sdk.show_instances.return_value = [{"id": 1, "actual_status": "stopped"}]
    assert InstanceManager(sdk).find_reusable_running_instance(min_dph=1, max_dph=2, reliability=0.9) is None

    sdk.show_instances.return_value = [{"id": 1, "actual_status": "running", "dph_total": 9, "reliability": 0.1}]
    assert InstanceManager(sdk).find_reusable_running_instance(min_dph=1, max_dph=2, reliability=0.9) is None

    sdk.destroy_instance.side_effect = Exception("boom")
    with pytest.raises(RuntimeError, match="Failed to destroy Vast.ai instance"):
        InstanceManager(sdk)._destroy_instance(1)

    manager = InstanceManager(MagicMock())
    # Use itertools.chain so that after the two controlled values (needed for the
    # timeout logic in _wait_until_instance_absent) the iterator never exhausts.
    # This is necessary because the monkeypatch targets the global time module,
    # so span() calls in subsequent workflow code also hit this lambda.
    import itertools
    values = itertools.chain(iter([0.0, 10.0]), itertools.repeat(999.0))
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.monotonic", lambda: next(values))
    monkeypatch.setattr(manager, "_fetch_instance", lambda _cid: {"id": _cid})
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.sleep", lambda *_: None)
    assert manager._wait_until_instance_absent(1, timeout_seconds=1) is False

    sdk = MagicMock()
    sdk.show_instances.return_value = {"bad": True}
    assert InstanceManager(sdk).list_instances() == []

    sdk = MagicMock()
    sdk.show_instances.return_value = [
        {"id": 1, "actual_status": "running", "gpu_total_ram": 100, "dph_total": 1.9, "reliability": 0.95},
        {"id": 2, "actual_status": "running", "gpu_total_ram": 100, "dph_total": 1.2, "reliability": 0.95},
    ]
    assert InstanceManager(sdk).find_reusable_running_instance(min_dph=1, max_dph=2, reliability=0.9)["id"] == 2

    called = []
    manager = InstanceManager(MagicMock())
    monkeypatch.setattr(manager, "_destroy_instance", lambda contract_id: called.append(contract_id))
    manager.stop_instance(9)
    assert called == [9]

    details = InstanceManager.build_connection_details(1, {"actual_status": "running", "ports": {"svc2": [{"HostIp": None, "HostPort": None}]}})
    assert details.service_urls == []

    sdk = MagicMock()
    sdk.show_volumes.side_effect = Exception("nope")
    assert InstanceManager(sdk)._list_volumes_optional() is None

    monkeypatch.setattr("rune_bench.backends.ollama.OllamaClient", lambda _url: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError, match="bad"):
        workflows.normalize_backend_url("x")

    class FakeManager:
        def __init__(self, _sdk):
            pass

        def find_reusable_running_instance(self, **kwargs):
            return {"id": 1, "gpu_total_ram": 10}

        @staticmethod
        def build_connection_details(contract_id, _info):
            return workflows.ConnectionDetails(contract_id=contract_id, status="running", ssh_host=None, ssh_port=None, machine_id=None, service_urls=[])

    monkeypatch.setattr(workflows, "InstanceManager", FakeManager)
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _vram: (_ for _ in ()).throw(RuntimeError("small")))
    monkeypatch.setattr(workflows.OfferFinder, "find_best", lambda self, **_: type("Offer", (), {"offer_id": 5, "total_vram_mb": 24000})())
    monkeypatch.setattr(workflows.TemplateLoader, "load", lambda self, _hash: type("Tpl", (), {"env": "ENV=1", "image": "img"})())

    class CreatedManager(FakeManager):
        def __init__(self, _sdk):
            pass

        def find_reusable_running_instance(self, **kwargs):
            return {"id": 1, "gpu_total_ram": 10}

        def create(self, *args, **kwargs):
            return 7

        def wait_until_running(self, *args, **kwargs):
            return {"id": 7}

        def pull_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr(workflows, "InstanceManager", CreatedManager)
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _vram: type("M", (), {"name": "m", "vram_mb": 1, "required_disk_gb": 2})())
    monkeypatch.setattr(workflows, "list_backend_models", lambda _url: [])
    monkeypatch.setattr(workflows, "list_running_backend_models", lambda _url: [])
    monkeypatch.setattr(workflows, "normalize_backend_model_for_api", lambda model: model)
    monkeypatch.setattr(workflows, "warmup_backend_model", lambda *_args, **_kwargs: "m")
    res = workflows.provision_vastai_backend(MagicMock(), template_hash="t", min_dph=1, max_dph=2, reliability=0.9, confirm_create=lambda: True)
    assert res.pull_warning is not None


def test_api_backend_and_workflow_last_edges(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    monkeypatch.setattr(api_backend, "_make_agent_runner", lambda **kwargs: type("R", (), {"ask": lambda self, **_: "a"})())
    result = api_backend.run_agentic_agent(api_backend.RunAgenticAgentRequest(question="q", model="m", backend_url=None, backend_warmup=False, backend_warmup_timeout=1, kubeconfig=str(kubeconfig)))
    assert result == {"answer": "a"}

    fake_client = MagicMock()
    fake_client.get_available_models.return_value = ["x"]
    from rune_bench.workflows import use_existing_backend_server
    assert callable(use_existing_backend_server)


def test_cost_estimate_backend_and_server_endpoints(monkeypatch, tmp_path):
    """Cover _get_cost_estimate_backend (lines 55-58) and the /v1/estimates,
    /v1/metrics/summary and /v1/jobs/{id}/events server endpoints."""
    import threading
    from rune_bench.api_contracts import CostEstimationRequest

    # --- _get_cost_estimate_backend direct call ---
    result = api_server._get_cost_estimate_backend(
        CostEstimationRequest(vastai=True, min_dph=2.0, max_dph=3.0, estimated_duration_seconds=3600)
    )
    assert "projected_cost_usd" in result

    with pytest.raises(RuntimeError, match="invalid request type"):
        api_server._get_cost_estimate_backend(
            api_server.RunAgenticAgentRequest(question="q", model="m", backend_url=None, backend_warmup=False, backend_warmup_timeout=1, kubeconfig="/k")
        )

    # --- live server for endpoint coverage ---
    store = api_server.JobStore(tmp_path / "jobs.db")
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=False, tenant_tokens={"t": hashlib.sha256(b"tok").hexdigest()}),
        backend_functions={
            "cost-estimate": lambda req: {"projected_cost_usd": 1.5, "cost_driver": "vastai", "resource_impact": "low", "local_energy_kwh": 0.0, "confidence_score": 1.0, "warning": None},
        },
    )
    monkeypatch.setattr(api_server, "list_backend_models", lambda backend_url, **kw: {"backend_url": backend_url, "backend_type": kw.get("backend_type", "ollama"), "models": [], "running_models": []})
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        auth_headers = {"Authorization": "Bearer tok", "X-Tenant-ID": "t"}

        # /v1/estimates POST (lines 226-231)
        req = Request(
            f"{base}/v1/estimates",
            method="POST",
            data=b'{"vastai": true, "min_dph": 2.0, "max_dph": 3.0, "estimated_duration_seconds": 3600}',
            headers={**auth_headers, "Content-Type": "application/json"},
        )
        with urlopen(req) as resp:  # nosec  # test request mock/local execution
            import json as _json
            payload = _json.loads(resp.read())
        assert payload["projected_cost_usd"] == 1.5

        # /v1/metrics/summary GET (lines 183-188)
        req = Request(f"{base}/v1/metrics/summary", headers=auth_headers)
        with urlopen(req) as resp:  # nosec  # test request mock/local execution
            payload = _json.loads(resp.read())
        assert "events" in payload

        # /v1/jobs/{id}/events GET — job not found (lines 193-197)
        req = Request(f"{base}/v1/jobs/nonexistent/events", headers=auth_headers)
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution

        # /v1/jobs/{id}/events GET — job exists (lines 198-200)
        job_id, _ = store.create_job(tenant_id="t", kind="agentic-agent", request_payload={"question": "q"})
        req = Request(f"{base}/v1/jobs/{job_id}/events", headers=auth_headers)
        with urlopen(req) as resp:  # nosec  # test request mock/local execution
            payload = _json.loads(resp.read())
        assert payload["job_id"] == job_id
        assert "events" in payload

    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

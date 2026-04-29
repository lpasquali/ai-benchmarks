# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from urllib.request import Request, urlopen

import pytest
from argon2 import PasswordHasher

import rune_bench.api_backend as api_backend
import rune_bench.api_client as api_client_module
import rune_bench.api_server as api_server
from rune_bench.agents.base import AgentResult
from rune_bench.agents.sre.holmes import HolmesRunner
from rune_bench.api_client import RuneApiClient
from rune_bench.resources.vastai import OfferFinder
from rune_bench.resources.vastai import TemplateLoader

_ph = PasswordHasher()
_COMPREHENSIVE_API_TOKEN = "test-token-32-chars-long-12345678"


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
    monkeypatch.setattr(
        holmes_driver_module, "get_backend", lambda *_args, **_kw: bad_backend
    )
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

    monkeypatch.setattr(
        "rune_bench.common.http_client.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(NoDetailHTTPError()),
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client._request("GET", "/x")

    monkeypatch.setattr(
        "rune_bench.common.http_client.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        client._request("GET", "/x")

    monkeypatch.setattr(
        "rune_bench.common.http_client.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("late")),
    )
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

    monkeypatch.setattr(
        "rune_bench.common.http_client.urlopen", lambda *args, **kwargs: Response(b"[]")
    )
    with pytest.raises(RuntimeError, match="Unexpected JSON payload"):
        client._request("GET", "/x")

    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="missing 'models' list"):
        client.get_vastai_models()
    with pytest.raises(RuntimeError, match="missing 'models' list"):
        client.get_ollama_models("http://x")

    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: {"models": [], "running_models": "bad"},
    )
    with pytest.raises(RuntimeError, match="missing 'running_models' list"):
        client.get_ollama_models("http://x")

    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    for submit in (
        client.submit_agentic_agent_job,
        client.submit_benchmark_job,
        client.submit_ollama_instance_job,
    ):
        with pytest.raises(RuntimeError, match="missing 'job_id'"):
            submit({})
    with pytest.raises(RuntimeError, match="missing 'status'"):
        client.get_job_status("job")

    statuses = iter([{"status": "running"}, {"status": "cancelled", "message": "bye"}])
    monkeypatch.setattr(
        client, "get_job_status", lambda *_args, **_kwargs: next(statuses)
    )
    monkeypatch.setattr(api_client_module.time, "sleep", lambda *_args, **_kwargs: None)
    updates = []
    with pytest.raises(RuntimeError, match="bye"):
        client.wait_for_job(
            "job",
            timeout_seconds=5,
            poll_interval_seconds=0,
            on_update=lambda status, message: updates.append((status, message)),
        )
    assert updates[0] == ("running", None)

    values = iter([0.0, 10.0])
    monkeypatch.setattr(api_client_module.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(
        client, "get_job_status", lambda *_args, **_kwargs: {"status": "running"}
    )
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        client.wait_for_job("job", timeout_seconds=1, poll_interval_seconds=0)


@pytest.mark.asyncio
async def test_api_server_remaining_paths(monkeypatch, tmp_path):
    store = api_server.JobStore(tmp_path / "jobs.db")
    created = []

    async def backend_ollama(request, **kwargs):
        created.append(request.backend_url)
        return {"mode": "existing"}

    async def backend_bench(request, **kwargs):
        return {"answer": request.question}

    async def backend_agentic_agent(request, **kwargs):
        raise RuntimeError("bad-run")

    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(
            auth_disabled=False, tenant_tokens={"tenant": "token"}
        ),
        backend_functions={
            "llm-instance": backend_ollama,
            "ollama-instance": backend_ollama,
            "benchmark": backend_bench,
            "agentic-agent": backend_agentic_agent,
        },
    )
    monkeypatch.setattr(
        api_server,
        "list_backend_models",
        lambda backend_url, **kw: {
            "backend_url": backend_url,
            "backend_type": kw.get("backend_type", "ollama"),
            "models": [],
            "running_models": [],
        },
    )
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        req = Request(
            f"{base}/v1/ollama/models?backend_url=http://x",
            headers={"Authorization": "Bearer token", "X-Tenant-ID": "tenant"},
        )
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            assert response.status == 200

        bad_auth = Request(
            f"{base}/v1/jobs/whatever",
            method="POST",
            data=b"{}",
            headers={
                "Authorization": "Bearer wrong",
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(Exception):
            urlopen(bad_auth)  # nosec  # test request mock/local execution

        bad_kind = Request(
            f"{base}/v1/jobs/whatever",
            method="POST",
            data=b"{}",
            headers={
                "Authorization": f"Bearer {_COMPREHENSIVE_API_TOKEN}",
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(Exception):
            urlopen(bad_kind)  # nosec  # test request mock/local execution

        req = Request(
            f"{base}/v1/jobs/ollama-instance",
            method="POST",
            data=b'{"provisioning": {"vastai": {"template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9}}, "backend_url": "http://x"}',
            headers={
                "Authorization": "Bearer token",
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
            },
        )
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            payload = response.read().decode("utf-8")
            assert "job_id" in payload

        req = Request(
            f"{base}/v1/jobs/benchmark",
            method="POST",
            data=b'{"provisioning": {"vastai": {"template_hash": "t", "min_dph": 1, "max_dph": 2, "reliability": 0.9, "stop_instance": false}}, "backend_url": null, "question": "q", "model": "m", "backend_warmup": false, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"}',
            headers={
                "Authorization": "Bearer token",
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
                "Idempotency-Key": "id1",
            },
        )
        with urlopen(req) as response:  # nosec  # test request mock/local execution
            assert response.status == 202
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        store.close()

    jobs = [
        store.get_job(job_id)
        for job_id, _ in [
            store.create_job(
                tenant_id="tenant", kind="agentic-agent", request_payload={}
            )
        ]
    ]
    assert jobs[0] is not None

    job_id, _ = store.create_job(
        tenant_id="tenant",
        kind="agentic-agent",
        request_payload={
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        },
    )  # nosec  # test artifact paths
    await app._execute_job(
        job_id,
        app.backend_functions["agentic-agent"],
        "agentic-agent",
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        },
    )  # nosec  # test artifact paths
    assert store.get_job(job_id).status == "failed"

    assert await app._dispatch(
        "llm-instance", {"provisioning": None, "backend_url": "http://x"}
    ) == {"mode": "existing"}
    assert await app._dispatch(
        "benchmark",
        {
            "provisioning": None,
            "backend_url": None,
            "question": "qq",
            "model": "m",
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        },
    ) == {"answer": "qq"}  # nosec  # test artifact paths

    monkeypatch.setattr(
        api_server,
        "ThreadingHTTPServer",
        lambda *args, **kwargs: type(
            "S",
            (),
            {"serve_forever": lambda self: None, "server_close": lambda self: None},
        )(),
    )
    api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
    ).serve("127.0.0.1", 0)


@pytest.mark.asyncio
async def test_api_server_error_paths(monkeypatch, tmp_path):
    class ExplodingStore(api_server.JobStore):
        def create_job(self, **kwargs):
            raise RuntimeError("db-down")

    store = ExplodingStore(tmp_path / "jobs.db")
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(
            auth_disabled=False, tenant_tokens={"tenant": "token"}
        ),
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
            headers={
                "X-API-Key": _COMPREHENSIVE_API_TOKEN,
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution

        req = Request(
            f"{base}/v1/jobs/agentic-agent",
            method="POST",
            data=b"{}",
            headers={
                "X-API-Key": _COMPREHENSIVE_API_TOKEN,
                "X-Tenant-ID": "tenant",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution

        req = Request(
            f"{base}/v1/jobs/missing",
            headers={"X-API-Key": _COMPREHENSIVE_API_TOKEN, "X-Tenant-ID": "tenant"},
        )
        with pytest.raises(Exception):
            urlopen(req)  # nosec  # test request mock/local execution
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        store.close()

    real_store = api_server.JobStore(tmp_path / "ok.db")

    async def mock_agent(request, **kwargs):
        return {"answer": request.question}

    app = api_server.RuneApiApplication(
        store=real_store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
        backend_functions={"agentic-agent": mock_agent},
    )
    job_id, _ = real_store.create_job(
        tenant_id="default",
        kind="agentic-agent",
        request_payload={
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        },  # nosec  # test artifact paths
    )
    await app._execute_job(
        job_id,
        app.backend_functions["agentic-agent"],
        "agentic-agent",
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        },
    )  # nosec  # test artifact paths
    assert real_store.get_job(job_id).status == "succeeded"


@pytest.mark.asyncio
async def test_offer_template_backend_instance_and_workflow_remaining(
    monkeypatch, tmp_path
):
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

    mock_provider = AsyncMock()
    mock_provider.provision.return_value = ProvisioningResult(
        backend_url="http://x", model="m", provider_handle=8
    )

    async def mock_stop(*a, **k):
        stopped.append(True)

    mock_provider.teardown.side_effect = mock_stop

    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda req: mock_provider,
    )

    mock_agent_run = AsyncMock()
    mock_agent_run.ask_structured.return_value = AgentResult(answer="a")
    monkeypatch.setattr(
        api_backend,
        "_make_agent_runner",
        lambda path: mock_agent_run,
    )

    async def mock_cost(*a, **k):
        return 0.0

    monkeypatch.setattr(api_backend, "calculate_run_cost", mock_cost)

    vast_req = api_backend.RunBenchmarkRequest.from_cli(
        vastai=True,
        template_hash="t",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        backend_url=None,
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=1,
        kubeconfig=Path(kubeconfig),
        vastai_stop_instance=True,
    )
    result = await api_backend.run_benchmark(vast_req)
    assert result["contract_id"] == 8
    assert stopped == [True]

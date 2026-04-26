# SPDX-License-Identifier: Apache-2.0
import pytest
import json
import os
import sys
import time
from unittest.mock import MagicMock, AsyncMock, patch
from urllib.request import Request, urlopen
from pathlib import Path

import rune
import rune_bench.api_backend as api_backend
import rune_bench.api_server as api_server
from rune_bench.api_server import JobStore
from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
    RunTelemetry,
    TokenBreakdown,
)
from rune_bench.backends.base import ModelCapabilities
from rune_bench.agents.base import AgentResult


@pytest.fixture
def sqlite_store(tmp_path):
    db_path = tmp_path / "jobs.db"
    store = JobStore(db_path)
    try:
        yield store
    finally:
        store.close()


@pytest.mark.asyncio
async def test_api_server_sse_errors(sqlite_store):
    store = sqlite_store
    job_id, _ = store.create_job(
        tenant_id="t1", kind="agentic-agent", request_payload={"q": "a"}
    )
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
    )

    HandlerClass = app.create_handler()
    handler_instance = MagicMock(spec=HandlerClass)
    handler_instance.path = f"/v1/runs/{job_id}/trace"
    handler_instance.headers = {"X-Tenant-ID": "t1"}
    handler_instance.wfile = MagicMock()
    handler_instance.wfile.write.side_effect = ConnectionResetError()

    job_mock_running = MagicMock()
    job_mock_running.status = "running"
    job_mock_done = MagicMock()
    job_mock_done.status = "succeeded"

    with patch.object(
        app.store,
        "get_job",
        side_effect=[job_mock_running, job_mock_done, job_mock_running, job_mock_done],
    ):
        with patch.object(
            app.store,
            "get_events_for_job",
            return_value=[{"recorded_at": time.time(), "event": "e", "status": "ok"}],
        ):
            with patch("time.sleep"):
                with patch("time.gmtime"):
                    with patch("time.strftime"):
                        HandlerClass.do_GET(handler_instance)

        handler_instance.wfile.write.side_effect = Exception("generic")
        with patch("time.sleep"):
            HandlerClass.do_GET(handler_instance)


def test_sqlite_storage_extra(sqlite_store):
    store = sqlite_store
    # _compute_overall_chain_status branches
    assert store._compute_overall_chain_status([]) == "pending"
    assert (
        store._compute_overall_chain_status(
            [{"status": "failed"}, {"status": "success"}]
        )
        == "failed"
    )
    assert (
        store._compute_overall_chain_status(
            [{"status": "running"}, {"status": "success"}]
        )
        == "running"
    )
    assert (
        store._compute_overall_chain_status(
            [{"status": "pending"}, {"status": "success"}]
        )
        == "pending"
    )
    assert store._compute_overall_chain_status([{"status": "skipped"}]) == "skipped"

    store.record_chain_initialized(job_id="j1", nodes=[{"id": "n1"}], edges=[])
    state = store.get_chain_state("j1")
    assert state["overall_status"] == "pending"

    store.record_chain_node_transition(job_id="j1", node_id="n1", status="success")
    state = store.get_chain_state("j1")
    assert state["overall_status"] == "success"

    aid = store.record_audit_artifact(
        job_id="j1", kind="tpm_attestation", name="a1", content=b"content"
    )
    art_tuple = store.get_audit_artifact(job_id="j1", artifact_id=aid)
    assert art_tuple[0] == b"content"
    assert store.get_audit_artifact(job_id="j1", artifact_id="missing") is None
    artifacts = store.list_audit_artifacts("j1")
    assert any(a["name"] == "a1" for a in artifacts)


def test_metrics_extra():
    from rune_bench.metrics import InMemoryCollector, MetricsEvent

    collector = InMemoryCollector()
    collector.record(
        MetricsEvent(
            event="e1",
            status="ok",
            duration_ms=100.0,
            labels={},
            recorded_at=time.time(),
        )
    )
    summary = collector.summary_rows()
    assert any(r["event"] == "e1" for r in summary)
    rune._print_metrics_summary(collector)


def test_attestation_factory_extra():
    from rune_bench.attestation.factory import get_driver

    with patch("os.environ", {"RUNE_ATTESTATION_DRIVER": "tpm2"}):
        with patch("rune_bench.attestation.tpm2.TPM2Driver") as mock_tpm:
            get_driver()
            mock_tpm.assert_called_once()

    with pytest.raises(ValueError):
        get_driver({"driver": "unknown"})


@pytest.mark.asyncio
async def test_rune_helpers_extra(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    with patch("pathlib.Path.exists", return_value=False):
        assert not rune._is_containerized()

    with patch("rune._is_containerized", return_value=True):
        assert rune._resolve_serve_port() == 8080
    with patch("rune._is_containerized", return_value=False):
        with patch("rune._find_free_port", return_value=1234):
            with patch("rune.console"):
                assert rune._resolve_serve_port() == 1234

    with patch("rune.VastAI", None):
        with pytest.raises(RuntimeError, match="vastai"):
            rune._vastai_sdk()


def test_rune_serve_api_coverage():
    with patch("rune_bench.api_server.RuneApiApplication") as mock_app:
        with patch("rune._resolve_serve_port", return_value=8080):
            with patch("rune.console"):
                rune.serve_api(api_host="127.0.0.1", api_port=None)
                mock_app.from_env.return_value.serve.assert_called_with(
                    host="127.0.0.1", port=8080
                )

                mock_app.from_env.return_value.serve.side_effect = KeyboardInterrupt()
                with pytest.raises(rune.typer.Exit):
                    rune.serve_api()

                mock_app.from_env.return_value.serve.side_effect = Exception("err")
                with pytest.raises(rune.typer.Exit):
                    rune.serve_api()


@pytest.mark.asyncio
async def test_rune_vastai_provisioning_errors():
    from rune_bench.workflows import UserAbortedError

    with patch("rune._vastai_sdk"):
        with patch("rune.Progress"):
            with patch("rune.provision_vastai_backend", side_effect=UserAbortedError()):
                with patch("rune.console"):
                    with pytest.raises(rune.typer.Exit) as exc:
                        await rune._run_vastai_provisioning(
                            template_hash="t", min_dph=1, max_dph=2, reliability=0.9
                        )
                    assert exc.value.exit_code == 0

            with patch(
                "rune.provision_vastai_backend", side_effect=RuntimeError("err")
            ):
                with patch("rune.console"):
                    with pytest.raises(rune.typer.Exit):
                        await rune._run_vastai_provisioning(
                            template_hash="t", min_dph=1, max_dph=2, reliability=0.9
                        )


@pytest.mark.asyncio
async def test_rune_preflight_cost_check_extra(monkeypatch):
    monkeypatch.setattr(os, "isatty", lambda fd: True)
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)
    with patch("builtins.input", return_value="n"):
        with patch(
            "rune.run_preflight_cost_check",
            AsyncMock(return_value={"projected_cost_usd": 10.0, "threshold_usd": 5.0}),
        ):
            with patch("rune.console"):
                with pytest.raises(rune.typer.Exit):
                    await rune._run_preflight_cost_check(
                        vastai=True, max_dph=1, min_dph=0, yes=False
                    )


def test_rune_init_misc():
    caps = ModelCapabilities(model_name="m", context_window=100, max_output_tokens=50)
    with patch("rune.console"):
        with pytest.raises(rune.typer.Exit):
            rune._print_error_and_exit("test error")

        server = MagicMock()
        server.url = "http://u"
        server.model_name = "m"
        rune._print_existing_ollama(server)
        rune._print_existing_ollama(server, capabilities=caps)

        vast_res = MagicMock()
        vast_res.offer_id = 1
        vast_res.total_vram_mb = 1000
        vast_res.model_name = "m"
        vast_res.model_vram_mb = 500
        vast_res.required_disk_gb = 10
        vast_res.template_env = "env"
        vast_res.reused_existing_instance = False
        vast_res.contract_id = 123
        vast_res.details.contract_id = 123
        vast_res.details.status = "running"
        vast_res.details.ssh_host = "h"
        vast_res.details.ssh_port = 22
        vast_res.details.service_urls = [{"name": "n", "direct": "d", "proxy": "p"}]
        vast_res.backend_url = "http://v"
        vast_res.pull_warning = None
        rune._print_vastai_result(vast_res)
        rune._print_vastai_result(vast_res, capabilities=caps)

        with patch("rune.ModelSelector") as mock_selector:
            mock_model = MagicMock()
            mock_model.name = "m1"
            mock_model.vram_mb = 100
            mock_model.required_disk_gb = 10
            mock_selector.return_value.list_models.return_value = [mock_model]
            rune._print_vastai_models()

        rune._print_ollama_models(
            backend_url="http://u", models=["m1"], running_models={"m1"}
        )

        with patch("rune.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.get_model_capabilities.return_value = caps
            mock_backend.normalize_model_name.side_effect = lambda x: x
            mock_get_backend.return_value = mock_backend
            assert rune._fetch_model_capabilities("http://u", "m") == caps


@pytest.mark.asyncio
async def test_api_server_job_submission(sqlite_store):
    store = sqlite_store
    backend_functions = {
        "benchmark": AsyncMock(return_value={"answer": "ok"}),
        "ollama-instance": AsyncMock(return_value={"mode": "existing"}),
        "llm-instance": AsyncMock(return_value={"mode": "existing"}),
    }
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
        backend_functions=backend_functions,
    )
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    common_headers = {"X-Tenant-ID": "default"}
    try:
        req_data = {
            "provisioning": None,
            "backend_url": "http://x",
            "question": "q",
            "model": "m",
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/k",
        }
        with urlopen(
            Request(
                f"{base}/v1/jobs/benchmark",
                method="POST",
                data=json.dumps(req_data).encode(),
                headers=common_headers,
            )
        ) as resp:
            assert resp.status == 202

        req_llm = {"provisioning": None, "backend_url": "http://x"}
        with urlopen(
            Request(
                f"{base}/v1/jobs/ollama-instance",
                method="POST",
                data=json.dumps(req_llm).encode(),
                headers=common_headers,
            )
        ) as resp:
            assert resp.status == 202

        mock_raw = {"profiles": {"p1": {"api_token": "secret"}}}
        with patch("rune_bench.api_server.get_raw_config", return_value=mock_raw):
            with urlopen(
                Request(f"{base}/v1/settings", headers=common_headers)
            ) as resp:
                data = json.loads(resp.read())
                assert data["profiles"]["p1"]["api_token"] == "[REDACTED]"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        store.close()


@pytest.mark.asyncio
async def test_api_server_extra_branches(sqlite_store):
    _ = api_server.RuneApiApplication(
        store=sqlite_store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
    )
    req_b = RunBenchmarkRequest(
        provisioning=None,
        backend_url="u",
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=0,
        kubeconfig="/tmp/k",
    )
    req_a = RunAgenticAgentRequest(
        question="q",
        model="m",
        backend_url="u",
        backend_warmup=False,
        backend_warmup_timeout=0,
    )

    with pytest.raises(RuntimeError):
        await api_server._run_agentic_backend(req_b)
    with pytest.raises(RuntimeError):
        await api_server._run_benchmark_backend(req_a)
    with pytest.raises(RuntimeError):
        await api_server._run_llm_instance_backend(req_b)
    with pytest.raises(RuntimeError):
        await api_server._get_cost_estimate_backend(req_b)


@pytest.mark.asyncio
async def test_api_backend_remaining_branches():
    from rune_bench.metrics.cost import calculate_run_cost

    assert await calculate_run_cost("unknown", "m", 10) >= 0
    req = RunAgenticAgentRequest(
        question="q",
        model="m",
        backend_url="http://x",
        backend_warmup=False,
        backend_warmup_timeout=0,
        kubeconfig="/tmp/k",
    )
    mock_runner = AsyncMock()
    mock_runner.ask_structured.return_value = AgentResult(
        answer="ans",
        telemetry=RunTelemetry(tokens=TokenBreakdown(total=100)),
        result_type="success",
    )
    with patch("rune_bench.api_backend.get_agent", return_value=mock_runner):
        res = await api_backend.run_agentic_agent(req)
        assert res["metadata"]["cost"] >= 0

    with patch("rune_bench.api_backend.VastAI", None):
        with pytest.raises(RuntimeError, match="vastai"):
            api_backend._vastai_sdk()

    from rune_bench.api_contracts import Provisioning, VastAIProvisioning

    req_vast = RunBenchmarkRequest(
        provisioning=Provisioning(
            vastai=VastAIProvisioning(
                template_hash="t",
                min_dph=1,
                max_dph=2,
                reliability=0.9,
                stop_instance=True,
            )
        ),
        backend_url=None,
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=0,
        kubeconfig="",
    )
    with patch("rune_bench.api_backend._vastai_sdk"):
        provider = api_backend._make_resource_provider_for_benchmark(req_vast)
        assert provider.__class__.__name__ == "VastAIProvider"
        req_llm_vast = RunLLMInstanceRequest(
            provisioning=Provisioning(
                vastai=VastAIProvisioning(
                    template_hash="t", min_dph=1, max_dph=2, reliability=0.9
                )
            ),
            backend_url=None,
        )
        provider_llm = api_backend._make_resource_provider_for_ollama_instance(
            req_llm_vast
        )
        assert provider_llm.__class__.__name__ == "VastAIProvider"


@pytest.mark.asyncio
async def test_rune_init_extra_coverage(monkeypatch):
    monkeypatch.delenv("RUNE_API_BASE_URL", raising=False)
    assert rune._http_client().base_url.startswith("http")
    await rune._run_preflight_cost_check(vastai=False, max_dph=1, min_dph=0, yes=True)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)
    with patch(
        "rune.run_preflight_cost_check",
        AsyncMock(return_value={"projected_cost_usd": 100.0}),
    ):
        with pytest.raises(rune.typer.Exit):
            await rune._run_preflight_cost_check(
                vastai=True, max_dph=1, min_dph=0, yes=False
            )


@pytest.mark.asyncio
async def test_rune_commands_full_coverage(monkeypatch):
    with patch("rune.console"):
        with patch("rune._run_preflight_cost_check", AsyncMock()):
            with patch("rune._vastai_sdk") as mock_sdk_factory:
                mock_sdk = mock_sdk_factory.return_value
                # 1. find_reusable_running_instance call 1
                # 2. find_reusable_running_instance call 2 (after SDK refresh)
                # 3. wait_until_running poll 1
                running_inst = [
                    {
                        "id": 1,
                        "actual_status": "running",
                        "state": "running",
                        "ssh_host": "h",
                        "ssh_port": 22,
                        "service_urls": [],
                        "gpu_total_ram": 24,
                        "dph_total": 0.5,
                        "ports": {
                            "11434/tcp": [{"HostIp": "1.2.3.4", "HostPort": "11434"}]
                        },
                    }
                ]

                mock_sdk.show_instances.side_effect = [
                    [],
                    [],
                    running_inst,
                    running_inst,
                    running_inst,
                ]

                mock_sdk.search_offers.return_value = [
                    {
                        "id": 1,
                        "dph_total": 0.5,
                        "gpu_name": "RTX 4090",
                        "gpu_total_ram": 24000,
                        "reliability2": 0.99,
                        "num_gpus": 1,
                    }
                ]
                mock_sdk.create_instance.return_value = {
                    "success": True,
                    "new_contract": 1,
                }
                mock_sdk.show_templates.return_value = [
                    {"hash": "tpl-abc", "name": "test-template"}
                ]

                monkeypatch.setattr(rune, "DEFAULT_VASTAI_TEMPLATE", "tpl-abc")
                monkeypatch.setattr(rune, "BACKEND_MODE", "local")
                monkeypatch.setattr(
                    rune,
                    "use_existing_backend_server",
                    lambda *a, **k: MagicMock(url="u", model_name="m"),
                )
                monkeypatch.setattr(
                    rune, "_warmup_ollama_model", MagicMock(return_value=None)
                )
                monkeypatch.setattr(
                    rune,
                    "list_backend_models",
                    lambda *a, **k: {"models": [], "running_models": []},
                )
                monkeypatch.setattr(
                    rune, "list_running_backend_models", lambda *a, **k: []
                )
                monkeypatch.setattr(
                    rune,
                    "get_agent",
                    lambda *a, **k: AsyncMock(
                        ask_structured=AsyncMock(return_value=AgentResult(answer="a"))
                    ),
                )

                with patch(
                    "rune_bench.metrics.cost.calculate_run_cost",
                    AsyncMock(return_value=0.1),
                ):
                    with patch("time.sleep"):
                        await rune.run_agentic_agent(
                            debug=False, question="q", model="m"
                        )
                        await rune.run_benchmark(
                            debug=False,
                            question="q",
                            model="m",
                            kubeconfig=Path("/tmp/k"),
                            vastai=True,
                            min_dph=0.1,
                            max_dph=1.0,
                            reliability=0.9,
                            template_hash="tpl-abc",
                            yes=True,
                        )

                        monkeypatch.setattr(rune, "BACKEND_MODE", "http")
                        mock_client = MagicMock()
                        monkeypatch.setattr(rune, "_http_client", lambda: mock_client)
                        monkeypatch.setattr(
                            rune,
                            "_run_http_job_with_progress",
                            AsyncMock(
                                return_value={
                                    "result": {
                                        "answer": "a",
                                        "backend_url": "http://x",
                                        "mode": "existing",
                                    }
                                }
                            ),
                        )

                        await rune.run_agentic_agent(
                            debug=False, question="q", model="m"
                        )
                        await rune.run_benchmark(
                            debug=False,
                            question="q",
                            model="m",
                            kubeconfig=Path("/tmp/k"),
                            vastai=False,
                        )
                        await rune.run_llm_instance(debug=False)

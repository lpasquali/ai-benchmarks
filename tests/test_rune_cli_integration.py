# SPDX-License-Identifier: Apache-2.0
import pytest
import typer
from rich.console import Console

import rune
from rune_bench.workflows import ExistingOllamaServer, VastAIProvisioningResult, ConnectionDetails
from rune_bench.agents.base import AgentResult

def _result():
    details = ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host="h",
        ssh_port=22,
        machine_id="m",
        service_urls=[{"name": "ollama", "direct": "http://x", "proxy": None}]
    )
    return VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=24000,
        model_name="llama3.1:8b",
        model_vram_mb=8000,
        required_disk_gb=40,
        template_env="ENV=1",
        contract_id=1,
        details=details,
        backend_url="http://x",
    )

@pytest.mark.asyncio
async def test_run_http_job_with_progress_logic(monkeypatch):
    client = rune.RuneApiClient("http://api:8080")
    
    responses = [
        {"status": "queued"},
        {"status": "running"},
        {"status": "succeeded", "result": {"answer": "ok"}}
    ]
    # RuneApiClient.wait_for_job is sync
    def mock_wait(*a, **k):
        return {"status": "succeeded", "result": {"answer": "ok"}}
    
    monkeypatch.setattr(client, "wait_for_job", mock_wait)
    
    payload = await rune._run_http_job_with_progress(
        submit_description="sub",
        wait_description="wait",
        submit_job=lambda: "job-1",
        client=client,
        timeout_seconds=1,
        poll_interval_seconds=0,
    )
    assert payload["result"]["answer"] == "ok"

@pytest.mark.asyncio
async def test_run_llm_instance_paths(monkeypatch):
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_backend_server", lambda *_args, **_kwargs: ExistingOllamaServer(url="http://x", model_name="m"))
    
    await rune.run_llm_instance(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url="u", debug=False, idempotency_key=None)
    assert "Existing Ollama Server" in test_console.export_text()

    monkeypatch.setattr(rune, "use_existing_backend_server", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        await rune.run_llm_instance(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url="u", debug=False, idempotency_key=None)

    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_kwargs: _result())
    await rune.run_llm_instance(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url=None, debug=False, idempotency_key=None)
    assert "Provisioned contract" in test_console.export_text()

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    
    # client.wait_for_job mock (sync)
    def mock_wait(*a, **k):
        return {"result": {"mode": "vastai", "contract_id": 2, "backend_url": "http://x", "model_name": "m"}}
    
    fake_client = type("C", (), {
        "submit_ollama_instance_job": lambda self, payload, idempotency_key=None: "job-1",
        "wait_for_job": mock_wait,
        "get_cost_estimate": lambda self, *_a, **_k: {"projected_cost_usd": 1.0, "cost_driver": "vastai", "resource_impact": "medium", "local_energy_kwh": 0.0, "confidence_score": 1.0, "warning": None}
    })()
    monkeypatch.setattr(rune, "_http_client", lambda: fake_client)
    
    await rune.run_llm_instance(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url=None, debug=False, idempotency_key="id1")
    assert "Selected model" in test_console.export_text()

@pytest.mark.asyncio
async def test_run_agentic_agent_paths(monkeypatch, tmp_path):
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    kubeconfig = tmp_path / "kube"
    kubeconfig.write_text("apiVersion: v1\n")

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    
    def mock_wait(*a, **k):
        return {"result": {"answer": "http-answer"}}
    
    fake_client = type("C", (), {
        "submit_agentic_agent_job": lambda self, payload, idempotency_key=None: "job-1",
        "wait_for_job": mock_wait,
    })()
    monkeypatch.setattr(rune, "_http_client", lambda: fake_client)

    await rune.run_agentic_agent(debug=False, question="q", model="m", backend_url=None, backend_warmup=False, backend_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key="id1")
    assert "http-answer" in test_console.export_text()

    # Local mode
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_backend_server", lambda *_a, **_k: ExistingOllamaServer(url="http://x", model_name="m"))
    
    from unittest.mock import AsyncMock
    mock_runner = AsyncMock()
    mock_runner.ask_structured.return_value = AgentResult(answer="local-answer")
    monkeypatch.setattr(rune, "get_agent", lambda *a, **k: mock_runner)
    
    await rune.run_agentic_agent(debug=False, question="q", model="m", backend_url="http://x", backend_warmup=False, backend_warmup_timeout=1, kubeconfig=kubeconfig)
    assert "local-answer" in test_console.export_text()

@pytest.mark.asyncio
async def test_run_benchmark_paths(monkeypatch, tmp_path):
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    kubeconfig = tmp_path / "kube"
    kubeconfig.write_text("apiVersion: v1\n")

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    
    def mock_wait(*a, **k):
        return {"result": {"answer": "bench-http"}}
    
    fake_client = type("C", (), {
        "submit_benchmark_job": lambda self, payload, idempotency_key=None: "job-1",
        "wait_for_job": mock_wait,
    })()
    monkeypatch.setattr(rune, "_http_client", lambda: fake_client)

    await rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url=None, question="q", model="m", backend_warmup=False, backend_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key="i")
    assert "bench-http" in test_console.export_text()

    # Local mode
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_backend_server", lambda *_a, **_k: ExistingOllamaServer(url="http://x", model_name="m"))
    
    from unittest.mock import AsyncMock
    mock_runner = AsyncMock()
    mock_runner.ask_structured.return_value = AgentResult(answer="bench-local")
    monkeypatch.setattr(rune, "get_agent", lambda *a, **k: mock_runner)
    
    # Mock calculate_run_cost
    async def mock_cost(*a, **k): return 0.01
    monkeypatch.setattr(rune, "calculate_run_cost", mock_cost)

    await rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, backend_url="http://x", question="q", model="m", backend_warmup=False, backend_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True)
    assert "bench-local" in test_console.export_text()

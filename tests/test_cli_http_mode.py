# SPDX-License-Identifier: Apache-2.0
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

import rune
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore
from rune_bench.workflows import SpendGateAction


@pytest.fixture
def rune_api_server(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    state = {"agentic_calls": 0, "benchmark_calls": 0}

    async def run_agentic(request):
        state["agentic_calls"] += 1
        return {"answer": "agent-http-answer", "result_type": "text", "artifacts": []}

    async def run_benchmark(request):
        state["benchmark_calls"] += 1
        return {"answer": "benchmark-http-answer", "result_type": "text", "artifacts": [], "mode": "existing", "backend_url": "http://x"}

    app = RuneApiApplication(
        store=store,
        security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
        backend_functions={
            "agentic-agent": run_agentic,
            "benchmark": run_benchmark,
        },
    )

    # Use a basic handler that delegates to the app but with better logging
    handler_class = app.create_handler()

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    
    # Wait for server to be ready
    for _ in range(10):
        try:
            import socket
            with socket.create_connection((host, port), timeout=0.1):
                break
        except:
            time.sleep(0.1)
    
    try:
        yield base_url, state
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.mark.asyncio
async def test_cli_http_run_agentic_agent_job_flow(monkeypatch, rune_api_server, tmp_path):
    base_url, state = rune_api_server
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setenv("RUNE_API_BASE_URL", base_url)

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    # Mock preflight behavior to avoid any issues
    monkeypatch.setattr(rune, "_run_preflight_cost_check", AsyncMock(return_value=None))
    monkeypatch.setattr(rune, "evaluate_spend_gate", lambda *a, **k: SpendGateAction.ALLOW)
    
    # Typer commands might raise Exit(0) on success if they return, but here they just return None
    await rune.run_agentic_agent(
        debug=True,
        question="test-question",
        model="llama3.1:8b",
        backend_url="http://ollama:11434",
        backend_warmup=False,
        backend_warmup_timeout=1,
        kubeconfig=kubeconfig,
        idempotency_key=None,
    )

    output = test_console.export_text()
    assert "agent-http-answer" in output
    assert state["agentic_calls"] == 1


@pytest.mark.asyncio
async def test_cli_http_run_benchmark_job_flow(monkeypatch, rune_api_server, tmp_path):
    base_url, state = rune_api_server
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setenv("RUNE_API_BASE_URL", base_url)

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    monkeypatch.setattr(rune, "_run_preflight_cost_check", AsyncMock(return_value=None))
    monkeypatch.setattr(rune, "evaluate_spend_gate", lambda *a, **k: SpendGateAction.ALLOW)

    await rune.run_benchmark(
        debug=True,
        vastai=False,
        template_hash="t",
        min_dph=0.1,
        max_dph=1.5,
        reliability=0.9,
        backend_url="http://ollama:11434",
        question="bench-question",
        model="llama3.1:8b",
        backend_warmup=False,
        backend_warmup_timeout=1,
        kubeconfig=kubeconfig,
        vastai_stop_instance=True,
        idempotency_key=None,
    )

    output = test_console.export_text()
    assert "benchmark-http-answer" in output
    assert state["benchmark_calls"] == 1

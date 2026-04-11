# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
import hashlib
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch
from rune_bench.metrics.cost import calculate_run_cost
from rune_bench.metrics.pricing import PricingSoothSayer, PricingProjection
from rune_bench.api_contracts import TokenBreakdown, RunTelemetry, LatencyPhase, RunAgenticAgentRequest
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore
from rune_bench.drivers.holmes import HolmesDriverClient
from rune_bench.api_backend import run_agentic_agent

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
            tenant_tokens={
                "tenant-a": hashlib.sha256(b"token-a").hexdigest(), 
            }
        ),
        backend_functions={
            "agentic-agent": run_agentic,
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

@pytest.mark.asyncio
async def test_calculate_run_cost_fallback():
    mock_estimator = MagicMock()
    # Mock return value should be an awaitable if we want to force Exception during await
    # but calculate_run_cost catches exception from await estimator.estimate(...)
    mock_estimator.estimate.side_effect = Exception("mock error")
    cost = await calculate_run_cost("vastai", "model", 3600, estimator=mock_estimator)
    assert cost == 2.50

@pytest.mark.asyncio
async def test_pricing_sooth_sayer_simulate_basic():
    soothsayer = PricingSoothSayer()
    projection = await soothsayer.simulate("holmes", "gpt-4o")
    
    assert isinstance(projection, PricingProjection)
    assert projection.total_cost_usd > 0
    assert projection.historical_match is True
    assert projection.token_cost_usd > 0

@pytest.mark.asyncio
async def test_pricing_sooth_sayer_unknown_agent():
    soothsayer = PricingSoothSayer()
    projection = await soothsayer.simulate("unknown-agent", "llama3.1:8b")
    
    assert projection.historical_match is False
    assert projection.confidence == 0.4

@pytest.mark.asyncio
async def test_pricing_sooth_sayer_live_dph_fallback():
    soothsayer = PricingSoothSayer()
    dph = await soothsayer.get_live_dph("rtx4090")
    assert dph == 0.40
    
    dph = await soothsayer.get_live_dph("h100")
    assert dph == 3.50

def test_telemetry_to_dict():
    tokens = TokenBreakdown(system_prompt=10, total=10)
    latency = [LatencyPhase(phase="init", ms=100)]
    telemetry = RunTelemetry(tokens=tokens, latency=latency, cost_estimate_usd=0.05)
    
    d = telemetry.to_dict()
    assert d["tokens"]["system_prompt"] == 10
    assert d["latency"][0]["phase"] == "init"
    assert d["cost_estimate_usd"] == 0.05

@pytest.mark.asyncio
async def test_calculate_run_cost_vastai():
    from rune_bench.api_contracts import CostEstimationResponse
    mock_resp = CostEstimationResponse(
        projected_cost_usd=0.1234,
        cost_driver="vastai",
        resource_impact="high",
        confidence_score=1.0
    )
    
    mock_estimator = MagicMock()
    f = asyncio.Future()
    f.set_result(mock_resp)
    mock_estimator.estimate.return_value = f
    
    cost = await calculate_run_cost("vastai", "llama3.1:8b", 600, estimator=mock_estimator)
    assert cost == 0.1234

@pytest.mark.asyncio
async def test_pricing_sooth_sayer_with_sdk():
    mock_sdk = MagicMock()
    mock_sdk.search_offers.return_value = [{"dph_total": 0.25}]
    
    soothsayer = PricingSoothSayer(sdk=mock_sdk)
    dph = await soothsayer.get_live_dph("rtx4090")
    assert dph == 0.25
    mock_sdk.search_offers.assert_called_once()

@pytest.mark.asyncio
async def test_api_finops_simulate(rune_api_server):
    base_url, _ = rune_api_server
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{base_url}/v1/finops/simulate",
            params={"agent": "holmes", "model": "gpt-4o"},
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_cost_usd" in data
        assert data["historical_match"] is True

@pytest.mark.asyncio
async def test_api_run_trace_sse(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    import httpx
    
    # Create a dummy job
    job_id, _ = store.create_job(tenant_id="tenant-a", kind="agentic-agent", request_payload={})
    
    # Add an event
    from rune_bench.metrics import MetricsEvent
    store.record_workflow_event(MetricsEvent(
        job_id=job_id, event="test.event", status="ok", duration_ms=10.0, labels={}, recorded_at=time.time()
    ))
    
    # Mark job as succeeded to terminate the stream quickly
    store.update_job(job_id, status="succeeded")
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"{base_url}/v1/runs/{job_id}/trace",
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"}
        ) as response:
            assert response.status_code == 200
            
            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
                if "event: end" in line:
                    break
            
            assert any("event: log" in l for l in lines)
            assert any("test.event" in l for l in lines)
            assert any("event: end" in l for l in lines)

def test_holmes_parse_telemetry(tmp_path):
    k = tmp_path / "kubeconfig"
    k.write_text("dummy")
    
    client = HolmesDriverClient(kubeconfig=k)
    raw = {
        "tokens": {"total": 650},
        "latency": [{"phase": "planning", "ms": 1500}],
        "cost_estimate_usd": 0.015
    }
    
    tel = client._parse_telemetry(raw)
    assert tel.tokens.total == 650
    assert tel.cost_estimate_usd == 0.015

@pytest.mark.asyncio
async def test_run_agentic_agent_telemetry():
    from rune_bench.agents.base import AgentResult
    
    from unittest.mock import AsyncMock
    
    mock_runner = MagicMock()
    mock_runner.ask_structured = AsyncMock(return_value=AgentResult(
        answer="ok",
        telemetry=RunTelemetry(tokens=TokenBreakdown(total=100))
    ))
    
    with patch("rune_bench.api_backend.get_agent", return_value=mock_runner), \
         patch("rune_bench.api_backend.calculate_run_cost", new_callable=AsyncMock) as mock_calc:
        
        mock_calc.return_value = 0.05
        
        req = RunAgenticAgentRequest(
            agent="holmes",
            question="test",
            model="llama",
            backend_url=None,
            backend_warmup=False,
            backend_warmup_timeout=30,
            kubeconfig="/tmp/dummy"
        )
        
        res = await run_agentic_agent(req)
        assert res["telemetry"]["cost_estimate_usd"] == 0.05
        assert res["metadata"]["cost"] == 0.05

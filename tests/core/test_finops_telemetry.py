# SPDX-License-Identifier: Apache-2.0
import pytest
import time
from unittest.mock import MagicMock
from rune_bench.metrics.pricing import PricingSoothSayer


@pytest.fixture
def rune_api_server(tmp_path):
    from rune_bench.api_server import (
        RuneApiApplication,
        SQLiteStorageAdapter,
        ApiSecurityConfig,
        ThreadingHTTPServer,
    )
    import threading

    tmp_db = tmp_path / "jobs.db"
    store = SQLiteStorageAdapter(tmp_db)
    state = {"agentic_calls": 0, "store": store}

    def run_agentic(request):
        state["agentic_calls"] += 1
        return {
            "answer": f"ok:{request.question}",
            "telemetry": {"tokens": {"total": 100}},
        }

    app = RuneApiApplication(
        store=store,
        security=ApiSecurityConfig(
            auth_disabled=False, tenant_tokens={"tenant-a": "token-a"}
        ),
        backend_functions={
            "agentic-agent": run_agentic,
            "benchmark": lambda request: {"answer": "bench"},
            "llm-instance": lambda request: {"mode": "existing"},
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


@pytest.mark.asyncio
async def test_pricing_sooth_sayer_simulate_basic():
    soothsayer = PricingSoothSayer()
    projection = await soothsayer.simulate(agent="holmes", model="gpt-4o")

    assert isinstance(projection, dict)
    assert projection["total_cost_usd"] > 0
    assert projection["historical_match"] is False
    assert projection["token_cost_usd"] > 0


@pytest.mark.asyncio
async def test_pricing_sooth_sayer_unknown_agent():
    soothsayer = PricingSoothSayer()
    projection = await soothsayer.simulate(agent="unknown-agent", model="llama3.1:8b")

    assert projection["historical_match"] is False
    assert projection["confidence"] == "low"


@pytest.mark.asyncio
async def test_pricing_sooth_sayer_live_dph_fallback():
    soothsayer = PricingSoothSayer()
    dph = await soothsayer.get_live_dph("rtx4090")
    assert dph == 0.40


@pytest.mark.asyncio
async def test_pricing_sooth_sayer_with_sdk():
    mock_sdk = MagicMock()
    # Match substring check: 'rtx4090' in 'rtx4090'
    mock_sdk.search_offers.return_value = [{"gpu_name": "rtx4090", "dph_total": 0.25}]

    soothsayer = PricingSoothSayer(vast_search_offers=mock_sdk.search_offers)
    dph = await soothsayer.get_live_dph("rtx4090")
    assert dph == 0.25
    mock_sdk.search_offers.assert_called_once()


@pytest.mark.asyncio
async def test_api_finops_simulate(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]

    # Create a historical job to ensure historical_match is True
    job_id, _ = store.create_job(
        tenant_id="tenant-a",
        kind="agentic-agent",
        request_payload={"agent": "holmes", "model": "gpt-4o"},
    )
    store.update_job(
        job_id,
        status="succeeded",
        result_payload={"eval_count": 100, "prompt_eval_count": 50},
    )

    import httpx

    async with httpx.AsyncClient() as client:
        # Test GET
        response = await client.get(
            f"{base_url}/v1/finops/simulate",
            params={"agent": "holmes", "model": "gpt-4o"},
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_cost_usd" in data
        assert data["historical_match"] is True

        # Test POST (Webhook use case)
        response_post = await client.post(
            f"{base_url}/v1/finops/simulate",
            params={"agent": "holmes", "model": "gpt-4o"},
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"},
        )
        assert response_post.status_code == 200
        data_post = response_post.json()
        assert "cost_high_usd" in data_post
        assert "projected_cost_usd" in data_post
        assert "confidence_score" in data_post


@pytest.mark.asyncio
async def test_api_run_trace_sse(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    import httpx

    # Create a dummy job
    job_id, _ = store.create_job(
        tenant_id="tenant-a", kind="agentic-agent", request_payload={}
    )

    # Add an event
    from rune_bench.metrics import MetricsEvent

    store.record_workflow_event(
        MetricsEvent(
            job_id=job_id,
            event="test.event",
            status="ok",
            duration_ms=10.0,
            labels={},
            recorded_at=time.time(),
        )
    )

    # Mark job as succeeded to terminate the stream quickly
    store.update_job(job_id, status="succeeded")

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"{base_url}/v1/runs/{job_id}/trace",
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"},
        ) as response:
            assert response.status_code == 200

            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
                if "event: end" in line:
                    break

            assert any("event: log" in line for line in lines)
            assert any("data: {" in line for line in lines)

@pytest.mark.asyncio
async def test_api_estimates_webhook_pattern(rune_api_server):
    base_url, state = rune_api_server
    import httpx

    async with httpx.AsyncClient() as client:
        # Webhook consumption pattern: /v1/estimates
        response = await client.post(
            f"{base_url}/v1/estimates",
            json={
                "aws": True,
                "model": "g4dn.xlarge",
                "estimated_duration_seconds": 3600
            },
            headers={"X-Tenant-ID": "tenant-a", "Authorization": "Bearer token-a"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence_score" in data
        assert data["cost_driver"] == "aws"
        assert data["projected_cost_usd"] > 0

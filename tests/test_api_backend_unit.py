# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.api_backend import get_cost_estimate, run_agentic_agent, run_benchmark, _vastai_sdk, _make_agent_runner, run_llm_instance, list_backend_models
from rune_bench.api_contracts import CostEstimationRequest, RunAgenticAgentRequest, RunBenchmarkRequest, Provisioning, RunLLMInstanceRequest, VastAIProvisioning

def test_get_cost_estimate_cloud():
    req = CostEstimationRequest(estimated_duration_seconds=3600, aws=True, min_dph=1.0, max_dph=2.0)
    res = get_cost_estimate(req)
    assert res["projected_cost_usd"] == 1.5
    
    req = CostEstimationRequest(estimated_duration_seconds=3600, gcp=True, min_dph=1.0, max_dph=2.0)
    res = get_cost_estimate(req)
    assert res["projected_cost_usd"] == 1.5
    
    req = CostEstimationRequest(estimated_duration_seconds=3600, azure=True, min_dph=1.0, max_dph=2.0)
    res = get_cost_estimate(req)
    assert res["projected_cost_usd"] == 1.5

def test_get_cost_estimate_local_hardware():
    req = CostEstimationRequest(
        estimated_duration_seconds=3600,
        local_hardware=True,
        local_tdp_watts=1000,
        local_energy_rate_kwh=0.1,
        local_hardware_purchase_price=8760,
        local_hardware_lifespan_years=1
    )
    res = get_cost_estimate(req)
    assert res["projected_cost_usd"] == 1.1

def test_get_cost_estimate_unknown():
    req = CostEstimationRequest(estimated_duration_seconds=3600)
    res = get_cost_estimate(req)
    assert res["projected_cost_usd"] == 0.0

def test_vastai_sdk_error():
    with patch("rune_bench.api_backend.VastAI", None):
        with pytest.raises(RuntimeError, match="vastai' package is required"):
            _vastai_sdk()

def test_make_agent_runner_legacy():
    with patch("rune_bench.api_backend.get_agent") as mock_get:
        _make_agent_runner(Path("/tmp/kubeconfig"))
        # Should be called with agent_name="holmes" and kubeconfig=Path(...)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "holmes"
        assert kwargs["kubeconfig"] == Path("/tmp/kubeconfig")

def test_list_backend_models_unit():
    with patch("rune_bench.backends.get_backend") as mock_get:
        mock_backend = MagicMock()
        mock_get.return_value = mock_backend
        mock_backend.base_url = "http://u"
        mock_backend.list_models.return_value = []
        mock_backend.list_running_models.return_value = []
        res = list_backend_models("http://u")
        assert res["backend_url"] == "http://u"

@pytest.mark.asyncio
async def test_run_llm_instance_vastai_mode():
    req = RunLLMInstanceRequest(
        provisioning=Provisioning(vastai=VastAIProvisioning(template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9)),
        backend_url=None
    )
    with patch("rune_bench.api_backend._make_resource_provider_for_ollama_instance") as mock_factory:
        mock_prov = MagicMock()
        mock_factory.return_value = mock_prov
        mock_prov.provision = AsyncMock(return_value=MagicMock(backend_url="http://u", model="m1", provider_handle="c1"))
        res = await run_llm_instance(req)
        assert res["mode"] == "vastai"
        assert res["contract_id"] == "c1"

@pytest.mark.asyncio
async def test_run_agentic_agent_no_kubeconfig():
    req = RunAgenticAgentRequest(
        question="q", model="m", backend_url="u", 
        backend_warmup=False, backend_warmup_timeout=10, 
        kubeconfig=None, agent="holmes"
    )
    with pytest.raises(RuntimeError, match="requires a kubeconfig path"):
        await run_agentic_agent(req)

@pytest.mark.asyncio
async def test_run_agentic_agent_error():
    req = RunAgenticAgentRequest(
        question="q", model="m", backend_url="u", 
        backend_warmup=False, backend_warmup_timeout=10, 
        kubeconfig="k", agent="holmes"
    )
    with patch("rune_bench.api_backend.get_agent", side_effect=RuntimeError("agent fail")):
        with pytest.raises(RuntimeError, match="Agent error"):
            await run_agentic_agent(req)

@pytest.mark.asyncio
async def test_run_benchmark_attestation():
    req = RunBenchmarkRequest(
        question="q", model="m", backend_url="u",
        backend_warmup=False, backend_warmup_timeout=10,
        kubeconfig="k", attestation_required=True,
        provisioning=Provisioning()
    )
    with patch("rune_bench.api_backend._verify_attestation") as mock_verify, \
         patch("rune_bench.api_backend._make_resource_provider_for_benchmark") as mock_prov_factory:
        mock_prov = MagicMock()
        mock_prov_factory.return_value = mock_prov
        mock_prov.provision = AsyncMock(return_value=MagicMock(backend_url="http://u"))
        mock_prov.teardown = AsyncMock()
        
        mock_agent = MagicMock()
        mock_agent.ask_structured = AsyncMock()
        with patch("rune_bench.api_backend._make_agent_runner", return_value=mock_agent), \
             patch("rune_bench.api_backend.calculate_run_cost", return_value=0.1):
            await run_benchmark(req)
            mock_verify.assert_called_once_with("k")

@pytest.mark.asyncio
async def test_run_benchmark_no_url_error():
    req = RunBenchmarkRequest(
        question="q", model="m", backend_url=None,
        backend_warmup=False, backend_warmup_timeout=10,
        kubeconfig="k", provisioning=Provisioning()
    )
    with patch("rune_bench.api_backend._make_resource_provider_for_benchmark") as mock_factory:
        mock_prov = MagicMock()
        mock_factory.return_value = mock_prov
        mock_prov.provision = AsyncMock(return_value=MagicMock(backend_url=None))
        with pytest.raises(RuntimeError, match="Could not determine Ollama URL"):
            await run_benchmark(req)

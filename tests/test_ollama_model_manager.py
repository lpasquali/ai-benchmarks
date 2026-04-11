# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from rune_bench.backends.ollama import OllamaModelManager
from rune_bench.resources.base import ProvisioningResult
from rune_bench.agents.base import AgentResult

def test_ollama_model_manager_normalize_model_name():
    manager = OllamaModelManager.create("http://localhost:11434")
    assert manager.normalize_model_name("llama3.1:8b") == "llama3.1:8b"
    assert manager.normalize_model_name("llama3.1") == "llama3.1"

@pytest.mark.asyncio
async def test_api_backend_functions(tmp_path, monkeypatch):
    from rune_bench import api_backend
    
    async def mock_cost(*a, **k):
        return 0.0
    monkeypatch.setattr(api_backend, "calculate_run_cost", mock_cost)

    monkeypatch.setattr(
        api_backend,
        "list_backend_models",
        lambda _u, _t: ["m1"],
    )
    assert api_backend.list_backend_models("u", "ollama") == ["m1"]

    monkeypatch.setattr(
        api_backend,
        "list_vastai_models",
        lambda: ["v1"],
    )
    assert api_backend.list_vastai_models() == ["v1"]

    monkeypatch.setattr(
        api_backend,
        "get_cost_estimate",
        lambda _r: {"cost": 1.0},
    )
    assert api_backend.get_cost_estimate(None)["cost"] == 1.0

    monkeypatch.setattr(
        api_backend,
        "warmup_backend_model",
        lambda *a, **k: None,
    )
    
    monkeypatch.setattr(
        api_backend,
        "run_llm_instance",
        AsyncMock(return_value={"mode": "existing", "backend_url": "norm:u"})
    )
    req = api_backend.RunLLMInstanceRequest(provisioning=None, backend_url="u")
    assert await api_backend.run_llm_instance(req) == {"mode": "existing", "backend_url": "norm:u"}

    mock_prov_ollama = AsyncMock()
    mock_prov_ollama.provision.return_value = ProvisioningResult(backend_url="http://x", model="m", provider_handle=3)
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_ollama_instance",
        lambda r: mock_prov_ollama,
    )

    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    warmed = []
    monkeypatch.setattr(api_backend, "warmup_backend_model", lambda *_args, **_kwargs: warmed.append(True) or "m")
    
    mock_agent_run = AsyncMock()
    mock_agent_run.ask_structured.return_value = AgentResult(answer="answer")
    monkeypatch.setattr(api_backend, "get_agent", lambda *_args, **_kwargs: mock_agent_run)
    
    areq = api_backend.RunAgenticAgentRequest(question="q", model="m", backend_url="http://x", backend_warmup=True, backend_warmup_timeout=1, kubeconfig=str(kubeconfig))
    result = await api_backend.run_agentic_agent(areq)
    assert result["answer"] == "answer"
    assert warmed == [True]

    mock_prov_bench = AsyncMock()
    mock_prov_bench.provision.return_value = ProvisioningResult(backend_url="http://existing", model="m")
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda r: mock_prov_bench,
    )
    breq = api_backend.RunBenchmarkRequest(provisioning=None, backend_url="u", question="q", model="m", backend_warmup=False, backend_warmup_timeout=1, kubeconfig=str(kubeconfig))
    result = await api_backend.run_benchmark(breq)
    assert result["answer"] == "answer"
    assert result["backend_url"] == "http://existing"

    mock_prov_fail = AsyncMock()
    mock_prov_fail.provision.return_value = ProvisioningResult(backend_url=None, model="m", provider_handle=7)
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda r: mock_prov_fail,
    )
    with pytest.raises(RuntimeError, match="Could not determine Ollama URL"):
        vast_req = api_backend.RunBenchmarkRequest.from_cli(
            vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, 
            backend_url=None, question="q", model="m", backend_warmup=False, 
            backend_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=False
        )
        await api_backend.run_benchmark(vast_req)

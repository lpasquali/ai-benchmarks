import pytest
from rune_bench.api_backend import _vastai_sdk, _make_resource_provider_for_benchmark, _make_resource_provider_for_ollama_instance
from rune_bench.api_contracts import RunBenchmarkRequest, RunLLMInstanceRequest
from unittest.mock import patch
import os

def test_vastai_sdk_missing():
    # Mock ImportError
    with patch("rune_bench.api_backend.VastAI", None):
        with pytest.raises(RuntimeError, match="vastai"):
            _vastai_sdk()

def _make_benchmark_req(vastai=False, backend_url="http://local"):
    return RunBenchmarkRequest(
        question="test",
        model="test",
        vastai=vastai,
        template_hash="xyz",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.9,
        backend_url=backend_url,
        backend_warmup=False,
        backend_warmup_timeout=0,
        kubeconfig="/dev/null",
        vastai_stop_instance=False,
    )

def _make_llm_req(vastai=False, backend_url="http://local"):
    return RunLLMInstanceRequest(
        vastai=vastai,
        template_hash="xyz",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.9,
        backend_url=backend_url,
    )

def test_make_resource_provider_vastai_benchmark(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "123")
    req = _make_benchmark_req(vastai=True)
    prov = _make_resource_provider_for_benchmark(req)
    assert prov.__class__.__name__ == "VastAIProvider"

def test_make_resource_provider_vastai_llm_instance(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "123")
    req = _make_llm_req(vastai=True)
    prov = _make_resource_provider_for_ollama_instance(req)
    assert prov.__class__.__name__ == "VastAIProvider"

def test_make_resource_provider_local_benchmark():
    req = _make_benchmark_req(vastai=False, backend_url="http://local")
    prov = _make_resource_provider_for_benchmark(req)
    assert prov.__class__.__name__ == "ExistingBackendProvider"
    
def test_make_resource_provider_local_llm_instance():
    req = _make_llm_req(vastai=False, backend_url="http://local")
    prov = _make_resource_provider_for_ollama_instance(req)
    assert prov.__class__.__name__ == "ExistingBackendProvider"

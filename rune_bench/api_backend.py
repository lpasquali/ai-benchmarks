"""Local execution backend behind the RUNE HTTP API."""

from __future__ import annotations

import os
from pathlib import Path

from rune_bench.agents.base import AgentRunner
from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunOllamaInstanceRequest,
)
from rune_bench.common import ModelSelector
from rune_bench.metrics import span
from rune_bench.resources.base import LLMResourceProvider
from rune_bench.resources.existing_ollama_provider import ExistingOllamaProvider
from rune_bench.workflows import (
    list_existing_ollama_models,
    list_running_ollama_models,
    use_existing_ollama_server,
    warmup_existing_ollama_model,
)

try:
    from vastai import VastAI
except ImportError:
    VastAI = None  # type: ignore[assignment,misc]


def _vastai_sdk() -> "VastAI":
    """Instantiate VastAI SDK reading the API key from the environment."""
    if VastAI is None:
        raise RuntimeError(
            "The 'vastai' package is required for Vast.ai provisioning. "
            "Install it with: pip install 'rune-bench[vastai]'"
        )
    api_key = os.environ.get("VAST_API_KEY", "")
    return VastAI(api_key=api_key, raw=True)


def _make_resource_provider_for_benchmark(request: RunBenchmarkRequest) -> LLMResourceProvider:
    """Factory: return the LLM resource provider for a benchmark run."""
    if request.vastai:
        from rune_bench.resources.vastai import VastAIProvider
        return VastAIProvider(
            _vastai_sdk(),
            template_hash=request.template_hash,
            min_dph=request.min_dph,
            max_dph=request.max_dph,
            reliability=request.reliability,
            stop_on_teardown=request.vastai_stop_instance,
        )
    return ExistingOllamaProvider(
        request.ollama_url,
        model=request.model,
        warmup=request.ollama_warmup,
        warmup_timeout=request.ollama_warmup_timeout,
    )


def _make_resource_provider_for_ollama_instance(request: RunOllamaInstanceRequest) -> LLMResourceProvider:
    """Factory: return the LLM resource provider for an Ollama instance run."""
    if request.vastai:
        from rune_bench.resources.vastai import VastAIProvider
        return VastAIProvider(
            _vastai_sdk(),
            template_hash=request.template_hash,
            min_dph=request.min_dph,
            max_dph=request.max_dph,
            reliability=request.reliability,
            stop_on_teardown=False,
        )
    return ExistingOllamaProvider(request.ollama_url)


def _make_agent_runner(kubeconfig: Path) -> AgentRunner:
    """Lazy factory: load HolmesRunner only when an agent run is requested.

    Replace this function (via monkeypatch or dependency injection) to swap
    in a different AgentRunner implementation.
    """
    from rune_bench.agents.sre.holmes import HolmesRunner

    return HolmesRunner(kubeconfig)


def list_vastai_models() -> list[dict]:
    return [
        {
            "name": model.name,
            "vram_mb": model.vram_mb,
            "required_disk_gb": model.required_disk_gb,
        }
        for model in ModelSelector().list_models()
    ]


def list_ollama_models(ollama_url: str) -> dict:
    server = use_existing_ollama_server(ollama_url, model_name="<n/a>")
    return {
        "ollama_url": server.url,
        "models": list_existing_ollama_models(server.url),
        "running_models": list_running_ollama_models(server.url),
    }


def run_ollama_instance(request: RunOllamaInstanceRequest) -> dict:
    provider = _make_resource_provider_for_ollama_instance(request)
    result = provider.provision()
    out: dict = {"mode": "vastai" if request.vastai else "existing", "ollama_url": result.ollama_url}
    if request.vastai:
        out["model_name"] = result.model
        out["contract_id"] = result.provider_handle
    return out


def run_agentic_agent(request: RunAgenticAgentRequest) -> dict:
    if request.ollama_url and request.ollama_warmup:
        warmup_existing_ollama_model(
            request.ollama_url,
            request.model,
            timeout_seconds=request.ollama_warmup_timeout,
        )
    runner = _make_agent_runner(Path(request.kubeconfig))
    with span("agent.ask", model=request.model, backend="existing"):
        answer = runner.ask(
            question=request.question,
            model=request.model,
            ollama_url=request.ollama_url,
        )
    return {"answer": answer}


def run_benchmark(request: RunBenchmarkRequest) -> dict:
    backend = "vastai" if request.vastai else "existing"
    provider = _make_resource_provider_for_benchmark(request)
    with span("workflow.provision", backend=backend):
        result = provider.provision()

    if not result.ollama_url:
        raise RuntimeError(
            "Could not determine Ollama URL from the Vast.ai instance service mappings. "
            "Ensure port 11434 is exposed in the template."
        )

    effective_model = result.model or request.model
    try:
        runner = _make_agent_runner(Path(request.kubeconfig))
        with span("agent.ask", model=effective_model, backend=backend):
            answer = runner.ask(
                question=request.question,
                model=effective_model,
                ollama_url=result.ollama_url,
            )
    finally:
        provider.teardown(result)

    return {
        "answer": answer,
        "model_name": effective_model,
        "ollama_url": result.ollama_url,
        "contract_id": result.provider_handle,
    }


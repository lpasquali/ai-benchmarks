"""Local execution backend behind the RUNE HTTP API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rune_bench.agents.registry import get_agent
from rune_bench.api_contracts import (
    CostEstimationRequest,
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunOllamaInstanceRequest,
)
from rune_bench.common import ModelSelector
from rune_bench.metrics import span  # noqa: F401 (used by workflows layer)
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


def _make_agent_runner(agent_name: str | Path = "holmes", *, kubeconfig: Path | None = None) -> Any:
    """Lazy factory: resolve an agent via the registry.

    Accepts either the new ``(agent_name, *, kubeconfig=...)`` signature or
    the legacy ``(kubeconfig_path)`` positional call used by existing tests
    and monkeypatches.
    """
    if isinstance(agent_name, Path) or (
        isinstance(agent_name, str) and "/" in agent_name
    ):
        kubeconfig = Path(agent_name)
        agent_name = "holmes"

    kwargs: dict[str, Any] = {}
    if kubeconfig is not None:
        kwargs["kubeconfig"] = kubeconfig
    return get_agent(agent_name, **kwargs)


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
    agent_name = getattr(request, "agent", "holmes")
    if agent_name != "holmes":
        # Pass all potentially useful kwargs; get_agent() filters based on required_config
        agent_kwargs: dict[str, Any] = {"kubeconfig": Path(request.kubeconfig)}
        runner = get_agent(agent_name, **agent_kwargs)
    else:
        runner = _make_agent_runner(Path(request.kubeconfig))
    answer = runner.ask(
        question=request.question,
        model=request.model,
        ollama_url=request.ollama_url,
    )
    return {"answer": answer}


def _verify_attestation(target: str) -> None:
    """Run PCR attestation for *target*; raises RuntimeError on failure."""
    from rune_bench.attestation.factory import get_driver

    driver = get_driver()
    result = driver.verify(target)
    if not result.passed:
        raise RuntimeError(
            f"Attestation failed for scheduling target {target!r}: {result.message}"
        )


def run_benchmark(request: RunBenchmarkRequest) -> dict:
    if request.attestation_required:
        _verify_attestation(request.kubeconfig)

    provider = _make_resource_provider_for_benchmark(request)
    result = provider.provision()

    if not result.ollama_url:
        raise RuntimeError(
            "Could not determine Ollama URL from the Vast.ai instance service mappings. "
            "Ensure port 11434 is exposed in the template."
        )

    effective_model = result.model or request.model
    try:
        runner = _make_agent_runner(Path(request.kubeconfig))
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


def get_cost_estimate(request: CostEstimationRequest) -> dict:
    """Estimate cost for a benchmark run based on cloud or local hardware parameters."""
    duration_hours = request.estimated_duration_seconds / 3600

    if request.vastai:
        dph = request.max_dph if request.max_dph > 0 else request.min_dph
        projected_cost_usd = dph * duration_hours
        local_energy_kwh = 0.0
        cost_driver = "vastai"
    elif request.local_hardware:
        local_energy_kwh = (request.local_tdp_watts * duration_hours) / 1000
        energy_cost = local_energy_kwh * request.local_energy_rate_kwh
        depreciation_cost = 0.0
        if request.local_hardware_lifespan_years > 0:
            depreciation_cost = (
                request.local_hardware_purchase_price
                / (request.local_hardware_lifespan_years * 8760)
                * duration_hours
            )
        projected_cost_usd = energy_cost + depreciation_cost
        cost_driver = "local"
    elif request.aws:
        projected_cost_usd = (request.min_dph + request.max_dph) / 2 * duration_hours
        local_energy_kwh = 0.0
        cost_driver = "aws"
    elif request.gcp:
        projected_cost_usd = (request.min_dph + request.max_dph) / 2 * duration_hours
        local_energy_kwh = 0.0
        cost_driver = "gcp"
    elif request.azure:
        projected_cost_usd = (request.min_dph + request.max_dph) / 2 * duration_hours
        local_energy_kwh = 0.0
        cost_driver = "azure"
    else:
        projected_cost_usd = 0.0
        local_energy_kwh = 0.0
        cost_driver = "unknown"

    if projected_cost_usd < 1.0:
        resource_impact = "low"
    elif projected_cost_usd < 10.0:
        resource_impact = "medium"
    else:
        resource_impact = "high"

    return {
        "projected_cost_usd": round(projected_cost_usd, 4),
        "cost_driver": cost_driver,
        "resource_impact": resource_impact,
        "local_energy_kwh": round(local_energy_kwh, 4),
        "confidence_score": 1.0,
        "warning": None,
    }

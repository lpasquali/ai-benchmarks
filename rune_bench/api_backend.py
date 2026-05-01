# SPDX-License-Identifier: Apache-2.0
"""Local execution backend behind the RUNE HTTP API."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from rune_bench.agents.registry import get_agent
from rune_bench.api_contracts import (
    CostEstimationRequest,
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
    RunTelemetry,
)
from rune_bench.common import ModelSelector
from rune_bench.metrics import span  # noqa: F401 (used by workflows layer)
from rune_bench.metrics.cost import calculate_run_cost
from rune_bench.resources.base import LLMResourceProvider
from rune_bench.resources.existing_backend_provider import ExistingBackendProvider
from rune_bench.workflows import warmup_backend_model
from rune_bench.common.artifact_utils import process_agent_artifacts

try:
    from rune_bench.resources.vastai.sdk import VastAI
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


def _make_resource_provider_for_benchmark(
    request: RunBenchmarkRequest,
) -> LLMResourceProvider:
    """Factory: return the LLM resource provider for a benchmark run."""
    backend_type = getattr(request, "backend_type", "ollama")
    
    if backend_type == "vastai" or (request.provisioning and request.provisioning.vastai):
        from rune_bench.api_contracts import VastAIProvisioning
        from rune_bench.resources.vastai import VastAIProvider
        v = (request.provisioning.vastai if request.provisioning else None) or VastAIProvisioning(
            template_hash=os.environ.get("RUNE_VASTAI_TEMPLATE", "c166c11f035d3a97871a23bd32ca6aba"),
            min_dph=float(os.environ.get("RUNE_VASTAI_MIN_DPH", "0.0")),
            max_dph=float(os.environ.get("RUNE_VASTAI_MAX_DPH", "10.0")),
            reliability=float(os.environ.get("RUNE_VASTAI_RELIABILITY", "0.9")),
        )

        return VastAIProvider(
            _vastai_sdk(),
            template_hash=v.template_hash,
            min_dph=v.min_dph,
            max_dph=v.max_dph,
            reliability=v.reliability,
            stop_on_teardown=v.stop_instance,
        )
    return ExistingBackendProvider(
        request.backend_url,
        model=request.model,
        warmup=request.backend_warmup,
        warmup_timeout=request.backend_warmup_timeout,
        backend_type=backend_type,
    )


def _make_resource_provider_for_ollama_instance(
    request: RunLLMInstanceRequest,
) -> LLMResourceProvider:
    """Factory: return the LLM resource provider for an Ollama instance run."""
    backend_type = getattr(request, "backend_type", "ollama")
    
    if backend_type == "vastai" or (request.provisioning and request.provisioning.vastai):
        from rune_bench.api_contracts import VastAIProvisioning
        from rune_bench.resources.vastai import VastAIProvider
        v = (request.provisioning.vastai if request.provisioning else None) or VastAIProvisioning(
            template_hash=os.environ.get("RUNE_VASTAI_TEMPLATE", "c166c11f035d3a97871a23bd32ca6aba"),
            min_dph=0.0,
            max_dph=10.0,
            reliability=0.9,
        )
        return VastAIProvider(
            _vastai_sdk(),
            template_hash=v.template_hash,
            min_dph=v.min_dph,
            max_dph=v.max_dph,
            reliability=v.reliability,
            stop_on_teardown=False,
        )
    return ExistingBackendProvider(
        request.backend_url,
        backend_type=backend_type,
    )


def _make_agent_runner(
    agent_name: str = "holmes", *, kubeconfig: Path | None = None
) -> Any:
    """Lazy factory: resolve an agent via the registry."""
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


def list_backend_models(backend_url: str, *, backend_type: str = "ollama") -> dict:
    from rune_bench.backends import get_backend

    backend = get_backend(backend_type, backend_url)
    return {
        "backend_url": backend.base_url,
        "backend_type": backend_type,
        "models": backend.list_models(),
        "running_models": backend.list_running_models(),
    }


async def run_llm_instance(request: RunLLMInstanceRequest) -> dict:
    provider = _make_resource_provider_for_ollama_instance(request)
    result = await provider.provision()
    mode = (
        "vastai"
        if (request.provisioning and request.provisioning.vastai)
        else "existing"
    )
    out: dict = {"mode": mode, "backend_url": result.backend_url}
    if mode == "vastai":
        out["model_name"] = result.model
        out["contract_id"] = result.provider_handle
    return out


async def run_agentic_agent(
    request: RunAgenticAgentRequest,
    job_id: str | None = None,
    storage: Any | None = None,
) -> dict:
    if request.backend_url and request.backend_warmup:
        warmup_backend_model(
            request.backend_url,
            request.model,
            timeout_seconds=request.backend_warmup_timeout,
        )
    agent_name = getattr(request, "agent", "holmes")

    # Validate kubeconfig is provided when the agent requires it.
    from rune_bench.agents.registry import _BUILTIN_AGENTS

    builtin_entry = _BUILTIN_AGENTS.get(agent_name)
    if (
        builtin_entry
        and "kubeconfig" in builtin_entry[2]
        and request.kubeconfig is None
    ):
        raise RuntimeError(
            f"Agent '{agent_name}' requires a kubeconfig path; "
            "set KUBECONFIG or pass --kubeconfig"
        )

    kubeconfig_path = Path(request.kubeconfig) if request.kubeconfig else None
    agent_kwargs: dict[str, Any] = {}
    if kubeconfig_path is not None:
        agent_kwargs["kubeconfig"] = kubeconfig_path
    # Block 10 — Run agentic agent
    start_time = time.perf_counter()
    try:
        from rune_bench.metrics import span as _span

        runner = get_agent(agent_name, **agent_kwargs)
        with _span("agent.ask", model=request.model, backend="existing"):
            result = await runner.ask_structured(
                question=request.question,
                model=request.model,
                backend_url=request.backend_url,
                backend_type=getattr(request, "backend_type", "ollama"),
            )
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"Agent error: {exc}") from exc
    duration_s = int(time.perf_counter() - start_time)

    # Calculate and attach cost
    backend_mode = getattr(request, "backend_type", "local")
    cost_usd = await calculate_run_cost(backend_mode, request.model, duration_s)

    if result.metadata is None:
        result.metadata = {}
    result.metadata["cost"] = cost_usd
    result.metadata["duration_s"] = duration_s

    if result.telemetry:
        # RunTelemetry is frozen, so we would need to recreate it if we wanted to update cost_estimate_usd
        # but for now we just return it in metadata as well.
        # Actually, let's try to populate it if it was None.
        if result.telemetry.cost_estimate_usd is None:
            # Re-create with cost
            new_telemetry = RunTelemetry(
                tokens=result.telemetry.tokens,
                latency=result.telemetry.latency,
                cost_estimate_usd=cost_usd,
            )
            result.telemetry = new_telemetry
    else:
        result.telemetry = RunTelemetry(cost_estimate_usd=cost_usd)

    processed_artifacts = result.artifacts
    if job_id and storage and result.artifacts:
        processed_artifacts = process_agent_artifacts(job_id, result.artifacts, storage)

    return {
        "answer": result.answer,
        "result_type": result.result_type,
        "artifacts": processed_artifacts,
        "metadata": result.metadata,
        "telemetry": result.telemetry.to_dict() if result.telemetry else None,
    }


def _verify_attestation(target: str) -> None:
    """Run PCR attestation for *target*; raises RuntimeError on failure."""
    from rune_bench.attestation.factory import get_driver

    driver = get_driver()
    result = driver.verify(target)
    if not result.passed:
        raise RuntimeError(
            f"Attestation failed for scheduling target {target!r}: {result.message}"
        )


async def run_benchmark(
    request: RunBenchmarkRequest,
    job_id: str | None = None,
    storage: Any | None = None,
) -> dict:
    if request.attestation_required:
        _verify_attestation(request.kubeconfig)

    provider = _make_resource_provider_for_benchmark(request)
    result = await provider.provision()

    if not result.backend_url:
        raise RuntimeError(
            "Could not determine Ollama URL from the Vast.ai instance service mappings. "
            "Ensure port 11434 is exposed in the template."
        )

    effective_model = result.model or request.model
    start_time = time.perf_counter()
    try:
        runner = _make_agent_runner(kubeconfig=Path(request.kubeconfig))
        agent_result = await runner.ask_structured(
            question=request.question,
            model=effective_model,
            backend_url=result.backend_url,
            backend_type=getattr(request, "backend_type", "ollama"),
        )
    finally:
        await provider.teardown(result)
    duration_s = int(time.perf_counter() - start_time)

    # Calculate and attach cost
    mode = (
        "vastai"
        if (request.provisioning and request.provisioning.vastai)
        else getattr(request, "backend_type", "local")
    )
    cost_usd = await calculate_run_cost(mode, effective_model, duration_s)

    if agent_result.metadata is None:
        agent_result.metadata = {}
    agent_result.metadata["cost"] = cost_usd
    agent_result.metadata["duration_s"] = duration_s

    if agent_result.telemetry:
        if agent_result.telemetry.cost_estimate_usd is None:
            agent_result.telemetry = RunTelemetry(
                tokens=agent_result.telemetry.tokens,
                latency=agent_result.telemetry.latency,
                cost_estimate_usd=cost_usd,
            )
    else:
        agent_result.telemetry = RunTelemetry(cost_estimate_usd=cost_usd)

    processed_artifacts = agent_result.artifacts
    if job_id and storage and agent_result.artifacts:
        processed_artifacts = process_agent_artifacts(job_id, agent_result.artifacts, storage)

    return {
        "answer": agent_result.answer,
        "result_type": agent_result.result_type,
        "artifacts": processed_artifacts,
        "metadata": agent_result.metadata,
        "telemetry": agent_result.telemetry.to_dict()
        if agent_result.telemetry
        else None,
        "model_name": effective_model,
        "backend_url": result.backend_url,
        "contract_id": result.provider_handle,
    }


async def get_cost_estimate(request: CostEstimationRequest) -> dict:
    """Estimate cost for a benchmark run based on cloud or local hardware parameters."""
    from rune_bench.common.costs import CostEstimator

    estimator = CostEstimator()
    try:
        response = await estimator.estimate(request)
        return {
            "projected_cost_usd": response.projected_cost_usd,
            "cost_driver": response.cost_driver,
            "resource_impact": response.resource_impact,
            "local_energy_kwh": response.local_energy_kwh,
            "confidence_score": response.confidence_score,
            "warning": response.warning,
        }
    except Exception as exc:
        return {
            "projected_cost_usd": 0.0,
            "cost_driver": "error",
            "resource_impact": "low",
            "local_energy_kwh": 0.0,
            "confidence_score": 0.0,
            "warning": f"Cost estimation failed: {exc}",
        }

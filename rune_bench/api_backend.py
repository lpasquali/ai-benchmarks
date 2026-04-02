"""Local execution backend behind the RUNE HTTP API."""

import os
from pathlib import Path

from vastai import VastAI

from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunOllamaInstanceRequest,
)
from rune_bench.common import ModelSelector
from rune_bench.workflows import (
    provision_vastai_ollama,
    stop_vastai_instance,
    use_existing_ollama_server,
    list_existing_ollama_models,
    list_running_ollama_models,
    warmup_existing_ollama_model,
)

# Lazily imported so the API server starts without requiring the holmes/holmesgpt package
HolmesRunner = None


def _get_holmes_runner():
    """Lazy loader for HolmesRunner to allow API-only deployments."""
    global HolmesRunner
    if HolmesRunner is None:
        from rune_bench.agents.holmes import HolmesRunner as _HolmesRunner
        HolmesRunner = _HolmesRunner
    return HolmesRunner


def _vastai_sdk() -> VastAI:
    """Instantiate VastAI SDK reading the API key from the environment."""
    api_key = os.environ.get("VAST_API_KEY", "")
    return VastAI(api_key=api_key, raw=True)


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
    if not request.vastai:
        server = use_existing_ollama_server(request.ollama_url, model_name="<user-selected>")
        return {
            "mode": "existing",
            "ollama_url": server.url,
        }

    result = provision_vastai_ollama(
        _vastai_sdk(),
        template_hash=request.template_hash,
        min_dph=request.min_dph,
        max_dph=request.max_dph,
        reliability=request.reliability,
        confirm_create=lambda: True,
    )
    return {
        "mode": "vastai",
        "contract_id": result.contract_id,
        "ollama_url": result.ollama_url,
        "model_name": result.model_name,
    }


def run_agentic_agent(request: RunAgenticAgentRequest) -> dict:
    if request.ollama_url and request.ollama_warmup:
        warmup_existing_ollama_model(
            request.ollama_url,
            request.model,
            timeout_seconds=request.ollama_warmup_timeout,
        )

    from rune_bench.agents.holmes import HolmesRunner as _HR
    runner = (HolmesRunner or _HR)(Path(request.kubeconfig))
    answer = runner.ask(
        question=request.question,
        model=request.model,
        ollama_url=request.ollama_url,
    )
    return {"answer": answer}


def run_benchmark(request: RunBenchmarkRequest) -> dict:
    selected_model_name = request.model
    selected_ollama_url = request.ollama_url
    vastai_contract_to_stop: int | str | None = None

    if request.vastai:
        result = provision_vastai_ollama(
            _vastai_sdk(),
            template_hash=request.template_hash,
            min_dph=request.min_dph,
            max_dph=request.max_dph,
            reliability=request.reliability,
            confirm_create=lambda: True,
        )
        selected_model_name = result.model_name
        selected_ollama_url = result.ollama_url
        vastai_contract_to_stop = result.contract_id

        if not selected_ollama_url:
            raise RuntimeError(
                "Could not determine Ollama URL from the Vast.ai instance service mappings. "
                "Ensure port 11434 is exposed in the template."
            )
    else:
        server = use_existing_ollama_server(request.ollama_url, model_name=selected_model_name)
        selected_ollama_url = server.url

    if selected_ollama_url and request.ollama_warmup:
        warmup_existing_ollama_model(
            selected_ollama_url,
            selected_model_name,
            timeout_seconds=request.ollama_warmup_timeout,
        )

    try:
        from rune_bench.agents.holmes import HolmesRunner as _HR
        runner = (HolmesRunner or _HR)(Path(request.kubeconfig))
        answer = runner.ask(
            question=request.question,
            model=selected_model_name,
            ollama_url=selected_ollama_url,
        )
    finally:
        if request.vastai and request.vastai_stop_instance and vastai_contract_to_stop is not None:
            stop_vastai_instance(_vastai_sdk(), vastai_contract_to_stop)

    return {
        "answer": answer,
        "model_name": selected_model_name,
        "ollama_url": selected_ollama_url,
        "contract_id": vastai_contract_to_stop,
    }

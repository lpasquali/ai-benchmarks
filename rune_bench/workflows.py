"""Application workflows used by the RUNE CLI.

This module keeps orchestration/business logic out of the CLI layer.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from rune_bench.resources.vastai.sdk import VastAI

from .common import ModelSelector
from .debug import debug_log
from .metrics import span
from rune_bench.backends.ollama import OllamaClient, OllamaModelManager

try:
    from rune_bench.resources.vastai import ConnectionDetails, InstanceManager, OfferFinder, TeardownResult, TemplateLoader
except ImportError:  # vastai extra not installed
    ConnectionDetails = None  # type: ignore[assignment,misc]
    InstanceManager = None  # type: ignore[assignment,misc]
    OfferFinder = None  # type: ignore[assignment,misc]
    TeardownResult = None  # type: ignore[assignment,misc]
    TemplateLoader = None  # type: ignore[assignment,misc]


class UserAbortedError(RuntimeError):
    """Raised when an interactive confirmation is rejected by the user."""


DEFAULT_SPEND_THRESHOLD = 5.00


class SpendGateAction(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    BLOCK = "block"


def evaluate_spend_gate(
    projected_cost: float,
    *,
    threshold: float,
    yes: bool,
) -> SpendGateAction:
    """Determine the spend-gate action for a given projected cost.

    Returns ALLOW when cost is within threshold or --yes is set.
    Returns BLOCK when running in CI (non-interactive).
    Returns PROMPT otherwise.
    """
    if projected_cost <= threshold or yes:
        return SpendGateAction.ALLOW
    if os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}:
        return SpendGateAction.BLOCK
    return SpendGateAction.PROMPT


@dataclass
class ExistingOllamaServer:
    url: str
    model_name: str


@dataclass
class VastAIProvisioningResult:
    offer_id: int
    total_vram_mb: int
    model_name: str
    model_vram_mb: int
    required_disk_gb: int
    template_env: str
    contract_id: int | str
    details: ConnectionDetails
    backend_url: str | None = None
    reused_existing_instance: bool = False
    pull_warning: str | None = None


def normalize_backend_url(backend_url: str | None) -> str:
    """Validate and normalize an Ollama base URL.

    Adds ``http://`` when missing. This is a convenience wrapper that delegates
    to OllamaClient's normalization logic.
    """
    if backend_url is None:
        raise RuntimeError("Missing Ollama URL")
    client = OllamaClient(backend_url)
    return client.base_url


def use_existing_ollama_server(backend_url: str | None, model_name: str) -> ExistingOllamaServer:
    """Resolve an existing Ollama server target."""
    return ExistingOllamaServer(url=normalize_backend_url(backend_url), model_name=model_name)


def list_existing_ollama_models(backend_url: str | None) -> list[str]:
    """Return available model names from an existing Ollama server."""
    manager = OllamaModelManager.create(normalize_backend_url(backend_url))
    return manager.list_available_models()


def list_running_ollama_models(backend_url: str | None) -> list[str]:
    """Return model names currently loaded in memory on an existing Ollama server."""
    manager = OllamaModelManager.create(normalize_backend_url(backend_url))
    return manager.list_running_models()


def normalize_ollama_model_for_api(model_name: str) -> str:
    """Convert provider-prefixed model identifiers into plain Ollama model names."""
    manager = OllamaModelManager.create("http://localhost:11434")  # URL not used for normalization
    return manager.normalize_model_name(model_name)


def warmup_existing_ollama_model(
    backend_url: str | None,
    model_name: str,
    *,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
    keep_alive: str = "30m",
) -> str:
    """Load a model into an existing Ollama server and wait until it is running."""
    normalized_url = normalize_backend_url(backend_url)
    manager = OllamaModelManager.create(normalized_url)
    api_model_name = manager.normalize_model_name(model_name)
    debug_log(
        f"Workflow warmup: backend_url={normalized_url} requested_model={model_name} api_model={api_model_name}"
    )

    with span("ollama.model.warmup", model=api_model_name):
        return manager.warmup_model(
            api_model_name,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            keep_alive=keep_alive,
            unload_others=True,
        )



def provision_vastai_ollama(
    sdk: VastAI,
    *,
    template_hash: str,
    min_dph: float,
    max_dph: float,
    reliability: float,
    confirm_create: Callable[[], bool],
    on_poll: Callable[[str], None] | None = None,
) -> VastAIProvisioningResult:
    """Run the full Vast.ai + Ollama provisioning workflow.

    Reuses an already-running matching instance when possible; otherwise provisions a new one.
    Ensures the selected model is present and warmed up before returning.
    """
    manager = InstanceManager(sdk)

    with span("vastai.instance.reuse_check"):
        reusable = manager.find_reusable_running_instance(
            min_dph=min_dph,
            max_dph=max_dph,
            reliability=reliability,
        )

    template = None
    reused_existing_instance = False

    if reusable is not None:
        debug_log(f"Workflow reuse candidate found: contract_id={reusable.get('id')}")
        total_vram_mb = int(float(reusable.get("gpu_total_ram", 0)))
        try:
            selected_model = ModelSelector().select(total_vram_mb)
            reusable_contract_id = reusable.get("id")
            if reusable_contract_id is None or reusable_contract_id == "":
                raise RuntimeError("Reusable instance is missing contract id")
            contract_id: int | str = reusable_contract_id
            instance_info = reusable
            reused_existing_instance = True
            offer_id = int(float(reusable.get("ask_contract_id", reusable.get("id", 0))))
            debug_log(
                f"Workflow reusing instance: contract_id={contract_id} total_vram_mb={total_vram_mb} model={selected_model.name}"
            )
        except RuntimeError:
            reusable = None

    if reusable is None:
        debug_log("Workflow provisioning new Vast.ai instance")
        with span("vastai.offer_search", min_dph=min_dph, max_dph=max_dph, reliability=reliability):
            offer = OfferFinder(sdk).find_best(
                min_dph=min_dph,
                max_dph=max_dph,
                reliability=reliability,
            )
        selected_model = ModelSelector().select(offer.total_vram_mb)
        template = TemplateLoader(sdk).load(template_hash)

        if not confirm_create():
            raise UserAbortedError("User aborted instance creation.")

        with span("vastai.instance.create", model=selected_model.name):
            contract_id = manager.create(offer.offer_id, selected_model, template)
        with span("vastai.instance.wait_running"):
            instance_info = manager.wait_until_running(contract_id, on_poll=on_poll)
        total_vram_mb = offer.total_vram_mb
        offer_id = offer.offer_id

    details = InstanceManager.build_connection_details(contract_id, instance_info)
    backend_url = _extract_ollama_service_url(details)
    debug_log(f"Workflow detected Ollama URL: {backend_url or '<missing>'}")

    pull_warning = None
    try:
        if not backend_url:
            raise RuntimeError(
                "Could not determine Ollama URL from instance service mappings (port 11434 missing)."
            )

        available = set(list_existing_ollama_models(backend_url))
        running = set(list_running_ollama_models(backend_url))
        api_model = normalize_ollama_model_for_api(selected_model.name)
        debug_log(
            f"Workflow Ollama state: available={sorted(available)} running={sorted(running)} selected={api_model}"
        )

        if api_model not in running:
            if api_model not in available:
                with span("vastai.model.pull", model=selected_model.name):
                    manager.pull_model(contract_id, selected_model.name, backend_url=backend_url)

            warmup_existing_ollama_model(
                backend_url,
                selected_model.name,
                timeout_seconds=120,
            )
    except RuntimeError as exc:
        pull_warning = str(exc)

    return VastAIProvisioningResult(
        offer_id=offer_id,
        total_vram_mb=total_vram_mb,
        model_name=selected_model.name,
        model_vram_mb=selected_model.vram_mb,
        required_disk_gb=selected_model.required_disk_gb,
        template_env=(template.env if template is not None else "<reused-running-instance>"),
        contract_id=contract_id,
        details=details,
        backend_url=backend_url,
        reused_existing_instance=reused_existing_instance,
        pull_warning=pull_warning,
    )


def stop_vastai_instance(sdk: VastAI, contract_id: int | str) -> TeardownResult:
    """Destroy Vast.ai instance + related storage and verify cleanup."""
    return InstanceManager(sdk).destroy_instance_and_related_storage(contract_id)


def run_preflight_cost_check(
    *,
    vastai: bool,
    max_dph: float,
    min_dph: float,
    estimated_duration_seconds: int = 3600,
    backend_mode: str = "local",
    http_client=None,
) -> dict:
    """Estimate projected spend for a Vast.ai job.

    Returns the cost estimate dict (empty dict when vastai is False).
    Raises FailClosedError when no cost driver is configured.
    Raises RuntimeError when estimation is unavailable.
    """
    if not vastai:
        return {}

    from rune_bench.api_contracts import CostEstimationRequest
    from rune_bench.common.costs import CostEstimator, FailClosedError  # noqa: F401 (re-raised by caller)

    cost_req = CostEstimationRequest(
        vastai=vastai,
        max_dph=max_dph,
        min_dph=min_dph,
        estimated_duration_seconds=estimated_duration_seconds,
    )

    if backend_mode == "http":
        if http_client is None:
            raise RuntimeError("http_client is required when backend_mode='http'")
        return http_client.get_cost_estimate(cost_req.to_dict())

    estimator = CostEstimator()
    response = asyncio.run(estimator.estimate(cost_req))
    return response.to_dict()

def _extract_ollama_service_url(details: ConnectionDetails) -> str | None:
    for svc in details.service_urls:
        direct = str(svc.get("direct", ""))
        proxy = str(svc.get("proxy", "")) if svc.get("proxy") else ""
        if ":11434" in direct:
            return direct
        if ":11434" in proxy:
            return proxy
    return None

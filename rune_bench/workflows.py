# SPDX-License-Identifier: Apache-2.0
"""Application workflows used by the RUNE CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from rune_bench.resources.vastai.sdk import VastAI

from .common import ModelSelector
from .common.backend_utils import (
    list_backend_models,
    list_running_backend_models,
    normalize_backend_model_for_api,
    normalize_backend_url as normalize_backend_url,
    use_existing_backend_server,
    warmup_backend_model,
)
from .debug import debug_log
from .metrics import span  # noqa: F401
from rune_bench.backends.ollama import OllamaBackend
from rune_bench.resources.vastai.contracts import ConnectionDetails, TeardownResult

try:
    from rune_bench.resources.vastai.instance import InstanceManager
    from rune_bench.resources.vastai.offer import OfferFinder
    from rune_bench.resources.vastai.template import TemplateLoader
except ImportError:
    InstanceManager = None  # type: ignore[assignment,misc]
    OfferFinder = None  # type: ignore[assignment,misc]
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


@dataclass(frozen=True)
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




# ---------------------------------------------------------------------------
# Vast.ai provisioning workflow
# ---------------------------------------------------------------------------


def provision_vastai_backend(
    sdk: VastAI,
    *,
    template_hash: str,
    min_dph: float,
    max_dph: float,
    reliability: float,
    confirm_create: Callable[[], bool],
    on_poll: Callable[[str], None] | None = None,
) -> VastAIProvisioningResult:
    """Run the full Vast.ai + backend provisioning workflow.

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
            offer_id = int(
                float(reusable.get("ask_contract_id", reusable.get("id", 0)))
            )
            debug_log(
                f"Workflow reusing instance: contract_id={contract_id} total_vram_mb={total_vram_mb} model={selected_model.name}"
            )
        except RuntimeError:
            reusable = None

    if reusable is None:
        debug_log("Workflow provisioning new Vast.ai instance")
        with span(
            "vastai.offer_search",
            min_dph=min_dph,
            max_dph=max_dph,
            reliability=reliability,
        ):
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
    backend_url = OllamaBackend.extract_service_url(details)
    debug_log(f"Workflow detected backend URL: {backend_url or '<missing>'}")

    pull_warning = None
    try:
        if not backend_url:
            raise RuntimeError(
                "Could not determine Ollama URL from instance service mappings (port 11434 missing)."
            )

        available = set(list_backend_models(backend_url))
        running = set(list_running_backend_models(backend_url))
        api_model = normalize_backend_model_for_api(selected_model.name)
        debug_log(
            f"Workflow backend state: available={sorted(available)} running={sorted(running)} selected={api_model}"
        )

        if api_model not in running:
            if api_model not in available:
                with span("vastai.model.pull", model=selected_model.name):
                    manager.pull_model(
                        contract_id, selected_model.name, backend_url=backend_url
                    )

            warmup_backend_model(
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
        template_env=(
            template.env if template is not None else "<reused-running-instance>"
        ),
        contract_id=contract_id,
        details=details,
        backend_url=backend_url,
        reused_existing_instance=reused_existing_instance,
        pull_warning=pull_warning,
    )




def stop_vastai_instance(sdk: VastAI, contract_id: int | str) -> TeardownResult:
    """Destroy Vast.ai instance + related storage and verify cleanup."""
    return InstanceManager(sdk).destroy_instance_and_related_storage(contract_id)


async def run_preflight_cost_check(
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
    response = await estimator.estimate(cost_req)
    return response.to_dict()


def _extract_ollama_service_url(details: ConnectionDetails) -> str | None:
    """Backward-compatible alias for :meth:`OllamaBackend.extract_service_url`."""
    return OllamaBackend.extract_service_url(details)


# ── Multi-agent chain orchestration ───────────────────────────────────────────


class JobStoreChainRecorder:
    """Adapter that lets ``ChainExecutionEngine`` write its DAG state to a ``JobStore``.

    The engine emits ``initialize`` once and one ``transition`` per node state
    change (running / success / failed). This adapter forwards each call to the
    matching JobStore method, so a dashboard polling
    ``GET /v1/chains/{run_id}/state`` can render the live DAG.
    """

    def __init__(self, store: "JobStore") -> None:  # noqa: F821 — forward ref
        self._store = store

    def initialize(self, *, job_id: str, nodes: list[dict], edges: list[dict]) -> None:
        self._store.record_chain_initialized(job_id=job_id, nodes=nodes, edges=edges)

    def transition(
        self,
        *,
        job_id: str,
        node_id: str,
        status: str,
        started_at: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
    ) -> None:
        self._store.record_chain_node_transition(
            job_id=job_id,
            node_id=node_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )


async def run_chain_workflow(
    *,
    steps: "list[ChainStep]",  # noqa: F821 — forward ref
    initial_context: dict[str, object],
    model: str,
    job_id: str,
    store: "JobStore",  # noqa: F821 — forward ref
    backend_url: str | None = None,
) -> "ChainResult":  # noqa: F821 — forward ref
    """Run a multi-agent chain and persist its live DAG state to ``store``.

    This is the production entry point for executing a ``ChainExecutionEngine``.
    A ``JobStoreChainRecorder`` is wired into the engine so every step
    transition is durably recorded against ``job_id`` and made available via
    ``GET /v1/chains/{job_id}/state``.

    Parameters
    ----------
    steps:
        Ordered list of ``ChainStep`` instances forming a DAG.
    initial_context:
        Variables made available to the first step's question template.
    model:
        Model name passed to each agent's ``ask_async``.
    job_id:
        The job/run identifier under which chain state is persisted. The same
        id is used by ``GET /v1/chains/{job_id}/state`` so the dashboard can
        correlate the run with its DAG view.
    store:
        ``JobStore`` instance that will receive ``initialize`` and per-node
        ``transition`` calls via ``JobStoreChainRecorder``.
    backend_url:
        Optional backend URL forwarded to each agent.
    """
    # Local imports to avoid hard import cycles at module load time.
    from rune_bench.agents.chain import ChainExecutionEngine

    recorder = JobStoreChainRecorder(store)
    engine = ChainExecutionEngine(steps, recorder=recorder, job_id=job_id)
    return await engine.execute(initial_context, model=model, backend_url=backend_url)

# SPDX-License-Identifier: Apache-2.0
"""Stable contracts for CLI/API compatibility.

These dataclasses are transport-agnostic and define the operation payloads
used by the current CLI and future HTTP API backend.
"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLLMInstanceRequest:
    vastai: bool
    template_hash: str
    min_dph: float
    max_dph: float
    reliability: float
    backend_url: str | None
    backend_type: str = "ollama"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunAgenticAgentRequest:
    question: str
    model: str
    backend_url: str | None
    backend_warmup: bool
    backend_warmup_timeout: int
    backend_type: str = "ollama"
    kubeconfig: str | None = None
    agent: str = "holmes"

    @classmethod
    def from_cli(
        cls,
        *,
        question: str,
        model: str,
        backend_url: str | None,
        backend_warmup: bool,
        backend_warmup_timeout: int,
        kubeconfig: Path | None = None,
        agent: str = "holmes",
        backend_type: str = "ollama",
    ) -> "RunAgenticAgentRequest":
        return cls(
            question=question,
            model=model,
            backend_url=backend_url,
            backend_warmup=backend_warmup,
            backend_warmup_timeout=backend_warmup_timeout,
            backend_type=backend_type,
            kubeconfig=str(kubeconfig) if kubeconfig is not None else None,
            agent=agent,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunBenchmarkRequest:
    vastai: bool
    template_hash: str
    min_dph: float
    max_dph: float
    reliability: float
    backend_url: str | None
    question: str
    model: str
    backend_warmup: bool
    backend_warmup_timeout: int
    kubeconfig: str
    vastai_stop_instance: bool
    attestation_required: bool = False
    backend_type: str = "ollama"

    @classmethod
    def from_cli(
        cls,
        *,
        vastai: bool,
        template_hash: str,
        min_dph: float,
        max_dph: float,
        reliability: float,
        backend_url: str | None,
        question: str,
        model: str,
        backend_warmup: bool,
        backend_warmup_timeout: int,
        kubeconfig: Path,
        vastai_stop_instance: bool,
        attestation_required: bool = False,
        backend_type: str = "ollama",
    ) -> "RunBenchmarkRequest":
        return cls(
            vastai=vastai,
            template_hash=template_hash,
            min_dph=min_dph,
            max_dph=max_dph,
            reliability=reliability,
            backend_url=backend_url,
            question=question,
            model=model,
            backend_warmup=backend_warmup,
            backend_warmup_timeout=backend_warmup_timeout,
            kubeconfig=str(kubeconfig),
            vastai_stop_instance=vastai_stop_instance,
            attestation_required=attestation_required,
            backend_type=backend_type,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimationRequest:
    # Cloud parameters
    vastai: bool = False
    aws: bool = False
    gcp: bool = False
    azure: bool = False
    min_dph: float = 0.0
    max_dph: float = 0.0
    
    # Local hardware parameters
    local_hardware: bool = False
    local_tdp_watts: float = 0.0
    local_energy_rate_kwh: float = 0.0
    local_hardware_purchase_price: float = 0.0
    local_hardware_lifespan_years: float = 0.0
    
    # Run parameters
    model: str = ""
    estimated_duration_seconds: int = 3600

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimationResponse:
    projected_cost_usd: float
    cost_driver: str  # vastai, aws, gcp, azure, local
    resource_impact: str  # low, medium, high
    local_energy_kwh: float = 0.0
    confidence_score: float = 1.0
    warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChainStateResponse:
    """State of a multi-agent chain (DAG) execution, suitable for dashboard rendering.

    `nodes` items: {id, agent_name, status, started_at, finished_at, error}
    `edges` items: {from, to}
    `overall_status` ∈ {pending, running, success, failed, skipped}
    """

    run_id: str
    nodes: list[dict]
    edges: list[dict]
    overall_status: str

    def to_dict(self) -> dict:
        return asdict(self)

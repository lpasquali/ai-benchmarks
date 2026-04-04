"""Stable contracts for CLI/API compatibility.

These dataclasses are transport-agnostic and define the operation payloads
used by the current CLI and future HTTP API backend.
"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOllamaInstanceRequest:
    vastai: bool
    template_hash: str
    min_dph: float
    max_dph: float
    reliability: float
    ollama_url: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunAgenticAgentRequest:
    question: str
    model: str
    ollama_url: str | None
    ollama_warmup: bool
    ollama_warmup_timeout: int
    kubeconfig: str
    agent: str = "holmes"

    @classmethod
    def from_cli(
        cls,
        *,
        question: str,
        model: str,
        ollama_url: str | None,
        ollama_warmup: bool,
        ollama_warmup_timeout: int,
        kubeconfig: Path,
        agent: str = "holmes",
    ) -> "RunAgenticAgentRequest":
        return cls(
            question=question,
            model=model,
            ollama_url=ollama_url,
            ollama_warmup=ollama_warmup,
            ollama_warmup_timeout=ollama_warmup_timeout,
            kubeconfig=str(kubeconfig),
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
    ollama_url: str | None
    question: str
    model: str
    ollama_warmup: bool
    ollama_warmup_timeout: int
    kubeconfig: str
    vastai_stop_instance: bool
    attestation_required: bool = False

    @classmethod
    def from_cli(
        cls,
        *,
        vastai: bool,
        template_hash: str,
        min_dph: float,
        max_dph: float,
        reliability: float,
        ollama_url: str | None,
        question: str,
        model: str,
        ollama_warmup: bool,
        ollama_warmup_timeout: int,
        kubeconfig: Path,
        vastai_stop_instance: bool,
        attestation_required: bool = False,
    ) -> "RunBenchmarkRequest":
        return cls(
            vastai=vastai,
            template_hash=template_hash,
            min_dph=min_dph,
            max_dph=max_dph,
            reliability=reliability,
            ollama_url=ollama_url,
            question=question,
            model=model,
            ollama_warmup=ollama_warmup,
            ollama_warmup_timeout=ollama_warmup_timeout,
            kubeconfig=str(kubeconfig),
            vastai_stop_instance=vastai_stop_instance,
            attestation_required=attestation_required,
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

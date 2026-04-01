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
    ) -> "RunAgenticAgentRequest":
        return cls(
            question=question,
            model=model,
            ollama_url=ollama_url,
            ollama_warmup=ollama_warmup,
            ollama_warmup_timeout=ollama_warmup_timeout,
            kubeconfig=str(kubeconfig),
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
        )

    def to_dict(self) -> dict:
        return asdict(self)

# SPDX-License-Identifier: Apache-2.0
"""Protocol definition for agentic runner implementations."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AgentConfig:
    """Per-agent configuration resolution block."""
    api_key: str | None = None
    base_url: str | None = None
    kubeconfig: str | None = None
    model: str | None = None
    backend_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Structured result from any agent.

    New drivers that want structured output can return an ``AgentResult``
    from a dedicated method while keeping the plain-string ``ask()`` path
    for backward compatibility.
    """

    answer: str
    result_type: str = "text"  # "text" | "image" | "structured" | "report"
    artifacts: list[dict] | None = None
    metadata: dict | None = None


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for agentic runner implementations across all domains.

    Implement this protocol to add a new agent framework alongside HolmesGPT.
    The ``ask()`` method returns a plain string for backward compatibility.
    """

    def ask(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> str:
        """Run an investigation query and return the answer as a string."""
        ...

    def ask_structured(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Run an investigation query and return a structured AgentResult."""
        ...

"""Protocol definition for agentic runner implementations."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run an investigation query and return the answer as a string."""
        ...

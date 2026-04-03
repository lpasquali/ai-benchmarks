"""Protocol definition for agentic runner implementations."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for agents that investigate Kubernetes clusters.

    Implement this protocol to add a new agent framework alongside HolmesGPT.
    """

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Run an investigation query and return the answer as a string."""
        ...

"""Protocol and credential types for LLM backend implementations."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCapabilities:
    """Generic LLM model capability summary."""

    model_name: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    raw: dict[str, Any] | None = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class BackendCredentials:
    """Generic credential container for any LLM backend.

    ``extra`` absorbs vendor-specific quirks (e.g. AWS region, Azure tenant ID)
    without polluting the base class with provider-specific fields.
    """

    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


class LLMBackend(Protocol):
    """Protocol for LLM backend clients.

    Implement this protocol to add a new LLM backend (e.g. OpenAI, AWS Bedrock)
    alongside the existing Ollama backend.
    """

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Return best-effort capability metadata for the given model."""
        ...

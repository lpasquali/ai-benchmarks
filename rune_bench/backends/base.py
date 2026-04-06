"""Protocol and credential types for LLM backend implementations."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backend clients.

    Implement this protocol to add a new LLM backend (e.g. OpenAI, AWS Bedrock)
    alongside the existing Ollama backend.
    """

    @property
    def base_url(self) -> str:
        """Return the normalized backend endpoint URL."""
        ...

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Return best-effort capability metadata for the given model."""
        ...

    def list_models(self) -> list[str]:
        """Return model names available on this backend."""
        ...

    def list_running_models(self) -> list[str]:
        """Return model names currently loaded/active on this backend."""
        ...

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize a provider-prefixed model name to the backend's native form."""
        ...

    def warmup(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
    ) -> str:
        """Load a model and wait until it is ready. Return the resolved model name."""
        ...

"""LLM resource provisioning interfaces and implementations."""

from .base import LLMResourceProvider, ProvisioningResult
from .existing_backend_provider import ExistingBackendProvider
from .existing_ollama_provider import ExistingOllamaProvider

__all__ = [
    "LLMResourceProvider",
    "ProvisioningResult",
    "ExistingBackendProvider",
    "ExistingOllamaProvider",
]

def __getattr__(name: str) -> object:
    if name == "VastAIProvider":
        from .vastai import VastAIProvider
        return VastAIProvider
    raise AttributeError(f"module 'rune_bench.resources' has no attribute {name!r}")

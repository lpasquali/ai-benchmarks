"""LLM backend protocol, registry, and factory.

Backends define HOW to communicate with an LLM — handling auth, model
capability discovery, and inference calls — independent of WHERE the LLM
runs (resources/) or WHAT the agent does with it (agents/).

The registry supports two kinds of backends:

1. **Built-in backends** listed in :data:`_BUILTIN_BACKENDS` -- resolved via
   :func:`importlib.import_module` the first time they are requested.
2. **Custom backends** added at runtime via :func:`register_backend`.

Custom registrations take precedence over built-in entries so that
downstream integrations can override the default implementation of any
backend without modifying this module.
"""

from __future__ import annotations

import importlib

from .base import BackendCredentials, LLMBackend, ModelCapabilities
from .ollama import OllamaBackend, OllamaClient, OllamaModelCapabilities, OllamaModelManager

_BACKEND_REGISTRY: dict[str, type] = {}

_BUILTIN_BACKENDS: dict[str, tuple[str, str]] = {
    "ollama": ("rune_bench.backends.ollama", "OllamaBackend"),
}


def register_backend(name: str, cls: type) -> None:
    """Register a custom backend class under *name*.

    Custom registrations shadow built-in entries so callers can override
    the default implementation at runtime.
    """
    _BACKEND_REGISTRY[name] = cls


def get_backend(backend_type: str, base_url: str, **kwargs: object) -> LLMBackend:
    """Resolve and instantiate an LLM backend by type.

    Resolution order:
    1. Custom registry (populated by :func:`register_backend`).
    2. Built-in map (lazy ``importlib.import_module``).

    Raises:
        ValueError: if *backend_type* is not found in either source.
    """
    if backend_type in _BACKEND_REGISTRY:
        return _BACKEND_REGISTRY[backend_type](base_url, **kwargs)  # type: ignore[return-value]

    if backend_type in _BUILTIN_BACKENDS:
        module_path, class_name = _BUILTIN_BACKENDS[backend_type]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(base_url, **kwargs)  # type: ignore[return-value]

    available = sorted(set(list(_BACKEND_REGISTRY.keys()) + list(_BUILTIN_BACKENDS.keys())))
    raise ValueError(
        f"Unknown backend type {backend_type!r}. Available: {', '.join(available)}"
    )


def list_backends() -> list[str]:
    """Return a sorted list of all known backend type names."""
    return sorted(set(list(_BACKEND_REGISTRY.keys()) + list(_BUILTIN_BACKENDS.keys())))


__all__ = [
    "BackendCredentials",
    "LLMBackend",
    "ModelCapabilities",
    "OllamaBackend",
    "OllamaClient",
    "OllamaModelCapabilities",
    "OllamaModelManager",
    "get_backend",
    "list_backends",
    "register_backend",
]

# SPDX-License-Identifier: Apache-2.0
from rune_bench.backends.ollama import OllamaBackend
from dataclasses import dataclass


@dataclass
class ExistingOllamaServer:
    url: str
    model_name: str


def normalize_backend_url(backend_url: str | None) -> str:
    """Validate and normalize a backend base URL."""
    return OllamaBackend.normalize_url(backend_url)


def use_existing_backend_server(
    backend_url: str | None, model_name: str
) -> ExistingOllamaServer:
    """Resolve an existing backend server target."""
    return ExistingOllamaServer(
        url=normalize_backend_url(backend_url), model_name=model_name
    )


def list_backend_models(backend_url: str | None) -> list[str]:
    """Return available model names from an existing backend server."""
    url = normalize_backend_url(backend_url)
    backend = OllamaBackend(url)
    return backend.list_models()


def list_running_backend_models(backend_url: str | None) -> list[str]:
    """Return model names currently loaded in memory on an existing backend server."""
    url = normalize_backend_url(backend_url)
    backend = OllamaBackend(url)
    return backend.list_running_models()


def normalize_backend_model_for_api(model_name: str) -> str:
    """Convert provider-prefixed model identifiers into plain backend model names."""
    backend = OllamaBackend("http://localhost:11434")  # URL not used for normalization
    return backend.normalize_model_name(model_name)


def warmup_backend_model(
    backend_url: str | None,
    model_name: str,
    *,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 2.0,
    keep_alive: str = "30m",
) -> str:
    """Load a model into an existing backend server and wait until it is running."""
    url = normalize_backend_url(backend_url)
    backend = OllamaBackend(url)
    return backend.warmup(
        model_name,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        keep_alive=keep_alive,
    )

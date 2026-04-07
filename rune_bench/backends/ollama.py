# SPDX-License-Identifier: Apache-2.0
"""Ollama LLM backend — HTTP client and high-level model manager.

Consolidates the low-level API client (OllamaClient) and the high-level
model lifecycle manager (OllamaModelManager) into a single backend module.
The :class:`OllamaBackend` facade exposes every Ollama-specific operation
so that the workflow layer remains backend-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rune_bench.backends.base import ModelCapabilities
from rune_bench.common import make_http_request, normalize_url
from rune_bench.debug import debug_log
from rune_bench.metrics import span

# ---------------------------------------------------------------------------
# Public type alias — keeps existing callsites using OllamaModelCapabilities
# working without changes.
# ---------------------------------------------------------------------------
OllamaModelCapabilities = ModelCapabilities


@dataclass
class OllamaClient:
    """HTTP client for interacting with an Ollama server.

    Attributes:
        base_url: Normalized Ollama server URL (e.g., http://localhost:11434)
    """

    base_url: str

    def __post_init__(self) -> None:
        """Validate and normalize the base URL."""
        try:
            self.base_url = normalize_url(self.base_url, service_name="Ollama")
        except RuntimeError:
            raise RuntimeError(
                "Missing or invalid Ollama URL. Provide --backend-url when --vastai is not enabled. "
                "Expected format like http://host:11434"
            )

    def _make_request(self, endpoint: str, *, method: str, payload: dict | None, action: str) -> dict:
        url = self.base_url.rstrip("/") + endpoint
        return make_http_request(
            url,
            method=method,
            payload=payload,
            action=action,
            timeout_seconds=30,
            debug_prefix="Ollama API",
        )

    def get_available_models(self) -> list[str]:
        """List all models available on the Ollama server."""
        data = self._make_request("/api/tags", method="GET", payload=None, action="query available models")
        models = data.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"Ollama server at {self.base_url} returned an unexpected /api/tags payload")
        names = [name for item in models if isinstance(item, dict) and isinstance((name := item.get("name")), str)]
        return sorted(names)

    def get_running_models(self) -> set[str]:
        """List all models currently loaded in memory on the Ollama server."""
        data = self._make_request("/api/ps", method="GET", payload=None, action="list running models")
        models = data.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"Ollama server at {self.base_url} returned an unexpected /api/ps payload")
        names = set()
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
        return names

    def get_model_capabilities(self, model_name: str) -> OllamaModelCapabilities:
        """Return best-effort context/output capabilities for a model."""
        data = self._make_request(
            "/api/show",
            method="POST",
            payload={"model": model_name},
            action=f"inspect model {model_name}",
        )
        model_info = data.get("model_info")
        context_window = None
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if not isinstance(key, str):
                    continue
                if key == "context_length" or key.endswith(".context_length"):
                    try:
                        context_window = int(value)
                        break
                    except (TypeError, ValueError):
                        continue

        max_output_tokens = None
        if context_window and context_window > 0:
            max_output_tokens = min(64000, max(1024, context_window // 5))

        debug_log(
            f"Ollama model capabilities: model={model_name} context_window={context_window} "
            f"derived_max_output_tokens={max_output_tokens}"
        )
        return OllamaModelCapabilities(
            model_name=model_name,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            raw=data,
        )

    def load_model(self, model_name: str, *, keep_alive: str = "30m") -> None:
        """Load a model into memory on the Ollama server."""
        payload = {"model": model_name, "prompt": "", "stream": False, "keep_alive": keep_alive}
        self._make_request("/api/generate", method="POST", payload=payload, action=f"load model {model_name}")

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory on the Ollama server."""
        payload = {"model": model_name, "prompt": "", "stream": False, "keep_alive": 0}
        self._make_request("/api/generate", method="POST", payload=payload, action=f"unload model {model_name}")


@dataclass
class OllamaModelManager:
    """High-level manager for Ollama models on a specific server."""

    client: OllamaClient

    @classmethod
    def create(cls, base_url: str) -> "OllamaModelManager":
        """Create a new manager for an Ollama server at the given URL."""
        return cls(client=OllamaClient(base_url))

    def list_available_models(self) -> list[str]:
        """List all models available on the server."""
        return self.client.get_available_models()

    def list_running_models(self) -> list[str]:
        """List all models currently loaded in memory."""
        return sorted(self.client.get_running_models())

    def warmup_model(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
        unload_others: bool = True,
    ) -> str:
        """Load a model into memory and wait until it is ready."""
        if unload_others:
            self._unload_conflicting_models(model_name)
        self.client.load_model(model_name, keep_alive=keep_alive)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            running = self.client.get_running_models()
            if self._model_in_running(model_name, running):
                return model_name
            time.sleep(poll_interval_seconds)
        raise RuntimeError(
            f"Timed out waiting for Ollama model {model_name} to become ready at {self.client.base_url}"
        )

    def _unload_conflicting_models(self, target_model: str) -> None:
        running = self.client.get_running_models()
        for model in sorted(name for name in running if not self._model_in_running(target_model, {name})):
            self.client.unload_model(model)

    @staticmethod
    def _model_in_running(requested: str, running: set[str]) -> bool:
        """Return True when *requested* matches any entry in *running*.

        Ollama normalises untagged model names by appending ``:latest``
        (e.g. ``tinyllama`` → ``tinyllama:latest``) in ``/api/ps``.
        This helper checks both the bare name and the ``:latest`` variant so
        callers are not required to use the fully-qualified form.
        """
        if requested in running:
            return True
        if ":" not in requested and f"{requested}:latest" in running:
            return True
        return False

    def normalize_model_name(self, model_name: str) -> str:
        """Convert provider-prefixed model identifiers to plain Ollama names."""
        normalized = model_name.strip()
        for prefix in ("ollama/", "ollama_chat/"):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix)
        return normalized


class OllamaBackend:
    """Facade over OllamaClient + OllamaModelManager implementing LLMBackend.

    Provides every Ollama-specific operation that the workflow layer needs,
    keeping ``workflows.py`` free of Ollama implementation details.
    """

    def __init__(self, base_url: str) -> None:
        self._manager = OllamaModelManager.create(base_url)

    @property
    def base_url(self) -> str:
        """Return the normalized Ollama server URL."""
        return self._manager.client.base_url

    # -- URL helpers --------------------------------------------------------

    @staticmethod
    def normalize_url(url: str | None) -> str:
        """Validate and normalize an Ollama base URL.

        Adds ``http://`` when missing.  Raises ``RuntimeError`` when *url* is
        ``None``.
        """
        if url is None:
            raise RuntimeError("Missing Ollama URL")
        client = OllamaClient(url)
        return client.base_url

    @staticmethod
    def extract_service_url(details: Any) -> str | None:
        """Return the Ollama endpoint from Vast.ai connection details.

        Scans *details.service_urls* for port 11434 (Ollama default).
        """
        for svc in details.service_urls:
            direct = str(svc.get("direct", ""))
            proxy = str(svc.get("proxy", "")) if svc.get("proxy") else ""
            if ":11434" in direct:
                return direct
            if ":11434" in proxy:
                return proxy
        return None

    # -- Server / model operations -----------------------------------------

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Return best-effort capability metadata for the given model."""
        return self._manager.client.get_model_capabilities(model)

    def list_models(self) -> list[str]:
        """Return model names available on this backend."""
        return self._manager.list_available_models()

    def list_running_models(self) -> list[str]:
        """Return model names currently loaded/active on this backend."""
        return self._manager.list_running_models()

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize a provider-prefixed model name to the backend's native form."""
        return self._manager.normalize_model_name(model_name)

    def warmup(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
    ) -> str:
        """Load a model and wait until it is ready. Return the resolved model name."""
        api_model_name = self._manager.normalize_model_name(model_name)
        debug_log(
            f"OllamaBackend warmup: base_url={self.base_url} "
            f"requested_model={model_name} api_model={api_model_name}"
        )
        with span("ollama.model.warmup", model=api_model_name):
            return self._manager.warmup_model(
                api_model_name,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                keep_alive=keep_alive,
                unload_others=True,
            )

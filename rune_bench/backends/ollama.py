"""Ollama LLM backend — HTTP client and high-level model manager.

Consolidates the low-level API client (OllamaClient) and the high-level
model lifecycle manager (OllamaModelManager) into a single backend module.
"""

import time
from dataclasses import dataclass

from rune_bench.backends.base import ModelCapabilities
from rune_bench.common import make_http_request, normalize_url
from rune_bench.debug import debug_log

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
                "Missing or invalid Ollama URL. Provide --ollama-url when --vastai is not enabled. "
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
            if model_name in running:
                return model_name
            time.sleep(poll_interval_seconds)
        raise RuntimeError(
            f"Timed out waiting for Ollama model {model_name} to become ready at {self.client.base_url}"
        )

    def _unload_conflicting_models(self, target_model: str) -> None:
        running = self.client.get_running_models()
        for model in sorted(name for name in running if name != target_model):
            self.client.unload_model(model)

    def normalize_model_name(self, model_name: str) -> str:
        """Convert provider-prefixed model identifiers to plain Ollama names."""
        normalized = model_name.strip()
        for prefix in ("ollama/", "ollama_chat/"):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix)
        return normalized

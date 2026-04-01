"""Low-level HTTP client for Ollama API.

Handles URL validation, request formatting, and error handling.
"""

from dataclasses import dataclass
from typing import Any

from rune_bench.common import make_http_request, normalize_url
from rune_bench.debug import debug_log


@dataclass
class OllamaModelCapabilities:
    """Best-effort model capability summary derived from Ollama metadata."""

    model_name: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    raw: dict[str, Any] | None = None


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
        """Execute an HTTP request to the Ollama API.
        
        Args:
            endpoint: API endpoint path (e.g., /api/tags, /api/generate)
            method: HTTP method (GET, POST, etc.)
            payload: Optional JSON payload for POST requests
            action: Human-readable description of the operation (for error messages)
            
        Returns:
            Parsed JSON response as dict
            
        Raises:
            RuntimeError: If the request fails or response is invalid
        """
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
        """List all models available on the Ollama server.
        
        Returns:
            Sorted list of model names from /api/tags
            
        Raises:
            RuntimeError: If the query fails or response is invalid
        """
        data = self._make_request(
            "/api/tags",
            method="GET",
            payload=None,
            action="query available models",
        )

        models = data.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"Ollama server at {self.base_url} returned an unexpected /api/tags payload")

        names = [item.get("name") for item in models if isinstance(item, dict) and isinstance(item.get("name"), str)]
        return sorted(names)

    def get_running_models(self) -> set[str]:
        """List all models currently loaded in memory on the Ollama server.
        
        Returns:
            Set of currently running model names from /api/ps
            
        Raises:
            RuntimeError: If the query fails or response is invalid
        """
        data = self._make_request(
            "/api/ps",
            method="GET",
            payload=None,
            action="list running models",
        )

        models = data.get("models")
        if not isinstance(models, list):
            raise RuntimeError(f"Ollama server at {self.base_url} returned an unexpected /api/ps payload")

        names = set()
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.add(item["name"])
        return names

    def get_model_capabilities(self, model_name: str) -> OllamaModelCapabilities:
        """Return best-effort context/output capabilities for a model.

        Uses `/api/show` metadata. `context_window` is read from model metadata when
        present. `max_output_tokens` is derived conservatively when Ollama does not
        expose an explicit output limit.
        """
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
        """Load a model into memory on the Ollama server.
        
        Uses an empty prompt to trigger model loading without generating output.
        
        Args:
            model_name: Plain Ollama model name (without provider prefix)
            keep_alive: Duration to keep model in memory (default "30m")
            
        Raises:
            RuntimeError: If the load operation fails
        """
        payload = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        self._make_request(
            "/api/generate",
            method="POST",
            payload=payload,
            action=f"load model {model_name}",
        )

    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory on the Ollama server.
        
        Uses keep_alive=0 to immediately unload the model.
        
        Args:
            model_name: Plain Ollama model name (without provider prefix)
            
        Raises:
            RuntimeError: If the unload operation fails
        """
        payload = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        self._make_request(
            "/api/generate",
            method="POST",
            payload=payload,
            action=f"unload model {model_name}",
        )

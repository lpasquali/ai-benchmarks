"""Low-level HTTP client for Ollama API.

Handles URL validation, request formatting, and error handling.
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
        self.base_url = self._normalize_url(self.base_url)

    @staticmethod
    def _normalize_url(ollama_url: str | None) -> str:
        """Validate and normalize an Ollama base URL.
        
        Adds ``http://`` when missing.
        
        Raises:
            RuntimeError: If URL is missing, invalid scheme, or missing host.
        """
        if not ollama_url:
            raise RuntimeError(
                "Missing Ollama URL. Provide --ollama-url when --vastai is not enabled."
            )

        # Normalize: if no recognized scheme, prepend http://
        parsed = urlparse(ollama_url)
        if parsed.scheme not in {"http", "https"}:
            # No valid scheme found; treat entire input as host:port
            ollama_url = f"http://{ollama_url}"
            parsed = urlparse(ollama_url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(
                "Invalid --ollama-url. Expected format like http://host:11434"
            )

        return ollama_url

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
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=data, headers=headers, method=method)

        debug_log(
            f"Ollama API request: method={method} url={url} action={action} "
            f"payload={json.dumps(payload, sort_keys=True) if payload is not None else '<none>'}"
        )

        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                debug_log(
                    f"Ollama API response: method={method} url={url} status={getattr(response, 'status', '<unknown>')} "
                    f"body={raw}"
                )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            debug_log(f"Ollama API HTTP error: method={method} url={url} status={exc.code} detail={detail}")
            if detail:
                raise RuntimeError(f"Failed to {action} via Ollama API {url}: {detail}") from exc
            raise RuntimeError(f"Failed to {action} via Ollama API {url}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            debug_log(f"Ollama API transport error: method={method} url={url} error={exc}")
            raise RuntimeError(f"Failed to {action} via Ollama API {url}: {exc}") from exc

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama API {url} returned invalid JSON while attempting to {action}") from exc

        if not isinstance(result, dict):
            raise RuntimeError(f"Ollama API {url} returned an unexpected JSON payload while attempting to {action}")
        return result

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

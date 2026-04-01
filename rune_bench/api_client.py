"""HTTP client for RUNE API backend mode."""

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


@dataclass
class RuneApiClient:
    base_url: str

    def __post_init__(self) -> None:
        self.base_url = self._normalize_url(self.base_url)

    @staticmethod
    def _normalize_url(url: str | None) -> str:
        if not url:
            raise RuntimeError("Missing API base URL. Set --api-base-url or RUNE_API_BASE_URL.")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            url = f"http://{url}"
            parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("Invalid API base URL. Expected format like http://host:8080")

        return url.rstrip("/")

    def _request(self, method: str, path: str, *, query: dict[str, str] | None = None) -> dict:
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)

        request = Request(url, method=method)

        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                raise RuntimeError(f"API request failed {method} {url}: {detail}") from exc
            raise RuntimeError(f"API request failed {method} {url}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"API request failed {method} {url}: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned invalid JSON for {method} {url}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"API returned unexpected payload for {method} {url}")

        return payload

    def get_vastai_models(self) -> list[dict]:
        payload = self._request("GET", "/v1/catalog/vastai-models")
        models = payload.get("models")
        if not isinstance(models, list):
            raise RuntimeError("API payload missing 'models' list for Vast.ai model catalog")
        return [m for m in models if isinstance(m, dict)]

    def get_ollama_models(self, ollama_url: str) -> dict:
        payload = self._request("GET", "/v1/ollama/models", query={"ollama_url": ollama_url})
        if not isinstance(payload.get("models"), list):
            raise RuntimeError("API payload missing 'models' list for Ollama models endpoint")
        if not isinstance(payload.get("running_models"), list):
            raise RuntimeError("API payload missing 'running_models' list for Ollama models endpoint")
        return payload

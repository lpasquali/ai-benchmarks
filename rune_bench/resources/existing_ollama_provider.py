"""Existing Ollama server LLM resource provider."""

from rune_bench.resources.base import ProvisioningResult
from rune_bench.workflows import use_existing_ollama_server, warmup_existing_ollama_model


class ExistingOllamaProvider:
    """LLMResourceProvider for a pre-existing Ollama server.

    Optionally warms up the specified model during :meth:`provision`.
    :meth:`teardown` is a no-op — the external server is left running.
    """

    def __init__(
        self,
        ollama_url: str | None,
        *,
        model: str | None = None,
        warmup: bool = False,
        warmup_timeout: int = 120,
    ) -> None:
        self._ollama_url = ollama_url
        self._model = model
        self._warmup = warmup
        self._warmup_timeout = warmup_timeout

    def provision(self) -> ProvisioningResult:
        server = use_existing_ollama_server(
            self._ollama_url,
            model_name=self._model or "<user-selected>",
        )
        if self._warmup and self._model:
            warmup_existing_ollama_model(
                server.url,
                self._model,
                timeout_seconds=self._warmup_timeout,
            )
        return ProvisioningResult(ollama_url=server.url, model=self._model)

    def teardown(self, result: ProvisioningResult) -> None:
        pass

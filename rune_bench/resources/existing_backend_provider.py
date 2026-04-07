"""Existing LLM backend server resource provider."""

from rune_bench.resources.base import ProvisioningResult
from rune_bench.workflows import use_existing_backend_server, warmup_backend_model


class ExistingBackendProvider:
    """LLMResourceProvider for a pre-existing LLM backend server.

    Optionally warms up the specified model during :meth:`provision`.
    :meth:`teardown` is a no-op — the external server is left running.
    """

    def __init__(
        self,
        backend_url: str | None,
        *,
        model: str | None = None,
        warmup: bool = False,
        warmup_timeout: int = 120,
        backend_type: str = "ollama",
    ) -> None:
        self._backend_url = backend_url
        self._model = model
        self._warmup = warmup
        self._warmup_timeout = warmup_timeout
        self._backend_type = backend_type

    def provision(self) -> ProvisioningResult:
        server = use_existing_backend_server(
            self._backend_url,
            model_name=self._model or "<user-selected>",
        )
        if self._warmup and self._model:
            warmup_backend_model(
                server.url,
                self._model,
                timeout_seconds=self._warmup_timeout,
            )
        return ProvisioningResult(
            backend_url=server.url,
            model=self._model,
            backend_type=self._backend_type,
        )

    def teardown(self, result: ProvisioningResult) -> None:
        pass

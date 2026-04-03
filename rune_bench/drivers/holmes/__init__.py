"""Holmes driver client — delegates HolmesGPT queries to the holmes driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.holmes.__main__``) calls ``python -m holmes.main ask``
and therefore only requires holmesgpt to be installed in the *subprocess*
environment — not in the rune core process.
"""

from __future__ import annotations

from pathlib import Path

from rune_bench.backends.ollama import OllamaClient, OllamaModelManager
from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class HolmesDriverClient:
    """Investigate a Kubernetes cluster by delegating to the holmes driver process.

    The public interface is identical to the old ``HolmesRunner`` so existing
    call-sites in :mod:`rune_bench.api_backend` and the CLI require no changes.
    """

    def __init__(
        self,
        kubeconfig: Path,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig
        self._transport: DriverTransport = transport or make_driver_transport("holmes")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Dispatch a question to the holmes driver and return the answer.

        Fetches Ollama model capability limits (context window, max output tokens)
        when *ollama_url* is provided and passes them to the driver so it can set
        the appropriate environment overrides before calling HolmesGPT.

        Args:
            question: Natural-language question about the Kubernetes cluster.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            ollama_url: Base URL of the Ollama server (optional).

        Returns:
            HolmesGPT's textual answer.
        """
        resolved_model = model.strip()
        params: dict = {
            "question": question,
            "model": resolved_model,
            "kubeconfig_path": str(self._kubeconfig),
        }
        if ollama_url:
            params["ollama_url"] = ollama_url
            params.update(self._fetch_model_limits(model=resolved_model, ollama_url=ollama_url))

        debug_log(
            f"HolmesDriverClient.ask: question={question!r} model={resolved_model!r} "
            f"ollama_url={ollama_url or '<none>'}"
        )
        result = self._transport.call("ask", params)
        return str(result.get("answer", ""))

    def _fetch_model_limits(self, *, model: str, ollama_url: str) -> dict:
        """Return context_window / max_output_tokens for *model*, or ``{}`` on error."""
        try:
            normalized = OllamaModelManager.create(ollama_url).normalize_model_name(model)
            caps = OllamaClient(ollama_url).get_model_capabilities(normalized)
        except Exception as exc:  # noqa: BLE001
            debug_log(f"Could not fetch model limits for {model!r}: {exc}")
            return {}

        limits: dict = {}
        if caps.context_window:
            limits["context_window"] = caps.context_window
        if caps.max_output_tokens:
            limits["max_output_tokens"] = caps.max_output_tokens
        return limits

"""Metoro driver client -- delegates eBPF observability queries to the metoro driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.metoro.__main__``) calls the Metoro REST API and therefore
only requires network access to the Metoro instance -- not any local SDK.
"""

from __future__ import annotations

from pathlib import Path

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class MetoroDriverClient:
    """Investigate a Kubernetes cluster via Metoro eBPF observability.

    The public interface mirrors :class:`~rune_bench.drivers.holmes.HolmesDriverClient`
    so existing call-sites require no changes.
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
        self._transport: DriverTransport = transport or make_driver_transport("metoro")

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Dispatch a question to the Metoro driver and return the answer.

        Args:
            question: Natural-language question about the Kubernetes cluster.
            model: Model identifier (not forwarded to Metoro API; kept for interface compatibility).
            backend_url: Ollama server URL (not forwarded; kept for interface compatibility).

        Returns:
            Metoro's textual explanation.
        """
        params: dict = {
            "question": question,
            "kubeconfig_path": str(self._kubeconfig),
        }

        debug_log(
            f"MetoroDriverClient.ask: question={question!r} model={model!r}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Metoro driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Metoro driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Metoro driver returned an empty answer.")

        return answer_text


MetoroRunner = MetoroDriverClient

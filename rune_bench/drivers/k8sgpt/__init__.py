"""K8sGPT driver client — delegates k8sgpt analysis queries to the k8sgpt driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.k8sgpt.__main__``) calls ``k8sgpt analyze`` and therefore
only requires the k8sgpt binary to be installed in the *subprocess* environment
— not in the rune core process.
"""

from __future__ import annotations

from pathlib import Path

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class K8sGPTDriverClient:
    """Scan a Kubernetes cluster for issues by delegating to the k8sgpt driver process.

    The public interface mirrors :class:`~rune_bench.drivers.holmes.HolmesDriverClient`
    so that SRE agent call-sites can use either backend interchangeably.
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
        self._transport: DriverTransport = transport or make_driver_transport("k8sgpt")

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Dispatch an analysis request to the k8sgpt driver and return the answer.

        Args:
            question: Natural-language question or resource-kind hint
                      (e.g. ``"Pod"``, ``"Service"``, ``"Why is my pod failing?"``).
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            backend_url: Base URL of the Ollama server (optional).

        Returns:
            K8sGPT's textual analysis or ``"No issues detected"`` when the
            cluster is healthy.
        """
        resolved_model = model.strip()
        params: dict = {
            "question": question,
            "model": resolved_model,
            "kubeconfig_path": str(self._kubeconfig),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"K8sGPTDriverClient.ask: question={question!r} model={resolved_model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("K8sGPT driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("K8sGPT driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("K8sGPT driver returned an empty answer.")

        return answer_text

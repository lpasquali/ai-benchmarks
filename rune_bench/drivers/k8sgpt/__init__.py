# SPDX-License-Identifier: Apache-2.0
"""K8sGPT driver client — delegates k8sgpt analysis queries to the k8sgpt driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.k8sgpt.__main__``) calls ``k8sgpt analyze`` and therefore
only requires the k8sgpt binary to be installed in the *subprocess* environment
— not in the rune core process.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

from pathlib import Path

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, AsyncDriverTransport, make_driver_transport, make_async_driver_transport


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
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("k8sgpt")

    def ask(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> str:
        """Dispatch a question to the driver and return the answer string."""
        return self.ask_structured(
            question=question,
            model=model,
            backend_url=backend_url,
            backend_type=backend_type,
        ).answer

    def ask_structured(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Dispatch a question to the driver and return a structured AgentResult.

        Dispatch an analysis request to the k8sgpt driver and return the answer.

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

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )

    async def ask_async(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Dispatch a question to the driver asynchronously."""
        resolved_model = model.strip()
        params: dict = {
            "question": question,
            "model": resolved_model,
        }
        if backend_url:
            params["backend_url"] = backend_url
            if hasattr(self, "_fetch_model_limits"):
                params.update(self._fetch_model_limits(
                    model=resolved_model, backend_url=backend_url, backend_type=backend_type,
                ))

        debug_log(
            f"{self.__class__.__name__}.ask_async: question={question!r} model={resolved_model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = await self._async_transport.call_async("ask", params)

        if "answer" not in result:
            raise RuntimeError("Driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )


K8sGPTRunner = K8sGPTDriverClient

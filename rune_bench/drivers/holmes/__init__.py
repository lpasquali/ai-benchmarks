# SPDX-License-Identifier: Apache-2.0
"""Holmes driver client — delegates HolmesGPT queries to the holmes driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.holmes.__main__``) calls ``python -m holmes.main ask``
and therefore only requires holmesgpt to be installed in the *subprocess*
environment — not in the rune core process.
"""

from __future__ import annotations

from pathlib import Path

from rune_bench.agents.base import AgentResult
from rune_bench.backends import get_backend
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

    def ask(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> str:
        """Dispatch a question to the holmes driver and return the answer string."""
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
        """Dispatch a question to the holmes driver and return a structured AgentResult."""
        resolved_model = model.strip()
        params: dict = {
            "question": question,
            "model": resolved_model,
            "kubeconfig_path": str(self._kubeconfig),
        }
        if backend_url:
            params["backend_url"] = backend_url
            params.update(self._fetch_model_limits(
                model=resolved_model, backend_url=backend_url, backend_type=backend_type,
            ))

        debug_log(
            f"HolmesDriverClient.ask: question={question!r} model={resolved_model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Holmes driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Holmes driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Holmes driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )

    def _fetch_model_limits(
        self, *, model: str, backend_url: str, backend_type: str = "ollama",
    ) -> dict:
        """Return context_window / max_output_tokens for *model*, or ``{}`` on error."""
        try:
            backend = get_backend(backend_type, backend_url)
            normalized = backend.normalize_model_name(model)
            caps = backend.get_model_capabilities(normalized)
        except Exception as exc:  # noqa: BLE001
            debug_log(f"Could not fetch model limits for {model!r}: {exc}")
            return {}

        limits: dict = {}
        if caps.context_window:
            limits["context_window"] = caps.context_window
        if caps.max_output_tokens:
            limits["max_output_tokens"] = caps.max_output_tokens
        return limits

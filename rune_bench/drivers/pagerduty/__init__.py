# SPDX-License-Identifier: Apache-2.0
"""PagerDuty driver client -- delegates incident triage queries to the pagerduty driver process.

This is a hybrid agent: PagerDuty REST API for data retrieval plus Ollama for
triage synthesis.  The driver process
(``rune_bench.drivers.pagerduty.__main__``) calls the PagerDuty REST v2 API
via :func:`~rune_bench.common.http_client.make_http_request` and therefore requires no external dependencies
beyond a valid ``RUNE_PAGERDUTY_API_KEY`` env var.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

from pathlib import Path

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class PagerDutyDriverClient:
    """Fetch and triage PagerDuty incidents by delegating to the pagerduty driver.

    The public interface mirrors :class:`~rune_bench.drivers.holmes.HolmesDriverClient`
    so existing call-sites in :mod:`rune_bench.api_backend` and the CLI require
    no changes.
    """

    def __init__(
        self,
        kubeconfig: Path | None = None,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        if kubeconfig is not None and not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig
        self._transport: DriverTransport = transport or make_driver_transport("pagerduty")

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
        """Dispatch a question to the driver and return a structured AgentResult."""
        """Dispatch a triage question to the pagerduty driver and return the answer.

        Args:
            question: Natural-language triage question.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            backend_url: Base URL of the Ollama server (optional).

        Returns:
            A triage summary (LLM-synthesised when Ollama is available) or
            formatted raw incident data.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if self._kubeconfig is not None:
            params["kubeconfig_path"] = str(self._kubeconfig)
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"PagerDutyDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("PagerDuty driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("PagerDuty driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("PagerDuty driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )

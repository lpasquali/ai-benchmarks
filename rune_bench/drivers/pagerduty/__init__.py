"""PagerDuty driver client -- delegates incident triage queries to the pagerduty driver process.

This is a hybrid agent: PagerDuty REST API for data retrieval plus Ollama for
triage synthesis.  The driver process
(``rune_bench.drivers.pagerduty.__main__``) calls the PagerDuty REST v2 API
via :mod:`urllib.request` and therefore requires no external dependencies
beyond a valid ``RUNE_PAGERDUTY_API_KEY`` env var.
"""

from __future__ import annotations

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
        kubeconfig: Path,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig
        self._transport: DriverTransport = transport or make_driver_transport("pagerduty")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Dispatch a triage question to the pagerduty driver and return the answer.

        Args:
            question: Natural-language triage question.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            ollama_url: Base URL of the Ollama server (optional).

        Returns:
            A triage summary (LLM-synthesised when Ollama is available) or
            formatted raw incident data.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if ollama_url:
            params["ollama_url"] = ollama_url

        debug_log(
            f"PagerDutyDriverClient.ask: question={question!r} model={model!r} "
            f"ollama_url={ollama_url or '<none>'}"
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

        return answer_text

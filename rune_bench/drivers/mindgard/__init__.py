# SPDX-License-Identifier: Apache-2.0
"""Mindgard driver client — delegates AI red-teaming to the mindgard driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.mindgard.__main__``) invokes the ``mindgard`` CLI and
therefore only requires the mindgard package to be installed in the *subprocess*
environment — not in the rune core process.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class MindgardDriverClient:
    """Run AI red-teaming assessments by delegating to the mindgard driver process.

    The public interface mirrors the old ``MindgardRunner`` so existing call-sites
    require no changes.

    .. note::

        Unlike most drivers, ``backend_url`` here identifies the model endpoint
        being **attacked** (the target under test), not a backend LLM.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("mindgard")

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
        """Dispatch a red-team assessment to the mindgard driver and return findings.

        Args:
            question: Objective or prompt for the red-team assessment.
            model: Model identifier of the target under test.
            backend_url: Base URL of the model endpoint being **attacked**.

        Returns:
            Formatted text summarising risk scores and vulnerabilities.
        """
        params: dict = {
            "question": question,
            "model": model,
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"MindgardDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Mindgard driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Mindgard driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Mindgard driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )

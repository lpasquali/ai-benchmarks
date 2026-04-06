"""Glean driver client -- delegates enterprise search queries to the glean driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.glean.__main__``) calls the Glean REST API and therefore
only requires network access to the Glean instance -- not any local SDK.
"""

from __future__ import annotations

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class GleanDriverClient:
    """Research agent: autonomous internal knowledge discovery via Glean enterprise search.

    Unlike SRE drivers, Glean does not require a kubeconfig.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("glean")

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Dispatch a question to the Glean driver and return the answer.

        Args:
            question: Natural-language research question.
            model: Model identifier (unused -- Glean uses its own hosted model).
            backend_url: Ollama server URL (unused, kept for interface compatibility).

        Returns:
            Glean's synthesised answer with source citations.
        """
        params: dict = {
            "question": question,
        }

        debug_log(
            f"GleanDriverClient.ask: question={question!r}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Glean driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Glean driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Glean driver returned an empty answer.")

        return answer_text


GleanRunner = GleanDriverClient

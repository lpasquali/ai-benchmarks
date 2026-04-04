"""CrewAI driver client — delegates ops queries to the crewai driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.crewai.__main__``) imports CrewAI directly, so the
package only needs to be installed in the *subprocess* environment — not in
the rune core process.
"""

from __future__ import annotations

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class CrewAIDriverClient:
    """Orchestrate multi-agent ops tasks via CrewAI.

    Unlike the Holmes driver, CrewAI does not require a kubeconfig — it is a
    pure-Python framework that uses Ollama as its LLM backend.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("crewai")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Dispatch a question to the CrewAI driver and return the answer.

        Args:
            question: Natural-language ops/analysis question.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            ollama_url: Base URL of the Ollama server (optional).

        Returns:
            The CrewAI workflow's textual answer.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if ollama_url:
            params["ollama_url"] = ollama_url

        debug_log(
            f"CrewAIDriverClient.ask: question={question!r} model={model!r} "
            f"ollama_url={ollama_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("CrewAI driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("CrewAI driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("CrewAI driver returned an empty answer.")

        return answer_text

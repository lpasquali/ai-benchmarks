"""LangGraph driver client — delegates research queries to the langgraph driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.langgraph.__main__``) imports LangGraph and langchain_ollama
directly, so these packages only need to be installed in the *subprocess*
environment — not in the rune core process.
"""

from __future__ import annotations

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class LangGraphDriverClient:
    """Run stateful multi-agent research flows via LangGraph.

    Unlike the Holmes driver, LangGraph does not require a kubeconfig — it is a
    pure-Python framework that uses Ollama as its LLM backend.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("langgraph")

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Dispatch a question to the LangGraph driver and return the answer.

        Args:
            question: Natural-language research question.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            backend_url: Base URL of the Ollama server (optional).

        Returns:
            The LangGraph research workflow's textual answer.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"LangGraphDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("LangGraph driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("LangGraph driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("LangGraph driver returned an empty answer.")

        return answer_text

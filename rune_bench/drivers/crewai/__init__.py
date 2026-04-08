# SPDX-License-Identifier: Apache-2.0
"""CrewAI driver client — delegates ops queries to the crewai driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.crewai.__main__``) imports CrewAI directly, so the
package only needs to be installed in the *subprocess* environment — not in
the rune core process.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, AsyncDriverTransport, make_driver_transport, make_async_driver_transport


class CrewAIDriverClient:
    """Orchestrate multi-agent ops tasks via CrewAI.

    Unlike the Holmes driver, CrewAI does not require a kubeconfig — it is a
    pure-Python framework that uses Ollama as its LLM backend.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._transport: DriverTransport = transport or make_driver_transport("crewai")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("crewai")

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

        Dispatch a question to the CrewAI driver and return the answer.

        Args:
            question: Natural-language ops/analysis question.
            model: Ollama model identifier (e.g. ``"llama3.1:8b"``).
            backend_url: Base URL of the Ollama server (optional).

        Returns:
            The CrewAI workflow's textual answer.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"CrewAIDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
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

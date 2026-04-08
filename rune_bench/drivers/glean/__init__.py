# SPDX-License-Identifier: Apache-2.0
"""Glean driver client -- delegates enterprise search queries to the glean driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.glean.__main__``) calls the Glean REST API and therefore
only requires network access to the Glean instance -- not any local SDK.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, AsyncDriverTransport, make_driver_transport, make_async_driver_transport


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
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("glean")

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

        Dispatch a question to the Glean driver and return the answer.

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
            f"{self.__name__}.ask_async: question={question!r} model={resolved_model!r} "
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

GleanRunner = GleanDriverClient

# SPDX-License-Identifier: Apache-2.0
"""Browser-use driver client — delegates web automation queries to the browser-use driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult
from rune_bench.debug import debug_log
from rune_bench.drivers import (
    DriverTransport,
    AsyncDriverTransport,
    make_driver_transport,
    make_async_driver_transport,
)


class BrowserUseDriverClient:
    """Automate web tasks via the browser-use library in a driver process."""

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("browser_use")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("browser_use")

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
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"BrowserUseDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Browser-use driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Browser-use driver returned an empty answer.")

        return AgentResult(
            answer=str(answer),
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

        debug_log(
            f"{self.__class__.__name__}.ask_async: question={question!r} model={resolved_model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = await self._async_transport.call_async("ask", params)

        if "answer" not in result:
            raise RuntimeError("Browser-use driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Browser-use driver returned an empty answer.")

        return AgentResult(
            answer=str(answer),
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
        )

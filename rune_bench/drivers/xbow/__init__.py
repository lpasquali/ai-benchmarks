# SPDX-License-Identifier: Apache-2.0
"""XBOW driver client — enterprise stub pending API access."""
from __future__ import annotations
from rune_bench.debug import debug_log


from rune_bench.agents.base import AgentResult


import os

from rune_bench.drivers import DriverTransport, AsyncDriverTransport, make_driver_transport, make_async_driver_transport


class XbowDriverClient:
    """XBOW autonomous pentesting agent. Requires enterprise API access.

    Configure via environment variables:
        RUNE_XBOW_API_KEY   — API key/token
    """

    ONBOARDING_URL = "https://xbow.com/"

    def __init__(self, *, transport: DriverTransport | None = None) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("xbow")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("xbow")

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

        Send a question to the XBOW driver.

        Raises:
            RuntimeError: if ``RUNE_XBOW_API_KEY`` is not set.
        """
        api_key = os.getenv("RUNE_XBOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "XBOW requires an enterprise contract or API access. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                "Once provisioned, set RUNE_XBOW_API_KEY."
            )
        result = self._transport.call("ask", {
            "question": question,
            "model": model,
            "backend_url": backend_url,
        })
        answer = str(result.get("answer", ""))
        if not answer:
            raise RuntimeError("Driver returned an empty answer.")
        return AgentResult(
            answer=answer,
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

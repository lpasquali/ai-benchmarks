# SPDX-License-Identifier: Apache-2.0
"""Browser-Use driver client — AI-powered browser automation agent.

Browser-Use (https://github.com/browser-use/browser-use) is an open-source
framework for AI-driven browser automation. This driver delegates browser
automation tasks to the browseruse driver process via DriverTransport.

Tier 3 — requires Playwright and browser-use package in the subprocess.
"""

from __future__ import annotations

import os

from rune_bench.agents.base import AgentResult
from rune_bench.debug import debug_log
from rune_bench.drivers import (
    AsyncDriverTransport,
    DriverTransport,
    make_async_driver_transport,
    make_driver_transport,
)


class BrowserUseDriverClient:
    """AI-powered browser automation agent.

    Configure via environment variables:
        RUNE_BROWSERUSE_API_KEY  — OpenAI API key for LLM reasoning
        RUNE_BROWSERUSE_BASE_URL — Target URL for browser automation
    """

    ONBOARDING_URL = "https://github.com/browser-use/browser-use"

    def __init__(self, *, transport: DriverTransport | None = None) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("browseruse")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("browseruse")

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
        """Dispatch a browser automation task and return a structured AgentResult.

        Raises:
            RuntimeError: if ``RUNE_BROWSERUSE_API_KEY`` is not set.
        """
        api_key = os.getenv("RUNE_BROWSERUSE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Browser-Use requires an LLM API key for reasoning. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                "Set RUNE_BROWSERUSE_API_KEY (typically an OpenAI key)."
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
            token_usage=result.get("token_usage"),
            telemetry=result.get("telemetry"),
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
            token_usage=result.get("token_usage"),
            telemetry=result.get("telemetry"),
        )

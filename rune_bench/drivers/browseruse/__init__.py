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
from rune_bench.api_contracts import LatencyPhase, RunTelemetry, TokenBreakdown
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
        self._transport: DriverTransport = transport or make_driver_transport(
            "browseruse"
        )
        self._async_transport: AsyncDriverTransport = make_async_driver_transport(
            "browseruse"
        )

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
        """Dispatch a browser automation task and return a structured AgentResult."""
        self._check_auth()
        params: dict = {
            "question": question,
            "model": model,
            "backend_url": backend_url,
            "backend_type": backend_type,
        }
        if backend_url:
            params.update(
                self._fetch_model_limits(
                    model=model,
                    backend_url=backend_url,
                    backend_type=backend_type,
                )
            )

        result = self._transport.call("ask", params)
        answer = str(result.get("answer", ""))
        if not answer:
            raise RuntimeError("Driver returned an empty answer.")
        return AgentResult(
            answer=answer,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
            telemetry=self._parse_telemetry(result.get("telemetry")),
        )

    async def ask_async(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Dispatch a question to the driver asynchronously."""
        self._check_auth()
        resolved_model = model.strip()
        params: dict = {
            "question": question,
            "model": resolved_model,
            "backend_type": backend_type,
        }
        if backend_url:
            params["backend_url"] = backend_url
            params.update(
                self._fetch_model_limits(
                    model=resolved_model,
                    backend_url=backend_url,
                    backend_type=backend_type,
                )
            )

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
            telemetry=self._parse_telemetry(result.get("telemetry")),
        )

    def _check_auth(self) -> None:
        """Raise RuntimeError if RUNE_BROWSERUSE_API_KEY is not set."""
        api_key = os.getenv("RUNE_BROWSERUSE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Browser-Use requires an LLM API key for reasoning. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                "Set RUNE_BROWSERUSE_API_KEY (typically an OpenAI key)."
            )

    def _fetch_model_limits(
        self,
        *,
        model: str,
        backend_url: str,
        backend_type: str = "ollama",
    ) -> dict:
        """Return context_window / max_output_tokens for *model*, or ``{}`` on error."""
        from rune_bench.backends import get_backend

        try:
            backend = get_backend(backend_type, backend_url)
            normalized = backend.normalize_model_name(model)
            caps = backend.get_model_capabilities(normalized)
        except Exception as exc:  # noqa: BLE001
            debug_log(f"Could not fetch model limits for {model!r}: {exc}")
            return {}

        limits: dict = {}
        if caps.context_window:
            limits["context_window"] = caps.context_window
        if caps.max_output_tokens:
            limits["max_output_tokens"] = caps.max_output_tokens
        return limits

    def _parse_telemetry(self, raw: dict | None) -> RunTelemetry | None:
        """Parse raw telemetry dict into a RunTelemetry object."""
        if not raw:
            return None

        tokens_raw = raw.get("tokens", {})
        tokens = TokenBreakdown(
            system_prompt=tokens_raw.get("system_prompt", 0),
            tool_calls=tokens_raw.get("tool_calls", 0),
            agent_reasoning=tokens_raw.get("agent_reasoning", 0),
            output=tokens_raw.get("output", 0),
            total=tokens_raw.get("total", 0),
        )

        latency_raw = raw.get("latency", [])
        latency = [
            LatencyPhase(phase=p.get("phase", "unknown"), ms=p.get("ms", 0))
            for p in latency_raw
            if isinstance(p, dict)
        ]

        return RunTelemetry(
            tokens=tokens,
            latency=latency,
            cost_estimate_usd=raw.get("cost_estimate_usd"),
        )

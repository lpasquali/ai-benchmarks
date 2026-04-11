# SPDX-License-Identifier: Apache-2.0
"""Perplexity driver client — delegates research queries to the perplexity driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.perplexity.__main__``) calls the Perplexity REST API
using ``urllib.request`` and therefore requires no extra dependencies in the
*subprocess* environment.
"""

from __future__ import annotations

from rune_bench.agents.base import AgentResult
from rune_bench.api_contracts import LatencyPhase, RunTelemetry, TokenBreakdown
from rune_bench.debug import debug_log
from rune_bench.drivers import (
    DriverTransport,
    AsyncDriverTransport,
    make_driver_transport,
    make_async_driver_transport,
)


class PerplexityDriverClient:
    """Submit research queries to the Perplexity API via the driver process.

    The public interface mirrors :class:`~rune_bench.drivers.holmes.HolmesDriverClient`
    so existing call-sites (CLI, API backend) can use either interchangeably.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("perplexity")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport("perplexity")

    def ask(
        self,
        question: str,
        model: str = "sonar-pro",
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> str:
        """Dispatch a research question to the Perplexity driver and return the answer string."""
        return self.ask_structured(
            question=question,
            model=model,
            backend_url=backend_url,
            backend_type=backend_type,
        ).answer

    def ask_structured(
        self,
        question: str,
        model: str = "sonar-pro",
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Dispatch a research question to the Perplexity driver and return a structured AgentResult.

        Args:
            question: Natural-language research question.
            model: Perplexity model name (e.g. ``"sonar"``, ``"sonar-pro"``,
                ``"sonar-deep-research"``).  Default: ``"sonar-pro"``.
            backend_url: Ignored — Perplexity is cloud-only.

        Returns:
            The textual answer from Perplexity.
        """
        normalized_model = model.strip()
        if not normalized_model:
            raise RuntimeError("Perplexity model must be a non-empty string.")
        params: dict = {
            "question": question,
            "model": normalized_model,
        }

        debug_log(
            f"PerplexityDriverClient.ask: question={question!r} model={model!r}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Perplexity driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Perplexity driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Perplexity driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
            telemetry=self._parse_telemetry(result.get("telemetry")),
        )

    async def ask_async(
        self,
        question: str,
        model: str = "sonar-pro",
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> AgentResult:
        """Dispatch a research question to the Perplexity driver asynchronously."""
        normalized_model = model.strip()
        if not normalized_model:
            raise RuntimeError("Perplexity model must be a non-empty string.")
        params: dict = {
            "question": question,
            "model": normalized_model,
        }

        debug_log(
            f"PerplexityDriverClient.ask_async: question={question!r} model={model!r}"
        )
        result = await self._async_transport.call_async("ask", params)

        if "answer" not in result:
            raise RuntimeError("Perplexity driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Perplexity driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Perplexity driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
            telemetry=self._parse_telemetry(result.get("telemetry")),
        )



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
            for p in latency_raw if isinstance(p, dict)
        ]

        return RunTelemetry(
            tokens=tokens,
            latency=latency,
            cost_estimate_usd=raw.get("cost_estimate_usd"),
        )

PerplexityRunner = PerplexityDriverClient
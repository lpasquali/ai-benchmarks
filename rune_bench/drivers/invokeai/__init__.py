# SPDX-License-Identifier: Apache-2.0
"""InvokeAI driver client — delegates art generation queries to the InvokeAI driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`.
InvokeAI runs as a local server (Python) or via Docker.
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


class InvokeAIDriverClient:
    """Run autonomous art generation workflows via InvokeAI."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._base_url = base_url or "http://127.0.0.1:9090"
        self._transport: DriverTransport = transport or make_driver_transport(
            "invokeai"
        )
        self._async_transport: AsyncDriverTransport = make_async_driver_transport(
            "invokeai"
        )

    def ask(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "invokeai",
    ) -> str:
        """Dispatch a prompt to InvokeAI and return the path/URL to the generated image."""
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
        backend_type: str = "invokeai",
    ) -> AgentResult:
        params: dict = {
            "prompt": question,
            "model": model.strip(),
            "base_url": backend_url or self._base_url,
        }

        debug_log(f"InvokeAIDriverClient.ask: prompt={question!r} model={model!r}")
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError(
                "InvokeAI driver response did not include an answer (image path/URL)."
            )

        return AgentResult(
            answer=str(result["answer"]),
            result_type="image",
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
            telemetry=self._parse_telemetry(result.get("telemetry")),
        )

    async def ask_async(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "invokeai",
    ) -> AgentResult:
        params: dict = {
            "prompt": question,
            "model": model.strip(),
            "base_url": backend_url or self._base_url,
        }

        debug_log(f"InvokeAIDriverClient.ask_async: prompt={question!r}")
        result = await self._async_transport.call_async("ask", params)

        if "answer" not in result:
            raise RuntimeError("InvokeAI driver response did not include an answer.")

        return AgentResult(
            answer=str(result["answer"]),
            result_type="image",
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
            for p in latency_raw
            if isinstance(p, dict)
        ]

        return RunTelemetry(
            tokens=tokens,
            latency=latency,
            cost_estimate_usd=raw.get("cost_estimate_usd"),
        )

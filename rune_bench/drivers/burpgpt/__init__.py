# SPDX-License-Identifier: Apache-2.0
"""BurpGPT driver client -- delegates web vulnerability scanning to the burpgpt driver process.

The driver process (``rune_bench.drivers.burpgpt.__main__``) talks to the
Burp Suite REST API to launch scans and retrieve findings.  Burp Suite Pro
must be running locally (or remotely) with the REST API enabled.

Configuration:
    RUNE_BURPGPT_BURP_API_URL  Base URL of the Burp REST API
                                (default: http://localhost:1337)

Security notice: only scan targets you own or have explicit written
authorization to test.
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


class BurpGPTDriverClient:
    """Run web vulnerability scans by delegating to the burpgpt driver.

    Unlike Holmes, BurpGPT does **not** require a kubeconfig -- it operates
    against the Burp Suite REST API.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("burpgpt")
        self._async_transport: AsyncDriverTransport = make_async_driver_transport(
            "burpgpt"
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
        """Dispatch a question to the driver and return a structured AgentResult.

        Dispatch a scan request to the burpgpt driver and return findings.

        Args:
            question: Target URL or scan objective.
            model: Ollama model identifier (currently unused by BurpGPT but
                   kept for interface consistency).
            backend_url: Base URL of the Ollama server (currently unused).

        Returns:
            Formatted vulnerability findings from the Burp scan.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"BurpGPTDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("BurpGPT driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("BurpGPT driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("BurpGPT driver returned an empty answer.")

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


BurpGPTRunner = BurpGPTDriverClient

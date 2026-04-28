# SPDX-License-Identifier: Apache-2.0
"""Consensus driver client — delegates research queries to the consensus driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.consensus.__main__``) queries the Semantic Scholar API
and optionally synthesizes answers via Ollama — no external Python packages
are required.
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


class ConsensusDriverClient:
    """Research agent: evidence-based synthesis from academic papers.

    Uses the Semantic Scholar API for paper search and optionally Ollama
    for answer synthesis.  The public interface mirrors the
    ``ConsensusRunner`` so existing call-sites require no changes.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport(
            "consensus"
        )
        self._async_transport: AsyncDriverTransport = make_async_driver_transport(
            "consensus"
        )

    def ask(
        self,
        question: str,
        model: str = "",
        backend_url: str | None = None,
        backend_type: str = "ollama",
        limit: int | None = None,
    ) -> str:
        """Dispatch a research question to the consensus driver and return the answer string."""
        return self.ask_structured(
            question=question,
            model=model,
            backend_url=backend_url,
            backend_type=backend_type,
            limit=limit,
        ).answer

    def ask_structured(
        self,
        question: str,
        model: str = "",
        backend_url: str | None = None,
        backend_type: str = "ollama",
        limit: int | None = None,
    ) -> AgentResult:
        """Dispatch a research question to the consensus driver and return a structured AgentResult.

        Args:
            question: Natural-language research question.
            model: Ollama model identifier for synthesis (optional).
            backend_url: Base URL of the Ollama server (optional).
            limit: Maximum number of papers to retrieve (optional, driver default 10).

        Returns:
            Synthesized answer (if model + backend_url provided) or a
            formatted list of relevant papers.
        """
        params: dict = {"question": question}
        normalized_model = model.strip()
        if normalized_model:
            params["model"] = normalized_model
        if backend_url:
            params["backend_url"] = backend_url
            params.update(
                self._fetch_model_limits(
                    model=normalized_model,
                    backend_url=backend_url,
                    backend_type=backend_type,
                )
            )
        if limit is not None:
            params["limit"] = limit

        debug_log(
            f"ConsensusDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Consensus driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Consensus driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Consensus driver returned an empty answer.")

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
        model: str = "",
        backend_url: str | None = None,
        backend_type: str = "ollama",
        limit: int | None = None,
    ) -> AgentResult:
        """Dispatch a research question to the consensus driver asynchronously."""
        params: dict = {"question": question}
        normalized_model = model.strip()
        if normalized_model:
            params["model"] = normalized_model
        if backend_url:
            params["backend_url"] = backend_url
            params.update(
                self._fetch_model_limits(
                    model=normalized_model,
                    backend_url=backend_url,
                    backend_type=backend_type,
                )
            )
        if limit is not None:
            params["limit"] = limit

        debug_log(
            f"ConsensusDriverClient.ask_async: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = await self._async_transport.call_async("ask", params)

        if "answer" not in result:
            raise RuntimeError("Consensus driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Consensus driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Consensus driver returned an empty answer.")

        return AgentResult(
            answer=answer_text,
            result_type=result.get("result_type", "text"),
            artifacts=result.get("artifacts"),
            metadata=result.get("metadata"),
            telemetry=self._parse_telemetry(result.get("telemetry")),
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

        if not model:
            return {}

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


ConsensusRunner = ConsensusDriverClient

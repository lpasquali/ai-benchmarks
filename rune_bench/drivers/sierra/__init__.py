# SPDX-License-Identifier: Apache-2.0
"""Sierra driver client — enterprise stub pending API access."""

from __future__ import annotations

from rune_bench.agents.base import AgentResult

import os

from rune_bench.drivers import DriverTransport, make_driver_transport


class SierraDriverClient:
    """Sierra conversational AI platform. Requires enterprise API access.

    Configure via environment variables:
        RUNE_SIERRA_API_KEY   — API key/token
    """

    ONBOARDING_URL = "https://sierra.ai/"

    def __init__(self, *, transport: DriverTransport | None = None) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("sierra")

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
        """Send a question to the Sierra driver.

        Raises:
            RuntimeError: if ``RUNE_SIERRA_API_KEY`` is not set.
        """
        api_key = os.getenv("RUNE_SIERRA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Sierra requires an enterprise contract or API access. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                "Once provisioned, set RUNE_SIERRA_API_KEY."
            )
        result = self._transport.call("ask", {
            "question": question,
            "model": model,
            "backend_url": backend_url,
        })
        return result.get("answer", "")

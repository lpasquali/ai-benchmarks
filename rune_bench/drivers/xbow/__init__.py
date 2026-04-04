"""XBOW driver client — enterprise stub pending API access."""

from __future__ import annotations

import os

from rune_bench.drivers import DriverTransport, make_driver_transport


class XbowDriverClient:
    """XBOW autonomous pentesting agent. Requires enterprise API access.

    Configure via environment variables:
        RUNE_XBOW_API_KEY   — API key/token
    """

    ONBOARDING_URL = "https://xbow.com/"

    def __init__(self, *, transport: DriverTransport | None = None) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("xbow")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Send a question to the XBOW driver.

        Raises:
            RuntimeError: if ``RUNE_XBOW_API_KEY`` is not set.
        """
        api_key = os.getenv("RUNE_XBOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"XBOW requires an enterprise contract or API access. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                f"Once provisioned, set RUNE_XBOW_API_KEY."
            )
        result = self._transport.call("ask", {
            "question": question,
            "model": model,
            "ollama_url": ollama_url,
        })
        return result.get("answer", "")

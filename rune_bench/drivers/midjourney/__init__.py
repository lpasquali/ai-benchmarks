"""Midjourney driver client — enterprise stub pending API access."""

from __future__ import annotations

import os

from rune_bench.drivers import DriverTransport, make_driver_transport


class MidjourneyDriverClient:
    """Midjourney image generation. Requires proxy service (no official API).

    There is no official Midjourney API. Access typically requires a proxy
    service such as useapi.net or similar.

    Configure via environment variables:
        RUNE_MIDJOURNEY_API_KEY    — API key/token for proxy service
        RUNE_MIDJOURNEY_API_BASE   — Base URL for the proxy API
    """

    ONBOARDING_URL = "https://docs.midjourney.com/"

    def __init__(self, *, transport: DriverTransport | None = None) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("midjourney")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Send a question to the Midjourney driver.

        Raises:
            RuntimeError: if ``RUNE_MIDJOURNEY_API_KEY`` is not set.
        """
        api_key = os.getenv("RUNE_MIDJOURNEY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Midjourney requires an enterprise contract or API access. "
                f"Visit {self.ONBOARDING_URL} to get started. "
                "Once provisioned, set RUNE_MIDJOURNEY_API_KEY."
            )
        result = self._transport.call("ask", {
            "question": question,
            "model": model,
            "ollama_url": ollama_url,
        })
        return result.get("answer", "")

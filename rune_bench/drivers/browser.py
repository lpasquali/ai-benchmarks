# SPDX-License-Identifier: Apache-2.0
"""BrowserDriverTransport — browser-based automation for agents with no public API."""

from __future__ import annotations

import asyncio
import os

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None

from rune_bench.debug import debug_log


class BrowserDriverTransport:
    """Automates a browser via Playwright to interact with web-only agents."""

    def __init__(self, driver_name: str | None = None, headless: bool = True) -> None:
        self._driver_name = driver_name
        self._headless = headless

    def _get_default_url(self) -> str | None:
        if self._driver_name:
            return os.getenv(f"RUNE_{self._driver_name.upper()}_DRIVER_URL")
        return None

    def call(self, action: str, params: dict) -> dict:
        """Synchronous wrapper for browser actions."""
        return asyncio.run(self.call_async(action, params))

    async def call_async(self, action: str, params: dict) -> dict:
        if async_playwright is None:
            raise ImportError(
                "BrowserDriverTransport requires playwright. "
                "Install with: pip install playwright && playwright install"
            )

        debug_log(f"BrowserDriverTransport.call_async: action={action!r}")

        if action == "ask":
            url = params.get("url") or self._get_default_url()
            question = params.get("question")
            if not url:
                raise ValueError(
                    f"Browser 'ask' action requires a URL. Set RUNE_{self._driver_name.upper()}_DRIVER_URL "
                    "or pass 'url' in params."
                    if self._driver_name
                    else "Browser 'ask' action requires a 'url' parameter."
                )

            return await self._run_browser_task(url, question)

        raise NotImplementedError(
            f"Action {action!r} not implemented in BrowserDriverTransport."
        )

    async def _run_browser_task(self, url: str, question: str | None) -> dict:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self._headless)
            page = await browser.new_page()

            debug_log(f"BrowserDriverTransport: navigating to {url}")
            await page.goto(url)

            # Basic implementation: extract text
            # Future: Use 'question' to drive interactions via LLM (browser-use)
            text = await page.evaluate("() => document.body.innerText")

            await browser.close()

            return {"answer": text, "metadata": {"url": url, "automated": True}}

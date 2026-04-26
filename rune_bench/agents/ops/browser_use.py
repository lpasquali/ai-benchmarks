# SPDX-License-Identifier: Apache-2.0
"""Browser-Use agentic runner implementation.

Scope:      Ops/Misc  |  Rank 5  |  Rating 4.0
Capability: Autonomous web automation using the browser-use library.
Docs:       https://github.com/browser-use/browser-use
Ecosystem:  Web / Playwright
"""

from __future__ import annotations

import asyncio
import os


class BrowserUseRunner:
    """Ops agent: autonomous web automation via browser-use."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # browser-use requires an LLM to drive the browser.
        # It usually expects a LangChain-compatible chat model.
        self._api_key = api_key or os.getenv("BROWSER_USE_API_KEY")
        self._model_name = model or os.getenv("BROWSER_USE_MODEL", "gpt-4o")

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run a browser automation task and return the result."""
        # Use asyncio.run for sync->async bridge
        try:
            return asyncio.run(self._run_task(question))
        except Exception as exc:
            return f"Browser-Use error: {exc}"

    async def _run_task(self, question: str) -> str:
        """Execute the task asynchronously."""
        if not self._api_key:
            return "Error: BROWSER_USE_API_KEY not set."
        try:
            from browser_use import Agent
            # For simplicity in this implementation, we assume the environment
            # has already configured the default LLM (e.g. via env vars).
            # Future: Use get_backend() to provide a local Ollama model to Agent.

            agent = Agent(
                task=question,
                llm=None,  # uses default from env
            )
            result = await agent.run()
            return f"Browser-Use task result: {result}"
        except ImportError:
            return "Error: browser-use package not installed. Run 'pip install browser-use'."
        except Exception as exc:
            return f"Browser-Use execution failed: {exc}"

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

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Run a browser automation task and return the result."""
        # Use asyncio.run for sync->async bridge
        try:
            return asyncio.run(self._run_task(question, model, backend_url, backend_type))
        except Exception as exc:
            return f"Browser-Use error: {exc}"

    async def _run_task(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Execute the task asynchronously."""
        if not self._api_key:
            return "Error: BROWSER_USE_API_KEY not set."
        try:
            from browser_use import Agent
            
            # Resolve the LLM based on backend_type
            llm = None
            if backend_url:
                if backend_type == "ollama":
                    try:
                        from langchain_ollama import ChatOllama
                        llm = ChatOllama(model=model, base_url=backend_url)
                    except ImportError:
                        return "Error: langchain-ollama not installed. Run 'pip install langchain-ollama'."
                elif backend_type == "bedrock":
                    try:
                        from langchain_aws import ChatBedrock
                        # Bedrock doesn't use a backend_url in the same way, but we use the type
                        llm = ChatBedrock(model_id=model)
                    except ImportError:
                        return "Error: langchain-aws not installed. Run 'pip install langchain-aws'."

            agent = Agent(
                task=question,
                llm=llm,
            )
            result = await agent.run()
            return f"Browser-Use task result: {result}"
        except ImportError:
            return "Error: browser-use package not installed. Run 'pip install browser-use'."
        except Exception as exc:
            return f"Browser-Use execution failed: {exc}"

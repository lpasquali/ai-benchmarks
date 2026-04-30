# SPDX-License-Identifier: Apache-2.0
"""MultiOn agentic runner implementation.

Scope:      Ops/Misc  |  Rank 5  |  Rating 4.0
Capability: Autonomous web browsing and web-based task execution.
Docs:       https://docs.multion.ai/
Ecosystem:  Web / SaaS API
"""

from __future__ import annotations

import os

import httpx


class MultiOnRunner:
    """MultiOn agent: autonomous web navigation and execution."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("MULTION_API_KEY")
        self._api_base = "https://api.multion.ai/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Execute a web-based task and return the result summary."""
        if not self._api_key:
            return "Error: MULTION_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=60.0
        ) as client:
            try:
                # 1. Browse (Run step-by-step or atomic)
                # Note: MultiOn 'browse' is often long-running.
                payload = {
                    "cmd": question,
                    "url": "https://www.google.com",  # Default starting point
                    "local": False,
                }
                resp = client.post("/browse", json=payload)
                resp.raise_for_status()
                data = resp.json()

                # MultiOn returns a summary/message of the action taken
                return f"MultiOn browsing result: {data.get('message', 'No summary provided.')}"
            except Exception as exc:
                return f"MultiOn error: {exc}"

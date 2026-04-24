# SPDX-License-Identifier: Apache-2.0
"""Sierra agentic runner implementation.

Scope:      Ops/Misc  |  Rank 3  |  Rating 4.3
Capability: Autonomous business operations and customer experience orchestration.
Docs:       https://sierra.ai/
Ecosystem:  Enterprise SaaS
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rune_bench.debug import debug_log


class SierraRunner:
    """Sierra agent: autonomous business process orchestration."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("SIERRA_API_KEY")
        self._api_base = "https://api.sierra.ai/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run an autonomous operation and return the result."""
        if not self._api_key:
            return "Error: SIERRA_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}
        
        with httpx.Client(base_url=self._api_base, headers=headers, timeout=30.0) as client:
            try:
                # 1. Submit operation
                payload = {"objective": question, "agent_id": model or "default"}
                resp = client.post("/runs", json=payload)
                resp.raise_for_status()
                run_id = resp.json()["id"]
                debug_log(f"Sierra: Run started (ID: {run_id})")

                # 2. Poll for completion (max 5 mins)
                for _ in range(60):
                    time.sleep(5)
                    run_resp = client.get(f"/runs/{run_id}")
                    if run_resp.status_code == 200:
                        run_data = run_resp.json()
                        status = run_data.get("status", "").lower()
                        
                        if status == "completed":
                            return f"Sierra execution result: {run_data.get('summary')}"
                        
                        if status == "failed":
                            return f"Sierra: Run failed: {run_data.get('error')}"

                return "Sierra: Timeout waiting for execution."
            except Exception as exc:
                return f"Sierra error: {exc}"

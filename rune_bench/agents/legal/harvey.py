# SPDX-License-Identifier: Apache-2.0
"""Harvey AI agentic runner implementation.

Scope:      Legal  |  Rank 1  |  Rating 4.8
Capability: Autonomous legal disclosure and risk analysis.
Docs:       https://www.harvey.ai/
Ecosystem:  Transparency Manifestos
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rune_bench.debug import debug_log


class HarveyAIRunner:
    """Legal agent: autonomous legal disclosure and risk analysis via Harvey AI."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("HARVEY_API_KEY")
        self._api_base = "https://api.harvey.ai/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Submit a legal query to Harvey AI and return the analysis."""
        if not self._api_key:
            return "Error: HARVEY_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}
        
        with httpx.Client(base_url=self._api_base, headers=headers, timeout=60.0) as client:
            try:
                # 1. Submit Completion
                payload = {"prompt": question, "matter_type": model or "general"}
                resp = client.post("/completions", json=payload)
                resp.raise_for_status()
                task_id = resp.json()["id"]
                debug_log(f"Harvey: Task started (ID: {task_id})")

                # 2. Poll for result
                for _ in range(60):
                    time.sleep(5)
                    job_resp = client.get(f"/completions/{task_id}")
                    if job_resp.status_code == 200:
                        job_data = job_resp.json()
                        status = job_data.get("status", "").lower()
                        
                        if status == "completed":
                            return f"Harvey AI Analysis: {job_data.get('analysis')}"
                        
                        if status == "failed":
                            return f"Harvey: Analysis failed: {job_data.get('error')}"

                return "Harvey: Timeout waiting for analysis."
            except Exception as exc:
                return f"Harvey error: {exc}"

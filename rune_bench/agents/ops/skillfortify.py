# SPDX-License-Identifier: Apache-2.0
"""SkillFortify agentic runner implementation.

Scope:      Ops/Misc  |  Rank 4  |  Rating 4.1
Capability: Autonomous technical skill gap analysis and training orchestration.
Docs:       https://skillfortify.com/
Ecosystem:  EdTech / Enterprise SaaS
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rune_bench.debug import debug_log


class SkillFortifyRunner:
    """Ops agent: autonomous skill development orchestration via SkillFortify."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("SKILLFORTIFY_API_KEY")
        self._api_base = "https://api.skillfortify.com/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Analyze skill requirements and return training plan."""
        if not self._api_key:
            return "Error: SKILLFORTIFY_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}
        
        with httpx.Client(base_url=self._api_base, headers=headers, timeout=30.0) as client:
            try:
                # 1. Submit gap analysis request
                payload = {"goal": question, "domain": model or "engineering"}
                resp = client.post("/analyses", json=payload)
                resp.raise_for_status()
                analysis_id = resp.json()["id"]
                debug_log(f"SkillFortify: Analysis started (ID: {analysis_id})")

                # 2. Poll for completion
                for _ in range(60):
                    time.sleep(2)
                    task_resp = client.get(f"/analyses/{analysis_id}")
                    if task_resp.status_code == 200:
                        task_data = task_resp.json()
                        status = task_data.get("status", "").lower()
                        
                        if status == "completed":
                            return f"SkillFortify Plan: {task_data.get('plan_summary')}"
                        
                        if status == "failed":
                            return f"SkillFortify: Analysis failed: {task_data.get('error')}"

                return "SkillFortify: Timeout waiting for analysis."
            except Exception as exc:
                return f"SkillFortify error: {exc}"

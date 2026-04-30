# SPDX-License-Identifier: Apache-2.0
"""XBOW agentic runner implementation.

Scope:      Cybersec  |  Rank 2  |  Rating 4.8
Capability: Autonomous offensive security and pentesting.
Docs:       https://xbow.com/
Ecosystem:  SaaS API
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class XBOWRunner:
    """Cybersec agent: autonomous offensive security via XBOW."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("XBOW_API_KEY")
        self._api_base = "https://api.xbow.com/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run a security assessment and return the findings."""
        if not self._api_key:
            return "Error: XBOW_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=60.0
        ) as client:
            try:
                # 1. Start assessment
                payload = {"target": question, "scan_type": model or "full"}
                resp = client.post("/assessments", json=payload)
                resp.raise_for_status()
                assessment_id = resp.json()["id"]
                debug_log(f"XBOW: Assessment started (ID: {assessment_id})")

                # 2. Poll for completion
                for _ in range(120):
                    time.sleep(5)
                    job_resp = client.get(f"/assessments/{assessment_id}")
                    if job_resp.status_code == 200:
                        job_data = job_resp.json()
                        status = job_data.get("status", "").lower()

                        if status == "finished":
                            findings = job_data.get("findings", [])
                            return f"XBOW Findings: {len(findings)} issues found. Summary: {job_data.get('summary')}"

                        if status == "error":
                            return f"XBOW: Assessment failed: {job_data.get('error')}"

                return "XBOW: Timeout waiting for assessment."
            except Exception as exc:
                return f"XBOW error: {exc}"

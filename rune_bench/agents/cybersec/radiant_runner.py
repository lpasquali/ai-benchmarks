# SPDX-License-Identifier: Apache-2.0
"""Radiant Security agentic runner implementation.

Scope:      Cybersec  |  Rank 2  |  Rating 4.5
Capability: Autonomous SOC incident investigation and response.
Docs:       https://radiantsecurity.ai/
Ecosystem:  Enterprise SaaS
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class RadiantSecurityRunner:
    """Cybersec agent: autonomous SOC incident investigation via Radiant Security."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.getenv("RADIANT_API_KEY")
        self._api_base = base_url or os.getenv(
            "RADIANT_API_BASE", "https://api.radiantsecurity.ai/v1"
        )

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Submit a security incident to Radiant and return the investigation report."""
        if not self._api_key:
            return "Error: RADIANT_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=30.0
        ) as client:
            try:
                # 1. Create Incident / Alert
                payload = {"description": question, "source": "RUNE-Bench"}
                resp = client.post("/incidents", json=payload)
                resp.raise_for_status()
                incident_id = resp.json()["id"]
                debug_log(f"Radiant: Incident created (ID: {incident_id})")

                # 2. Poll for investigation report
                for _ in range(60):
                    time.sleep(5)
                    report_resp = client.get(f"/incidents/{incident_id}/report")
                    if report_resp.status_code == 200:
                        report_data = report_resp.json()
                        status = report_data.get("status", "").lower()

                        if status == "completed":
                            verdict = report_data.get("verdict", "Unknown")
                            summary = report_data.get("summary", "No summary provided.")
                            return (
                                f"Radiant Investigation: {verdict}. Summary: {summary}"
                            )

                        if status == "failed":
                            return f"Radiant: Investigation failed: {report_data.get('error')}"

                return "Radiant: Timeout waiting for investigation."
            except Exception as exc:
                return f"Radiant error: {exc}"

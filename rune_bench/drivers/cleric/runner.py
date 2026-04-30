# SPDX-License-Identifier: Apache-2.0
"""Cleric agentic runner implementation.

Scope:      SRE  |  Rank 5  |  Rating 3.5
Capability: Mimics an engineer's "parallel investigation" loop.
Docs:       https://github.com/ClericHQ/cleric
Ecosystem:  Infra Interoperability
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class ClericRunner:
    """SRE agent: parallel investigation loop mimicking human debugging."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs) -> None:
        self._api_key = api_key or os.getenv("CLERIC_API_KEY")
        self._api_base = base_url or os.getenv(
            "CLERIC_API_BASE", "http://localhost:8080/v1"
        )

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama", **kwargs) -> str:
        """Run a Cleric investigation and return the findings."""
        if not self._api_key and "localhost" not in self._api_base:
            return "Error: CLERIC_API_KEY not set (required for remote API)."

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=60.0
        ) as client:
            try:
                # 1. Start Investigation
                payload = {"goal": question, "model": model or "gpt-4"}
                resp = client.post("/investigations", json=payload)
                resp.raise_for_status()
                inv_id = resp.json()["id"]
                debug_log(f"Cleric: Investigation started (ID: {inv_id})")

                # 2. Poll for conclusion
                for _ in range(120):
                    time.sleep(5)
                    job_resp = client.get(f"/investigations/{inv_id}")
                    if job_resp.status_code == 200:
                        job_data = job_resp.json()
                        status = job_data.get("status", "").lower()

                        if status == "finished":
                            return f"Cleric Root Cause Analysis: {job_data.get('conclusion')}"

                        if status == "failed":
                            return (
                                f"Cleric: Investigation failed: {job_data.get('error')}"
                            )

                return "Cleric: Timeout waiting for investigation."
            except Exception as exc:
                return f"Cleric error: {exc}"

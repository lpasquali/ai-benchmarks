# SPDX-License-Identifier: Apache-2.0
"""Spellbook agentic runner implementation.

Scope:      Legal/Ops  |  Rank 1  |  Rating 4.9
Capability: Autonomous legal document drafting and contract review.
Docs:       https://www.spellbook.legal/
Ecosystem:  Legal SaaS
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class SpellbookRunner:
    """Legal/Ops agent: autonomous contract review via Spellbook."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("SPELLBOOK_API_KEY")
        self._api_base = "https://api.spellbook.legal/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Run a legal document analysis and return the review result."""
        if not self._api_key:
            return "Error: SPELLBOOK_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=60.0
        ) as client:
            try:
                # 1. Start review
                payload = {
                    "document_text": question,
                    "review_type": model or "standard",
                }
                resp = client.post("/reviews", json=payload)
                resp.raise_for_status()
                review_id = resp.json()["id"]
                debug_log(f"Spellbook: Review started (ID: {review_id})")

                # 2. Poll for completion
                for _ in range(60):
                    time.sleep(5)
                    job_resp = client.get(f"/reviews/{review_id}")
                    if job_resp.status_code == 200:
                        job_data = job_resp.json()
                        status = job_data.get("status", "").lower()

                        if status == "completed":
                            return f"Spellbook legal review result: {job_data.get('summary')}"

                        if status == "error":
                            return f"Spellbook: Review failed: {job_data.get('error')}"

                return "Spellbook: Timeout waiting for review."
            except Exception as exc:
                return f"Spellbook error: {exc}"

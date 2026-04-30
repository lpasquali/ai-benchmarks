# SPDX-License-Identifier: Apache-2.0
"""Midjourney agentic runner implementation via proxy API.

Scope:      Art/Creative  |  Rank 1  |  Rating 5.0
Capability: Iterative agentic refinement via "Remix" modes.
Docs:       https://docs.midjourney.com/
Ecosystem:  Generative AI Ethics
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class MidjourneyRunner:
    """Art/Creative agent: iterative image generation via Midjourney Remix."""

    def __init__(self, api_base: str | None = None, api_key: str | None = None, **kwargs) -> None:
        self._api_base = api_base or os.getenv(
            "MIDJOURNEY_API_BASE", "https://api.useapi.net/v1"
        )
        self._api_key = api_key or os.getenv("MIDJOURNEY_API_KEY")

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama", **kwargs) -> str:
        """Generate an image from the prompt and return the result URL."""
        if not self._api_key:
            return "Error: MIDJOURNEY_API_KEY not set. Unofficial Midjourney API requires a proxy key."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=30.0
        ) as client:
            try:
                # 1. Start generation (Imagine)
                payload = {"prompt": question}
                resp = client.post("/midjourney/imagine", json=payload)
                resp.raise_for_status()
                job_id = resp.json()["jobid"]
                debug_log(f"Midjourney: Imagine job started (ID: {job_id})")

                # 2. Poll for completion (max 5 mins)
                for _ in range(30):
                    time.sleep(10)
                    job_resp = client.get(f"/midjourney/jobs/{job_id}")
                    if job_resp.status_code == 200:
                        job_data = job_resp.json()
                        status = job_data.get("status", "").lower()

                        if status == "completed":
                            attachments = job_data.get("attachments", [])
                            if attachments:
                                url = attachments[0].get("url")
                                return f"Midjourney generated image: {url}"
                            return "Midjourney: Generation complete but no image URL found."

                        if status == "failed":
                            return f"Midjourney: Job failed: {job_data.get('error')}"

                return "Midjourney: Timeout waiting for generation."
            except Exception as exc:
                return f"Midjourney error: {exc}"

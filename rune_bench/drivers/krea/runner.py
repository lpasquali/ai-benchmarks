# SPDX-License-Identifier: Apache-2.0
"""Krea AI agentic runner implementation.

Scope:      Art/Creative  |  Rank 3  |  Rating 4.2
Capability: Real-time autonomous image enhancement and upscaling.
Docs:       https://www.krea.ai/
            https://www.krea.ai/docs/api
Ecosystem:  SaaS API
"""

from __future__ import annotations

import os
import time

import httpx

from rune_bench.debug import debug_log


class KreaRunner:
    """Art/Creative agent: real-time image enhancement via Krea AI."""

    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        self._api_key = api_key or os.getenv("KREA_API_KEY")
        self._api_base = "https://api.krea.ai/v1"

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama", **kwargs) -> str:
        """Generate/enhance an image and return the result URL."""
        if not self._api_key:
            return "Error: KREA_API_KEY not set."

        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.Client(
            base_url=self._api_base, headers=headers, timeout=30.0
        ) as client:
            try:
                # 1. Submit task
                payload = {"prompt": question, "model": model or "krea-v1"}
                resp = client.post("/image/generate", json=payload)
                resp.raise_for_status()
                task_id = resp.json()["id"]
                debug_log(f"Krea: Task submitted (ID: {task_id})")

                # 2. Poll for completion
                for _ in range(60):
                    time.sleep(2)
                    task_resp = client.get(f"/tasks/{task_id}")
                    if task_resp.status_code == 200:
                        task_data = task_resp.json()
                        status = task_data.get("status", "").lower()

                        if status == "succeeded":
                            image_url = task_data.get("output_url")
                            return f"Krea AI generated image: {image_url}"

                        if status == "failed":
                            return f"Krea: Task failed: {task_data.get('error')}"

                return "Krea: Timeout waiting for generation."
            except Exception as exc:
                return f"Krea error: {exc}"

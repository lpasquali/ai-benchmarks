# SPDX-License-Identifier: Apache-2.0
# WARNING: ComfyUI is GPL-3.0. This driver MUST integrate via REST API only (DriverTransport).
# Do NOT import any ComfyUI Python modules directly -- doing so triggers GPL-3.0 copyleft obligations.
"""ComfyUI agentic runner implementation.

Scope:      Art/Creative  |  Rank 2  |  Rating 4.5
Capability: Node-based autonomous art pipeline orchestration.
Docs:       https://github.com/comfy-org/ComfyUI
            https://github.com/comfy-org/ComfyUI/wiki/API
Ecosystem:  OSS Community
"""

from __future__ import annotations

import time
import uuid

import httpx

from rune_bench.debug import debug_log


class ComfyUIRunner:
    """Art/Creative agent: autonomous art pipeline orchestration via ComfyUI node graph."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", **kwargs) -> None:
        self._base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama", **kwargs) -> str:
        """Run a ComfyUI workflow with the given prompt and return output image info."""
        # Minimal SDXL workflow template (API format)
        # Node IDs are standard for default ComfyUI simple workflow
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "seed": int(time.time()),
                    "steps": 20,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model or "sd_xl_base_1.0.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": 1024, "width": 1024},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": question},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": "text, watermark"},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "RUNE_Art", "images": ["8", 0]},
            },
        }

        with httpx.Client(base_url=self._base_url, timeout=30.0) as client:
            try:
                # 1. Queue prompt
                payload = {"prompt": workflow, "client_id": self._client_id}
                resp = client.post("/prompt", json=payload)
                resp.raise_for_status()
                prompt_id = resp.json()["prompt_id"]
                debug_log(f"ComfyUI: prompt queued (ID: {prompt_id})")

                # 2. Poll for history (max 2 mins)
                for _ in range(120):
                    time.sleep(1)
                    hist_resp = client.get(f"/history/{prompt_id}")
                    if hist_resp.status_code == 200:
                        hist_data = hist_resp.json()
                        if prompt_id in hist_data:
                            # Success
                            outputs = hist_data[prompt_id].get("outputs", {})
                            # Node 9 is our SaveImage node
                            if "9" in outputs:
                                image_info = outputs["9"]["images"][0]
                                filename = image_info["filename"]
                                subfolder = image_info["subfolder"]
                                type_ = image_info["type"]
                                view_url = f"{self._base_url}/view?filename={filename}&subfolder={subfolder}&type={type_}"
                                return f"Generated image: {view_url}"
                            return "ComfyUI: Generation finished but no output image found."

                return "ComfyUI: Timeout waiting for generation."
            except Exception as exc:
                return f"ComfyUI error: {exc}"

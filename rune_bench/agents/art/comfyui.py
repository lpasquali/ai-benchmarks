# SPDX-License-Identifier: Apache-2.0
"""ComfyUI agentic runner — delegates to the ComfyUI driver.

Scope:      Art/Creative  |  Rank 2  |  Rating 4.5
Capability: Node-based autonomous art pipeline orchestration.
Docs:       https://github.com/comfy-org/ComfyUI
Ecosystem:  OSS Community
"""

from rune_bench.drivers.comfyui import ComfyUIDriverClient

# Backwards-compatible alias so existing imports of ComfyUIRunner keep working.
ComfyUIRunner = ComfyUIDriverClient

__all__ = ["ComfyUIRunner", "ComfyUIDriverClient"]

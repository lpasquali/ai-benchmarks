# SPDX-License-Identifier: Apache-2.0
"""InvokeAI agentic runner — delegates to the InvokeAI driver.

Scope:      Art/Creative  |  Rank 3  |  Rating 4.0
Capability: Autonomous art generation and image-to-image refinement.
Docs:       https://github.com/invoke-ai/InvokeAI
Ecosystem:  OSS Community / Local Server
"""

from __future__ import annotations

from rune_bench.drivers.invokeai import InvokeAIDriverClient

# Alias for registry resolution
InvokeAIRunner = InvokeAIDriverClient

__all__ = ["InvokeAIRunner", "InvokeAIDriverClient"]

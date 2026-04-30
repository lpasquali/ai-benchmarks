# SPDX-License-Identifier: Apache-2.0
"""Krea AI agentic runner — delegates to the Krea driver.

Scope:      Art/Creative  |  Rank 3  |  Rating 4.2
Capability: Real-time autonomous image enhancement and upscaling.
Docs:       https://www.krea.ai/
Ecosystem:  SaaS API
"""

from rune_bench.drivers.krea import KreaDriverClient

# Backwards-compatible alias so existing imports of KreaRunner keep working.
KreaRunner = KreaDriverClient

__all__ = ["KreaRunner", "KreaDriverClient"]

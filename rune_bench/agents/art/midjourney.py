# SPDX-License-Identifier: Apache-2.0
"""Midjourney agentic runner — delegates to the Midjourney driver.

Scope:      Art/Creative  |  Rank 1  |  Rating 5.0
Capability: Iterative agentic refinement via "Remix" modes.
Docs:       https://docs.midjourney.com/
Ecosystem:  Generative AI Ethics
"""

from rune_bench.drivers.midjourney import MidjourneyDriverClient

# Backwards-compatible alias so existing imports of MidjourneyRunner keep working.
MidjourneyRunner = MidjourneyDriverClient

__all__ = ["MidjourneyRunner", "MidjourneyDriverClient"]

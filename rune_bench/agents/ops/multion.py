# SPDX-License-Identifier: Apache-2.0
"""MultiOn agentic runner — delegates to the MultiOn driver.

Scope:      Ops/Misc  |  Rank 3  |  Rating 4.5
Capability: Browser-based "Action" agent (The Hands).
Docs:       https://docs.multion.ai/
Ecosystem:  AAIF (Agentic)
"""

from rune_bench.drivers.multion import MultiOnDriverClient

# Backwards-compatible alias so existing imports of MultiOnRunner keep working.
MultiOnRunner = MultiOnDriverClient

__all__ = ["MultiOnRunner", "MultiOnDriverClient"]

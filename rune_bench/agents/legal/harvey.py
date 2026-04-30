# SPDX-License-Identifier: Apache-2.0
"""Harvey AI agentic runner — delegates to the Harvey driver.

Scope:      Legal  |  Rank 1  |  Rating 4.8
Capability: Autonomous legal disclosure and risk analysis.
Docs:       https://www.harvey.ai/
Ecosystem:  Transparency Manifestos
"""

from rune_bench.drivers.harvey import HarveyDriverClient

# Backwards-compatible alias so existing imports of HarveyAIRunner keep working.
HarveyAIRunner = HarveyDriverClient

__all__ = ["HarveyAIRunner", "HarveyDriverClient"]

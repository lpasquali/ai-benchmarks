# SPDX-License-Identifier: Apache-2.0
"""Sierra agentic runner — delegates to the Sierra driver.

Scope:      Legal/Ops  |  Rank 7  |  Rating 4.3
Capability: Autonomous business process orchestration.
Docs:       https://sierra.ai/
Ecosystem:  Enterprise CX
"""

from rune_bench.drivers.sierra import SierraDriverClient

# Backwards-compatible alias so existing imports of SierraRunner keep working.
SierraRunner = SierraDriverClient

__all__ = ["SierraRunner", "SierraDriverClient"]

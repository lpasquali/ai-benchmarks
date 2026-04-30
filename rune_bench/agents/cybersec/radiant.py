# SPDX-License-Identifier: Apache-2.0
"""Radiant Security agentic runner — delegates to the Radiant driver.

Scope:      Cybersec  |  Rank 2  |  Rating 4.5
Capability: Autonomous SOC incident investigation and response.
Docs:       https://radiantsecurity.ai/
Ecosystem:  Enterprise SaaS
"""

from rune_bench.drivers.radiant import RadiantSecurityDriverClient

# Backwards-compatible alias so existing imports of RadiantSecurityRunner keep working.
RadiantSecurityRunner = RadiantSecurityDriverClient

__all__ = ["RadiantSecurityRunner", "RadiantSecurityDriverClient"]

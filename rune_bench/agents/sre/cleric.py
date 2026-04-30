# SPDX-License-Identifier: Apache-2.0
"""Cleric agentic runner — delegates to the Cleric driver.

Scope:      SRE  |  Rank 5  |  Rating 3.5
Capability: Mimics an engineer's "parallel investigation" loop.
Docs:       https://github.com/ClericHQ/cleric
Ecosystem:  Infra Interoperability
"""

from rune_bench.drivers.cleric import ClericDriverClient

# Backwards-compatible alias so existing imports of ClericRunner keep working.
ClericRunner = ClericDriverClient

__all__ = ["ClericRunner", "ClericDriverClient"]

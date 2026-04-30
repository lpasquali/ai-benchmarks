# SPDX-License-Identifier: Apache-2.0
"""Spellbook agentic runner — delegates to the Spellbook driver.

Scope:      Legal/Ops  |  Rank 1  |  Rating 4.9
Capability: Autonomous legal document drafting and contract review.
Docs:       https://www.spellbook.legal/
Ecosystem:  Legal SaaS
"""

from rune_bench.drivers.spellbook import SpellbookDriverClient

# Backwards-compatible alias so existing imports of SpellbookRunner keep working.
SpellbookRunner = SpellbookDriverClient

__all__ = ["SpellbookRunner", "SpellbookDriverClient"]

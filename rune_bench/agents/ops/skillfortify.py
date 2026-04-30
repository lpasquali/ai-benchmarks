# SPDX-License-Identifier: Apache-2.0
"""SkillFortify agentic runner — delegates to the SkillFortify driver.

Scope:      Legal/Ops  |  Rank 7  |  Rating 4.0
Capability: Autonomous professional skill gap analysis.
Docs:       https://skillfortify.com/
Ecosystem:  EduTech Standards
"""

from rune_bench.drivers.skillfortify import SkillFortifyDriverClient

# Backwards-compatible alias so existing imports of SkillFortifyRunner keep working.
SkillFortifyRunner = SkillFortifyDriverClient

__all__ = ["SkillFortifyRunner", "SkillFortifyDriverClient"]

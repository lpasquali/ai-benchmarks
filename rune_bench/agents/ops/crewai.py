# SPDX-License-Identifier: Apache-2.0
"""CrewAI agentic runner — delegates to the CrewAI driver.

Scope:      Ops/Misc  |  Rank 5  |  Rating 4.0
Capability: Orchestrates groups of agents to complete complex tasks.
Docs:       https://docs.crewai.com/
            https://docs.crewai.com/concepts/crews
            https://docs.crewai.com/concepts/agents
            https://docs.crewai.com/concepts/tasks
Ecosystem:  OSS Framework

The actual implementation lives in :mod:`rune_bench.drivers.crewai`.
This module re-exports :class:`CrewAIDriverClient` as ``CrewAIRunner``
for backward compatibility with existing call-sites.
"""

from rune_bench.drivers.crewai import CrewAIDriverClient as CrewAIRunner

__all__ = ["CrewAIRunner"]

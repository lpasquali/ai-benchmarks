"""Mindgard agentic runner — delegates to the Mindgard driver for AI red-teaming.

Scope:      Cybersec  |  Rank 3  |  Rating 4.0
Capability: Autonomous "Red Teaming" for AI model safety.
Docs:       https://mindgard.ai/
            https://docs.mindgard.ai/
            https://github.com/Mindgard/cli
Ecosystem:  AI Security

Implementation notes:
- Install:  pip install mindgard  (CLI + Python SDK)
            https://github.com/Mindgard/cli
- Auth:     RUNE_MINDGARD_API_KEY env var  (register at https://mindgard.ai/)
- Approach: Run automated red-team attacks against an AI model endpoint.
            Mindgard tests for jailbreaks, prompt injection, data extraction, etc.
- ``backend_url`` is the model endpoint being **attacked** (target under test).
- `model` identifies the target model.
- `question` maps to the red-team prompt/objective.
"""

from rune_bench.drivers.mindgard import MindgardDriverClient

MindgardRunner = MindgardDriverClient

__all__ = ["MindgardRunner"]

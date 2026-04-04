"""Elicit agentic runner — delegates to the Elicit driver for literature review.

Scope:      Research  |  Rank 3  |  Rating 4.0
Capability: Automates literature review and data extraction.
Docs:       https://elicit.com/
            https://elicit.com/api  (API access via waitlist as of writing)
Ecosystem:  Open Science

Implementation notes:
- Auth:     RUNE_ELICIT_API_KEY env var (request access at https://elicit.com/api)
- SDK:      REST API via driver (no public Python SDK)
- Approach: Submit a research question; Elicit searches academic databases,
            extracts structured data from papers, and returns a synthesis.
- The `question` maps to the research search query.
- `model` and `ollama_url` are passed through but not used by Elicit.
"""

from rune_bench.drivers.elicit import ElicitDriverClient

ElicitRunner = ElicitDriverClient

__all__ = ["ElicitRunner"]

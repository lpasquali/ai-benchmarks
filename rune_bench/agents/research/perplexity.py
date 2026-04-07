# SPDX-License-Identifier: Apache-2.0
"""Perplexity Pro agentic runner — delegates to the Perplexity driver.

Scope:      Research  |  Rank 1  |  Rating 5.0
Capability: Multi-step research with autonomous source validation.
Docs:       https://docs.perplexity.ai/
            https://docs.perplexity.ai/reference/post_chat_completions
Ecosystem:  Open Web Standards
"""

from rune_bench.drivers.perplexity import PerplexityDriverClient

PerplexityRunner = PerplexityDriverClient

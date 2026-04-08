# SPDX-License-Identifier: Apache-2.0
"""LangGraph agentic runner — delegates to the LangGraph driver.

Scope:      Research  |  Rank 4  |  Rating 4.0
Capability: Framework for building stateful multi-agent research flows.
Docs:       https://langchain-ai.github.io/langgraph/
            https://langchain-ai.github.io/langgraph/concepts/
            https://langchain-ai.github.io/langgraph/tutorials/introduction/
Ecosystem:  OSS / LangChain

The actual implementation lives in :mod:`rune_bench.drivers.langgraph`.
This module re-exports :class:`LangGraphDriverClient` as ``LangGraphRunner``
for backward compatibility with existing call-sites.
"""

from rune_bench.drivers.langgraph import LangGraphDriverClient as LangGraphRunner

__all__ = ["LangGraphRunner"]

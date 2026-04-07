# SPDX-License-Identifier: Apache-2.0
"""Experimental Multi-Tier Memory Provider for autonomous agents.

Supports episodic (short-term session logs), semantic (long-term domain knowledge),
and procedural (cached tool sequences) memory.
"""

import json
from typing import Any, Dict, List, Optional


class MemoryProvider:
    """In-memory provider for agentic memory tiers."""

    def __init__(self) -> None:
        self._episodic: List[Dict[str, Any]] = []
        self._semantic: Dict[str, str] = {}
        self._procedural: Dict[str, List[str]] = {}

    def append_episodic(self, action: str, result: str) -> None:
        """Log a short-term action and result from the current session."""
        self._episodic.append({"action": action, "result": result})

    def get_episodic_context(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the last N actions taken in the current session."""
        return self._episodic[-last_n:]

    def store_semantic(self, key: str, knowledge: str) -> None:
        """Store long-term domain knowledge (e.g., error codes)."""
        self._semantic[key] = knowledge

    def retrieve_semantic(self, key: str) -> Optional[str]:
        """Retrieve domain knowledge by key."""
        return self._semantic.get(key)

    def cache_procedure(self, goal: str, steps: List[str]) -> None:
        """Cache a successful sequence of tool executions."""
        self._procedural[goal] = steps

    def get_procedure(self, goal: str) -> Optional[List[str]]:
        """Retrieve a cached sequence of tool executions for a given goal."""
        return self._procedural.get(goal)

    def dump_memory_state(self) -> str:
        """Serialize the entire memory state for debugging or persistence."""
        return json.dumps(
            {
                "episodic": self._episodic,
                "semantic": self._semantic,
                "procedural": self._procedural,
            }
        )

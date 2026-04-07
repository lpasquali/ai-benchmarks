# SPDX-License-Identifier: Apache-2.0
"""Consensus agentic runner — evidence-based synthesis from academic papers.

Scope:      Research  |  Rank 5  |  Rating 3.5
Capability: Synthesizes answers from 200M+ academic papers via Semantic Scholar.
Docs:       https://consensus.app/
Ecosystem:  Evidence-Based Research

Implementation: delegates to the consensus driver process which queries the
Semantic Scholar API and optionally synthesizes answers via Ollama.
"""

from rune_bench.drivers.consensus import ConsensusDriverClient

ConsensusRunner = ConsensusDriverClient

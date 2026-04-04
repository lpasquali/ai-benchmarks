"""Catalog data model for RUNE benchmark scopes, agents, and chain definitions.

All types are plain dataclasses (no third-party dependencies) so they are
usable anywhere the catalog is loaded, regardless of whether PyYAML is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class QuestionSpec:
    """A single question posed to an agent and the expected agentic action."""

    text: str
    action: str  # description of what the agent does in response


@dataclass
class AgentSpec:
    """One agent entry within a scope."""

    name: str
    rank: int              # 1 = best in scope
    rating: float          # 0.0–5.0 community rating
    capability: str        # one-line description of the agent's agentic ability
    questions: list[QuestionSpec]  # Q1 (technical), Q2 (investigation), Q3 (optimisation)
    github: str            # link to docs / repo
    ecosystem: str         # e.g. "CNCF Sandbox", "OSS Framework"


@dataclass
class ChainStep:
    """One step in a multi-agent chain pipeline.

    ``input_from`` references the ``id`` of the preceding step whose output is
    injected as context into this step's prompt.  A value of ``None`` marks the
    chain entry point.
    """

    id: str
    agent: str             # must match an AgentSpec.name in the same scope
    role: str              # human label, e.g. "Legal Analyst", "Orchestrator"
    question: str          # question asked at this step in the chain context
    input_from: str | None = None  # predecessor step id; None = entry point


@dataclass
class ChainSpec:
    """Full pipeline definition for a chain-mode scope."""

    scope: str             # must match a ScopeSpec.name
    name: str              # human name, e.g. "NDA Processing Pipeline"
    trigger: str           # the top-level task description that kicks off the chain
    steps: list[ChainStep]

    def entry_point(self) -> ChainStep:
        """Return the first step (the one with no predecessor)."""
        for step in self.steps:
            if step.input_from is None:
                return step
        raise ValueError(f"Chain {self.name!r} has no entry point (no step with input_from=null)")

    def step_by_id(self, step_id: str) -> ChainStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def ordered_steps(self) -> list[ChainStep]:
        """Return steps in execution order (topological, linear chains only)."""
        by_id = {s.id: s for s in self.steps}
        by_input: dict[str | None, ChainStep] = {s.input_from: s for s in self.steps}
        ordered: list[ChainStep] = []
        current: ChainStep | None = by_input.get(None)
        while current is not None:
            ordered.append(current)
            current = by_input.get(current.id)
        return ordered


@dataclass
class ScopeSpec:
    """A benchmark scope containing agents and an optional chain definition.

    ``mode`` is either ``"atomic"`` (each agent evaluated independently on its
    three questions) or ``"chain"`` (agents collaborate in a pipeline).
    """

    name: str              # e.g. "SRE", "Legal/Ops"
    model: str             # default Ollama model for this scope
    mode: str              # "atomic" | "chain"
    agents: list[AgentSpec]
    chain: ChainSpec | None = None  # populated when mode="chain"

    def get_agent(self, name: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.name == name), None)


@dataclass
class Catalog:
    """The full RUNE benchmark catalog: all scopes, agents, questions, and chains."""

    scopes: list[ScopeSpec]

    def get_scope(self, name: str) -> ScopeSpec | None:
        return next((s for s in self.scopes if s.name == name), None)

    def atomic_scopes(self) -> list[ScopeSpec]:
        """Return scopes that use the independent per-agent evaluation mode."""
        return [s for s in self.scopes if s.mode == "atomic"]

    def chain_scopes(self) -> list[ScopeSpec]:
        """Return scopes that use the multi-agent pipeline mode."""
        return [s for s in self.scopes if s.mode == "chain"]

    def __iter__(self) -> Iterator[ScopeSpec]:
        return iter(self.scopes)

    def __len__(self) -> int:
        return len(self.scopes)

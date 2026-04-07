# SPDX-License-Identifier: Apache-2.0
"""RUNE benchmark catalog — scopes, agents, questions, and chain definitions.

Quick start::

    from rune_bench.catalog import load_catalog, Catalog, ScopeSpec

    catalog = load_catalog()          # uses bundled defaults (CSV + chains.yaml)
    for scope in catalog:
        print(scope.name, scope.mode, scope.model)

    legal = catalog.get_scope("Legal/Ops")
    if legal and legal.chain:
        for step in legal.chain.ordered_steps():
            print(step.role, "→", step.agent, ":", step.question)
"""

from .loader import load as load_catalog
from .loader import load_from_csv, load_from_yaml, merge_chains
from .models import (
    AgentSpec,
    Catalog,
    ChainSpec,
    ChainStep,
    QuestionSpec,
    ScopeSpec,
)

__all__ = [
    "load_catalog",
    "load_from_csv",
    "load_from_yaml",
    "merge_chains",
    "AgentSpec",
    "Catalog",
    "ChainSpec",
    "ChainStep",
    "QuestionSpec",
    "ScopeSpec",
]

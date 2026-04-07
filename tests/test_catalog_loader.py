# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.catalog — CSV loading, YAML overlay, and auto-detection."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rune_bench.catalog import (
    ChainSpec,
    load_catalog,
    load_from_csv,
    merge_chains,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal CSV and YAML written to tmp_path
# ---------------------------------------------------------------------------

_MINIMAL_CSV = textwrap.dedent("""\
    Scope,Rank,Agent Name,Rating,Agentic Capability,Q1 (Technical/Standard),Q1 Agentic Action,Q2 (Investigation/Deep),Q2 Agentic Action,Q3 (Optimization/Logic),Q3 Agentic Action,GitHub/Docs Link,Ecosystem
    TestScope,1,AgentAlpha,4.5,Does alpha things.,Q1 text,Q1 action,Q2 text,Q2 action,Q3 text,Q3 action,https://alpha.example,OSS
    TestScope,2,AgentBeta,3.0,Does beta things.,Q1b text,Q1b action,Q2b text,Q2b action,Q3b text,Q3b action,https://beta.example,Enterprise
""")

_MINIMAL_CHAINS_CSV = textwrap.dedent("""\
    Scope,Rank,Agent Name,Rating,Agentic Capability,Q1 (Technical/Standard),Q1 Agentic Action,Q2 (Investigation/Deep),Q2 Agentic Action,Q3 (Optimization/Logic),Q3 Agentic Action,GitHub/Docs Link,Ecosystem,Ollama Model (2026)
    TestScope,1,AgentAlpha,4.5,Does alpha things.,Q1 text,Q1 action,Q2 text,Q2 action,Q3 text,Q3 action,https://alpha.example,OSS,deepseek-r1:32b
    TestScope,2,AgentBeta,3.0,Does beta things.,Q1b text,Q1b action,Q2b text,Q2b action,Q3b text,Q3b action,https://beta.example,Enterprise,deepseek-r1:32b
""")

_CHAIN_YAML = textwrap.dedent("""\
    chains:
      - scope: TestScope
        name: Test Pipeline
        trigger: "Run the test pipeline end-to-end."
        steps:
          - id: start
            agent: AgentAlpha
            role: Orchestrator
            question: "Kick off the pipeline."
            input_from: null
          - id: finish
            agent: AgentBeta
            role: Executor
            question: "Complete the pipeline task."
            input_from: start
""")

_SCOPES_YAML = textwrap.dedent("""\
    scopes:
      - name: YamlScope
        model: llama3:8b
        mode: atomic
        agents:
          - name: YamlAgent
            rank: 1
            rating: 4.0
            capability: "YAML-defined agent."
            github: https://example.com
            ecosystem: OSS
            questions:
              - text: "YAML Q1?"
                action: "Does YAML Q1 action."
              - text: "YAML Q2?"
                action: "Does YAML Q2 action."
              - text: "YAML Q3?"
                action: "Does YAML Q3 action."
""")


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    p = tmp_path / "scopes.csv"
    p.write_text(_MINIMAL_CSV, encoding="utf-8")
    return p


@pytest.fixture
def tmp_chains_csv(tmp_path: Path) -> Path:
    p = tmp_path / "chains.csv"
    p.write_text(_MINIMAL_CHAINS_CSV, encoding="utf-8")
    return p


@pytest.fixture
def tmp_chain_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "chains.yaml"
    p.write_text(_CHAIN_YAML, encoding="utf-8")
    return p


@pytest.fixture
def tmp_scopes_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "scopes.yaml"
    p.write_text(_SCOPES_YAML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_from_csv — basic structure
# ---------------------------------------------------------------------------


def test_load_from_csv_scope_count(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert len(catalog.scopes) == 1


def test_load_from_csv_scope_name(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert catalog.scopes[0].name == "TestScope"


def test_load_from_csv_agent_count(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert len(catalog.scopes[0].agents) == 2


def test_load_from_csv_agent_ordering_by_rank(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    ranks = [a.rank for a in catalog.scopes[0].agents]
    assert ranks == sorted(ranks)


def test_load_from_csv_agent_fields(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    alpha = catalog.scopes[0].agents[0]
    assert alpha.name == "AgentAlpha"
    assert alpha.rank == 1
    assert alpha.rating == 4.5
    assert alpha.capability == "Does alpha things."
    assert alpha.github == "https://alpha.example"
    assert alpha.ecosystem == "OSS"


def test_load_from_csv_questions(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    alpha = catalog.scopes[0].agents[0]
    assert len(alpha.questions) == 3
    assert alpha.questions[0].text == "Q1 text"
    assert alpha.questions[0].action == "Q1 action"
    assert alpha.questions[2].text == "Q3 text"


def test_load_from_csv_default_mode_is_atomic(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert catalog.scopes[0].mode == "atomic"


def test_load_from_csv_no_chain_attached(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert catalog.scopes[0].chain is None


# ---------------------------------------------------------------------------
# load_from_csv — model enrichment from chains.csv
# ---------------------------------------------------------------------------


def test_load_from_chains_csv_extracts_model(tmp_chains_csv: Path):
    catalog = load_from_csv(tmp_chains_csv)
    assert catalog.scopes[0].model == "deepseek-r1:32b"


def test_load_from_scopes_csv_uses_default_model(tmp_csv: Path):
    from rune_bench.catalog.loader import _DEFAULT_MODEL
    catalog = load_from_csv(tmp_csv)
    assert catalog.scopes[0].model == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# merge_chains — YAML topology overlay
# ---------------------------------------------------------------------------


def test_merge_chains_upgrades_mode(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    assert merged.get_scope("TestScope").mode == "chain"


def test_merge_chains_attaches_chain_spec(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    chain = merged.get_scope("TestScope").chain
    assert chain is not None
    assert chain.name == "Test Pipeline"
    assert chain.trigger == "Run the test pipeline end-to-end."


def test_merge_chains_step_count(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    assert len(merged.get_scope("TestScope").chain.steps) == 2


def test_merge_chains_entry_point(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    entry = merged.get_scope("TestScope").chain.entry_point()
    assert entry.id == "start"
    assert entry.agent == "AgentAlpha"
    assert entry.input_from is None


def test_merge_chains_ordered_steps(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    steps = merged.get_scope("TestScope").chain.ordered_steps()
    assert [s.id for s in steps] == ["start", "finish"]


def test_merge_chains_step_by_id(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    step = merged.get_scope("TestScope").chain.step_by_id("finish")
    assert step is not None
    assert step.agent == "AgentBeta"
    assert step.role == "Executor"
    assert step.input_from == "start"


def test_merge_chains_unrelated_scope_stays_atomic(tmp_csv: Path, tmp_chain_yaml: Path):
    """Scopes not mentioned in chains.yaml must remain atomic."""
    extra_csv = tmp_csv.parent / "extra.csv"
    extra_csv.write_text(
        _MINIMAL_CSV.replace("TestScope", "OtherScope"),
        encoding="utf-8",
    )
    catalog = load_from_csv(extra_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    assert merged.get_scope("OtherScope").mode == "atomic"


def test_merge_chains_missing_yaml_raises(tmp_csv: Path, tmp_path: Path):
    catalog = load_from_csv(tmp_csv)
    with pytest.raises(FileNotFoundError):
        merge_chains(catalog, tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# load_from_yaml
# ---------------------------------------------------------------------------


def test_load_from_yaml_scope(tmp_scopes_yaml: Path):
    pytest.importorskip("yaml", reason="pyyaml not installed")
    from rune_bench.catalog import load_from_yaml
    catalog = load_from_yaml(tmp_scopes_yaml)
    assert len(catalog.scopes) == 1
    scope = catalog.scopes[0]
    assert scope.name == "YamlScope"
    assert scope.model == "llama3:8b"
    assert scope.mode == "atomic"


def test_load_from_yaml_agent(tmp_scopes_yaml: Path):
    pytest.importorskip("yaml", reason="pyyaml not installed")
    from rune_bench.catalog import load_from_yaml
    catalog = load_from_yaml(tmp_scopes_yaml)
    agent = catalog.scopes[0].agents[0]
    assert agent.name == "YamlAgent"
    assert agent.rating == 4.0
    assert agent.questions[0].text == "YAML Q1?"


def test_load_from_yaml_with_chain(tmp_scopes_yaml: Path, tmp_path: Path):
    pytest.importorskip("yaml", reason="pyyaml not installed")
    from rune_bench.catalog import load_from_yaml
    chain_yaml = tmp_path / "chains.yaml"
    chain_yaml.write_text(
        textwrap.dedent("""\
            chains:
              - scope: YamlScope
                name: Yaml Chain
                trigger: "Go."
                steps:
                  - id: only
                    agent: YamlAgent
                    role: Solo
                    question: "Do the thing."
                    input_from: null
        """),
        encoding="utf-8",
    )
    catalog = load_from_yaml(tmp_scopes_yaml, chain_yaml)
    assert catalog.get_scope("YamlScope").mode == "chain"
    assert catalog.get_scope("YamlScope").chain.name == "Yaml Chain"


# ---------------------------------------------------------------------------
# load() auto-detection
# ---------------------------------------------------------------------------


def test_load_auto_detects_csv(tmp_path: Path):
    (tmp_path / "scopes.csv").write_text(_MINIMAL_CSV, encoding="utf-8")
    catalog = load_catalog(catalog_dir=tmp_path)
    assert catalog.get_scope("TestScope") is not None


def test_load_auto_prefers_chains_csv(tmp_path: Path):
    (tmp_path / "scopes.csv").write_text(_MINIMAL_CSV, encoding="utf-8")
    (tmp_path / "chains.csv").write_text(_MINIMAL_CHAINS_CSV, encoding="utf-8")
    catalog = load_catalog(catalog_dir=tmp_path)
    # chains.csv has model column; scopes.csv does not
    assert catalog.get_scope("TestScope").model == "deepseek-r1:32b"


def test_load_merges_chains_yaml_automatically(tmp_path: Path):
    (tmp_path / "scopes.csv").write_text(_MINIMAL_CSV, encoding="utf-8")
    (tmp_path / "chains.yaml").write_text(_CHAIN_YAML, encoding="utf-8")
    catalog = load_catalog(catalog_dir=tmp_path)
    assert catalog.get_scope("TestScope").mode == "chain"


def test_load_prefers_scopes_yaml_over_csv(tmp_path: Path):
    pytest.importorskip("yaml", reason="pyyaml not installed")
    (tmp_path / "scopes.csv").write_text(_MINIMAL_CSV, encoding="utf-8")
    (tmp_path / "scopes.yaml").write_text(_SCOPES_YAML, encoding="utf-8")
    catalog = load_catalog(catalog_dir=tmp_path)
    # scopes.yaml defines YamlScope only — CSV's TestScope must not appear
    assert catalog.get_scope("YamlScope") is not None
    assert catalog.get_scope("TestScope") is None


def test_load_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_catalog(catalog_dir=tmp_path / "nonexistent")


def test_load_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_catalog(catalog_dir=tmp_path)


# ---------------------------------------------------------------------------
# Bundled defaults
# ---------------------------------------------------------------------------


def test_bundled_defaults_load():
    """The shipped CSV files must load without error."""
    catalog = load_catalog()
    assert len(catalog.scopes) > 0


def test_bundled_defaults_all_five_scopes():
    catalog = load_catalog()
    names = {s.name for s in catalog.scopes}
    assert names == {"SRE", "Research", "Art/Creative", "Cybersec", "Legal/Ops"}


def test_bundled_defaults_each_scope_has_agents():
    catalog = load_catalog()
    for scope in catalog:
        assert len(scope.agents) > 0, f"{scope.name} has no agents"


def test_bundled_defaults_each_agent_has_three_questions():
    catalog = load_catalog()
    for scope in catalog:
        for agent in scope.agents:
            assert len(agent.questions) == 3, (
                f"{scope.name}/{agent.name} does not have exactly 3 questions"
            )


def test_bundled_defaults_models_assigned():
    catalog = load_catalog()
    for scope in catalog:
        assert scope.model, f"{scope.name} has no model assigned"


def test_bundled_defaults_legal_ops_is_chain():
    """Legal/Ops must be chain mode because chains.yaml ships alongside."""
    catalog = load_catalog()
    legal = catalog.get_scope("Legal/Ops")
    assert legal is not None
    assert legal.mode == "chain"
    assert legal.chain is not None


def test_bundled_defaults_legal_ops_chain_has_five_steps():
    catalog = load_catalog()
    chain = catalog.get_scope("Legal/Ops").chain
    assert len(chain.steps) == 5


def test_bundled_defaults_legal_ops_chain_entry_point():
    catalog = load_catalog()
    entry = catalog.get_scope("Legal/Ops").chain.entry_point()
    assert entry.agent == "CrewAI"
    assert entry.input_from is None


def test_bundled_defaults_legal_ops_chain_ordered():
    catalog = load_catalog()
    steps = catalog.get_scope("Legal/Ops").chain.ordered_steps()
    agents = [s.agent for s in steps]
    # Linear chain: CrewAI → Harvey AI → Spellbook → MultiOn → Dagger
    assert agents[0] == "CrewAI"
    assert agents[-1] == "Dagger"
    assert len(agents) == 5


def test_bundled_defaults_atomic_scopes():
    catalog = load_catalog()
    atomic = {s.name for s in catalog.atomic_scopes()}
    assert "SRE" in atomic
    assert "Research" in atomic
    assert "Legal/Ops" not in atomic


def test_bundled_defaults_chain_scopes():
    catalog = load_catalog()
    chain_names = {s.name for s in catalog.chain_scopes()}
    assert "Legal/Ops" in chain_names


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def test_catalog_get_scope_returns_none_for_unknown(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert catalog.get_scope("DoesNotExist") is None


def test_catalog_iteration(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    scopes = list(catalog)
    assert len(scopes) == 1


def test_catalog_len(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    assert len(catalog) == 1


def test_scope_get_agent(tmp_csv: Path):
    catalog = load_from_csv(tmp_csv)
    scope = catalog.scopes[0]
    assert scope.get_agent("AgentAlpha") is not None
    assert scope.get_agent("NoSuchAgent") is None


# ---------------------------------------------------------------------------
# ChainSpec helpers
# ---------------------------------------------------------------------------


def test_chain_spec_entry_point_raises_when_none(tmp_csv: Path):
    """A chain where all steps have a predecessor must raise."""
    broken = ChainSpec(
        scope="X",
        name="Broken",
        trigger="",
        steps=[],
    )
    with pytest.raises(ValueError, match="no entry point"):
        broken.entry_point()


def test_chain_spec_step_by_id_none_when_missing(tmp_csv: Path, tmp_chain_yaml: Path):
    catalog = load_from_csv(tmp_csv)
    merged = merge_chains(catalog, tmp_chain_yaml)
    chain = merged.get_scope("TestScope").chain
    assert chain.step_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# rune_bench lazy __getattr__ and resources __getattr__
# ---------------------------------------------------------------------------


def test_rune_bench_getattr_lazy_loads_vastai():
    import rune_bench
    # Should return the class without error
    klass = rune_bench.OfferFinder
    assert klass.__name__ == "OfferFinder"


def test_rune_bench_getattr_raises_for_unknown():
    import rune_bench
    with pytest.raises(AttributeError):
        _ = rune_bench.NonExistentThing


def test_resources_getattr_lazy_loads_vastai_provider():
    import rune_bench.resources as resources_pkg
    klass = resources_pkg.VastAIProvider
    assert klass.__name__ == "VastAIProvider"


def test_resources_getattr_raises_for_unknown():
    import rune_bench.resources as resources_pkg
    with pytest.raises(AttributeError):
        _ = resources_pkg.NonExistentThing

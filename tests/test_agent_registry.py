"""Tests for the agent registry, protocol, and stubs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune_bench.agents.base import AgentResult, AgentRunner
from rune_bench.agents.registry import (
    _BUILTIN_AGENTS,
    _REGISTRY,
    get_agent,
    list_agents,
    register_agent,
)
from rune_bench.agents.stubs import NotConfiguredError


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


def test_list_agents_returns_all_builtin():
    agents = list_agents()
    assert len(agents) == len(_BUILTIN_AGENTS)
    for name in _BUILTIN_AGENTS:
        assert name in agents


def test_list_agents_is_sorted():
    agents = list_agents()
    assert agents == sorted(agents)


def test_list_agents_includes_custom_after_register():
    _REGISTRY.pop("test_custom_agent", None)
    try:
        register_agent("test_custom_agent", lambda: MagicMock())
        assert "test_custom_agent" in list_agents()
    finally:
        _REGISTRY.pop("test_custom_agent", None)


# ---------------------------------------------------------------------------
# get_agent -- Holmes (real builtin)
# ---------------------------------------------------------------------------


def test_get_agent_holmes(tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1")
    runner = get_agent("holmes", kubeconfig=kubeconfig)
    assert hasattr(runner, "ask")


# ---------------------------------------------------------------------------
# get_agent -- unknown
# ---------------------------------------------------------------------------


def test_get_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown agent 'does_not_exist'"):
        get_agent("does_not_exist")


# ---------------------------------------------------------------------------
# register_agent -- custom factory
# ---------------------------------------------------------------------------


def test_register_agent_custom_factory():
    sentinel = object()
    _REGISTRY.pop("my_custom", None)
    try:
        register_agent("my_custom", lambda: sentinel)
        assert get_agent("my_custom") is sentinel
    finally:
        _REGISTRY.pop("my_custom", None)


def test_register_agent_shadows_builtin():
    """Custom registration should take priority over the built-in map."""
    sentinel = object()
    _REGISTRY.pop("holmes", None)
    try:
        register_agent("holmes", lambda **kw: sentinel)
        assert get_agent("holmes") is sentinel
    finally:
        _REGISTRY.pop("holmes", None)


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


def test_agent_result_defaults():
    r = AgentResult(answer="hello")
    assert r.answer == "hello"
    assert r.result_type == "text"
    assert r.artifacts is None
    assert r.metadata is None


def test_agent_result_structured():
    r = AgentResult(
        answer="report",
        result_type="report",
        artifacts=[{"path": "/tmp/out.pdf"}],  # nosec  # test artifact paths
        metadata={"pages": 3},
    )
    assert r.result_type == "report"
    assert r.artifacts == [{"path": "/tmp/out.pdf"}]  # nosec  # test artifact paths


# ---------------------------------------------------------------------------
# AgentRunner protocol -- structural subtyping
# ---------------------------------------------------------------------------


def test_agent_runner_protocol_satisfied():
    """Any object with an ``ask`` method matching the signature satisfies the protocol."""

    class FakeRunner:
        def ask(self, question: str, model: str, ollama_url: str | None = None, **kwargs) -> str:
            return "ok"

    assert isinstance(FakeRunner(), AgentRunner)


# ---------------------------------------------------------------------------
# NotConfiguredError
# ---------------------------------------------------------------------------


def test_not_configured_error_is_runtime_error():
    with pytest.raises(RuntimeError):
        raise NotConfiguredError("missing key")


# ---------------------------------------------------------------------------
# Builtin count
# ---------------------------------------------------------------------------


def test_builtin_agent_count():
    """Ensure the built-in map contains known core agents and a reasonable size."""
    for expected in ("holmes", "dagger", "perplexity"):
        assert expected in _BUILTIN_AGENTS, f"Expected built-in agent {expected!r} missing"
    assert len(_BUILTIN_AGENTS) >= 20, (
        f"Expected at least 20 built-in agents, got {len(_BUILTIN_AGENTS)}"
    )

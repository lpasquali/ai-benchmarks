"""Tests for agent registry and configuration validation."""

import pytest
from pathlib import Path
from rune_bench.agents.registry import get_agent, register_agent, list_agents


def test_get_agent_success(monkeypatch, tmp_path):
    kube_path = tmp_path / "config"
    kube_path.touch()
    monkeypatch.setenv("KUBECONFIG", str(kube_path))
    agent = get_agent("holmes", kubeconfig=kube_path)
    assert agent is not None


def test_get_agent_missing_config():
    with pytest.raises(RuntimeError, match="Agent 'holmes' requires KUBECONFIG to be set"):
        get_agent("holmes")


def test_get_agent_custom_registration(monkeypatch):
    class DummyAgent:
        def __init__(self, api_key=None):
            self.api_key = api_key
            
    register_agent("dummy", DummyAgent, required_config=["api_key"])
    
    with pytest.raises(RuntimeError, match="Agent 'dummy' requires RUNE_DUMMY_API_KEY to be set"):
        get_agent("dummy")
        
    monkeypatch.setenv("RUNE_DUMMY_API_KEY", "dummy-key")
    agent = get_agent("dummy")
    assert agent.api_key == "dummy-key"


def test_get_agent_unknown():
    with pytest.raises(ValueError, match="Unknown agent 'unknown'"):
        get_agent("unknown")


def test_get_agent_crewai_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # CrewAI might have other init requirements in tests, but we just want to ensure it passes the config resolution
    # However, CrewAIRunner takes api_key kwargs.
    try:
        agent = get_agent("crewai")
        # CrewAI agent might not store api_key as attribute, but the initialization shouldn't raise a RuntimeError from our check
    except Exception as e:
        # Ignore other exceptions like ModuleNotFoundError or CrewAI init errors, just verify our RuntimeError isn't raised
        assert "requires OPENAI_API_KEY to be set" not in str(e)


def test_get_agent_crewai_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RUNE_CREWAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY to be set"):
        get_agent("crewai")


def test_list_agents():
    agents = list_agents()
    assert "holmes" in agents
    assert "crewai" in agents

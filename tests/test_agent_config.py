"""Tests for agent configuration resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from rune_bench.agents.config import AgentConfig, resolve_agent_config


def test_resolve_reads_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "pplx-secret")
    monkeypatch.setenv("RUNE_PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    monkeypatch.setenv("KUBECONFIG", "/home/user/.kube/config")

    cfg = resolve_agent_config("perplexity")

    assert cfg.api_key == "pplx-secret"
    assert cfg.base_url == "https://api.perplexity.ai"
    assert cfg.kubeconfig == Path("/home/user/.kube/config")
    assert cfg.extra == {}


def test_resolve_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("RUNE_FOOBAR_API_KEY", raising=False)
    monkeypatch.delenv("RUNE_FOOBAR_BASE_URL", raising=False)
    monkeypatch.delenv("RUNE_KUBECONFIG", raising=False)
    monkeypatch.delenv("KUBECONFIG", raising=False)

    cfg = resolve_agent_config("foobar")

    assert cfg.api_key is None
    assert cfg.base_url is None
    assert cfg.kubeconfig is None


def test_resolve_uppercases_agent_name(monkeypatch):
    monkeypatch.setenv("RUNE_HOLMES_API_KEY", "h-key")

    cfg = resolve_agent_config("holmes")
    assert cfg.api_key == "h-key"


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.api_key is None
    assert cfg.base_url is None
    assert cfg.kubeconfig is None
    assert cfg.extra == {}

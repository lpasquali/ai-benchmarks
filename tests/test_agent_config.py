# SPDX-License-Identifier: Apache-2.0
"""Tests for agent configuration resolution."""

from __future__ import annotations


from rune_bench.agents.config import AgentConfig, resolve_agent_config


def test_resolve_reads_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "pplx-secret")
    monkeypatch.setenv("RUNE_PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    monkeypatch.setenv("KUBECONFIG", "/home/user/.kube/config")

    cfg = resolve_agent_config("perplexity")

    assert cfg.api_key == "pplx-secret"
    assert cfg.base_url == "https://api.perplexity.ai"
    assert cfg.kubeconfig == "/home/user/.kube/config"
    assert cfg.extra == {}


def test_resolve_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("RUNE_FOOBAR_API_KEY", raising=False)
    monkeypatch.delenv("RUNE_FOOBAR_BASE_URL", raising=False)
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

def test_resolve_crewai_fallback(monkeypatch):
    monkeypatch.delenv("RUNE_CREWAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    cfg = resolve_agent_config("crewai")
    assert cfg.api_key == "openai-key"

def test_resolve_comfyui_fallback(monkeypatch):
    monkeypatch.delenv("RUNE_COMFYUI_BASE_URL", raising=False)
    monkeypatch.setenv("COMFYUI_BASE_URL", "http://comfyui")
    cfg = resolve_agent_config("comfyui")
    assert cfg.base_url == "http://comfyui"

def test_resolve_burpgpt_fallback(monkeypatch):
    monkeypatch.delenv("RUNE_BURPGPT_BASE_URL", raising=False)
    monkeypatch.setenv("BURP_API_URL", "http://burp")
    cfg = resolve_agent_config("burpgpt")
    assert cfg.base_url == "http://burp"

def test_resolve_dagger_extra(monkeypatch):
    monkeypatch.setenv("DAGGER_CLOUD_TOKEN", "dagger-token")
    cfg = resolve_agent_config("dagger")
    assert cfg.extra["dagger_cloud_token"] == "dagger-token"

def test_resolve_langgraph_extra(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lang-key")
    cfg = resolve_agent_config("langgraph")
    assert cfg.extra["langchain_api_key"] == "lang-key"

def test_resolve_glean_extra(monkeypatch):
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "glean-inst")
    cfg = resolve_agent_config("glean")
    assert cfg.extra["instance"] == "glean-inst"

def test_resolve_with_kwargs():
    cfg = resolve_agent_config("test", {"kubeconfig": "kwarg-kube", "api_key": "kwarg-api", "base_url": "kwarg-url", "model": "kwarg-model", "backend_url": "kwarg-backend"})
    assert cfg.kubeconfig == "kwarg-kube"
    assert cfg.api_key == "kwarg-api"
    assert cfg.base_url == "kwarg-url"
    assert cfg.model == "kwarg-model"
    assert cfg.backend_url == "kwarg-backend"

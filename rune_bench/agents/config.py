"""Agent configuration resolution framework."""

import os
from typing import Any

from rune_bench.agents.base import AgentConfig

def resolve_agent_config(agent_name: str, kwargs: dict[str, Any] | None = None) -> AgentConfig:
    """Resolve configuration for an agent from environment variables and kwargs.
    
    Env vars follow the pattern RUNE_<AGENT>_<VAR>.
    """
    kwargs = kwargs or {}
    prefix = f"RUNE_{agent_name.upper()}_"
    
    # Kubeconfig is a special case often passed directly in kwargs from the API
    kubeconfig = kwargs.get("kubeconfig") or os.environ.get("KUBECONFIG")
    
    # API Key might be mapped directly or via OPENAI_API_KEY for some agents like crewai
    api_key = kwargs.get("api_key") or os.environ.get(f"{prefix}API_KEY")
    if not api_key and agent_name in ("crewai", "pentestgpt", "burpgpt"):
        api_key = os.environ.get("OPENAI_API_KEY")
        
    base_url = kwargs.get("base_url") or os.environ.get(f"{prefix}BASE_URL")
    if not base_url and agent_name == "comfyui":
        base_url = os.environ.get("COMFYUI_BASE_URL")
    if not base_url and agent_name == "burpgpt":
        base_url = os.environ.get("BURP_API_URL")

    model = kwargs.get("model") or os.environ.get(f"{prefix}MODEL")
    ollama_url = kwargs.get("ollama_url") or os.environ.get(f"{prefix}OLLAMA_URL")
    
    extra = {}
    if agent_name == "dagger":
        extra["dagger_cloud_token"] = os.environ.get("DAGGER_CLOUD_TOKEN", "")
    if agent_name == "langgraph":
        extra["langchain_api_key"] = os.environ.get("LANGCHAIN_API_KEY", "")
    if agent_name == "glean":
        extra["instance"] = os.environ.get(f"{prefix}INSTANCE", "")
        
    return AgentConfig(
        api_key=api_key,
        base_url=base_url,
        kubeconfig=kubeconfig,
        model=model,
        ollama_url=ollama_url,
        extra=extra,
    )

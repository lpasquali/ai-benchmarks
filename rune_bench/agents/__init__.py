"""Multi-domain agent registry, configuration, and execution APIs."""

from .base import AgentResult, AgentRunner
from .config import AgentConfig, resolve_agent_config
from .registry import get_agent, list_agents, register_agent
from .stubs import NotConfiguredError

__all__ = [
    "AgentConfig",
    "AgentResult",
    "AgentRunner",
    "NotConfiguredError",
    "get_agent",
    "list_agents",
    "register_agent",
    "resolve_agent_config",
]

"""Agent authentication and configuration resolution.

Each agent reads its credentials from environment variables following the
convention ``RUNE_<AGENT_NAME>_<KEY>``.  The :func:`resolve_agent_config`
helper materialises a :class:`AgentConfig` from the environment so that
agent factories never need to touch ``os.getenv`` directly.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Bag of credentials / connection details for a single agent."""

    api_key: str | None = None
    base_url: str | None = None
    kubeconfig: Path | None = None
    extra: dict = field(default_factory=dict)


def resolve_agent_config(agent_name: str) -> AgentConfig:
    """Build an :class:`AgentConfig` from environment variables."""
    prefix = f"RUNE_{agent_name.upper()}_"
    kubeconfig_raw = os.getenv("RUNE_KUBECONFIG") or os.getenv("KUBECONFIG")
    return AgentConfig(
        api_key=os.getenv(f"{prefix}API_KEY"),
        base_url=os.getenv(f"{prefix}BASE_URL"),
        kubeconfig=Path(kubeconfig_raw) if kubeconfig_raw else None,
        extra={},
    )

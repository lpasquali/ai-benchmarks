"""Agent authentication and configuration resolution.

Each agent reads its credentials from environment variables following the
convention ``RUNE_<AGENT_NAME>_<KEY>``.  The :func:`resolve_agent_config`
helper materialises a :class:`AgentConfig` from the environment so that
agent factories never need to touch ``os.getenv`` directly.
"""

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Bag of credentials / connection details for a single agent."""

    api_key: str | None = None
    base_url: str | None = None
    kubeconfig: str | None = None
    extra: dict = field(default_factory=dict)


def resolve_agent_config(agent_name: str) -> AgentConfig:
    """Build an :class:`AgentConfig` from environment variables.

    Variables inspected::

        RUNE_<AGENT>_API_KEY
        RUNE_<AGENT>_BASE_URL
        KUBECONFIG

    where ``<AGENT>`` is *agent_name* uppercased.
    """
    prefix = f"RUNE_{agent_name.upper()}_"
    return AgentConfig(
        api_key=os.getenv(f"{prefix}API_KEY"),
        base_url=os.getenv(f"{prefix}BASE_URL"),
        kubeconfig=os.getenv("KUBECONFIG"),
        extra={},
    )

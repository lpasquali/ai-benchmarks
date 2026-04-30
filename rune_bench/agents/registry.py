# SPDX-License-Identifier: Apache-2.0
"""Dynamic agent registry with lazy import resolution.

The registry supports two kinds of agents:

1. **Built-in agents** listed in :data:`_BUILTIN_AGENTS` -- resolved via
   :func:`importlib.import_module` the first time they are requested.
2. **Custom agents** added at runtime via :func:`register_agent`.

Custom registrations take precedence over built-in entries so that
downstream integrations can override the default implementation of any
agent without modifying this module.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from rune_bench.agents.config import resolve_agent_config

_REGISTRY: dict[str, dict] = {}

# Built-in agent map: agent_name -> (module_path, class_name, required_config)
_BUILTIN_AGENTS: dict[str, tuple[str, str, list[str]]] = {
    "holmes": ("rune_bench.drivers.holmes", "HolmesDriverClient", ["kubeconfig"]),
    "k8sgpt": ("rune_bench.drivers.k8sgpt", "K8sGPTDriverClient", ["kubeconfig"]),
    "metoro": (
        "rune_bench.drivers.metoro",
        "MetoroDriverClient",
        ["kubeconfig", "api_key"],
    ),
    "pagerduty": (
        "rune_bench.drivers.pagerduty",
        "PagerDutyDriverClient",
        ["kubeconfig", "api_key"],
    ),
    "perplexity": (
        "rune_bench.drivers.perplexity",
        "PerplexityDriverClient",
        ["api_key"],
    ),
    "glean": ("rune_bench.drivers.glean", "GleanDriverClient", ["api_key"]),
    "elicit": ("rune_bench.drivers.elicit", "ElicitDriverClient", ["api_key"]),
    "langgraph": (
        "rune_bench.drivers.langgraph",
        "LangGraphDriverClient",
        ["kubeconfig"],
    ),
    "consensus": ("rune_bench.drivers.consensus", "ConsensusDriverClient", []),
    "pentestgpt": (
        "rune_bench.drivers.pentestgpt",
        "PentestGPTDriverClient",
        ["api_key"],
    ),
    "radiant": (
        "rune_bench.drivers.radiant",
        "RadiantDriverClient",
        ["api_key", "base_url"],
    ),
    "mindgard": ("rune_bench.drivers.mindgard", "MindgardDriverClient", ["api_key"]),
    "burpgpt": (
        "rune_bench.drivers.burpgpt",
        "BurpGPTDriverClient",
        ["api_key", "base_url"],
    ),
    "xbow": ("rune_bench.drivers.xbow", "XBOWDriverClient", ["api_key"]),
    "harvey": ("rune_bench.drivers.harvey", "HarveyDriverClient", ["api_key"]),
    "spellbook": ("rune_bench.drivers.spellbook", "SpellbookDriverClient", ["api_key"]),
    "dagger": ("rune_bench.drivers.dagger", "DaggerDriverClient", []),
    "crewai": ("rune_bench.drivers.crewai", "CrewAIDriverClient", ["api_key"]),
    "browseruse": ("rune_bench.drivers.browseruse", "BrowserUseDriverClient", ["api_key"]),
    "multion": ("rune_bench.drivers.multion", "MultiOnDriverClient", ["api_key"]),
    "sierra": ("rune_bench.drivers.sierra", "SierraDriverClient", ["api_key"]),
    "cleric": ("rune_bench.drivers.cleric", "ClericDriverClient", ["api_key"]),
    "skillfortify": (
        "rune_bench.drivers.skillfortify",
        "SkillFortifyDriverClient",
        ["api_key"],
    ),
    "midjourney": (
        "rune_bench.drivers.midjourney",
        "MidjourneyDriverClient",
        ["api_key", "base_url"],
    ),
    "invokeai": ("rune_bench.drivers.invokeai", "InvokeAIDriverClient", ["base_url"]),
    "comfyui": ("rune_bench.drivers.comfyui", "ComfyUIDriverClient", ["base_url"]),
    "krea": ("rune_bench.drivers.krea", "KreaDriverClient", ["api_key"]),
}


def register_agent(
    name: str,
    factory: Callable[..., Any],
    *,
    required_config: list[str] | None = None,
) -> None:
    """Register a custom agent factory under *name*.

    Custom registrations shadow built-in entries so callers can override
    the default implementation at runtime.
    """
    _REGISTRY[name] = {
        "factory": factory,
        "required_config": required_config or [],
    }


def get_agent(name: str, **kwargs: Any) -> Any:
    """Return an instantiated agent for *name*.

    Resolution order:
    1. Custom registry (populated by :func:`register_agent`).
    2. Built-in map (lazy ``importlib.import_module``).

    Only kwargs matching the agent's ``required_config`` are forwarded to
    the constructor, so extra kwargs (like ``kubeconfig``) are silently
    dropped for agents that don't declare them.

    Raises:
        ValueError: if *name* is not found in either source.
        RuntimeError: if required configuration is missing.
    """
    if name in _REGISTRY:
        entry = _REGISTRY[name]
        req_config: list[str] = entry.get("required_config", [])
        factory = entry["factory"]
    elif name in _BUILTIN_AGENTS:
        module_path, class_name, req_config = _BUILTIN_AGENTS[name]
        mod = importlib.import_module(module_path)
        factory = getattr(mod, class_name)
    else:
        available = sorted(set(list(_REGISTRY.keys()) + list(_BUILTIN_AGENTS.keys())))
        raise ValueError(f"Unknown agent {name!r}. Available: {', '.join(available)}")

    config = resolve_agent_config(name, kwargs)
    filtered = {}

    for req in req_config:
        val = getattr(config, req, None)
        if not val:
            var_name = f"RUNE_{name.upper()}_{req.upper()}"
            if req == "kubeconfig":
                var_name = "KUBECONFIG"
            elif name in ("crewai", "pentestgpt", "burpgpt") and req == "api_key":
                var_name = "OPENAI_API_KEY"
            raise RuntimeError(f"Agent '{name}' requires {var_name} to be set")
        filtered[req] = val

    return factory(**filtered)


def list_agents() -> list[str]:
    """Return a sorted list of all known agent names."""
    return sorted(set(list(_REGISTRY.keys()) + list(_BUILTIN_AGENTS.keys())))

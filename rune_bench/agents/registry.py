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
    "holmes": ("rune_bench.agents.sre.holmes", "HolmesRunner", ["kubeconfig"]),
    "k8sgpt": ("rune_bench.agents.sre.k8sgpt", "K8sGPTRunner", ["kubeconfig"]),
    "metoro": ("rune_bench.agents.sre.metoro", "MetoroRunner", ["kubeconfig", "api_key"]),
    "pagerduty": ("rune_bench.agents.sre.pagerduty", "PagerDutyAIRunner", ["kubeconfig", "api_key"]),
    "perplexity": ("rune_bench.agents.research.perplexity", "PerplexityRunner", ["api_key"]),
    "glean": ("rune_bench.agents.research.glean", "GleanRunner", ["api_key"]),
    "elicit": ("rune_bench.agents.research.elicit", "ElicitRunner", ["api_key"]),
    "langgraph": ("rune_bench.agents.research.langgraph", "LangGraphRunner", ["kubeconfig"]),
    "consensus": ("rune_bench.agents.research.consensus", "ConsensusRunner", []),
    "pentestgpt": ("rune_bench.agents.cybersec.pentestgpt", "PentestGPTRunner", ["api_key"]),
    "radiant": ("rune_bench.agents.cybersec.radiant", "RadiantSecurityRunner", ["api_key", "base_url"]),
    "mindgard": ("rune_bench.agents.cybersec.mindgard", "MindgardRunner", ["api_key"]),
    "burpgpt": ("rune_bench.agents.cybersec.burpgpt", "BurpGPTRunner", ["api_key", "base_url"]),
    "xbow": ("rune_bench.agents.cybersec.xbow", "XBOWRunner", ["api_key"]),
    "harvey": ("rune_bench.agents.legal.harvey", "HarveyAIRunner", ["api_key"]),
    "spellbook": ("rune_bench.agents.legal.spellbook", "SpellbookRunner", ["api_key"]),
    "dagger": ("rune_bench.agents.ops.dagger", "DaggerRunner", []),
    "crewai": ("rune_bench.agents.ops.crewai", "CrewAIRunner", ["api_key"]),
    "sierra": ("rune_bench.agents.ops.sierra", "SierraRunner", ["api_key"]),
    "skillfortify": ("rune_bench.agents.ops.skillfortify", "SkillFortifyRunner", ["api_key"]),
    "midjourney": ("rune_bench.agents.art.midjourney", "MidjourneyRunner", ["api_key", "base_url"]),
    "invokeai": ("rune_bench.agents.art.invokeai", "InvokeAIRunner", ["base_url"]),
    "comfyui": ("rune_bench.agents.art.comfyui", "ComfyUIRunner", ["base_url"]),
    "krea": ("rune_bench.agents.art.krea", "KreaRunner", ["api_key"]),
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
        raise ValueError(
            f"Unknown agent {name!r}. Available: {', '.join(available)}"
        )

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

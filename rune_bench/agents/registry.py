"""Dynamic agent registry with lazy import resolution."""

from __future__ import annotations

import importlib
from typing import Any, Callable

_REGISTRY: dict[str, dict] = {}

_BUILTIN_AGENTS: dict[str, tuple[str, str, list[str]]] = {
    "holmes": ("rune_bench.agents.sre.holmes", "HolmesRunner", ["kubeconfig"]),
    "k8sgpt": ("rune_bench.agents.sre.k8sgpt", "K8sGPTRunner", ["kubeconfig"]),
    "metoro": ("rune_bench.agents.sre.metoro", "MetoroRunner", ["kubeconfig"]),
    "pagerduty": ("rune_bench.agents.sre.pagerduty", "PagerDutyAIRunner", ["kubeconfig"]),
    "perplexity": ("rune_bench.agents.research.perplexity", "PerplexityRunner", []),
    "glean": ("rune_bench.agents.research.glean", "GleanRunner", []),
    "elicit": ("rune_bench.agents.research.elicit", "ElicitRunner", []),
    "langgraph": ("rune_bench.agents.research.langgraph", "LangGraphRunner", []),
    "consensus": ("rune_bench.agents.research.consensus", "ConsensusRunner", []),
    "pentestgpt": ("rune_bench.agents.cybersec.pentestgpt", "PentestGPTRunner", []),
    "radiant": ("rune_bench.agents.cybersec.radiant", "RadiantSecurityRunner", []),
    "mindgard": ("rune_bench.agents.cybersec.mindgard", "MindgardRunner", []),
    "burpgpt": ("rune_bench.agents.cybersec.burpgpt", "BurpGPTRunner", []),
    "xbow": ("rune_bench.agents.cybersec.xbow", "XBOWRunner", []),
    "harvey": ("rune_bench.agents.legal.harvey", "HarveyAIRunner", []),
    "spellbook": ("rune_bench.agents.legal.spellbook", "SpellbookRunner", []),
    "dagger": ("rune_bench.agents.ops.dagger", "DaggerRunner", []),
    "crewai": ("rune_bench.agents.ops.crewai", "CrewAIRunner", []),
    "sierra": ("rune_bench.agents.ops.sierra", "SierraRunner", []),
    "skillfortify": ("rune_bench.agents.ops.skillfortify", "SkillFortifyRunner", []),
    "midjourney": ("rune_bench.agents.art.midjourney", "MidjourneyRunner", []),
    "comfyui": ("rune_bench.agents.art.comfyui", "ComfyUIRunner", []),
    "krea": ("rune_bench.agents.art.krea", "KreaRunner", []),
}


def register_agent(
    name: str,
    factory: Callable[..., Any],
    *,
    required_config: list[str] | None = None,
) -> None:
    """Register a custom agent factory under *name*."""
    _REGISTRY[name] = {
        "factory": factory,
        "required_config": required_config or [],
    }


def get_agent(name: str, **kwargs: Any) -> Any:
    """Return an instantiated agent for *name*."""
    if name in _REGISTRY:
        return _REGISTRY[name]["factory"](**kwargs)

    if name in _BUILTIN_AGENTS:
        module_path, class_name, _req_config = _BUILTIN_AGENTS[name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**kwargs)

    available = sorted(set(list(_REGISTRY.keys()) + list(_BUILTIN_AGENTS.keys())))
    raise ValueError(
        f"Unknown agent {name!r}. Available: {', '.join(available)}"
    )


def list_agents() -> list[str]:
    """Return a sorted list of all known agent names."""
    return sorted(set(list(_REGISTRY.keys()) + list(_BUILTIN_AGENTS.keys())))

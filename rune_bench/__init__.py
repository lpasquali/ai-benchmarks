# SPDX-License-Identifier: Apache-2.0
"""
rune_bench — modular benchmarking toolkit for RUNE.

Exposes the top-level classes used by rune.py:
    OfferFinder      — Block 1: search and select the best GPU offer
    ModelSelector    — Block 2+3: pick the best-fitting model and calculate disk size
    TemplateLoader   — Block 4: load and extract Vast.ai template configuration
    InstanceManager  — Block 6+7+8+9: create, poll, execute on, and inspect an instance
    HolmesRunner     — Block 10: run HolmesGPT against a Kubernetes cluster
"""

from .common import ModelSelector

__all__ = [
    "ModelSelector",
]

def __getattr__(name: str) -> object:
    """Lazily expose vastai resources only when the 'vastai' extra is installed."""
    if name in ("OfferFinder", "TemplateLoader", "InstanceManager"):
        from rune_bench.resources.vastai import InstanceManager, OfferFinder, TemplateLoader  # noqa: PLC0415
        _map = {"OfferFinder": OfferFinder, "TemplateLoader": TemplateLoader, "InstanceManager": InstanceManager}
        return _map[name]
    raise AttributeError(f"module 'rune_bench' has no attribute {name!r}")

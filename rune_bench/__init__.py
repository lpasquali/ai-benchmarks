"""
rune_bench — modular benchmarking toolkit for RUNE.

Exposes the top-level classes used by rune.py:
    OfferFinder      — Block 1: search and select the best GPU offer
    ModelSelector    — Block 2+3: pick the best-fitting model and calculate disk size
    TemplateLoader   — Block 4: load and extract Vast.ai template configuration
    InstanceManager  — Block 6+7+8+9: create, poll, execute on, and inspect an instance
    HolmesRunner     — Block 10: run HolmesGPT against a Kubernetes cluster
"""

from .agents import HolmesRunner
from .common import ModelSelector
from .vastai import InstanceManager, OfferFinder, TemplateLoader

__all__ = [
    "OfferFinder",
    "ModelSelector",
    "TemplateLoader",
    "InstanceManager",
    "HolmesRunner",
]

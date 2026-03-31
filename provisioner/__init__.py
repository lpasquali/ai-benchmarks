"""
provisioner — modular Vast.ai GPU provisioning library.

Exposes the top-level classes used by provision.py:
    OfferFinder      — Block 1: search and select the best GPU offer
    ModelSelector    — Block 2+3: pick the best-fitting model and calculate disk size
    TemplateLoader   — Block 4: load and extract Vast.ai template configuration
    InstanceManager  — Block 6+7+8+9: create, poll, execute on, and inspect an instance
    HolmesRunner     — Block 10: run HolmesGPT against a Kubernetes cluster
"""

from .holmes import HolmesRunner
from .instance import InstanceManager
from .models import ModelSelector
from .offer import OfferFinder
from .template import TemplateLoader

__all__ = [
    "OfferFinder",
    "ModelSelector",
    "TemplateLoader",
    "InstanceManager",
    "HolmesRunner",
]

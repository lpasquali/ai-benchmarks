# SPDX-License-Identifier: Apache-2.0
"""LLM resource provisioning interfaces and implementations."""

from .base import LLMResourceProvider, ProvisioningResult
from .existing_backend_provider import ExistingBackendProvider

__all__ = [
    "LLMResourceProvider",
    "ProvisioningResult",
    "ExistingBackendProvider",
]


def __getattr__(name: str) -> object:
    if name == "VastAIProvider":
        from .vastai import VastAIProvider

        return VastAIProvider
    raise AttributeError(f"module 'rune_bench.resources' has no attribute {name!r}")

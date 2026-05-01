# SPDX-License-Identifier: Apache-2.0
"""K8s Gateway API Inference Extension LLM backend.

Scope:      In-cluster Kubernetes Gateway
Ecosystem:  Kubernetes

Implementation notes:
- Acts as a pass-through router.
- The K8s Gateway API handles routing to specific model pools (e.g. vLLM)
  based on the requested model name.
- Models are typically always "running" or auto-scaled by the cluster.
"""

from __future__ import annotations

from typing import Any

from rune_bench.backends.base import ModelCapabilities
from rune_bench.debug import debug_log
from rune_bench.metrics import span


class K8sInferenceBackend:
    """LLM backend for K8s Gateway API Inference Extension."""

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        if not base_url:
            raise ValueError(
                "K8sInferenceBackend requires a base_url (the gateway endpoint)."
            )
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        """Return the K8s Gateway URL."""
        return self._base_url

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Capabilities are handled downstream by the actual model pool."""
        api_model_name = self.normalize_model_name(model)
        return ModelCapabilities(model_name=api_model_name)

    def list_models(self) -> list[str]:
        """Gateway routing handles model availability dynamically."""
        return []

    def list_running_models(self) -> list[str]:
        """In-cluster models are managed by K8s autoscalers."""
        return []

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize provider-prefixed model names."""
        normalized = model_name.strip()
        for prefix in ("k8s/", "gateway/"):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix)
        return normalized

    def warmup(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
    ) -> str:
        """Warmup is delegated to the cluster's autoscaler (e.g. KEDA)."""
        api_model_name = self.normalize_model_name(model_name)
        with span("k8s_inference.model.warmup", model=api_model_name):
            debug_log(
                f"K8sInferenceBackend routing model {api_model_name} "
                f"to gateway {self._base_url}"
            )
            return api_model_name

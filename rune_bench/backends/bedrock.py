# SPDX-License-Identifier: Apache-2.0
"""AWS Bedrock LLM backend implementation.

Scope:      SaaS API
Docs:       https://docs.aws.amazon.com/bedrock/latest/APIReference/
            https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
Ecosystem:  AWS
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rune_bench.backends.base import BackendCredentials, ModelCapabilities
from rune_bench.debug import debug_log
from rune_bench.metrics import span

if TYPE_CHECKING:
    pass


class BedrockBackend:
    """LLM backend for AWS Bedrock.

    Vendor quirk: region is mandatory and passed via BackendCredentials.extra["region"]
    or as the first positional argument (base_url in factory).
    """

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        """Initialize Bedrock backend.

        Args:
            base_url: Optional region name (e.g. "us-east-1").
            **kwargs: Backend configuration, must include 'credentials'.
        """
        import boto3

        self._credentials = kwargs.get("credentials")
        if not isinstance(self._credentials, BackendCredentials):
            # Fallback for direct instantiation
            self._credentials = BackendCredentials(extra={})

        self._region = (
            base_url or self._credentials.extra.get("region") or kwargs.get("region")
        )

        if not self._region:
            raise ValueError(
                "BedrockBackend requires a region. Provide it via base_url, "
                "credentials.extra['region'], or region keyword argument."
            )

        self._client = boto3.client("bedrock", region_name=self._region)
        self._runtime = boto3.client("bedrock-runtime", region_name=self._region)

    @property
    def base_url(self) -> str:
        """Return the AWS region as the base URL."""
        return str(self._region)

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Fetch model metadata from AWS Bedrock."""
        api_model_name = self.normalize_model_name(model)
        try:
            response = self._client.get_foundation_model(modelIdentifier=api_model_name)
            details = response.get("modelDetails", {})

            # Bedrock doesn't always return context window in the API call for all models.
            # We provide known defaults for common models.
            context_window = details.get("inputModalities", [])  # just a placeholder
            # Extract actual context length if available
            context_window = details.get("maxContextTokens")

            return ModelCapabilities(
                model_name=api_model_name,
                context_window=context_window,
                max_output_tokens=None,  # Usually defined per request in Bedrock
                raw=response,
            )
        except Exception as exc:
            debug_log(
                f"Failed to fetch Bedrock capabilities for {api_model_name}: {exc}"
            )
            return ModelCapabilities(model_name=api_model_name)

    def list_models(self) -> list[str]:
        """Return model names available on this backend."""
        try:
            response = self._client.list_foundation_models()
            return [m["modelId"] for m in response.get("modelSummaries", [])]
        except Exception as exc:
            debug_log(f"Failed to list Bedrock models: {exc}")
            return []

    def list_running_models(self) -> list[str]:
        """Bedrock models are always 'running' (SaaS)."""
        return []

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize a provider-prefixed model name."""
        normalized = model_name.strip()
        for prefix in ("bedrock/", "aws/"):
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
        """Warmup for Bedrock is a no-op (SaaS)."""
        api_model_name = self.normalize_model_name(model_name)
        with span("bedrock.model.warmup", model=api_model_name):
            return api_model_name

    def invoke(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        """Invoke a model on AWS Bedrock."""
        api_model_name = self.normalize_model_name(model)
        response = self._runtime.invoke_model(
            modelId=api_model_name, body=json.dumps(body)
        )
        return json.loads(response.get("body").read())

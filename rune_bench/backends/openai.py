"""OpenAI API LLM backend stub.

Scope:      SaaS API
Docs:       https://platform.openai.com/docs/api-reference
            https://platform.openai.com/docs/models
Ecosystem:  OpenAI

Implementation notes:
- Install:  pip install openai
- Auth:     OPENAI_API_KEY env var; optional OPENAI_ORG_ID
- SDK:      from openai import OpenAI
            client = OpenAI(api_key=credentials.api_key)
- Models:   gpt-4o, gpt-4-turbo, gpt-3.5-turbo, o1, o3-mini, ...
- Capabilities endpoint:
            GET https://api.openai.com/v1/models/{model}
            Returns context_window in model object
- BackendCredentials:
            api_key  = OPENAI_API_KEY
            base_url = custom base URL for Azure OpenAI or proxies
            extra    = {"organization": "org-xxx"}
"""

from rune_bench.backends.base import BackendCredentials, ModelCapabilities


class OpenAIBackend:
    """LLM backend for the OpenAI API (and OpenAI-compatible endpoints)."""

    def __init__(self, credentials: BackendCredentials) -> None:
        self._credentials = credentials

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Fetch model metadata from the OpenAI API."""
        raise NotImplementedError(
            "OpenAIBackend is not yet implemented. "
            "See https://platform.openai.com/docs/api-reference for details."
        )

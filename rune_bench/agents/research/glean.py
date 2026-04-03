"""Glean agentic runner stub.

Scope:      Research  |  Rank 2  |  Rating 4.8
Capability: Autonomous internal knowledge discovery for enterprises.
Docs:       https://developers.glean.com/
            https://developers.glean.com/docs/search_api/
Ecosystem:  Enterprise Search

Implementation notes:
- Auth:     GLEAN_API_TOKEN + GLEAN_INSTANCE (subdomain) env vars
- SDK:      REST API; no official Python SDK
            Base URL: https://<instance>-be.glean.com/api/v1
- Key endpoints:
    POST /search          # full-text + semantic search
    POST /chat            # agentic chat with internal knowledge
            body: { messages: [{role, content}], stream: false }
- The `question` maps to the chat message content.
- `model` and `ollama_url` are not used (Glean uses its own hosted model).
"""


class GleanRunner:
    """Research agent: autonomous internal knowledge discovery via Glean enterprise search."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Query Glean's agentic chat and return the synthesised answer."""
        raise NotImplementedError(
            "GleanRunner is not yet implemented. "
            "See https://developers.glean.com/docs/search_api/ for details."
        )

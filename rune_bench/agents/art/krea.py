"""Krea AI agentic runner stub.

Scope:      Art/Creative  |  Rank 3  |  Rating 4.0
Capability: Real-time generative enhancement and upscaling.
Docs:       https://www.krea.ai/
            https://docs.krea.ai/  (API docs, access via waitlist)
Ecosystem:  Open Weights

Implementation notes:
- Auth:     KREA_API_KEY env var (request access at https://docs.krea.ai/)
- SDK:      REST API (no official Python SDK)
- Key endpoints:
    POST /generations                  # text-to-image or image enhancement
        body: { prompt, style, enhance: bool }
    GET  /generations/{id}             # poll status
- The `question` maps to the generation prompt.
- `model` and `backend_url` are not used (Krea uses its own models).
- Returns image URLs as the answer string.
"""


class KreaRunner:
    """Art/Creative agent: real-time generative image enhancement via Krea AI."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Generate or enhance an image via Krea AI and return the result URL."""
        raise NotImplementedError(
            "KreaRunner is not yet implemented. "
            "See https://docs.krea.ai/ for API access and details."
        )

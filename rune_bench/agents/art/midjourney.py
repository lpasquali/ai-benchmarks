# SPDX-License-Identifier: Apache-2.0
"""Midjourney agentic runner stub.

Scope:      Art/Creative  |  Rank 1  |  Rating 5.0
Capability: Iterative agentic refinement via "Remix" modes.
Docs:       https://docs.midjourney.com/
            https://docs.midjourney.com/docs/remix-mode
            https://docs.midjourney.com/docs/quick-start
Ecosystem:  Generative AI Ethics

Implementation notes:
- Auth:     Midjourney has NO official public API as of writing.
            Unofficial options: useapi.net, thenextleg.io, or Discord bot automation.
            MIDJOURNEY_API_KEY + MIDJOURNEY_API_BASE env vars for proxy services.
- Approach via proxy API:
    POST /imagine     body: { prompt: str, aspect_ratio: str }
    GET  /jobs/{id}   poll until status == "complete"
    Returns: image URLs
- The `question` maps to the image generation prompt.
- `model` and `backend_url` are not used.
- Returns URLs or base64-encoded image data as the answer string.
"""


class MidjourneyRunner:
    """Art/Creative agent: iterative image generation via Midjourney Remix."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Generate an image from the prompt and return the result URL(s)."""
        raise NotImplementedError(
            "MidjourneyRunner is not yet implemented. "
            "See https://docs.midjourney.com/ — note: no official public API exists; "
            "consider a proxy service such as useapi.net."
        )

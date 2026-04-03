"""Harvey AI agentic runner stub.

Scope:      Legal  |  Rank 1  |  Rating 4.8
Capability: Autonomous legal disclosure and risk analysis.
Docs:       https://www.harvey.ai/
            https://developer.harvey.ai/  (API docs, enterprise access)
Ecosystem:  Transparency Manifestos

Implementation notes:
- Auth:     HARVEY_API_KEY env var (enterprise contract required)
- SDK:      REST API (no public Python SDK)
- Approach: Submit a legal document or question; Harvey autonomously
            analyses it for risks, disclosure obligations, and precedents.
- Key endpoints (expected):
    POST /completions         body: { prompt: str, matter_type: str }
    GET  /completions/{id}    poll for async tasks
    Returns: { analysis: str, risks: list, citations: list }
- `question` maps to the legal query or document excerpt.
- `model` and `ollama_url` are not used (Harvey uses its own fine-tuned models).
"""


class HarveyAIRunner:
    """Legal agent: autonomous legal disclosure and risk analysis via Harvey AI."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Submit a legal query to Harvey AI and return the analysis."""
        raise NotImplementedError(
            "HarveyAIRunner is not yet implemented. "
            "See https://developer.harvey.ai/ for enterprise API access."
        )

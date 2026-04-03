"""Elicit agentic runner stub.

Scope:      Research  |  Rank 3  |  Rating 4.0
Capability: Automates literature review and data extraction.
Docs:       https://elicit.com/
            https://elicit.com/api  (API access via waitlist as of writing)
Ecosystem:  Open Science

Implementation notes:
- Auth:     ELICIT_API_KEY env var (request access at https://elicit.com/api)
- SDK:      REST API (no public Python SDK)
- Approach: Submit a research question; Elicit searches academic databases,
            extracts structured data from papers, and returns a synthesis.
- Key flow:
    POST /tasks          # create a research task with question
    GET  /tasks/{id}     # poll until complete
    GET  /tasks/{id}/results  # structured paper + synthesis results
- The `question` maps to the research task question.
- `model` and `ollama_url` are not used (Elicit uses its own models).
"""


class ElicitRunner:
    """Research agent: automated literature review and data extraction via Elicit."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Submit a literature review task to Elicit and return the synthesis."""
        raise NotImplementedError(
            "ElicitRunner is not yet implemented. "
            "See https://elicit.com/api for API access and details."
        )

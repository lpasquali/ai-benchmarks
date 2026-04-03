"""Consensus agentic runner stub.

Scope:      Research  |  Rank 5  |  Rating 3.5
Capability: Synthesizes answers from 200M+ academic papers.
Docs:       https://consensus.app/
            https://consensus.app/tools/api  (API access)
Ecosystem:  Evidence-Based Research

Implementation notes:
- Auth:     CONSENSUS_API_KEY env var (request at https://consensus.app/tools/api)
- SDK:      REST API (no official Python SDK)
- Key endpoint:
    POST /search
        body: { query: str, limit: int }
    Returns: list of papers with abstracts + a GPT-4-generated synthesis
- The `question` maps to the search query.
- `model` and `ollama_url` are not used (Consensus uses its own hosted model).
- Extract .synthesis from response for the final answer string.
"""


class ConsensusRunner:
    """Research agent: evidence-based synthesis from 200M+ academic papers."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Search Consensus and return the paper-backed synthesised answer."""
        raise NotImplementedError(
            "ConsensusRunner is not yet implemented. "
            "See https://consensus.app/tools/api for API access details."
        )

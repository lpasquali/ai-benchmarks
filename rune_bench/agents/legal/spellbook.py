"""Spellbook agentic runner stub.

Scope:      Legal  |  Rank 2  |  Rating 4.0
Capability: Agentic contract review and risk flagging.
Docs:       https://www.spellbook.legal/
            https://spellbook.legal/api-docs  (API docs, enterprise access)
Ecosystem:  Legal Tech Standards

Implementation notes:
- Auth:     SPELLBOOK_API_KEY env var (enterprise/law firm contract required)
- SDK:      REST API (no public Python SDK); also available as Word add-in
- Approach: Submit a contract document; Spellbook autonomously reviews
            it clause by clause, flags risks, and suggests redlines.
- Key endpoints (expected):
    POST /reviews             body: { document: str, review_type: str }
    GET  /reviews/{id}        poll until status == "complete"
    Returns: { summary, risk_flags: list, suggested_redlines: list }
- `question` maps to the contract text or review instruction.
- `model` and `backend_url` are not used (Spellbook uses GPT-4 backend).
"""


class SpellbookRunner:
    """Legal agent: agentic contract review and risk flagging via Spellbook."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Submit a contract to Spellbook for review and return the risk analysis."""
        raise NotImplementedError(
            "SpellbookRunner is not yet implemented. "
            "See https://www.spellbook.legal/ for enterprise API access."
        )

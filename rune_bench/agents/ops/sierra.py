"""Sierra agentic runner stub.

Scope:      Ops/Misc  |  Rank 3  |  Rating 4.0
Capability: Agentic customer operations and task execution.
Docs:       https://sierra.ai/
            https://sierra.ai/docs  (API docs, enterprise access)
Ecosystem:  Enterprise CX

Implementation notes:
- Auth:     SIERRA_API_KEY env var (enterprise contract required)
- SDK:      REST API (no public Python SDK)
- Approach: Sierra acts as an autonomous AI agent for customer-facing operations.
            It can take actions (process refunds, update records, escalate tickets)
            and integrates with CRM/ops systems via tools.
- Key endpoints (expected):
    POST /conversations       body: { message: str, context: dict }
    GET  /conversations/{id}  retrieve conversation and actions taken
    Returns: { response: str, actions_taken: list }
- `question` maps to the customer/ops task description.
- `model` and `backend_url` are not used (Sierra uses its own hosted models).
"""


class SierraRunner:
    """Ops/Misc agent: agentic customer operations and task execution via Sierra."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Submit an ops task to Sierra and return the outcome."""
        raise NotImplementedError(
            "SierraRunner is not yet implemented. "
            "See https://sierra.ai/ for enterprise API access."
        )

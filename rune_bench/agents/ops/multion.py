# SPDX-License-Identifier: Apache-2.0
"""MultiOn agentic runner stub.

Scope:      Ops/Misc  |  Rank 1  |  Rating 4.5
Capability: Browser-based agent that performs web tasks.
Docs:       https://docs.multion.ai/
            https://docs.multion.ai/api-reference/
Ecosystem:  AAIF (Agentic AI)

Implementation notes:
- Install:  pip install multion
- Auth:     MULTION_API_KEY env var (get key at https://app.multion.ai/)
- SDK:      Official Python SDK available
            from multion.client import MultiOn
            client = MultiOn(api_key=os.environ["MULTION_API_KEY"])
- Key methods:
    client.browse(cmd=question, url=start_url)   # perform web task
    client.create_session(url=start_url)         # stateful browser session
    client.step_session(session_id, cmd=step)    # step-by-step execution
- `question` maps to the browser task command (e.g. "find the price of X on amazon.com")
- `model` and `backend_url` are not used (MultiOn uses its own cloud agent).
"""


class MultiOnRunner:
    """Ops/Misc agent: browser-based autonomous web task execution via MultiOn."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Execute a browser-based web task and return the result."""
        raise NotImplementedError(
            "MultiOnRunner is not yet implemented. "
            "See https://docs.multion.ai/api-reference/ for SDK details."
        )

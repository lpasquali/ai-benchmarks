# SPDX-License-Identifier: Apache-2.0
"""Cleric agentic runner stub.

Scope:      SRE  |  Rank 5  |  Rating 3.5
Capability: Mimics an engineer's "parallel investigation" loop.
Docs:       https://github.com/ClericHQ
            https://github.com/ClericHQ/cleric  (main repo)
Ecosystem:  Infra Interoperability

Implementation notes:
- Auth:     CLERIC_API_KEY env var (if using hosted); or run OSS version locally
- SDK:      Python package from GitHub; no PyPI release as of writing
            git+https://github.com/ClericHQ/cleric
- Approach: Cleric runs a ReAct-style loop, calling kubectl/CLI tools in parallel.
            It accepts a goal/question and iterates until it reaches a conclusion.
- Key interface (expected):
    from cleric import Cleric
    agent = Cleric(kubeconfig=str(kubeconfig))
    result = agent.investigate(question)
- The `model` and `backend_url` may need to be injected via env vars:
    CLERIC_MODEL, CLERIC_OLLAMA_BASE_URL
"""

from pathlib import Path


class ClericRunner:
    """SRE agent: parallel investigation loop mimicking a human SRE's debugging process."""

    def __init__(self, kubeconfig: Path) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run a Cleric parallel investigation and return the findings."""
        raise NotImplementedError(
            "ClericRunner is not yet implemented. "
            "See https://github.com/ClericHQ/cleric for implementation details."
        )

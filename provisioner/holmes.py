"""Block 10 — HolmesGPT Runner.

Runs HolmesGPT via its Python SDK against a Kubernetes cluster,
using the provisioned model as the LLM backend.
"""

from pathlib import Path

import holmesgpt  # type: ignore


class HolmesRunner:
    """Investigate a Kubernetes cluster using the HolmesGPT SDK."""

    def __init__(self, kubeconfig: Path) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig

    def ask(self, question: str, model: str) -> str:
        """Run a HolmesGPT query and return the answer as a string.

        Tries module-level ``holmesgpt.ask(...)`` first, then falls back to
        the class-based ``holmesgpt.HolmesGPT(...).ask(...)`` pattern.

        Raises:
            RuntimeError: if no supported SDK entry-point is found.
        """
        answer = None

        if hasattr(holmesgpt, "ask"):
            try:
                answer = holmesgpt.ask(
                    question=question,
                    model=model,
                    kubeconfig=str(self._kubeconfig),
                )
            except TypeError:
                pass

        if answer is None and hasattr(holmesgpt, "HolmesGPT"):
            try:
                client = holmesgpt.HolmesGPT(
                    model=model,
                    kubeconfig=str(self._kubeconfig),
                )
                answer = client.ask(question=question)
            except TypeError:
                client = holmesgpt.HolmesGPT(kubeconfig=str(self._kubeconfig))
                answer = client.ask(question=question, model=model)

        if answer is None:
            raise RuntimeError(
                "Unsupported HolmesGPT SDK API shape. "
                "Expected holmesgpt.ask(...) or holmesgpt.HolmesGPT(...).ask(...)."
            )

        return str(answer)

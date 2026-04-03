"""Metoro agentic runner stub.

Scope:      SRE  |  Rank 4  |  Rating 4.0
Capability: Uses eBPF for autonomous service mapping and debugging.
Docs:       https://metoro.io/docs
            https://metoro.io/docs/api
Ecosystem:  CNCF / eBPF

Implementation notes:
- Auth:     METORO_API_KEY env var
- SDK:      REST API (no official Python SDK as of writing)
            Base URL: https://app.metoro.io/api  (or self-hosted endpoint)
- Key endpoints:
    GET  /services                    # list detected services
    GET  /traces?service=<s>          # eBPF-captured traces
    POST /ai/explain                  # LLM-powered incident explanation
            body: { question, service, time_range }
- The `question` maps to the explanation query.
- `model` / `ollama_url` may be passed to the self-hosted Metoro AI endpoint.
"""

from pathlib import Path


class MetoroRunner:
    """SRE agent: eBPF-powered service mapping and autonomous debugging via Metoro."""

    def __init__(self, kubeconfig: Path) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Invoke Metoro AI explanation and return the analysis as a string."""
        raise NotImplementedError(
            "MetoroRunner is not yet implemented. "
            "See https://metoro.io/docs/api for implementation details."
        )

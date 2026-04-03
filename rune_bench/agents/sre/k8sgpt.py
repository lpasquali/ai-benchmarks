"""K8sGPT agentic runner stub.

Scope:      SRE  |  Rank 1  |  Rating 5.0
Capability: Scans clusters for issues and provides automated RCA.
Docs:       https://github.com/k8sgpt-ai/k8sgpt
Ecosystem:  CNCF Sandbox

Implementation notes:
- Install:  pip install k8sgpt  OR use the k8sgpt CLI binary
- Auth:     kubeconfig for cluster access; optional AI backend API key
- SDK:      https://github.com/k8sgpt-ai/k8sgpt#python-sdk (if available)
            Alternatively invoke CLI: k8sgpt analyze --explain --backend ollama
- Key CLI flags:
    k8sgpt analyze --explain           # scan all namespaces, explain with LLM
    k8sgpt analyze --filter <kind>     # scope to specific resource kind
    k8sgpt analyze --backend ollama --model <model> --base-url <ollama_url>
- Returns structured JSON results; extract .results[].error + .results[].details
"""

from pathlib import Path


class K8sGPTRunner:
    """SRE agent: scans a Kubernetes cluster and returns AI-powered RCA.

    Uses k8sgpt CLI with the Ollama backend so the same model/ollama_url
    parameters from the RUNE benchmark flow are forwarded transparently.
    """

    def __init__(self, kubeconfig: Path) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Run a k8sgpt analysis and return the explanation as a string."""
        raise NotImplementedError(
            "K8sGPTRunner is not yet implemented. "
            "See https://github.com/k8sgpt-ai/k8sgpt for SDK/CLI details."
        )

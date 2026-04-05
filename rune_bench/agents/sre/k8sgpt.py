"""K8sGPT agentic runner — delegates to the K8sGPT driver.

Scope:      SRE  |  Rank 1  |  Rating 5.0
Capability: Scans clusters for issues and provides automated RCA.
Docs:       https://github.com/k8sgpt-ai/k8sgpt
Ecosystem:  CNCF Sandbox
"""

from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

# Backwards-compatible alias so existing imports of K8sGPTRunner keep working.
K8sGPTRunner = K8sGPTDriverClient

__all__ = ["K8sGPTRunner", "K8sGPTDriverClient"]

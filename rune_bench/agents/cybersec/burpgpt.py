# SPDX-License-Identifier: Apache-2.0
"""BurpGPT agentic runner -- delegates to the burpgpt driver.

Scope:      Cybersec  |  Rank 4  |  Rating 3.5
Capability: Autonomous web vulnerability scanning via LLM.
Docs:       https://github.com/v87/burpgpt
            https://github.com/v87/burpgpt/blob/main/README.md
Ecosystem:  OWASP Standards

Implementation notes:
- Install:  BurpGPT is a Burp Suite extension (.jar); not a Python package.
            Download from: https://github.com/v87/burpgpt/releases
            Load into Burp Suite: Extender → Add → select .jar
- Auth:     Configured via Burp Suite UI; requires OpenAI API key or Ollama endpoint.
- Approach: Burp Suite proxies HTTP traffic; BurpGPT sends requests/responses
            to an LLM to identify vulnerabilities autonomously.
- Integration with RUNE:
    Option A: Drive via Burp REST API (requires Burp Suite Pro):
              POST http://localhost:1337/v0.1/scan  body: { urls: [target] }
    Option B: Run Burp headless with a Python wrapper script.
- `question` maps to the target URL or scan objective.
- `model` and `backend_url` configure the LLM backend within Burp/BurpGPT.
"""

from rune_bench.drivers.burpgpt import BurpGPTDriverClient


class BurpGPTRunner:
    """Cybersec agent: autonomous web vulnerability scanning via BurpGPT + LLM.

    Delegates to :class:`~rune_bench.drivers.burpgpt.BurpGPTDriverClient`.
    """

    def __init__(self) -> None:
        self._client = BurpGPTDriverClient()

    def ask(self, question: str, model: str, backend_url: str | None = None) -> str:
        """Run a BurpGPT-assisted scan and return identified vulnerabilities."""
        return self._client.ask(question, model, backend_url)

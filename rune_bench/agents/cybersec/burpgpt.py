"""BurpGPT agentic runner stub.

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
- `model` and `ollama_url` configure the LLM backend within Burp/BurpGPT.
"""


class BurpGPTRunner:
    """Cybersec agent: autonomous web vulnerability scanning via BurpGPT + LLM."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Run a BurpGPT-assisted scan and return identified vulnerabilities."""
        raise NotImplementedError(
            "BurpGPTRunner is not yet implemented. "
            "See https://github.com/v87/burpgpt — note: requires Burp Suite installation."
        )

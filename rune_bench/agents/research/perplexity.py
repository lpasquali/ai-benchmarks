"""Perplexity Pro agentic runner stub.

Scope:      Research  |  Rank 1  |  Rating 5.0
Capability: Multi-step research with autonomous source validation.
Docs:       https://docs.perplexity.ai/
            https://docs.perplexity.ai/reference/post_chat_completions
Ecosystem:  Open Web Standards

Implementation notes:
- Auth:     PERPLEXITY_API_KEY env var
- SDK:      OpenAI-compatible REST API; use openai Python client with custom base_url
            pip install openai
            client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"],
                            base_url="https://api.perplexity.ai")
- Models:   sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro
- The `question` maps to the user message.
- `model` can override the Perplexity model name (default: sonar-pro).
- `ollama_url` is not used; Perplexity is cloud-only.
- Response includes citations; extract .choices[0].message.content.
"""


class PerplexityRunner:
    """Research agent: multi-step web research with autonomous source validation."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Submit a research query to Perplexity and return the sourced answer."""
        raise NotImplementedError(
            "PerplexityRunner is not yet implemented. "
            "See https://docs.perplexity.ai/reference/post_chat_completions for details."
        )

"""CrewAI driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.crewai

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), ollama_url (str, optional)
    result: {"answer": str}

info
    params: (none)
    result: {"name": "crewai", "version": "1", "actions": [...]}

Dependencies
------------
Requires ``crewai`` to be installed::

    pip install rune[crewai]
"""

from __future__ import annotations

import json
import os
import sys


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    ollama_url: str | None = params.get("ollama_url")

    try:
        from crewai import Agent, Crew, Task
    except ImportError as exc:
        raise RuntimeError(
            "CrewAI driver requires: pip install rune[crewai]  "
            "(crewai package)"
        ) from exc

    # Configure Ollama via LiteLLM environment variables
    if ollama_url:
        os.environ["OPENAI_API_BASE"] = f"{ollama_url}/v1"

    agent = Agent(
        role="Analyst",
        goal=question,
        llm=f"ollama/{model}",
    )
    task = Task(
        description=question,
        agent=agent,
        expected_output="A detailed analysis",
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = crew.kickoff()

    return {"answer": result.raw}


def _handle_info(_params: dict) -> dict:
    return {
        "name": "crewai",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "Requires optional dependencies: pip install rune[crewai]",
    }


_HANDLERS: dict = {
    "ask": "_handle_ask",
    "info": "_handle_info",
}


def main() -> None:
    """Read JSON requests from stdin and write JSON responses to stdout."""
    current_module = sys.modules[__name__]
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = ""
        try:
            request = json.loads(line)
            req_id = str(request.get("id", ""))
            action = str(request.get("action", ""))
            params = request.get("params") or {}

            handler_name = _HANDLERS.get(action)
            if handler_name is None:
                raise RuntimeError(f"Unknown action: {action!r}")
            handler = getattr(current_module, handler_name)

            result = handler(params)
            print(json.dumps({"status": "ok", "result": result, "id": req_id}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps({"status": "error", "error": str(exc), "id": req_id}),
                flush=True,
            )


if __name__ == "__main__":
    main()

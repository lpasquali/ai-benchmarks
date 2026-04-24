# SPDX-License-Identifier: Apache-2.0
"""CrewAI driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.crewai

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), backend_url (str, optional)
    result: {"answer": str}

info
    params: (none)
    result: {"name": "crewai", "version": "1", "actions": [...]}

Dependencies
------------
Requires ``crewai`` to be installed::

    pip install crewai
"""

from __future__ import annotations

import json
import os
import sys

_MODEL_PREFIXES = ("ollama/", "ollama_chat/")

_SENTINEL = object()


def _normalize_model(model: str) -> str:
    """Strip provider prefixes (e.g. 'ollama/', 'ollama_chat/') from model name."""
    for prefix in _MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    backend_url: str | None = params.get("backend_url")

    try:
        from crewai import Agent, Crew, Task  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("CrewAI driver requires: pip install crewai") from exc

    normalized = _normalize_model(model)

    # Configure Ollama via LiteLLM environment variables, restoring previous
    # value after the request to avoid env var leaking across requests in the
    # long-lived stdio loop.
    prev_api_base = os.environ.get("OPENAI_API_BASE", _SENTINEL)
    if backend_url:
        os.environ["OPENAI_API_BASE"] = f"{backend_url.rstrip('/')}/v1"

    try:
        agent = Agent(
            role="Analyst",
            goal=question,
            llm=f"ollama/{normalized}",
        )
        task = Task(
            description=question,
            agent=agent,
            expected_output="A detailed analysis",
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
    finally:
        # Restore previous OPENAI_API_BASE value
        if prev_api_base is _SENTINEL:
            os.environ.pop("OPENAI_API_BASE", None)
        else:
            os.environ["OPENAI_API_BASE"] = prev_api_base  # type: ignore[assignment]

    return {"answer": result.raw}


def _handle_info(_params: dict) -> dict:
    return {
        "name": "crewai",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "Requires optional dependencies: pip install crewai",
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
            print(
                json.dumps({"status": "ok", "result": result, "id": req_id}), flush=True
            )
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps({"status": "error", "error": str(exc), "id": req_id}),
                flush=True,
            )


if __name__ == "__main__":
    main()

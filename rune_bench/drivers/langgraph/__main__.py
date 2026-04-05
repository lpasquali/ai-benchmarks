"""LangGraph driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.langgraph

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
    result: {"name": "langgraph", "version": "1", "actions": [...]}

Dependencies
------------
Requires ``langgraph`` and ``langchain-openai`` to be installed::

    pip install rune[langgraph]
"""

from __future__ import annotations

import json
import sys


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    ollama_url: str | None = params.get("ollama_url")

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        from langgraph.prebuilt import create_react_agent  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph driver requires: pip install rune[langgraph]  "
            "(langgraph and langchain-openai packages)"
        ) from exc

    # LangChain's ChatOpenAI can be used to talk to Ollama's OpenAI-compatible API
    llm = ChatOpenAI(
        model=model,
        base_url=f"{ollama_url.rstrip('/')}/v1" if ollama_url else "http://localhost:11434/v1",
        api_key="ollama",  # Required but ignored by Ollama
    )

    # A simple ReAct agent with no tools - can be extended later
    agent = create_react_agent(llm, tools=[])
    result = agent.invoke({"messages": [("user", question)]})

    # Extract the last message content
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("LangGraph agent returned no messages.")

    answer = messages[-1].content
    return {"answer": answer}


def _handle_info(_params: dict) -> dict:
    return {
        "name": "langgraph",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "Requires optional dependencies: pip install rune[langgraph]",
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

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
Requires ``langgraph`` and ``langchain-ollama`` to be installed::

    pip install langgraph
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    ollama_url: str | None = params.get("ollama_url")

    try:
        from langchain_ollama import ChatOllama
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph driver requires: pip install langgraph  "
            "(langgraph and langchain-ollama packages)"
        ) from exc

    # Build ChatOllama LLM
    llm_kwargs: dict[str, Any] = {"model": model}
    if ollama_url:
        llm_kwargs["base_url"] = ollama_url
    llm = ChatOllama(**llm_kwargs)

    # Define a simple single-node research graph.
    # This is intentionally minimal — a foundation for users to extend.
    from typing import TypedDict

    class GraphState(TypedDict):
        question: str
        answer: str

    def research_node(state: GraphState) -> dict:
        response = llm.invoke(state["question"])
        return {"answer": response.content}

    graph = StateGraph(GraphState)
    graph.add_node("research", research_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", END)

    compiled = graph.compile()
    result = compiled.invoke({"question": question, "answer": ""})

    return {"answer": result["answer"]}


def _handle_info(_params: dict) -> dict:
    return {
        "name": "langgraph",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "Requires optional dependencies: pip install langgraph",
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

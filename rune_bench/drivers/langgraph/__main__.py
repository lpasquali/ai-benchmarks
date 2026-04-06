"""LangGraph driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.langgraph

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
    result: {"name": "langgraph", "version": "1", "actions": [...]}

Dependencies
------------
Requires ``langgraph`` and ``langchain-ollama`` to be installed::

    pip install langgraph langchain-ollama
"""

from __future__ import annotations

import json
import sys
from typing import Any, TypedDict

_MODEL_PREFIXES = ("ollama/", "ollama_chat/")


def _normalize_model(model: str) -> str:
    """Strip provider prefixes (e.g. 'ollama/', 'ollama_chat/') from model name."""
    for prefix in _MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model

try:
    from langchain_ollama import ChatOllama  # type: ignore[import-not-found]
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
except ImportError:
    # Optional dependencies handled in _handle_ask
    ChatOllama = None  # type: ignore
    StateGraph = None  # type: ignore
    START = None  # type: ignore
    END = None  # type: ignore


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    question: str
    answer: str


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    backend_url: str | None = params.get("backend_url")

    if StateGraph is None or ChatOllama is None:
        raise RuntimeError(
            "LangGraph driver requires: pip install langgraph langchain-ollama"
        )

    # Build ChatOllama LLM
    llm_kwargs: dict[str, Any] = {"model": _normalize_model(model)}
    if backend_url:
        llm_kwargs["base_url"] = backend_url
    llm = ChatOllama(**llm_kwargs)

    # Define a simple single-node research graph.
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
        "note": "Requires optional dependencies: pip install langgraph langchain-ollama",
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

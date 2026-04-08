# SPDX-License-Identifier: Apache-2.0
"""LangGraph driver entry point — receives JSON actions on stdin, writes results to stdout.

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), kubeconfig_path (str, optional),
            backend_url (str, optional)
    result: {"answer": str}

info
    params: (none)
    result: {"name": "langgraph", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import sys
from typing import Any, TypedDict

_MODEL_PREFIXES = ("ollama/", "ollama_chat/")

def _normalize_model(model: str) -> str:
    """Strip provider prefixes from model name."""
    for prefix in _MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model

try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langchain_core.messages import HumanMessage, BaseMessage
except ImportError:
    ChatOllama = None
    StateGraph = None
    START = None
    END = None
    HumanMessage = None

class GraphState(TypedDict):
    """State for the LangGraph SRE/Research workflow."""
    question: str
    kubeconfig: str | None
    history: list[BaseMessage]
    next_step: str
    answer: str | None

def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    kubeconfig_path: str | None = params.get("kubeconfig_path")
    backend_url: str | None = params.get("backend_url")

    if StateGraph is None or ChatOllama is None:
        raise RuntimeError(
            "LangGraph driver requires: pip install langgraph langchain-ollama langchain-core"
        )

    llm_kwargs: dict[str, Any] = {"model": _normalize_model(model)}
    if backend_url:
        llm_kwargs["base_url"] = backend_url
    llm = ChatOllama(**llm_kwargs)

    def diagnostic_node(state: GraphState) -> dict:
        """Analyze the situation, possibly using tools if it were a full ReAct loop."""
        prompt = f"You are an SRE assistant. Question: {state['question']}\n"
        if state['kubeconfig']:
            prompt += f"Context: Kubernetes cluster diagnostics enabled via {state['kubeconfig']}\n"
        
        # In a real SRE workflow, we would bind tools here (kubectl, etc.)
        # For this Tier 1 implementation, we simulate a multi-step diagnostic logic
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content, "history": state["history"] + [response]}

    # Build the graph
    workflow = StateGraph(GraphState)
    workflow.add_node("diagnose", diagnostic_node)
    workflow.add_edge(START, "diagnose")
    workflow.add_edge("diagnose", END)

    compiled = workflow.compile()
    
    initial_state: GraphState = {
        "question": question,
        "kubeconfig": kubeconfig_path,
        "history": [],
        "next_step": "",
        "answer": None
    }
    
    result = compiled.invoke(initial_state)
    return {"answer": result["answer"]}

def _handle_info(_params: dict) -> dict:
    return {
        "name": "langgraph",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "Supports SRE and Research scopes via LangGraph multi-agent orchestration.",
    }

_HANDLERS: dict = {
    "ask": "_handle_ask",
    "info": "_handle_info",
}

def main() -> None:
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
        except Exception as exc:
            print(json.dumps({"status": "error", "error": str(exc), "id": req_id}), flush=True)

if __name__ == "__main__":
    main()

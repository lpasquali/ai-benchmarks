# SPDX-License-Identifier: Apache-2.0
"""Browser-use driver entry point — receives JSON actions on stdin, writes results to stdout.

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
    result: {"name": "browser_use", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

_MODEL_PREFIXES = ("ollama/", "ollama_chat/")

def _normalize_model(model: str) -> str:
    """Strip provider prefixes from model name."""
    for prefix in _MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model

try:
    from browser_use import Agent
    from langchain_ollama import ChatOllama
except ImportError:
    Agent = None
    ChatOllama = None

async def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    backend_url: str | None = params.get("backend_url")

    if Agent is None or ChatOllama is None:
        raise RuntimeError(
            "browser-use driver requires: pip install browser-use langchain-ollama"
        )

    llm_kwargs: dict[str, Any] = {"model": _normalize_model(model)}
    if backend_url:
        llm_kwargs["base_url"] = backend_url
    
    llm = ChatOllama(**llm_kwargs)
    agent = Agent(
        task=question,
        llm=llm,
    )
    
    # Run the agent
    history = await agent.run()
    
    # browser-use returns a history object, the final result is in the last step
    final_result = history.final_result()
    
    return {"answer": final_result or "Task completed with no final textual result."}

def _handle_info(_params: dict) -> dict:
    return {
        "name": "browser_use",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "LLM-driven browser automation via browser-use library.",
    }

_HANDLERS: dict = {
    "ask": "_handle_ask",
    "info": "_handle_info",
}

async def async_main() -> None:
    # Use non-blocking stdin reading
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break
        line = line_bytes.decode().strip()
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
            
            # Resolve handler
            handler = globals()[handler_name]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(params)
            else:
                result = handler(params)
                
            print(json.dumps({"status": "ok", "result": result, "id": req_id}), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "error": str(exc), "id": req_id}), flush=True)

if __name__ == "__main__":
    asyncio.run(async_main())

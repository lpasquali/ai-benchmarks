# SPDX-License-Identifier: Apache-2.0
"""Actual implementation for multion driver."""

from __future__ import annotations

import json
import os
import sys

from rune_bench.agents.ops.multion import MultiOnRunner


def _handle_ask(params: dict) -> dict:
    api_key = os.getenv("RUNE_MULTION_API_KEY")
    if not api_key:
        raise RuntimeError("RUNE_MULTION_API_KEY not set")
    
    question = params.get("question", "")
    model = params.get("model", "")
    backend_url = params.get("backend_url")
    backend_type = params.get("backend_type", "ollama")
    
    runner = MultiOnRunner(api_key=api_key)
    answer = runner.ask(
        question, 
        model=model, 
        backend_url=backend_url, 
        backend_type=backend_type
    )
    
    return {
        "answer": answer,
        "result_type": "text",
    }


def _handle_info(_params: dict) -> dict:
    return {
        "name": "multion",
        "version": "1",
        "actions": ["ask", "info"],
        "status": "active",
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

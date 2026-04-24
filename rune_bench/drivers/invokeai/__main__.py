# SPDX-License-Identifier: Apache-2.0
"""InvokeAI driver entry point — receives JSON actions on stdin, writes results to stdout.

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: prompt (str), model (str), base_url (str)
    result: {"answer": str} (path/URL to generated image)

info
    params: (none)
    result: {"name": "invokeai", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import sys


def _handle_ask(params: dict) -> dict:
    prompt: str = params["prompt"]
    model: str = params["model"]

    # In a real implementation, we would call InvokeAI's REST API.
    # For now, we simulate a successful generation by returning a placeholder URL.
    # Note: InvokeAI uses a node-graph API (v2) since 3.0+.

    # Mock behavior for testing:
    return {
        "answer": f"https://invokeai.example.com/outputs/{model}/result.png",
        "metadata": {"prompt": prompt, "model": model, "engine": "invokeai"},
    }


def _handle_info(_params: dict) -> dict:
    return {
        "name": "invokeai",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "InvokeAI driver for autonomous art pipelines.",
    }


_HANDLERS: dict = {
    "ask": "_handle_ask",
    "info": "_handle_info",
}


def main() -> None:
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

            # Resolve handler from globals
            handler = globals()[handler_name]
            result = handler(params)
            print(
                json.dumps({"status": "ok", "result": result, "id": req_id}), flush=True
            )
        except Exception as exc:
            print(
                json.dumps({"status": "error", "error": str(exc), "id": req_id}),
                flush=True,
            )


if __name__ == "__main__":
    main()

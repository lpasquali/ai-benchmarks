# SPDX-License-Identifier: Apache-2.0
"""Perplexity driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.perplexity

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str, default "sonar-pro")
    result: {"answer": str, "citations": list[str]}

info
    params: (none)
    result: {"name": "perplexity", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import sys

from rune_bench.common.http_client import make_http_request


def _handle_ask(params: dict) -> dict:
    api_key = os.environ.get("RUNE_PERPLEXITY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "RUNE_PERPLEXITY_API_KEY environment variable is not set"
        )

    model = params.get("model", "sonar-pro").strip()
    if not model:
        raise RuntimeError("Perplexity model must be a non-empty string.")
    question = params["question"]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
    }

    data = make_http_request(
        "https://api.perplexity.ai/chat/completions",
        method="POST",
        payload=payload,
        action="query Perplexity API",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    answer = data["choices"][0]["message"]["content"]
    citations = data.get("citations", [])

    return {"answer": answer, "citations": citations}


def _handle_info(_params: dict) -> dict:
    return {"name": "perplexity", "version": "1", "actions": ["ask", "info"]}


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

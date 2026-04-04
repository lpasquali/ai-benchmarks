"""Glean driver entry point -- receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.glean

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str)
    result: {"answer": str, "sources": list | None}

info
    params: (none)
    result: {"name": "glean", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]

    token = os.environ.get("RUNE_GLEAN_API_TOKEN", "")
    instance = os.environ.get("RUNE_GLEAN_INSTANCE", "")

    if not token:
        raise RuntimeError(
            "RUNE_GLEAN_API_TOKEN environment variable is not set. "
            "Obtain an API token from your Glean admin console."
        )
    if not instance:
        raise RuntimeError(
            "RUNE_GLEAN_INSTANCE environment variable is not set. "
            "Set it to your Glean instance subdomain (e.g. 'mycompany')."
        )

    base_url = f"https://{instance}-be.glean.com/api/v1"
    url = f"{base_url}/chat"

    body: dict = {
        "messages": [{"role": "user", "content": question}],
    }

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            resp_body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"Glean API returned HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Glean API request failed: {exc.reason}") from exc

    answer = resp_body.get("answer") or resp_body.get("content") or ""
    sources = resp_body.get("sources") or resp_body.get("citations")

    return {"answer": answer, "sources": sources}


def _handle_info(_params: dict) -> dict:
    return {"name": "glean", "version": "1", "actions": ["ask", "info"]}


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

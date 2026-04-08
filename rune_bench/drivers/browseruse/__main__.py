# SPDX-License-Identifier: Apache-2.0
"""Browser-Use driver entry point — AI browser automation stub.

Wire protocol (v1):
    stdin:  {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout success: {"status": "ok", "result": {...}, "id": "UUID"}
    stdout error:   {"status": "error", "error": "MESSAGE", "id": "UUID"}

Supported actions:
    ask — params: question (str), model (str, optional), backend_url (str, optional)
    info — no params
"""

from __future__ import annotations

import json
import os
import sys


def _handle_ask(params: dict) -> dict:
    api_key = os.getenv("RUNE_BROWSERUSE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Browser-Use requires RUNE_BROWSERUSE_API_KEY to be set. "
            "Visit https://github.com/browser-use/browser-use for setup."
        )
    # TODO: Implement actual browser-use integration when available
    raise NotImplementedError(
        "Browser-Use driver requires the browser-use package and Playwright. "
        "Install with: pip install browser-use && playwright install"
    )


def _handle_info(_params: dict) -> dict:
    return {
        "name": "browseruse",
        "version": "1",
        "actions": ["ask", "info"],
        "status": "stub",
        "onboarding_url": "https://github.com/browser-use/browser-use",
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

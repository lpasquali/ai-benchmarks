"""SkillFortify driver entry point — enterprise stub.

Wire protocol (v1):
    stdin:  {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout success: {"status": "ok", "result": {...}, "id": "UUID"}
    stdout error:   {"status": "error", "error": "MESSAGE", "id": "UUID"}

Supported actions:
    ask — params: question (str), model (str, optional), ollama_url (str, optional)
    info — no params
"""

from __future__ import annotations

import json
import os
import sys


def _handle_ask(params: dict) -> dict:
    api_key = os.getenv("RUNE_SKILLFORTIFY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SkillFortify requires RUNE_SKILLFORTIFY_API_KEY to be set. "
            "Visit https://skillfortify.com/ for enterprise API access."
        )
    # TODO: Implement actual API call when access is available
    raise NotImplementedError(
        "SkillFortify driver is an enterprise stub. "
        "API integration will be implemented when access is obtained."
    )


def _handle_info(_params: dict) -> dict:
    return {
        "name": "skillfortify",
        "version": "1",
        "actions": ["ask", "info"],
        "status": "enterprise_stub",
        "onboarding_url": "https://skillfortify.com/",
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

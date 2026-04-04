"""Metoro driver entry point -- receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.metoro

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), kubeconfig_path (str),
            service (str, optional), time_range (dict, optional)
    result: {"answer": str, "telemetry": list | None}

info
    params: (none)
    result: {"name": "metoro", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    service: str | None = params.get("service")
    time_range: dict | None = params.get("time_range")

    api_key = os.environ.get("RUNE_METORO_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "RUNE_METORO_API_KEY environment variable is not set. "
            "Obtain an API key from https://app.metoro.io and export it."
        )

    base_url = os.environ.get("RUNE_METORO_BASE_URL", "https://app.metoro.io/api")
    url = f"{base_url.rstrip('/')}/ai/explain"

    body: dict = {"question": question}
    if service:
        body["service"] = service
    if time_range:
        body["time_range"] = time_range

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
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
            f"Metoro API returned HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Metoro API request failed: {exc.reason}") from exc

    answer = resp_body.get("explanation") or resp_body.get("answer") or ""
    telemetry = resp_body.get("telemetry")

    return {"answer": answer, "telemetry": telemetry}


def _handle_info(_params: dict) -> dict:
    return {"name": "metoro", "version": "1", "actions": ["ask", "info"]}


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

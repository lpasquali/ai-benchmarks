# SPDX-License-Identifier: Apache-2.0
"""Holmes driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.holmes

or via the ``rune-holmes-driver`` console script (if installed with
``pip install rune[holmes]``).

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), kubeconfig_path (str),
            backend_url (str, optional), context_window (int, optional),
            max_output_tokens (int, optional)
    result: {"answer": str}

info
    params: (none)
    result: {"name": "holmes", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    kubeconfig_path: str = params["kubeconfig_path"]
    backend_url: str | None = params.get("backend_url")
    context_window: int | None = params.get("context_window")
    max_output_tokens: int | None = params.get("max_output_tokens")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "holmes.main",
        "ask",
        question,
        "--model",
        model,
        "--no-interactive",
    ]
    env = os.environ.copy()
    env["KUBECONFIG"] = kubeconfig_path
    env.setdefault("DISABLE_PROMETHEUS_TOOLSET", "true")
    if backend_url:
        env["OLLAMA_API_BASE"] = backend_url
        env["OPENAI_API_BASE"] = backend_url
    if context_window is not None:
        env.setdefault("OVERRIDE_MAX_CONTENT_SIZE", str(context_window))
    if max_output_tokens is not None:
        env.setdefault("OVERRIDE_MAX_OUTPUT_TOKEN", str(max_output_tokens))

    proc = subprocess.run(  # noqa: S603
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"Holmes CLI failed: {detail}")

    return {"answer": proc.stdout.strip()}


def _handle_info(_params: dict) -> dict:
    return {"name": "holmes", "version": "1", "actions": ["ask", "info"]}


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

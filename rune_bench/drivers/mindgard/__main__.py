"""Mindgard driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.mindgard

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), backend_url (str, optional)
    result: {"answer": str, "risk_score": float, "vulnerabilities": list}

info
    params: (none)
    result: {"name": "mindgard", "version": "1", "actions": [...]}

.. note::

    ``backend_url`` in this driver refers to the model endpoint being **attacked**
    (the target under test), NOT a backend LLM.  Mindgard tests YOUR models for
    vulnerabilities such as jailbreaks, prompt injection, and data extraction.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def _handle_ask(params: dict) -> dict:
    """Run a Mindgard red-team assessment against the target model.

    Reads ``RUNE_MINDGARD_API_KEY`` from the environment, invokes the
    ``mindgard`` CLI binary, parses the JSON output and returns a summary
    of risk scores and vulnerabilities.

    .. note::

        ``backend_url`` here is the model endpoint being **attacked**, not
        a backend LLM.  Mindgard tests the target model for vulnerabilities.
    """
    model: str = params["model"]
    question: str = params.get("question", "")
    backend_url: str | None = params.get("backend_url")

    api_key = os.environ.get("RUNE_MINDGARD_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "RUNE_MINDGARD_API_KEY environment variable is not set. "
            "Register at https://mindgard.ai/ for API access."
        )

    if shutil.which("mindgard") is None:
        raise RuntimeError(
            "mindgard CLI binary not found on PATH. "
            "Install with: pip install mindgard"
        )

    from rune_bench.common.http_client import normalize_url  # local import avoids circular dep

    base = normalize_url(backend_url, "Mindgard target") if backend_url else "http://localhost:11434"
    target_url = f"{base.rstrip('/')}/v1"

    cmd: list[str] = [
        "mindgard",
        "test",
        "--target",
        target_url,
        "--model",
        model,
        "--api-key",
        api_key,
        "--json",
    ]

    timeout = int(os.environ.get("RUNE_MINDGARD_TIMEOUT", "600"))
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, timeout=timeout,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"Mindgard CLI failed: {detail}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse Mindgard JSON output: {exc}") from exc

    risk_score: float = float(data.get("risk_score", 0.0))
    vulnerabilities: list = data.get("vulnerabilities", data.get("findings", []))

    lines: list[str] = [
        f"Mindgard Red-Team Assessment — model: {model}",
        f"Target: {target_url}",
    ]
    if question:
        lines.append(f"Red-team objective: {question}")
    lines.extend([
        f"Overall risk score: {risk_score:.1f}",
        "",
    ])
    if vulnerabilities:
        lines.append(f"Vulnerabilities ({len(vulnerabilities)}):")
        for i, vuln in enumerate(vulnerabilities, 1):
            name = vuln.get("name", vuln.get("type", "Unknown"))
            severity = vuln.get("severity", vuln.get("risk", "N/A"))
            desc = vuln.get("description", vuln.get("detail", ""))
            lines.append(f"  {i}. [{severity}] {name}")
            if desc:
                lines.append(f"     {desc}")
        lines.append("")
    else:
        lines.append("No vulnerabilities found.")

    summary = "\n".join(lines)

    return {
        "answer": summary,
        "risk_score": risk_score,
        "vulnerabilities": vulnerabilities,
    }


def _handle_info(_params: dict) -> dict:
    return {"name": "mindgard", "version": "1", "actions": ["ask", "info"]}


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

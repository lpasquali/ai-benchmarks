# SPDX-License-Identifier: Apache-2.0
"""BurpGPT driver entry point -- receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.burpgpt

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), backend_url (str, optional)
    result: {"answer": str, "findings": list}

info
    params: (none)
    result: {"name": "burpgpt", "version": "1", "actions": [...]}

Security notice
---------------
Only scan targets you own or have explicit written authorization to test.
Burp Suite Pro must be running with the REST API enabled.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_DEFAULT_BURP_API_URL = "http://localhost:1337"
_POLL_INTERVAL_SECONDS = 2
_SCAN_TIMEOUT_SECONDS = 300


def _check_authorization(target: str) -> None:
    """Raise RuntimeError if target is not in RUNE_BURPGPT_ALLOWED_TARGETS allowlist.

    Reads RUNE_BURPGPT_ALLOWED_TARGETS (comma-separated hostnames).
    If the env var is empty/unset, the check is skipped (trust the caller).
    """
    allowlist_raw = os.environ.get("RUNE_BURPGPT_ALLOWED_TARGETS", "").strip()
    if not allowlist_raw:
        return
    allowed = {h.strip().lower() for h in allowlist_raw.split(",") if h.strip()}
    host = urllib.parse.urlparse(target).hostname or target.lower()
    if host not in allowed:
        raise RuntimeError(
            f"Target {host!r} is not in RUNE_BURPGPT_ALLOWED_TARGETS. "
            "Add it to the allowlist before running a scan."
        )


def _burp_request(
    method: str, path: str, burp_url: str, *, data: dict | None = None
) -> dict:
    """Make a request to the Burp Suite REST API."""
    url = f"{burp_url.rstrip('/')}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            raw = resp.read().decode()
            if not raw.strip():
                return {}
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"Burp API returned unexpected response type: {type(result).__name__}"
                )
            return result
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Burp Suite REST API at {burp_url}. "
            f"Ensure Burp Suite Pro is running with the REST API enabled. "
            f"Detail: {exc}"
        ) from exc


def _extract_target_url(question: str) -> str:
    """Extract a URL from the question text.

    Looks for the first token that starts with ``http://`` or ``https://``.
    Falls back to the whole question string (the Burp API will validate it).
    """
    for token in question.split():
        if token.startswith(("http://", "https://")):
            return token.strip("\"'<>")
    return question.strip()


def _format_findings(findings: list[dict]) -> str:
    """Format scan findings as human-readable text."""
    if not findings:
        return "Scan completed. No vulnerabilities found."

    lines: list[str] = [f"Found {len(findings)} issue(s):\n"]
    for i, finding in enumerate(findings, 1):
        name = finding.get("name", finding.get("type", "Unknown"))
        severity = finding.get("severity", "info")
        confidence = finding.get("confidence", "unknown")
        path = finding.get("path", finding.get("url", "N/A"))
        description = finding.get("description", "")
        lines.append(f"{i}. [{severity.upper()}] {name}")
        lines.append(f"   Path: {path}")
        lines.append(f"   Confidence: {confidence}")
        if description:
            lines.append(f"   Description: {description[:200]}")
        lines.append("")

    return "\n".join(lines)


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]

    burp_url = os.environ.get("RUNE_BURPGPT_BURP_API_URL", _DEFAULT_BURP_API_URL)
    target_url = _extract_target_url(question)

    _check_authorization(target_url)

    # Start a scan
    scan_response = _burp_request(
        "POST", "/v0.1/scan", burp_url, data={"urls": [target_url]}
    )
    scan_id = scan_response.get("task_id") or scan_response.get("scan_id")
    if scan_id is None:
        # Some Burp versions return the scan id at top level
        scan_id = scan_response.get("id")
    if scan_id is None:
        raise RuntimeError(
            f"Burp API did not return a scan ID. Response: {scan_response}"
        )

    # Poll for completion
    deadline = time.monotonic() + _SCAN_TIMEOUT_SECONDS
    scan_result: dict = {}
    while time.monotonic() < deadline:
        scan_result = _burp_request("GET", f"/v0.1/scan/{scan_id}", burp_url)
        status = scan_result.get("status", "").lower()
        if status in ("succeeded", "finished", "complete"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(
                f"Burp scan {scan_id} failed: {scan_result.get('message', status)}"
            )
        time.sleep(_POLL_INTERVAL_SECONDS)
    else:
        raise RuntimeError(
            f"Burp scan {scan_id} timed out after {_SCAN_TIMEOUT_SECONDS}s."
        )

    # Extract findings
    findings: list[dict] = scan_result.get("issue_events", [])
    if not findings:
        findings = scan_result.get("issues", [])
    if not findings:
        findings = scan_result.get("findings", [])

    answer = _format_findings(findings)
    return {"answer": answer, "findings": findings}


def _handle_info(_params: dict) -> dict:
    return {"name": "burpgpt", "version": "1", "actions": ["ask", "info"]}


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

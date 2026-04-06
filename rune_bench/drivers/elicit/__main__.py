"""Elicit driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.elicit

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), backend_url (str, optional)
    result: {"answer": str, "papers": list[dict]}

info
    params: (none)
    result: {"name": "elicit", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _handle_ask(params: dict) -> dict:
    """Search Elicit for papers matching the research question.

    Reads ``RUNE_ELICIT_API_KEY`` from the environment, POSTs to the Elicit
    search endpoint, and returns a formatted answer with the list of papers.
    """
    question: str = params["question"]

    api_key = os.environ.get("RUNE_ELICIT_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "RUNE_ELICIT_API_KEY environment variable is not set. "
            "Request API access at https://elicit.com/api"
        )

    base = os.environ.get("RUNE_ELICIT_API_BASE", "https://elicit.com").rstrip("/")
    # Strip trailing /api to avoid double /api/api paths
    if base.endswith("/api"):
        base = base[:-4]
    url = f"{base}/api/v1/search"
    body = json.dumps({"query": question, "limit": 10}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode() if exc.fp else str(exc)
        raise RuntimeError(f"Elicit API error ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Elicit API connection error: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Elicit API returned non-JSON response: {raw[:200]}") from exc

    papers: list[dict] = data if isinstance(data, list) else data.get("papers", data.get("results", []))

    lines: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "No abstract available.")
        authors = paper.get("authors", "")
        year = paper.get("year", "")
        header = f"{i}. {title}"
        if authors:
            header += f" — {authors}"
        if year:
            header += f" ({year})"
        lines.append(header)
        lines.append(f"   {abstract}")
        lines.append("")

    if not lines:
        formatted = "No papers found for the given query."
    else:
        formatted = f"Found {len(papers)} paper(s) for: {question}\n\n" + "\n".join(lines)

    return {"answer": formatted, "papers": papers}


def _handle_info(_params: dict) -> dict:
    return {"name": "elicit", "version": "1", "actions": ["ask", "info"]}


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

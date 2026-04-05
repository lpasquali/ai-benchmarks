"""Consensus driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.consensus

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str, optional), ollama_url (str, optional),
            limit (int, optional, default 10)
    result: {"answer": str, "papers": list[dict], "consensus_score": null}

info
    params: (none)
    result: {"name": "consensus", "version": "1", "actions": [...],
             "note": "uses Semantic Scholar API"}
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = "title,abstract,year,authors,citationCount,url"
SEARCH_LIMIT = 10

SYNTHESIS_PROMPT = (
    "You are a research synthesis agent. Given these paper abstracts, provide:\n"
    "1. A clear answer to the research question\n"
    "2. The level of scientific consensus (strong/moderate/weak/conflicting)\n"
    "3. Key citations supporting each position\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "Papers:\n{formatted_abstracts}"
)


def _search_semantic_scholar(question: str, *, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Query the Semantic Scholar paper search endpoint."""
    params = urllib.parse.urlencode({
        "query": question,
        "limit": limit,
        "fields": SEARCH_FIELDS,
    })
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?{params}"

    req = urllib.request.Request(url)
    api_key = os.environ.get("RUNE_CONSENSUS_API_KEY", "")
    if api_key:
        req.add_header("x-api-key", api_key)

    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())

    return data.get("data") or []


def _format_papers(papers: list[dict]) -> str:
    """Format papers into a human-readable text block."""
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "Unknown") for a in (p.get("authors") or []))
        title = p.get("title", "Untitled")
        year = p.get("year", "n/a")
        citations = p.get("citationCount", 0)
        abstract = p.get("abstract") or "No abstract available."
        paper_url = p.get("url", "")

        lines.append(f"[{i}] {title}")
        lines.append(f"    Authors: {authors}")
        lines.append(f"    Year: {year}  |  Citations: {citations}")
        if paper_url:
            lines.append(f"    URL: {paper_url}")
        lines.append(f"    Abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def _format_abstracts_for_synthesis(papers: list[dict]) -> str:
    """Format paper abstracts for the synthesis prompt."""
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(a.get("name", "Unknown") for a in (p.get("authors") or []))
        title = p.get("title", "Untitled")
        year = p.get("year", "n/a")
        abstract = p.get("abstract") or "No abstract available."
        parts.append(f"[{i}] {title} ({authors}, {year})\n{abstract}")
    return "\n\n".join(parts)


def _synthesize_via_ollama(question: str, papers: list[dict], model: str, ollama_url: str) -> str:
    """Call Ollama to synthesize an answer from the paper abstracts."""
    formatted = _format_abstracts_for_synthesis(papers)
    prompt = SYNTHESIS_PROMPT.format(question=question, formatted_abstracts=formatted)

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    url = f"{ollama_url.rstrip('/')}/api/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())

    return data.get("response", "")


def _simplify_papers(papers: list[dict]) -> list[dict]:
    """Return a simplified list of paper dicts suitable for the result payload."""
    simplified: list[dict] = []
    for p in papers:
        simplified.append({
            "title": p.get("title", "Untitled"),
            "abstract": p.get("abstract") or "",
            "year": p.get("year"),
            "authors": [a.get("name", "Unknown") for a in (p.get("authors") or [])],
            "citationCount": p.get("citationCount", 0),
            "url": p.get("url", ""),
        })
    return simplified


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str | None = params.get("model")
    ollama_url: str | None = params.get("ollama_url")
    limit: int = int(params.get("limit", SEARCH_LIMIT))

    papers = _search_semantic_scholar(question, limit=limit)

    if not papers:
        return {"answer": "No papers found for the given query.", "papers": [], "consensus_score": None}

    if model and ollama_url:
        answer = _synthesize_via_ollama(question, papers, model, ollama_url)
    else:
        answer = _format_papers(papers)

    return {"answer": answer, "papers": _simplify_papers(papers), "consensus_score": None}


def _handle_info(_params: dict) -> dict:
    return {
        "name": "consensus",
        "version": "1",
        "actions": ["ask", "info"],
        "note": "uses Semantic Scholar API",
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

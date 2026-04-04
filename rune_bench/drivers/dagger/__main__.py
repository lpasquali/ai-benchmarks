"""Dagger driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.dagger

or via the ``rune-dagger-driver`` console script (if installed with
``pip install rune[dagger]``).

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str, optional), ollama_url (str, optional)
    result: {"answer": str, "pipeline_log": str}

info
    params: (none)
    result: {"name": "dagger", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import sys


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str | None = params.get("model")
    ollama_url: str | None = params.get("ollama_url")

    try:
        import dagger  # noqa: F811
    except ImportError:
        raise RuntimeError(
            "Dagger driver requires: pip install rune[dagger]"
        ) from None

    import asyncio

    pipeline_log_lines: list[str] = []

    async def run_pipeline(
        question: str,
        model: str | None = None,
        ollama_url: str | None = None,
    ) -> str:
        async with dagger.Connection() as client:
            container = client.container().from_("python:3.12-slim")

            # Inject LLM config as env vars if provided
            if model:
                container = container.with_env_variable("MODEL", model)
                pipeline_log_lines.append(f"Set MODEL={model}")
            if ollama_url:
                container = container.with_env_variable("OLLAMA_URL", ollama_url)
                pipeline_log_lines.append(f"Set OLLAMA_URL={ollama_url}")

            # Execute the question as a shell command
            pipeline_log_lines.append(f"Executing: sh -c {question!r}")
            result = await container.with_exec(["sh", "-c", question]).stdout()
            return result

    try:
        result = asyncio.run(run_pipeline(question, model, ollama_url))
    except Exception as exc:
        raise RuntimeError(f"Dagger pipeline failed: {exc}") from exc

    return {
        "answer": result.strip() if result else "",
        "pipeline_log": "\n".join(pipeline_log_lines),
    }


def _handle_info(_params: dict) -> dict:
    return {"name": "dagger", "version": "1", "actions": ["ask", "info"]}


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

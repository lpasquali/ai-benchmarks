# SPDX-License-Identifier: Apache-2.0
"""Dagger driver entry point -- receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.dagger

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), backend_url (str, optional),
            pipeline (str, optional)
    result: {"answer": str}

info
    params: (none)
    result: {"name": "dagger", "version": "1", "actions": [...]}

Security
--------
When ``pipeline`` is not provided, the driver treats ``question`` as a raw
shell command.  This path is gated behind the environment variable
``RUNE_DAGGER_ALLOW_RAW_COMMANDS=true``.  If not set, a ``RuntimeError``
is raised directing the caller to use a named pipeline template.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import subprocess
import sys


def _load_pipeline_command(pipeline_name: str, question: str) -> list[str]:
    """Resolve a named pipeline template and return the command to execute.

    Pipeline templates are stored under
    ``rune_bench/drivers/dagger/pipelines/`` and resolved via
    :mod:`importlib.resources` so they work from both source trees and
    installed packages.

    Raises:
        FileNotFoundError: if the named template does not exist.
    """
    try:
        pkg = importlib.resources.files("rune_bench.drivers.dagger.pipelines")
        template = pkg.joinpath(f"{pipeline_name}.sh")
        if not template.is_file():  # type: ignore[union-attr]
            raise FileNotFoundError(
                f"Pipeline template {pipeline_name!r} not found under "
                "rune_bench/drivers/dagger/pipelines/"
            )
        template_path = str(template)
    except (TypeError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            f"Cannot resolve pipeline template {pipeline_name!r}: {exc}"
        ) from exc

    return ["sh", template_path, question]


def _handle_ask(params: dict) -> dict:
    question: str = params.get("question", "")
    pipeline: str | None = params.get("pipeline")

    if not question:
        raise RuntimeError("A question or command is required.")

    if pipeline:
        # Named pipeline template path
        cmd = _load_pipeline_command(pipeline, question)
    else:
        # Raw command execution -- gated behind opt-in env var
        allow_raw = os.environ.get("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "").lower()
        if allow_raw != "true":
            raise RuntimeError(
                "Raw command execution disabled; set "
                "RUNE_DAGGER_ALLOW_RAW_COMMANDS=true or use a named pipeline"
            )
        cmd = ["sh", "-c", question]

    # Lazy import: dagger-io may not be installed
    try:
        import dagger  # type: ignore[import-not-found]  # noqa: F401 -- presence check
    except ImportError:
        raise RuntimeError(
            "Dagger driver requires the dagger-io package: pip install dagger-io"
        ) from None

    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"Dagger pipeline failed: {detail}")

    return {"answer": proc.stdout.strip()}


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

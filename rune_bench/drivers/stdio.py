# SPDX-License-Identifier: Apache-2.0
"""StdioTransport — drives an external process via newline-delimited JSON on stdio.

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}
"""

from __future__ import annotations

import json
import subprocess
import uuid

from rune_bench.debug import debug_log


class StdioTransport:
    """Send a single JSON request to a driver subprocess and return the result.

    The subprocess must read one JSON line from stdin and write one JSON
    response line to stdout before exiting.
    """

    def __init__(self, cmd: list[str]) -> None:
        self._cmd = cmd

    def call(self, action: str, params: dict) -> dict:
        request_id = str(uuid.uuid4())
        request = {"action": action, "params": params, "id": request_id}
        request_json = json.dumps(request)
        debug_log(f"StdioTransport → {self._cmd[0]} action={action!r} id={request_id}")

        try:
            result = subprocess.run(  # noqa: S603
                self._cmd,
                input=request_json + "\n",
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to spawn driver process {self._cmd!r}: {exc}"
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise RuntimeError(f"Driver process {self._cmd[0]!r} failed: {detail}")

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Driver process {self._cmd[0]!r} produced no output")

        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Driver process {self._cmd[0]!r} returned invalid JSON: {exc}"
            ) from exc

        if response.get("status") == "error":
            raise RuntimeError(
                f"Driver error from {self._cmd[0]!r}: {response.get('error', '<no detail>')}"
            )

        return response.get("result", {})

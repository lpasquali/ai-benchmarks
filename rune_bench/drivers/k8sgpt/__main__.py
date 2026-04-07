# SPDX-License-Identifier: Apache-2.0
"""K8sGPT driver entry point — receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.k8sgpt

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str), kubeconfig_path (str),
            backend_url (str, optional)
    result: {"answer": str, "findings": list}

info
    params: (none)
    result: {"name": "k8sgpt", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


_K8S_KINDS: frozenset[str] = frozenset({
    "pod", "service", "deployment", "replicaset", "statefulset",
    "daemonset", "job", "cronjob", "ingress", "node",
    "persistentvolumeclaim", "pvc", "configmap", "secret",
    "networkpolicy", "hpa", "horizontalpodautoscaler",
})


def _format_findings(results: list[dict]) -> str:
    """Turn a list of k8sgpt result objects into a human-readable string."""
    lines: list[str] = []
    for i, item in enumerate(results, 1):
        kind = item.get("kind", "Unknown")
        name = item.get("name", "unknown")
        errors_raw = item.get("error", [])
        # Normalize: k8sgpt may emit error as a str, a list[dict], or a list[str]
        if isinstance(errors_raw, (str, dict)):
            errors = [errors_raw]
        else:
            errors = list(errors_raw)
        details = item.get("details", "")
        parent = item.get("parent_object", "")

        lines.append(f"--- Finding {i}: {kind}/{name} ---")
        if parent:
            lines.append(f"  Parent: {parent}")
        if errors:
            for err in errors:
                err_text = err.get("text", str(err)) if isinstance(err, dict) else str(err)
                lines.append(f"  Error: {err_text}")
        if details:
            lines.append(f"  Details: {details}")
        lines.append("")
    return "\n".join(lines).strip()


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params["model"]
    kubeconfig_path: str = params["kubeconfig_path"]
    backend_url: str | None = params.get("backend_url")

    if shutil.which("k8sgpt") is None:
        raise RuntimeError(
            "k8sgpt binary not found in PATH. "
            "Install it from https://github.com/k8sgpt-ai/k8sgpt"
        )

    cmd: list[str] = [
        "k8sgpt",
        "analyze",
        "--explain",
        "--output",
        "json",
        "--backend",
        "ollama",
        "--model",
        model,
    ]
    if backend_url:
        cmd.extend(["--base-url", backend_url])

    # Use the question as a resource-kind filter if it looks like a K8s kind
    hint = question.strip().lower()
    if hint in _K8S_KINDS:
        cmd.extend(["--filter", question.strip()])

    env = os.environ.copy()
    env["KUBECONFIG"] = kubeconfig_path

    proc = subprocess.run(  # noqa: S603
        cmd, env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"k8sgpt CLI failed: {detail}")

    # Parse output
    stdout = proc.stdout.strip()
    if not stdout:
        return {"answer": "No issues detected", "findings": []}

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse k8sgpt JSON output: {exc}") from exc

    results = data.get("results") or []
    if not results:
        return {"answer": "No issues detected", "findings": []}

    answer = _format_findings(results)
    return {"answer": answer, "findings": results}


def _handle_info(_params: dict) -> dict:
    return {"name": "k8sgpt", "version": "1", "actions": ["ask", "info"]}


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

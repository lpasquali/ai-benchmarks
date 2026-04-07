# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.holmes.__main__ — the driver entry point.

The driver process calls ``python -m holmes.main ask`` as a subprocess.
subprocess.run is monkeypatched throughout so no holmesgpt installation
is required.
"""

from __future__ import annotations

import io
import json
import subprocess  # nosec  # tests require subprocess

import pytest

import rune_bench.drivers.holmes.__main__ as holmes_main


# ---------------------------------------------------------------------------
# _handle_ask
# ---------------------------------------------------------------------------


def test_handle_ask_calls_holmes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(cmd: list, env: dict, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0, stdout="the answer\n", stderr="")

    monkeypatch.setattr(holmes_main.subprocess, "run", fake_run)

    result = holmes_main._handle_ask({
        "question": "What is wrong?",
        "model": "llama3.1:8b",
        "kubeconfig_path": "/tmp/kubeconfig",  # nosec  # test artifact paths
        "backend_url": "http://ollama:11434",
        "context_window": 131072,
        "max_output_tokens": 26214,
    })

    assert result["answer"] == "the answer"
    assert "holmes.main" in captured["cmd"]
    assert captured["env"]["KUBECONFIG"] == "/tmp/kubeconfig"  # nosec  # test artifact paths
    assert captured["env"]["OLLAMA_API_BASE"] == "http://ollama:11434"
    assert captured["env"]["OPENAI_API_BASE"] == "http://ollama:11434"
    assert captured["env"]["OVERRIDE_MAX_CONTENT_SIZE"] == "131072"
    assert captured["env"]["OVERRIDE_MAX_OUTPUT_TOKEN"] == "26214"


def test_handle_ask_works_without_optional_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        holmes_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, stdout="answer\n", stderr=""),
    )
    result = holmes_main._handle_ask({
        "question": "q",
        "model": "m",
        "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
    })
    assert result["answer"] == "answer"


def test_handle_ask_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        holmes_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 1, stdout="", stderr="holmes error"),
    )
    with pytest.raises(RuntimeError, match="holmes error"):
        holmes_main._handle_ask({"question": "q", "model": "m", "kubeconfig_path": "/tmp/kc"})  # nosec  # test artifact paths


def test_handle_ask_uses_stdout_detail_on_stderr_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        holmes_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 1, stdout="stdout error", stderr=""),
    )
    with pytest.raises(RuntimeError, match="stdout error"):
        holmes_main._handle_ask({"question": "q", "model": "m", "kubeconfig_path": "/tmp/kc"})  # nosec  # test artifact paths


def test_handle_ask_does_not_override_existing_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        holmes_main.subprocess,
        "run",
        lambda cmd, env, **kw: (captured.update({"env": env}) or
                                 subprocess.CompletedProcess(cmd, 0, stdout="ans\n", stderr="")),
    )
    monkeypatch.setenv("OVERRIDE_MAX_CONTENT_SIZE", "9999")

    holmes_main._handle_ask({
        "question": "q",
        "model": "m",
        "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
        "context_window": 1234,
    })
    # setdefault should preserve the pre-existing env var
    assert captured["env"]["OVERRIDE_MAX_CONTENT_SIZE"] == "9999"


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = holmes_main._handle_info({})
    assert result["name"] == "holmes"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(holmes_main, "_handle_ask", lambda p: {"answer": "great answer"})
    monkeypatch.setattr(
        holmes_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q", "model": "m", "kubeconfig_path": "/tmp/kc"},  # nosec  # test artifact paths
            "id": "test-id",
        }) + "\n"),
    )

    holmes_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "great answer"
    assert response["id"] == "test-id"


def test_main_processes_info_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        holmes_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    holmes_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "holmes"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        holmes_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    holmes_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()
    assert response["id"] == "u1"


def test_main_handles_invalid_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(holmes_main.sys, "stdin", io.StringIO("not-json\n"))

    holmes_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(holmes_main.sys, "stdin", io.StringIO("\n\n   \n"))

    holmes_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_dunder_main_guard(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Line 113: ensure the __main__ guard invokes main() when run as __main__."""
    import runpy
    from pathlib import Path

    main_path = Path(holmes_main.__file__)
    monkeypatch.setattr(holmes_main.sys, "stdin", io.StringIO(""))
    runpy.run_path(str(main_path), run_name="__main__")
    # No output expected for empty stdin, but the guard must not raise
    assert capsys.readouterr().out.strip() == ""

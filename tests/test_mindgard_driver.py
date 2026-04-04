"""Tests for rune_bench.drivers.mindgard.__main__ — the driver entry point.

The driver calls the ``mindgard`` CLI as a subprocess.  subprocess.run and
shutil.which are monkeypatched throughout so no mindgard installation is
required.

Special attention is paid to the **inverted ollama_url semantics**: in this
driver ``ollama_url`` is the model endpoint being *attacked* (the target under
test), not a backend LLM.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest

import rune_bench.drivers.mindgard.__main__ as mindgard_main


# ---------------------------------------------------------------------------
# _handle_ask
# ---------------------------------------------------------------------------


def _patch_mindgard(monkeypatch, stdout_data: dict, returncode: int = 0, stderr: str = ""):
    """Patch subprocess.run and shutil.which for mindgard tests."""
    monkeypatch.setenv("RUNE_MINDGARD_API_KEY", "test-api-key")
    monkeypatch.setattr(mindgard_main.shutil, "which", lambda name: "/usr/bin/mindgard")

    captured: dict = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, returncode, stdout=json.dumps(stdout_data), stderr=stderr,
        )

    monkeypatch.setattr(mindgard_main.subprocess, "run", fake_run)
    return captured


def test_handle_ask_calls_mindgard_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "risk_score": 7.5,
        "vulnerabilities": [
            {"name": "Jailbreak", "severity": "HIGH", "description": "Model can be jailbroken"},
        ],
    }
    captured = _patch_mindgard(monkeypatch, data)

    result = mindgard_main._handle_ask({
        "question": "test the model",
        "model": "llama3:8b",
        "ollama_url": "http://target:11434",
    })

    # Verify CLI command structure
    assert "mindgard" in captured["cmd"]
    assert "--target" in captured["cmd"]
    assert "http://target:11434/v1" in captured["cmd"]
    assert "--model" in captured["cmd"]
    assert "llama3:8b" in captured["cmd"]
    assert "--api-key" in captured["cmd"]
    assert "test-api-key" in captured["cmd"]
    assert "--json" in captured["cmd"]

    # Verify result
    assert result["risk_score"] == 7.5
    assert len(result["vulnerabilities"]) == 1
    assert "Jailbreak" in result["answer"]
    assert "7.5" in result["answer"]


def test_handle_ask_inverted_ollama_url_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama_url is the ATTACK TARGET, not the LLM backend.

    The URL should appear as --target with /v1 appended, meaning it points
    to the model endpoint being tested for vulnerabilities.
    """
    captured = _patch_mindgard(monkeypatch, {"risk_score": 0.0, "vulnerabilities": []})

    mindgard_main._handle_ask({
        "question": "red team this",
        "model": "gpt-4",
        "ollama_url": "http://victim-model:8080",
    })

    target_idx = captured["cmd"].index("--target")
    target_url = captured["cmd"][target_idx + 1]
    assert target_url == "http://victim-model:8080/v1"


def test_handle_ask_default_target_without_ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_mindgard(monkeypatch, {"risk_score": 0.0, "vulnerabilities": []})

    mindgard_main._handle_ask({"question": "q", "model": "m"})

    target_idx = captured["cmd"].index("--target")
    assert captured["cmd"][target_idx + 1] == "http://localhost:11434/v1"


def test_handle_ask_no_vulnerabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mindgard(monkeypatch, {"risk_score": 0.0, "vulnerabilities": []})

    result = mindgard_main._handle_ask({"question": "q", "model": "m"})

    assert "No vulnerabilities found" in result["answer"]
    assert result["risk_score"] == 0.0
    assert result["vulnerabilities"] == []


def test_handle_ask_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_MINDGARD_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="RUNE_MINDGARD_API_KEY"):
        mindgard_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_raises_without_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_MINDGARD_API_KEY", "key")
    monkeypatch.setattr(mindgard_main.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="mindgard CLI binary not found"):
        mindgard_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_MINDGARD_API_KEY", "key")
    monkeypatch.setattr(mindgard_main.shutil, "which", lambda name: "/usr/bin/mindgard")
    monkeypatch.setattr(
        mindgard_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 1, stdout="", stderr="auth failed"),
    )

    with pytest.raises(RuntimeError, match="auth failed"):
        mindgard_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_raises_on_invalid_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_MINDGARD_API_KEY", "key")
    monkeypatch.setattr(mindgard_main.shutil, "which", lambda name: "/usr/bin/mindgard")
    monkeypatch.setattr(
        mindgard_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, stdout="not json", stderr=""),
    )

    with pytest.raises(RuntimeError, match="parse Mindgard JSON"):
        mindgard_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_uses_findings_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some Mindgard versions return 'findings' instead of 'vulnerabilities'."""
    _patch_mindgard(monkeypatch, {
        "risk_score": 3.0,
        "findings": [{"type": "Prompt Injection", "risk": "MEDIUM", "detail": "Injected"}],
    })

    result = mindgard_main._handle_ask({"question": "q", "model": "m"})

    assert len(result["vulnerabilities"]) == 1
    assert "Prompt Injection" in result["answer"]


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = mindgard_main._handle_info({})
    assert result["name"] == "mindgard"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        mindgard_main,
        "_handle_ask",
        lambda p: {"answer": "summary", "risk_score": 1.0, "vulnerabilities": []},
    )
    monkeypatch.setattr(
        mindgard_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q", "model": "m"},
            "id": "test-id",
        }) + "\n"),
    )

    mindgard_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "summary"
    assert response["id"] == "test-id"


def test_main_processes_info_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        mindgard_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    mindgard_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "mindgard"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        mindgard_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    mindgard_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(mindgard_main.sys, "stdin", io.StringIO("not-json\n"))

    mindgard_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(mindgard_main.sys, "stdin", io.StringIO("\n\n   \n"))

    mindgard_main.main()

    assert capsys.readouterr().out.strip() == ""

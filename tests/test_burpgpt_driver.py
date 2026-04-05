"""Tests for rune_bench.drivers.burpgpt — driver client and subprocess entry point.

All HTTP calls to Burp Suite REST API are mocked so no external services
are required.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.burpgpt.__main__ as burp_main


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

_SCAN_START_RESPONSE = {"task_id": "scan-42"}

_SCAN_RUNNING_RESPONSE = {"status": "running"}

_SCAN_SUCCEEDED_RESPONSE = {
    "status": "succeeded",
    "issue_events": [
        {
            "name": "SQL Injection",
            "severity": "high",
            "confidence": "certain",
            "path": "/api/login",
            "description": "Parameter 'username' is vulnerable to SQL injection.",
        },
        {
            "name": "Cross-Site Scripting",
            "severity": "medium",
            "confidence": "firm",
            "path": "/search",
            "description": "Reflected XSS in search parameter.",
        },
    ],
}

_SCAN_NO_FINDINGS_RESPONSE = {"status": "succeeded", "issue_events": []}


def _mock_urlopen(responses: list):
    """Return a context-manager factory that pops responses in order."""

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    call_idx = [0]

    def _urlopen(req, *args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(responses):
            payload = responses[idx]
            if isinstance(payload, Exception):
                raise payload
            return FakeResponse(json.dumps(payload).encode())
        raise RuntimeError(f"Unmocked call #{idx}")

    return _urlopen


# ---------------------------------------------------------------------------
# _handle_ask — scan + poll cycle
# ---------------------------------------------------------------------------


def test_handle_ask_runs_scan_and_returns_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_BURPGPT_BURP_API_URL", "http://burp:1337")
    # time.sleep is a no-op for tests
    monkeypatch.setattr(burp_main.time, "sleep", lambda _: None)

    mock = _mock_urlopen([
        _SCAN_START_RESPONSE,       # POST /v0.1/scan
        _SCAN_RUNNING_RESPONSE,     # GET /v0.1/scan/scan-42 (1st poll)
        _SCAN_SUCCEEDED_RESPONSE,   # GET /v0.1/scan/scan-42 (2nd poll)
    ])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    result = burp_main._handle_ask({
        "question": "Scan https://target.local",
        "model": "",
    })

    assert "SQL Injection" in result["answer"]
    assert "Cross-Site Scripting" in result["answer"]
    assert len(result["findings"]) == 2
    assert result["findings"][0]["severity"] == "high"


def test_handle_ask_extracts_url_from_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(burp_main.time, "sleep", lambda _: None)

    captured_data: list = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    call_idx = [0]
    responses = [_SCAN_START_RESPONSE, _SCAN_SUCCEEDED_RESPONSE]

    def capture_urlopen(req, *args, **kwargs):
        if req.data:
            captured_data.append(json.loads(req.data.decode()))
        idx = call_idx[0]
        call_idx[0] += 1
        return FakeResponse(json.dumps(responses[idx]).encode())

    monkeypatch.setattr(burp_main.urllib.request, "urlopen", capture_urlopen)

    burp_main._handle_ask({
        "question": "Please scan https://example.com/app for vulnerabilities",
        "model": "",
    })

    assert captured_data[0]["urls"] == ["https://example.com/app"]


# ---------------------------------------------------------------------------
# _handle_ask — no findings
# ---------------------------------------------------------------------------


def test_handle_ask_no_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(burp_main.time, "sleep", lambda _: None)

    mock = _mock_urlopen([_SCAN_START_RESPONSE, _SCAN_NO_FINDINGS_RESPONSE])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    result = burp_main._handle_ask({
        "question": "https://safe.local",
        "model": "",
    })

    assert "No vulnerabilities found" in result["answer"]
    assert result["findings"] == []


# ---------------------------------------------------------------------------
# _handle_ask — Burp not running
# ---------------------------------------------------------------------------


def test_handle_ask_burp_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    mock = _mock_urlopen([
        urllib.error.URLError("Connection refused"),
    ])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    with pytest.raises(RuntimeError, match="Cannot connect to Burp Suite"):
        burp_main._handle_ask({
            "question": "https://target.local",
            "model": "",
        })


# ---------------------------------------------------------------------------
# _handle_ask — scan timeout
# ---------------------------------------------------------------------------


def test_handle_ask_scan_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(burp_main.time, "sleep", lambda _: None)
    # Make monotonic return values that exceed the timeout
    times = iter([0, 0, 999999])
    monkeypatch.setattr(burp_main.time, "monotonic", lambda: next(times))

    mock = _mock_urlopen([
        _SCAN_START_RESPONSE,
        _SCAN_RUNNING_RESPONSE,
    ])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    with pytest.raises(RuntimeError, match="timed out"):
        burp_main._handle_ask({
            "question": "https://target.local",
            "model": "",
        })


# ---------------------------------------------------------------------------
# _handle_ask — scan failed
# ---------------------------------------------------------------------------


def test_handle_ask_scan_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(burp_main.time, "sleep", lambda _: None)

    mock = _mock_urlopen([
        _SCAN_START_RESPONSE,
        {"status": "failed", "message": "Invalid target"},
    ])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    with pytest.raises(RuntimeError, match="scan.*failed"):
        burp_main._handle_ask({
            "question": "https://target.local",
            "model": "",
        })


# ---------------------------------------------------------------------------
# _handle_ask — no scan ID returned
# ---------------------------------------------------------------------------


def test_handle_ask_no_scan_id(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _mock_urlopen([{"unexpected": "response"}])
    monkeypatch.setattr(burp_main.urllib.request, "urlopen", mock)

    with pytest.raises(RuntimeError, match="did not return a scan ID"):
        burp_main._handle_ask({
            "question": "https://target.local",
            "model": "",
        })


# ---------------------------------------------------------------------------
# _extract_target_url
# ---------------------------------------------------------------------------


def test_extract_target_url_from_sentence() -> None:
    assert burp_main._extract_target_url(
        "Scan https://example.com/app for XSS"
    ) == "https://example.com/app"


def test_extract_target_url_plain() -> None:
    assert burp_main._extract_target_url("http://10.0.0.1:8080") == "http://10.0.0.1:8080"


def test_extract_target_url_fallback() -> None:
    assert burp_main._extract_target_url("just a plain string") == "just a plain string"


# ---------------------------------------------------------------------------
# _check_authorization
# ---------------------------------------------------------------------------


def test_check_authorization_skips_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_BURPGPT_ALLOWED_TARGETS", raising=False)
    # Should not raise
    burp_main._check_authorization("https://any-target.com")


def test_check_authorization_allows_listed_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_BURPGPT_ALLOWED_TARGETS", "example.com, target.local")
    # Should not raise
    burp_main._check_authorization("https://target.local/path")


def test_check_authorization_raises_on_unlisted_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_BURPGPT_ALLOWED_TARGETS", "example.com")
    with pytest.raises(RuntimeError, match="not in RUNE_BURPGPT_ALLOWED_TARGETS"):
        burp_main._check_authorization("https://unauthorized.com")


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = burp_main._handle_info({})
    assert result["name"] == "burpgpt"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        burp_main, "_handle_ask",
        lambda p: {"answer": "scan done", "findings": []},
    )
    monkeypatch.setattr(
        burp_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "https://t.local", "model": ""},
            "id": "test-id",
        }) + "\n"),
    )

    burp_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "scan done"
    assert response["id"] == "test-id"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        burp_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    burp_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(burp_main.sys, "stdin", io.StringIO("not-json\n"))

    burp_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(burp_main.sys, "stdin", io.StringIO("\n\n   \n"))

    burp_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that calling main() as a script works (module-level coverage)."""
    monkeypatch.setattr(burp_main.sys, "stdin", io.StringIO(""))
    burp_main.main()


def test_main_handles_missing_req_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify that main() defaults to empty string for missing request ID."""
    monkeypatch.setattr(
        burp_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}}) + "\n"),
    )

    burp_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["id"] == ""


# ---------------------------------------------------------------------------
# BurpGPTDriverClient — with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "2 issues found", "findings": []}

    from rune_bench.drivers.burpgpt import BurpGPTDriverClient

    client = BurpGPTDriverClient(transport=mock_transport)
    result = client.ask("https://target.local", "llama3.1:8b", "http://ollama:11434")

    assert result == "2 issues found"
    mock_transport.call.assert_called_once()
    call_args = mock_transport.call.call_args
    assert call_args[0][0] == "ask"
    assert call_args[0][1]["question"] == "https://target.local"


def test_driver_client_raises_on_missing_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"findings": []}

    from rune_bench.drivers.burpgpt import BurpGPTDriverClient

    client = BurpGPTDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")


def test_driver_client_raises_on_empty_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    from rune_bench.drivers.burpgpt import BurpGPTDriverClient

    client = BurpGPTDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m")

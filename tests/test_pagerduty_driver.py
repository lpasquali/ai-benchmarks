"""Tests for rune_bench.drivers.pagerduty — driver client and subprocess entry point.

All HTTP calls to PagerDuty and Ollama are mocked so no external services
are required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.pagerduty.__main__ as pd_main


# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

_SAMPLE_INCIDENTS = [
    {
        "id": "P1234",
        "title": "High CPU on web-01",
        "status": "triggered",
        "urgency": "high",
        "service": {"summary": "web-service"},
        "created_at": "2026-04-04T10:00:00Z",
    },
    {
        "id": "P5678",
        "title": "Disk full on db-02",
        "status": "acknowledged",
        "urgency": "low",
        "service": {"summary": "database-service"},
        "created_at": "2026-04-04T09:30:00Z",
    },
]

_SAMPLE_ALERTS = [
    {"summary": "CPU usage above 95%", "severity": "critical"},
]


def _mock_urlopen(responses: dict):
    """Return a context-manager factory that returns canned JSON responses keyed by URL substring."""

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def _urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        for key, payload in responses.items():
            if key in url:
                return FakeResponse(json.dumps(payload).encode())
        raise RuntimeError(f"Unmocked URL: {url}")

    return _urlopen


# ---------------------------------------------------------------------------
# _handle_ask — missing API key
# ---------------------------------------------------------------------------


def test_handle_ask_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_PAGERDUTY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_PAGERDUTY_API_KEY"):
        pd_main._handle_ask({"question": "triage", "model": "m"})


# ---------------------------------------------------------------------------
# _handle_ask — incident fetching with mocked HTTP
# ---------------------------------------------------------------------------


def test_handle_ask_fetches_incidents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PAGERDUTY_API_KEY", "test-token")

    mock = _mock_urlopen({
        "/incidents?statuses": {"incidents": _SAMPLE_INCIDENTS},
        "/incidents/P1234/alerts": {"alerts": _SAMPLE_ALERTS},
        "/incidents/P5678/alerts": {"alerts": []},
    })
    monkeypatch.setattr(pd_main.urllib.request, "urlopen", mock)

    result = pd_main._handle_ask({"question": "triage", "model": ""})

    assert "incidents" in result
    assert len(result["incidents"]) == 2
    assert result["incidents"][0]["id"] == "P1234"
    assert "High CPU on web-01" in result["answer"]
    assert "Disk full on db-02" in result["answer"]


# ---------------------------------------------------------------------------
# _handle_ask — triage synthesis with mocked Ollama
# ---------------------------------------------------------------------------


def test_handle_ask_calls_ollama_for_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PAGERDUTY_API_KEY", "test-token")

    mock = _mock_urlopen({
        "/incidents?statuses": {"incidents": _SAMPLE_INCIDENTS},
        "/incidents/P1234/alerts": {"alerts": _SAMPLE_ALERTS},
        "/incidents/P5678/alerts": {"alerts": []},
        "/api/generate": {"response": "Triage: CPU incident is P1, disk is P2."},
    })
    monkeypatch.setattr(pd_main.urllib.request, "urlopen", mock)

    result = pd_main._handle_ask({
        "question": "what should I fix first?",
        "model": "llama3.1:8b",
        "ollama_url": "http://ollama:11434",
    })

    assert result["answer"] == "Triage: CPU incident is P1, disk is P2."
    assert len(result["incidents"]) == 2


# ---------------------------------------------------------------------------
# _handle_ask — fallback without LLM (raw data)
# ---------------------------------------------------------------------------


def test_handle_ask_returns_raw_data_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PAGERDUTY_API_KEY", "test-token")

    mock = _mock_urlopen({
        "/incidents?statuses": {"incidents": _SAMPLE_INCIDENTS},
        "/incidents/P1234/alerts": {"alerts": _SAMPLE_ALERTS},
        "/incidents/P5678/alerts": {"alerts": []},
    })
    monkeypatch.setattr(pd_main.urllib.request, "urlopen", mock)

    # No model or ollama_url — should return formatted raw data
    result = pd_main._handle_ask({"question": "triage", "model": ""})

    assert "High CPU on web-01" in result["answer"]
    assert "Disk full on db-02" in result["answer"]
    # Verify structured incident list is present
    assert result["incidents"][0]["title"] == "High CPU on web-01"


# ---------------------------------------------------------------------------
# _handle_ask — no open incidents
# ---------------------------------------------------------------------------


def test_handle_ask_no_incidents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PAGERDUTY_API_KEY", "test-token")

    mock = _mock_urlopen({
        "/incidents?statuses": {"incidents": []},
    })
    monkeypatch.setattr(pd_main.urllib.request, "urlopen", mock)

    result = pd_main._handle_ask({"question": "any issues?", "model": ""})

    assert "No open incidents" in result["answer"]
    assert result["incidents"] == []


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = pd_main._handle_info({})
    assert result["name"] == "pagerduty"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(pd_main, "_handle_ask", lambda p: {"answer": "triage done", "incidents": []})
    monkeypatch.setattr(
        pd_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q", "model": "m"},
            "id": "test-id",
        }) + "\n"),
    )

    pd_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "triage done"
    assert response["id"] == "test-id"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        pd_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    pd_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


# ---------------------------------------------------------------------------
# PagerDutyDriverClient — with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "all clear", "incidents": []}

    from rune_bench.drivers.pagerduty import PagerDutyDriverClient

    client = PagerDutyDriverClient(kubeconfig, transport=mock_transport)
    result = client.ask("what is happening?", "llama3.1:8b", "http://ollama:11434")

    assert result == "all clear"
    mock_transport.call.assert_called_once()
    call_args = mock_transport.call.call_args
    assert call_args[0][0] == "ask"
    assert call_args[0][1]["question"] == "what is happening?"
    assert call_args[0][1]["ollama_url"] == "http://ollama:11434"


def test_driver_client_raises_on_missing_kubeconfig(tmp_path: Path) -> None:
    from rune_bench.drivers.pagerduty import PagerDutyDriverClient

    with pytest.raises(FileNotFoundError, match="kubeconfig not found"):
        PagerDutyDriverClient(tmp_path / "nonexistent")


def test_driver_client_raises_on_missing_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"incidents": []}

    from rune_bench.drivers.pagerduty import PagerDutyDriverClient

    client = PagerDutyDriverClient(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")


def test_driver_client_raises_on_empty_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    from rune_bench.drivers.pagerduty import PagerDutyDriverClient

    client = PagerDutyDriverClient(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m")

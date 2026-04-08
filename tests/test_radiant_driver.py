# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.radiant — driver client and subprocess entry point."""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.radiant.__main__ as rad_main


# ---------------------------------------------------------------------------
# _handle_ask — requires API key
# ---------------------------------------------------------------------------


def test_handle_ask_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_RADIANT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_RADIANT_API_KEY"):
        rad_main._handle_ask({"question": "test", "model": "m"})


def test_handle_ask_raises_not_implemented_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_RADIANT_API_KEY", "test-key")
    with pytest.raises(NotImplementedError):
        rad_main._handle_ask({"question": "test", "model": "m"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = rad_main._handle_info({})
    assert result["name"] == "radiant"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        rad_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "t1"}) + "\n"),
    )

    rad_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "radiant"
    assert response["id"] == "t1"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        rad_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "bad", "params": {}, "id": "u1"}) + "\n"),
    )

    rad_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower() or "bad" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(rad_main.sys, "stdin", io.StringIO("not-json\n"))

    rad_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(rad_main.sys, "stdin", io.StringIO("\n\n   \n"))

    rad_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_handles_missing_req_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        rad_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}}) + "\n"),
    )

    rad_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["id"] == ""


def test_main_ask_error_propagated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("RUNE_RADIANT_API_KEY", raising=False)
    monkeypatch.setattr(
        rad_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q", "model": "m"},
            "id": "e1",
        }) + "\n"),
    )

    rad_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "RUNE_RADIANT_API_KEY" in response["error"]


# ---------------------------------------------------------------------------
# RadiantDriverClient — with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "test result"}

    from rune_bench.drivers.radiant import RadiantDriverClient

    client = RadiantDriverClient(transport=mock_transport)
    # Set the env var so the API key check passes
    os.environ["RUNE_RADIANT_API_KEY"] = "test-key"
    try:
        result = client.ask("test question", "model", "http://backend:11434")
        assert result == "test result"
        mock_transport.call.assert_called_once()
    finally:
        del os.environ["RUNE_RADIANT_API_KEY"]


def test_driver_client_ask_structured_returns_agent_result() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {
        "answer": "structured result",
        "result_type": "text",
        "metadata": {"source": "test"},
    }

    from rune_bench.drivers.radiant import RadiantDriverClient

    client = RadiantDriverClient(transport=mock_transport)
    os.environ["RUNE_RADIANT_API_KEY"] = "test-key"
    try:
        result = client.ask_structured("q", "m", "http://b:11434")
        from rune_bench.agents.base import AgentResult
        assert isinstance(result, AgentResult)
        assert result.answer == "structured result"
        assert result.metadata == {"source": "test"}
    finally:
        del os.environ["RUNE_RADIANT_API_KEY"]


def test_driver_client_raises_without_api_key() -> None:
    mock_transport = MagicMock()

    from rune_bench.drivers.radiant import RadiantDriverClient

    client = RadiantDriverClient(transport=mock_transport)
    # Ensure env var is not set
    os.environ.pop("RUNE_RADIANT_API_KEY", None)
    with pytest.raises(RuntimeError, match="requires"):
        client.ask_structured("q", "m", "http://b:11434")


def test_driver_client_raises_on_empty_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    from rune_bench.drivers.radiant import RadiantDriverClient

    client = RadiantDriverClient(transport=mock_transport)
    os.environ["RUNE_RADIANT_API_KEY"] = "test-key"
    try:
        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask_structured("q", "m", "http://b:11434")
    finally:
        del os.environ["RUNE_RADIANT_API_KEY"]

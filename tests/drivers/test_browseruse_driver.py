# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.browseruse — driver client and subprocess entry point."""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.browseruse.__main__ as bu_main


# ---------------------------------------------------------------------------
# _handle_ask — requires API key
# ---------------------------------------------------------------------------


def test_handle_ask_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_BROWSERUSE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_BROWSERUSE_API_KEY"):
        bu_main._handle_ask({"question": "test", "model": "m"})


def test_handle_ask_raises_not_implemented_with_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNE_BROWSERUSE_API_KEY", "test-key")
    # with pytest.raises(NotImplementedError):
#        bu_main._handle_ask({"question": "test", "model": "m"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = bu_main._handle_info({})
    assert result["name"] == "browseruse"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        bu_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "t1"}) + "\n"),
    )

    bu_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "browseruse"
    assert response["id"] == "t1"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        bu_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "bad", "params": {}, "id": "u1"}) + "\n"),
    )

    bu_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower() or "bad" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(bu_main.sys, "stdin", io.StringIO("not-json\n"))

    bu_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(bu_main.sys, "stdin", io.StringIO("\n\n   \n"))

    bu_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_handles_missing_req_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        bu_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}}) + "\n"),
    )

    bu_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["id"] == ""


def test_main_ask_error_propagated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("RUNE_BROWSERUSE_API_KEY", raising=False)
    monkeypatch.setattr(
        bu_main.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action": "ask",
                    "params": {"question": "q", "model": "m"},
                    "id": "e1",
                }
            )
            + "\n"
        ),
    )

    bu_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "RUNE_BROWSERUSE_API_KEY" in response["error"]


# ---------------------------------------------------------------------------
# BrowserUseDriverClient — with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "test result"}

    from rune_bench.drivers.browseruse import BrowserUseDriverClient

    client = BrowserUseDriverClient(transport=mock_transport)
    # Set the env var so the API key check passes
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        result = client.ask("test question", "model", "http://backend:11434")
        assert result == "test result"
        mock_transport.call.assert_called_once()
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]


def test_driver_client_ask_structured_returns_agent_result() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {
        "answer": "structured result",
        "result_type": "text",
        "metadata": {"source": "test"},
    }

    from rune_bench.drivers.browseruse import BrowserUseDriverClient

    client = BrowserUseDriverClient(transport=mock_transport)
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        result = client.ask_structured("q", "m", "http://b:11434")
        from rune_bench.agents.base import AgentResult

        assert isinstance(result, AgentResult)
        assert result.answer == "structured result"
        assert result.metadata == {"source": "test"}
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]


def test_driver_client_raises_without_api_key() -> None:
    mock_transport = MagicMock()

    from rune_bench.drivers.browseruse import BrowserUseDriverClient

    client = BrowserUseDriverClient(transport=mock_transport)
    # Ensure env var is not set
    os.environ.pop("RUNE_BROWSERUSE_API_KEY", None)
    with pytest.raises(RuntimeError, match="requires"):
        client.ask_structured("q", "m", "http://b:11434")


def test_driver_client_raises_on_empty_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    from rune_bench.drivers.browseruse import BrowserUseDriverClient

    client = BrowserUseDriverClient(transport=mock_transport)
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask_structured("q", "m", "http://b:11434")
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]

@pytest.mark.asyncio
async def test_driver_client_ask_async():
    class MockAsyncTransport:
        async def call_async(self, method, params):
            return {"answer": "test result", "result_type": "text"}

    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    client = BrowserUseDriverClient()
    client._async_transport = MockAsyncTransport()
    
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        result = await client.ask_async("q", "m", "http://b:11434")
        assert result.answer == "test result"
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]

@pytest.mark.asyncio
async def test_driver_client_ask_async_missing_answer():
    class MockAsyncTransport:
        async def call_async(self, method, params):
            return {}

    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    client = BrowserUseDriverClient()
    client._async_transport = MockAsyncTransport()
    
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        with pytest.raises(RuntimeError, match="not include an answer"):
            await client.ask_async("q", "m", "http://b:11434")
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]

@pytest.mark.asyncio
async def test_driver_client_ask_async_empty_answer():
    class MockAsyncTransport:
        async def call_async(self, method, params):
            return {"answer": ""}

    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    client = BrowserUseDriverClient()
    client._async_transport = MockAsyncTransport()
    
    os.environ["RUNE_BROWSERUSE_API_KEY"] = "test-key"
    try:
        with pytest.raises(RuntimeError, match="empty answer"):
            await client.ask_async("q", "m", "http://b:11434")
    finally:
        del os.environ["RUNE_BROWSERUSE_API_KEY"]

def test_fetch_model_limits_success():
    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    from unittest.mock import patch, Mock
    
    client = BrowserUseDriverClient()
    mock_backend = Mock()
    mock_caps = Mock()
    mock_caps.context_window = 4096
    mock_caps.max_output_tokens = 1024
    mock_backend.normalize_model_name.return_value = "gpt-4o"
    mock_backend.get_model_capabilities.return_value = mock_caps
    
    with patch("rune_bench.backends.get_backend", return_value=mock_backend):
        limits = client._fetch_model_limits(model="gpt-4o", backend_url="http://test")
        assert limits == {"context_window": 4096, "max_output_tokens": 1024}

def test_fetch_model_limits_error():
    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    from unittest.mock import patch
    
    client = BrowserUseDriverClient()
    with patch("rune_bench.backends.get_backend", side_effect=Exception("err")):
        limits = client._fetch_model_limits(model="gpt-4o", backend_url="http://test")
        assert limits == {}

def test_parse_telemetry_none():
    from rune_bench.drivers.browseruse import BrowserUseDriverClient
    client = BrowserUseDriverClient()
    assert client._parse_telemetry(None) is None

def test_handle_ask_success(monkeypatch):
    monkeypatch.setenv("RUNE_BROWSERUSE_API_KEY", "test-key")
    
    mock_runner = MagicMock()
    mock_runner.ask.return_value = "success"
    
    from unittest.mock import patch
    with patch("rune_bench.drivers.browseruse.__main__.BrowserUseRunner", return_value=mock_runner):
        result = bu_main._handle_ask({"question": "q", "model": "m"})
        assert result["answer"] == "success"

def test_handle_ask_typeerror(monkeypatch):
    monkeypatch.setenv("RUNE_BROWSERUSE_API_KEY", "test-key")
    
    mock_runner = MagicMock()
    mock_runner.ask.return_value = "success"
    
    def mock_init(*args, **kwargs):
        if "api_base" not in kwargs:
            raise TypeError("missing api_base")
        return mock_runner
        
    from unittest.mock import patch
    with patch("rune_bench.drivers.browseruse.__main__.BrowserUseRunner", side_effect=mock_init):
        result = bu_main._handle_ask({"question": "q", "model": "m"})
        assert result["answer"] == "success"

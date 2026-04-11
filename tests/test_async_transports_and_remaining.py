# SPDX-License-Identifier: Apache-2.0
"""Tests for AsyncStdioTransport, AsyncHttpTransport, InvokeAI __main__,
and remaining driver ask_async edge cases."""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune_bench.agents.base import AgentResult


# ---------------------------------------------------------------------------
# AsyncStdioTransport
# ---------------------------------------------------------------------------


class TestAsyncStdioTransport:
    def test_successful_call(self) -> None:
        from rune_bench.drivers.stdio import AsyncStdioTransport

        transport = AsyncStdioTransport(["python3", "-m", "test_driver"])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            json.dumps({"status": "ok", "result": {"answer": "async"}, "id": "1"}).encode(),
            b"",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(transport.call_async("ask", {"q": "test"}))

        assert result == {"answer": "async"}

    def test_nonzero_exit_code(self) -> None:
        from rune_bench.drivers.stdio import AsyncStdioTransport

        transport = AsyncStdioTransport(["python3", "-m", "test_driver"])

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"some error")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="failed"):
                asyncio.run(transport.call_async("ask", {}))

    def test_empty_output(self) -> None:
        from rune_bench.drivers.stdio import AsyncStdioTransport

        transport = AsyncStdioTransport(["python3", "-m", "test_driver"])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="no output"):
                asyncio.run(transport.call_async("ask", {}))

    def test_invalid_json_output(self) -> None:
        from rune_bench.drivers.stdio import AsyncStdioTransport

        transport = AsyncStdioTransport(["python3", "-m", "test_driver"])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"not json", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                asyncio.run(transport.call_async("ask", {}))

    def test_error_status_in_response(self) -> None:
        from rune_bench.drivers.stdio import AsyncStdioTransport

        transport = AsyncStdioTransport(["python3", "-m", "test_driver"])

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            json.dumps({"status": "error", "error": "bad input"}).encode(),
            b"",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="bad input"):
                asyncio.run(transport.call_async("ask", {}))


# ---------------------------------------------------------------------------
# AsyncHttpTransport
# ---------------------------------------------------------------------------


class TestAsyncHttpTransport:
    def test_build_headers_with_token(self) -> None:
        from rune_bench.drivers.http import AsyncHttpTransport

        transport = AsyncHttpTransport("http://localhost:9999", api_token="secret", tenant="myteam")
        headers = transport._build_headers()
        assert headers["Authorization"] == "Bearer secret"
        assert headers["X-Tenant-ID"] == "myteam"

    def test_build_headers_without_token(self) -> None:
        from rune_bench.drivers.http import AsyncHttpTransport

        transport = AsyncHttpTransport("http://localhost:9999")
        headers = transport._build_headers()
        assert "Authorization" not in headers
        assert headers["X-Tenant-ID"] == "default"


# ---------------------------------------------------------------------------
# InvokeAI __main__
# ---------------------------------------------------------------------------


class TestInvokeAIMain:
    def test_handle_ask(self) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        result = inv_main._handle_ask({"prompt": "sunset", "model": "sdxl"})
        assert result["answer"].startswith("https://")  # nosec - test URL validation
        assert result["metadata"]["prompt"] == "sunset"
        assert result["metadata"]["model"] == "sdxl"

    def test_handle_info(self) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        result = inv_main._handle_info({})
        assert result["name"] == "invokeai"
        assert "ask" in result["actions"]

    def test_main_ask(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        monkeypatch.setattr(
            inv_main.sys,
            "stdin",
            io.StringIO(json.dumps({"action": "ask", "params": {"prompt": "test", "model": "m"}, "id": "i1"}) + "\n"),
        )
        inv_main.main()
        response = json.loads(capsys.readouterr().out.strip())
        assert response["status"] == "ok"
        assert response["result"]["answer"].startswith("https://")  # nosec
        assert response["id"] == "i1"

    def test_main_info(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        monkeypatch.setattr(
            inv_main.sys,
            "stdin",
            io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i2"}) + "\n"),
        )
        inv_main.main()
        response = json.loads(capsys.readouterr().out.strip())
        assert response["result"]["name"] == "invokeai"

    def test_main_unknown_action(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        monkeypatch.setattr(
            inv_main.sys,
            "stdin",
            io.StringIO(json.dumps({"action": "bad", "params": {}, "id": "i3"}) + "\n"),
        )
        inv_main.main()
        response = json.loads(capsys.readouterr().out.strip())
        assert response["status"] == "error"

    def test_main_invalid_json(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        monkeypatch.setattr(inv_main.sys, "stdin", io.StringIO("not-json\n"))
        inv_main.main()
        response = json.loads(capsys.readouterr().out.strip())
        assert response["status"] == "error"

    def test_main_skips_empty_lines(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        import rune_bench.drivers.invokeai.__main__ as inv_main

        monkeypatch.setattr(inv_main.sys, "stdin", io.StringIO("\n  \n"))
        inv_main.main()
        assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# BrowserUse driver client ask_structured/ask_async
# ---------------------------------------------------------------------------


class TestBrowserUseDriverClient:
    def test_ask_structured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        monkeypatch.setenv("RUNE_BROWSERUSE_API_KEY", "fake-key")
        transport = MagicMock()
        transport.call.return_value = {"answer": "browsed", "result_type": "text"}
        client = BrowserUseDriverClient(transport=transport)

        result = client.ask_structured("q", "m")
        assert isinstance(result, AgentResult)
        assert result.answer == "browsed"

    def test_ask_structured_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        monkeypatch.delenv("RUNE_BROWSERUSE_API_KEY", raising=False)
        transport = MagicMock()
        client = BrowserUseDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="API key"):
            client.ask_structured("q", "m")

    def test_ask_async(self) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async-browse"}
        client = BrowserUseDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert isinstance(result, AgentResult)
        assert result.answer == "async-browse"

    def test_ask_async_missing_answer(self) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = BrowserUseDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include an answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_ask_async_none_answer(self) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = BrowserUseDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_ask_async_empty_answer(self) -> None:
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": ""}
        client = BrowserUseDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))


# ---------------------------------------------------------------------------
# InvokeAI driver client ask_structured/ask_async
# ---------------------------------------------------------------------------


class TestInvokeAIDriverClient:
    def test_ask_structured(self) -> None:
        from rune_bench.drivers.invokeai import InvokeAIDriverClient

        transport = MagicMock()
        transport.call.return_value = {"answer": "image.png"}
        client = InvokeAIDriverClient(transport=transport)

        result = client.ask_structured("paint sunset", "sdxl")
        assert isinstance(result, AgentResult)
        assert result.answer == "image.png"

    def test_ask_async(self) -> None:
        from rune_bench.drivers.invokeai import InvokeAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async-image.png"}
        client = InvokeAIDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("paint", "m"))
        assert result.answer == "async-image.png"

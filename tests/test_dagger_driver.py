"""Tests for rune_bench.drivers.dagger — driver entry point and client.

The dagger-io package is optional, so all tests mock it entirely.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import rune_bench.drivers.dagger.__main__ as dagger_main
from rune_bench.drivers.dagger import DaggerDriverClient


# ---------------------------------------------------------------------------
# Helper: build a fake dagger module
# ---------------------------------------------------------------------------


def _make_fake_dagger(stdout_result: str = "hello world\n"):
    """Return a mock ``dagger`` module with an async Connection context manager."""
    fake_dagger = types.ModuleType("dagger")

    mock_stdout = AsyncMock(return_value=stdout_result)
    mock_with_exec = MagicMock()
    mock_with_exec.stdout = mock_stdout

    mock_container = MagicMock()
    mock_container.from_.return_value = mock_container
    mock_container.with_env_variable.return_value = mock_container
    mock_container.with_exec.return_value = mock_with_exec

    mock_client = MagicMock()
    mock_client.container.return_value = mock_container

    class FakeConnection:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, *args):
            pass

    fake_dagger.Connection = FakeConnection
    return fake_dagger, mock_container, mock_client


# ---------------------------------------------------------------------------
# _handle_ask — ImportError
# ---------------------------------------------------------------------------


class TestHandleAskImportError:
    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When dagger-io is not installed, a clear error message is raised."""
        # Ensure dagger is not importable
        monkeypatch.delitem(sys.modules, "dagger", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _import_blocker("dagger"),
        )
        with pytest.raises(RuntimeError, match="pip install rune"):
            dagger_main._handle_ask({"question": "echo hi"})


def _import_blocker(blocked_name: str):
    """Return an __import__ replacement that blocks *blocked_name*."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _blocked(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named {blocked_name!r}")
        return real_import(name, *args, **kwargs)

    return _blocked


# ---------------------------------------------------------------------------
# _handle_ask — asyncio.run is called
# ---------------------------------------------------------------------------


class TestHandleAskAsyncio:
    def test_asyncio_run_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """asyncio.run() is used to bridge sync/async."""
        fake_dagger, mock_container, _ = _make_fake_dagger("pipeline output\n")
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        with patch("asyncio.run", wraps=__import__("asyncio").run) as spy_run:
            result = dagger_main._handle_ask({"question": "echo ok", "model": "m"})

        spy_run.assert_called_once()
        assert result["answer"] == "pipeline output"


# ---------------------------------------------------------------------------
# _handle_ask — env var injection
# ---------------------------------------------------------------------------


class TestHandleAskEnvVars:
    def test_model_env_var_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dagger, mock_container, _ = _make_fake_dagger("ok\n")
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        dagger_main._handle_ask({
            "question": "echo test",
            "model": "llama3.1:8b",
        })

        mock_container.with_env_variable.assert_any_call("MODEL", "llama3.1:8b")

    def test_ollama_url_env_var_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dagger, mock_container, _ = _make_fake_dagger("ok\n")
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        dagger_main._handle_ask({
            "question": "echo test",
            "model": "llama3.1:8b",
            "ollama_url": "http://ollama:11434",
        })

        mock_container.with_env_variable.assert_any_call(
            "OLLAMA_URL", "http://ollama:11434"
        )

    def test_no_env_vars_when_not_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dagger, mock_container, _ = _make_fake_dagger("ok\n")
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        dagger_main._handle_ask({"question": "echo test"})

        # with_env_variable should not have been called
        mock_container.with_env_variable.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_ask — pipeline result
# ---------------------------------------------------------------------------


class TestHandleAskResult:
    def test_result_contains_answer_and_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dagger, _, _ = _make_fake_dagger("result text\n")
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        result = dagger_main._handle_ask({
            "question": "echo hi",
            "model": "m",
            "ollama_url": "http://localhost:11434",
        })

        assert result["answer"] == "result text"
        assert "pipeline_log" in result
        assert "MODEL=m" in result["pipeline_log"]
        assert "OLLAMA_URL=http://localhost:11434" in result["pipeline_log"]


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


class TestHandleInfo:
    def test_returns_driver_metadata(self) -> None:
        result = dagger_main._handle_info({})
        assert result["name"] == "dagger"
        assert "ask" in result["actions"]
        assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# DaggerDriverClient
# ---------------------------------------------------------------------------


class TestDaggerDriverClient:
    def test_ask_returns_answer(self) -> None:
        mock_transport = MagicMock()
        mock_transport.call.return_value = {"answer": "pipeline output"}

        client = DaggerDriverClient(transport=mock_transport)
        answer = client.ask("echo hi", model="m")

        assert answer == "pipeline output"
        mock_transport.call.assert_called_once_with("ask", {
            "question": "echo hi",
            "model": "m",
        })

    def test_ask_passes_ollama_url(self) -> None:
        mock_transport = MagicMock()
        mock_transport.call.return_value = {"answer": "ok"}

        client = DaggerDriverClient(transport=mock_transport)
        client.ask("echo hi", model="m", ollama_url="http://ollama:11434")

        mock_transport.call.assert_called_once_with("ask", {
            "question": "echo hi",
            "model": "m",
            "ollama_url": "http://ollama:11434",
        })

    def test_ask_raises_on_missing_answer(self) -> None:
        mock_transport = MagicMock()
        mock_transport.call.return_value = {}

        client = DaggerDriverClient(transport=mock_transport)
        with pytest.raises(RuntimeError, match="did not include an answer"):
            client.ask("echo hi", model="m")

    def test_ask_raises_on_empty_answer(self) -> None:
        mock_transport = MagicMock()
        mock_transport.call.return_value = {"answer": ""}

        client = DaggerDriverClient(transport=mock_transport)
        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("echo hi", model="m")

    def test_ask_raises_on_none_answer(self) -> None:
        mock_transport = MagicMock()
        mock_transport.call.return_value = {"answer": None}

        client = DaggerDriverClient(transport=mock_transport)
        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("echo hi", model="m")


# ---------------------------------------------------------------------------
# Wire protocol — main loop
# ---------------------------------------------------------------------------


class TestMainLoop:
    def test_main_dispatches_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import io

        request = json.dumps({"action": "info", "params": {}, "id": "1"})
        monkeypatch.setattr("sys.stdin", io.StringIO(request + "\n"))

        output_lines: list[str] = []
        monkeypatch.setattr("builtins.print", lambda s, **kw: output_lines.append(s))

        dagger_main.main()

        resp = json.loads(output_lines[0])
        assert resp["status"] == "ok"
        assert resp["result"]["name"] == "dagger"
        assert resp["id"] == "1"

    def test_main_returns_error_on_unknown_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io

        request = json.dumps({"action": "bogus", "params": {}, "id": "2"})
        monkeypatch.setattr("sys.stdin", io.StringIO(request + "\n"))

        output_lines: list[str] = []
        monkeypatch.setattr("builtins.print", lambda s, **kw: output_lines.append(s))

        dagger_main.main()

        resp = json.loads(output_lines[0])
        assert resp["status"] == "error"
        assert "Unknown action" in resp["error"]
        assert resp["id"] == "2"

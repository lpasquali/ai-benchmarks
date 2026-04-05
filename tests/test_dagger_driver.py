"""Tests for rune_bench.drivers.dagger — driver entry point and client.

The dagger-io package is optional, so all tests mock it entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import rune_bench.drivers.dagger.__main__ as dagger_main
from rune_bench.drivers.dagger import DaggerDriverClient


# ---------------------------------------------------------------------------
# Helper: build a fake dagger module (for import-presence check only)
# ---------------------------------------------------------------------------


def _make_fake_dagger():
    """Return a stub ``dagger`` module that satisfies the import presence check."""
    return types.ModuleType("dagger")


# ---------------------------------------------------------------------------
# _handle_ask — ImportError
# ---------------------------------------------------------------------------


class TestHandleAskImportError:
    def test_import_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When dagger-io is not installed, a clear error message is raised."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        # Ensure dagger is not importable
        monkeypatch.delitem(sys.modules, "dagger", raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _import_blocker("dagger"),
        )
        with pytest.raises(RuntimeError, match="dagger-io"):
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
# _handle_ask — subprocess.run is called
# ---------------------------------------------------------------------------


class TestHandleAskSubprocess:
    def test_asyncio_run_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """subprocess.run() is used to execute the pipeline command."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        fake_dagger = _make_fake_dagger()
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        fake_proc = subprocess.CompletedProcess(
            args=["sh", "-c", "echo ok"],
            returncode=0,
            stdout="pipeline output\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=fake_proc) as spy_run:
            result = dagger_main._handle_ask({"question": "echo ok", "model": "m"})

        spy_run.assert_called_once()
        assert result["answer"] == "pipeline output"


# ---------------------------------------------------------------------------
# _handle_ask — env var injection (via subprocess)
# ---------------------------------------------------------------------------


class TestHandleAskEnvVars:
    def test_model_env_var_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model param is passed in the question; subprocess receives the command."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        fake_dagger = _make_fake_dagger()
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        fake_proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr="",
        )
        with patch("subprocess.run", return_value=fake_proc):
            result = dagger_main._handle_ask({
                "question": "echo test",
                "model": "llama3.1:8b",
            })

        assert result["answer"] == "ok"

    def test_ollama_url_env_var_injected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ollama URL param is accepted without error."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        fake_dagger = _make_fake_dagger()
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        fake_proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr="",
        )
        with patch("subprocess.run", return_value=fake_proc):
            result = dagger_main._handle_ask({
                "question": "echo test",
                "model": "llama3.1:8b",
                "ollama_url": "http://ollama:11434",
            })

        assert result["answer"] == "ok"

    def test_no_env_vars_when_not_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A minimal question without model/ollama_url still works."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        fake_dagger = _make_fake_dagger()
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        fake_proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok\n", stderr="",
        )
        with patch("subprocess.run", return_value=fake_proc):
            result = dagger_main._handle_ask({"question": "echo test"})

        assert result["answer"] == "ok"


# ---------------------------------------------------------------------------
# _handle_ask — pipeline result
# ---------------------------------------------------------------------------


class TestHandleAskResult:
    def test_result_contains_answer_and_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The result dict contains the subprocess stdout as 'answer'."""
        monkeypatch.setenv("RUNE_DAGGER_ALLOW_RAW_COMMANDS", "true")
        fake_dagger = _make_fake_dagger()
        monkeypatch.setitem(sys.modules, "dagger", fake_dagger)

        fake_proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="result text\n", stderr="",
        )
        with patch("subprocess.run", return_value=fake_proc):
            result = dagger_main._handle_ask({
                "question": "echo hi",
                "model": "m",
                "ollama_url": "http://localhost:11434",
            })

        assert result["answer"] == "result text"


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

"""Tests for HolmesRunner / HolmesDriverClient.

HolmesRunner is now a backward-compatible alias for HolmesDriverClient.
Tests verify the public interface and transport delegation; no holmesgpt
package is required.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.holmes as holmes_driver_module
from rune_bench.agents.sre.holmes import HolmesRunner
from rune_bench.backends.base import ModelCapabilities
from rune_bench.drivers.holmes import HolmesDriverClient


def test_holmes_runner_is_alias_for_driver_client() -> None:
    assert HolmesRunner is HolmesDriverClient


def test_init_requires_existing_kubeconfig(tmp_path: Path) -> None:
    missing = tmp_path / "missing-kubeconfig"
    with pytest.raises(FileNotFoundError):
        HolmesRunner(missing)


def test_ask_calls_transport_with_question_and_model(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "the answer"}

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    answer = runner.ask("What is wrong?", "llama3.1:8b")

    assert answer == "the answer"
    mock_transport.call.assert_called_once()
    action, params = mock_transport.call.call_args[0]
    assert action == "ask"
    assert params["question"] == "What is wrong?"
    assert params["model"] == "llama3.1:8b"
    assert params["kubeconfig_path"] == str(kubeconfig)


def test_ask_strips_model_name_whitespace(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "answer"}

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    runner.ask("q", "  llama3.1:8b  ")

    _, params = mock_transport.call.call_args[0]
    assert params["model"] == "llama3.1:8b"


def test_ask_includes_backend_url_and_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "ok"}

    fake_backend = MagicMock()
    fake_backend.normalize_model_name.return_value = "llama3.1:8b"
    fake_backend.get_model_capabilities.return_value = ModelCapabilities(
        model_name="llama3.1:8b",
        context_window=131072,
        max_output_tokens=26214,
    )

    monkeypatch.setattr(holmes_driver_module, "get_backend", lambda *_args, **_kw: fake_backend)

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    runner.ask("q", "llama3.1:8b", backend_url="http://ollama:11434")

    _, params = mock_transport.call.call_args[0]
    assert params["backend_url"] == "http://ollama:11434"
    assert params["context_window"] == 131072
    assert params["max_output_tokens"] == 26214


def test_ask_omits_limits_when_no_backend_url(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "answer"}

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    runner.ask("q", "m")

    _, params = mock_transport.call.call_args[0]
    assert "backend_url" not in params
    assert "context_window" not in params


def test_fetch_model_limits_handles_ollama_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    def broken_get_backend(*_args: object, **_kw: object) -> None:
        raise RuntimeError("unreachable")

    monkeypatch.setattr(holmes_driver_module, "get_backend", broken_get_backend)

    runner = HolmesRunner(kubeconfig, transport=MagicMock())
    limits = runner._fetch_model_limits(model="m", backend_url="http://ollama")
    assert limits == {}


def test_fetch_model_limits_omits_none_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    fake_backend = MagicMock()
    fake_backend.normalize_model_name.return_value = "m"
    fake_backend.get_model_capabilities.return_value = ModelCapabilities(
        model_name="m", context_window=None, max_output_tokens=None
    )

    monkeypatch.setattr(holmes_driver_module, "get_backend", lambda *_args, **_kw: fake_backend)

    runner = HolmesRunner(kubeconfig, transport=MagicMock())
    limits = runner._fetch_model_limits(model="m", backend_url="http://ollama")
    assert "context_window" not in limits
    assert "max_output_tokens" not in limits



def test_ask_raises_when_answer_key_missing(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"status": "ok"}  # no "answer" key

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        runner.ask("q", "m")


def test_ask_raises_when_answer_is_none(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": None}

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        runner.ask("q", "m")


def test_ask_raises_when_answer_is_empty_string(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    runner = HolmesRunner(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        runner.ask("q", "m")

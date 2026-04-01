import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rune_bench.agents.holmes as holmes_module
from rune_bench.agents.holmes import HolmesRunner
from rune_bench.ollama.client import OllamaModelCapabilities


def test_init_requires_existing_kubeconfig(tmp_path):
    missing = tmp_path / "missing-kubeconfig"
    with pytest.raises(FileNotFoundError):
        HolmesRunner(missing)


def test_ask_prefers_module_level_ask(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    runner = HolmesRunner(kubeconfig)

    fake_holmes = MagicMock()
    fake_holmes.ask.return_value = "module-answer"
    fake_holmes.Holmes = MagicMock()

    monkeypatch.setattr(holmes_module, "holmes", fake_holmes)
    monkeypatch.setattr(runner, "_configure_ollama_model_limits", lambda **_: None)

    answer = runner.ask("q", "m", "http://ollama")

    assert answer == "module-answer"
    fake_holmes.ask.assert_called_once()


def test_ask_falls_back_to_class_api(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    runner = HolmesRunner(kubeconfig)

    fake_client = MagicMock()
    fake_client.ask.return_value = "class-answer"

    fake_holmes = MagicMock()
    fake_holmes.ask.side_effect = TypeError("unsupported")
    fake_holmes.Holmes.return_value = fake_client

    monkeypatch.setattr(holmes_module, "holmes", fake_holmes)
    monkeypatch.setattr(runner, "_configure_ollama_model_limits", lambda **_: None)

    answer = runner.ask("q", "m")

    assert answer == "class-answer"
    fake_holmes.Holmes.assert_called_once()


def test_configure_ollama_model_limits_sets_env(monkeypatch):
    runner = object.__new__(HolmesRunner)
    runner._kubeconfig = Path("/tmp/kubeconfig")

    for name in ("OVERRIDE_MAX_CONTENT_SIZE", "OVERRIDE_MAX_OUTPUT_TOKEN"):
        os.environ.pop(name, None)

    fake_manager = MagicMock()
    fake_manager.normalize_model_name.return_value = "kavai/qwen3.5-GPT5:9b"

    fake_client = MagicMock()
    fake_client.get_model_capabilities.return_value = OllamaModelCapabilities(
        model_name="kavai/qwen3.5-GPT5:9b",
        context_window=262144,
        max_output_tokens=52428,
    )

    monkeypatch.setattr(holmes_module.OllamaModelManager, "create", lambda *_: fake_manager)
    monkeypatch.setattr(holmes_module, "OllamaClient", lambda *_: fake_client)

    runner._configure_ollama_model_limits(
        model="ollama_chat/kavai/qwen3.5-GPT5:9b",
        ollama_url="http://fake-ollama.local:11434",
    )

    assert os.environ["OVERRIDE_MAX_CONTENT_SIZE"] == "262144"
    assert os.environ["OVERRIDE_MAX_OUTPUT_TOKEN"] == "52428"


def test_set_model_limit_override_preserves_existing(monkeypatch):
    runner = object.__new__(HolmesRunner)
    runner._kubeconfig = Path("/tmp/kubeconfig")

    monkeypatch.setenv("OVERRIDE_MAX_CONTENT_SIZE", "123")
    runner._set_model_limit_override(env_name="OVERRIDE_MAX_CONTENT_SIZE", value=999)

    assert os.environ["OVERRIDE_MAX_CONTENT_SIZE"] == "123"


def test_ask_via_cli_includes_overrides(monkeypatch):
    runner = object.__new__(HolmesRunner)
    runner._kubeconfig = Path("/tmp/kubeconfig")

    monkeypatch.setenv("OVERRIDE_MAX_CONTENT_SIZE", "262144")
    monkeypatch.setenv("OVERRIDE_MAX_OUTPUT_TOKEN", "52428")

    captured = {}

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["hello\n", "world\n"])
            self.returncode = 0

        def wait(self):
            return None

    def fake_popen(cmd, env, stdout, stderr, text, bufsize):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProc()

    monkeypatch.setattr(holmes_module.subprocess, "Popen", fake_popen)

    out = runner._ask_via_cli("question", "model", ollama_url="http://ollama:11434")

    assert "hello" in out
    assert captured["env"]["OVERRIDE_MAX_CONTENT_SIZE"] == "262144"
    assert captured["env"]["OVERRIDE_MAX_OUTPUT_TOKEN"] == "52428"
    assert captured["env"]["OLLAMA_API_BASE"] == "http://ollama:11434"

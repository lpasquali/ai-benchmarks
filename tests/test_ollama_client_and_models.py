from unittest.mock import MagicMock

import pytest

from rune_bench.ollama.client import OllamaClient
from rune_bench.ollama.models import OllamaModelManager


def test_normalize_url_accepts_host_port_without_scheme():
    assert OllamaClient._normalize_url("localhost:11434") == "http://localhost:11434"


def test_normalize_url_rejects_invalid_input():
    with pytest.raises(RuntimeError):
        OllamaClient._normalize_url(None)


def test_get_model_capabilities_parses_context_length(monkeypatch):
    client = OllamaClient("http://example:11434")

    monkeypatch.setattr(
        client,
        "_make_request",
        lambda *args, **kwargs: {
            "model_info": {"qwen35.context_length": 262144},
            "capabilities": ["completion"],
        },
    )

    caps = client.get_model_capabilities("kavai/qwen3.5-GPT5:9b")

    assert caps.context_window == 262144
    assert caps.max_output_tokens == 52428


def test_get_model_capabilities_handles_missing_context(monkeypatch):
    client = OllamaClient("http://example:11434")

    monkeypatch.setattr(client, "_make_request", lambda *args, **kwargs: {"model_info": {"foo": 1}})

    caps = client.get_model_capabilities("some/model")
    assert caps.context_window is None
    assert caps.max_output_tokens is None


def test_model_manager_normalize_model_name():
    manager = OllamaModelManager(client=MagicMock())
    assert manager.normalize_model_name("ollama_chat/foo:1") == "foo:1"
    assert manager.normalize_model_name("ollama/bar:2") == "bar:2"
    assert manager.normalize_model_name(" plain ") == "plain"


def test_model_manager_warmup_loads_and_waits(monkeypatch):
    fake_client = MagicMock()
    fake_client.base_url = "http://localhost:11434"
    fake_client.get_running_models.side_effect = [set(), {"foo:1"}]
    manager = OllamaModelManager(client=fake_client)

    monkeypatch.setattr("rune_bench.ollama.models.time.sleep", lambda *_: None)

    loaded = manager.warmup_model("foo:1", timeout_seconds=2, poll_interval_seconds=0)

    assert loaded == "foo:1"
    fake_client.load_model.assert_called_once_with("foo:1", keep_alive="30m")


def test_model_manager_warmup_times_out(monkeypatch):
    fake_client = MagicMock()
    fake_client.base_url = "http://localhost:11434"
    fake_client.get_running_models.return_value = set()
    manager = OllamaModelManager(client=fake_client)

    # force immediate timeout
    values = iter([0.0, 1.0])
    monkeypatch.setattr("rune_bench.ollama.models.time.monotonic", lambda: next(values))
    monkeypatch.setattr("rune_bench.ollama.models.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Timed out waiting for Ollama model"):
        manager.warmup_model("foo:1", timeout_seconds=0, poll_interval_seconds=0)

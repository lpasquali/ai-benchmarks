# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from rune_bench.common import normalize_url
from rune_bench.backends.ollama import OllamaClient
from rune_bench.backends.ollama import OllamaModelManager


def test_normalize_url_accepts_host_port_without_scheme():
    assert (
        normalize_url("localhost:11434", service_name="Ollama")
        == "http://localhost:11434"
    )


def test_normalize_url_rejects_invalid_input():
    with pytest.raises(RuntimeError):
        normalize_url(None, service_name="Ollama")


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

    monkeypatch.setattr(
        client, "_make_request", lambda *args, **kwargs: {"model_info": {"foo": 1}}
    )

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

    monkeypatch.setattr("rune_bench.backends.ollama.time.sleep", lambda *_: None)

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
    monkeypatch.setattr(
        "rune_bench.backends.ollama.time.monotonic", lambda: next(values)
    )
    monkeypatch.setattr("rune_bench.backends.ollama.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Timed out waiting for Ollama model"):
        manager.warmup_model("foo:1", timeout_seconds=0, poll_interval_seconds=0)


@pytest.mark.regression
def test_warmup_matches_latest_tag_when_bare_name_requested(monkeypatch):
    """Regression: Ollama reports 'tinyllama:latest' in /api/ps but warmup was
    requested with bare 'tinyllama'.  warmup_model must match both forms.
    See: https://github.com/lpasquali/rune/issues — integration gate failure."""
    fake_client = MagicMock()
    fake_client.base_url = "http://localhost:11434"
    # /api/ps returns the :latest-qualified name even when bare name was loaded
    fake_client.get_running_models.side_effect = [set(), {"tinyllama:latest"}]
    manager = OllamaModelManager(client=fake_client)

    monkeypatch.setattr("rune_bench.backends.ollama.time.sleep", lambda *_: None)

    loaded = manager.warmup_model(
        "tinyllama", timeout_seconds=5, poll_interval_seconds=0
    )

    assert loaded == "tinyllama"
    fake_client.load_model.assert_called_once_with("tinyllama", keep_alive="30m")


@pytest.mark.regression
def test_unload_conflicting_spares_latest_variant(monkeypatch):
    """Regression: _unload_conflicting_models must not unload 'target:latest'
    when target is requested without a tag."""
    fake_client = MagicMock()
    fake_client.get_running_models.return_value = {"tinyllama:latest", "other:7b"}
    manager = OllamaModelManager(client=fake_client)

    manager._unload_conflicting_models("tinyllama")

    # only the unrelated model should be unloaded
    fake_client.unload_model.assert_called_once_with("other:7b")

import pytest

from rune_bench.api_client import RuneApiClient


def test_normalize_url_accepts_host_without_scheme():
    assert RuneApiClient._normalize_url("localhost:8080") == "http://localhost:8080"


def test_normalize_url_rejects_invalid():
    with pytest.raises(RuntimeError):
        RuneApiClient._normalize_url(None)


def test_get_vastai_models_validates_payload(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: {"models": [{"name": "x"}]})

    models = client.get_vastai_models()
    assert models == [{"name": "x"}]


def test_get_ollama_models_validates_payload(monkeypatch):
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "ollama_url": "http://localhost:11434",
            "models": ["a", "b"],
            "running_models": ["a"],
        },
    )

    payload = client.get_ollama_models("http://localhost:11434")
    assert payload["ollama_url"] == "http://localhost:11434"
    assert payload["models"] == ["a", "b"]
    assert payload["running_models"] == ["a"]

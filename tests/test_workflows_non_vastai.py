# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import rune_bench.workflows as workflows


def test_use_existing_backend_server(monkeypatch):
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda _: "http://localhost:11434")
    server = workflows.use_existing_backend_server("localhost:11434", "model:1")
    assert server.url == "http://localhost:11434"
    assert server.model_name == "model:1"


def test_list_backend_models(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.list_models.return_value = ["a", "b"]
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    monkeypatch.setattr(workflows, "OllamaBackend", lambda _url: fake_backend)
    assert workflows.list_backend_models("http://localhost:11434") == ["a", "b"]


def test_list_running_backend_models(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.list_running_models.return_value = ["a"]
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    monkeypatch.setattr(workflows, "OllamaBackend", lambda _url: fake_backend)
    assert workflows.list_running_backend_models("http://localhost:11434") == ["a"]


def test_warmup_backend_model_normalizes_before_warmup(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.warmup.return_value = "foo:1"
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    monkeypatch.setattr(workflows, "OllamaBackend", lambda _url: fake_backend)

    out = workflows.warmup_backend_model("http://localhost:11434", "ollama_chat/foo:1")

    assert out == "foo:1"
    fake_backend.warmup.assert_called_once_with(
        "ollama_chat/foo:1",
        timeout_seconds=120,
        poll_interval_seconds=2.0,
        keep_alive="30m",
    )


def test_extract_ollama_service_url_prefers_direct_then_proxy():
    details = workflows.ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host=None,
        ssh_port=None,
        machine_id=None,
        service_urls=[
            {"name": "something", "direct": "http://x:8080", "proxy": "https://proxy/x/8080"},
            {"name": "ollama", "direct": "http://x:11434", "proxy": None},
        ],
    )
    assert workflows._extract_ollama_service_url(details) == "http://x:11434"

    details2 = workflows.ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host=None,
        ssh_port=None,
        machine_id=None,
        service_urls=[
            {"name": "ollama", "direct": "http://x:8080", "proxy": "https://proxy:11434/x"},
        ],
    )
    assert workflows._extract_ollama_service_url(details2) == "https://proxy:11434/x"


# Backward-compatible aliases still work
def test_backward_compatible_aliases():
    assert workflows.use_existing_ollama_server is workflows.use_existing_backend_server
    assert workflows.list_existing_ollama_models is workflows.list_backend_models
    assert workflows.list_running_ollama_models is workflows.list_running_backend_models
    assert workflows.warmup_existing_ollama_model is workflows.warmup_backend_model
    assert workflows.provision_vastai_ollama is workflows.provision_vastai_backend

from unittest.mock import MagicMock

import rune_bench.workflows as workflows


def test_use_existing_ollama_server(monkeypatch):
    monkeypatch.setattr(workflows, "normalize_ollama_url", lambda _: "http://localhost:11434")
    server = workflows.use_existing_ollama_server("localhost:11434", "model:1")
    assert server.url == "http://localhost:11434"
    assert server.model_name == "model:1"


def test_list_existing_models(monkeypatch):
    fake_manager = MagicMock()
    fake_manager.list_available_models.return_value = ["a", "b"]
    monkeypatch.setattr(workflows.OllamaModelManager, "create", lambda *_: fake_manager)
    assert workflows.list_existing_ollama_models("http://localhost:11434") == ["a", "b"]


def test_list_running_models(monkeypatch):
    fake_manager = MagicMock()
    fake_manager.list_running_models.return_value = ["a"]
    monkeypatch.setattr(workflows.OllamaModelManager, "create", lambda *_: fake_manager)
    assert workflows.list_running_ollama_models("http://localhost:11434") == ["a"]


def test_warmup_existing_model_normalizes_before_warmup(monkeypatch):
    fake_manager = MagicMock()
    fake_manager.normalize_model_name.return_value = "foo:1"
    fake_manager.warmup_model.return_value = "foo:1"
    monkeypatch.setattr(workflows.OllamaModelManager, "create", lambda *_: fake_manager)

    out = workflows.warmup_existing_ollama_model("http://localhost:11434", "ollama_chat/foo:1")

    assert out == "foo:1"
    fake_manager.normalize_model_name.assert_called_once_with("ollama_chat/foo:1")
    fake_manager.warmup_model.assert_called_once()


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

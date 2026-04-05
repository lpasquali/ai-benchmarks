from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import rune_bench.api_backend as api_backend
import rune_bench.workflows as workflows
from rune_bench.common.models import ModelSelector
from rune_bench.backends.ollama import OllamaClient
from rune_bench.backends.ollama import OllamaModelManager


def test_model_selector_select_and_list_and_error():
    selector = ModelSelector([
        {"name": "big", "vram_mb": 10000},
        {"name": "small", "vram_mb": 1000},
    ])

    chosen = selector.select(5000)
    assert chosen.name == "small"
    assert selector.list_models()[0].required_disk_gb > 0

    with pytest.raises(RuntimeError, match="No configured model fits"):
        selector.select(1)


def test_ollama_client_make_request_and_helpers(monkeypatch):
    captured = {}

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"models": [{"name": "b"}, {"name": "a"}]}'

    def fake_urlopen(request, timeout, context=None):
        captured["timeout"] = timeout
        captured["method"] = request.get_method()
        return DummyResponse()

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", fake_urlopen)

    client = OllamaClient("localhost:11434")
    assert client.get_available_models() == ["a", "b"]
    assert captured["method"] == "GET"


def test_ollama_client_running_load_unload_and_errors(monkeypatch):
    client = OllamaClient("http://example:11434")
    calls = []

    def fake_make_request(endpoint, *, method, payload, action):
        calls.append((endpoint, method, payload, action))
        if endpoint == "/api/ps":
            return {"models": [{"name": "a"}, {"name": "b"}, {"bad": True}]}
        return {"ok": True}

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    assert client.get_running_models() == {"a", "b"}
    client.load_model("m1", keep_alive="5m")
    client.unload_model("m2")
    assert calls[1][2]["keep_alive"] == "5m"
    assert calls[2][2]["keep_alive"] == 0

    monkeypatch.setattr(client, "_make_request", lambda *args, **kwargs: {"models": "bad"})
    with pytest.raises(RuntimeError, match="unexpected /api/ps payload"):
        client.get_running_models()

    with pytest.raises(RuntimeError, match="unexpected /api/tags payload"):
        client.get_available_models()


def test_ollama_client_make_request_error_paths(monkeypatch):
    from urllib.error import HTTPError, URLError

    client = OllamaClient("http://example:11434")

    class FakeHTTPError(HTTPError):
        def __init__(self):
            super().__init__("http://example", 500, "boom", hdrs=None, fp=None)

        def read(self):
            return b"server exploded"

    def raise_http(*_args, **_kwargs):
        raise FakeHTTPError()

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", raise_http)
    with pytest.raises(RuntimeError, match="server exploded"):
        client._make_request("/api/tags", method="GET", payload=None, action="act")

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("nope")))
    with pytest.raises(RuntimeError, match="nope"):
        client._make_request("/api/tags", method="GET", payload=None, action="act")

    class BadJsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_args, **_kwargs: BadJsonResponse())
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        client._make_request("/api/tags", method="GET", payload=None, action="act")

    class ListResponse(BadJsonResponse):
        def read(self):
            return b"[]"

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_args, **_kwargs: ListResponse())
    with pytest.raises(RuntimeError, match="Unexpected JSON payload"):
        client._make_request("/api/tags", method="GET", payload=None, action="act")


def test_ollama_model_manager_helpers(monkeypatch):
    fake_client = MagicMock()
    fake_client.base_url = "http://localhost:11434"
    fake_client.get_running_models.return_value = {"keep", "drop"}
    manager = OllamaModelManager(client=fake_client)

    assert manager.list_running_models() == ["drop", "keep"]
    manager._unload_conflicting_models("keep")
    fake_client.unload_model.assert_called_once_with("drop")

    fake_client.reset_mock()
    fake_client.get_running_models.side_effect = [{"target"}]
    assert manager.warmup_model("target", unload_others=False) == "target"
    fake_client.load_model.assert_called_once_with("target", keep_alive="30m")

    created = OllamaModelManager.create("http://example:11434")
    assert isinstance(created.client, OllamaClient)


@dataclass
class _SelectedModel:
    name: str
    vram_mb: int
    required_disk_gb: int


def test_workflow_normalizers_and_stop(monkeypatch):
    monkeypatch.setattr(workflows, "OllamaClient", lambda url: type("C", (), {"base_url": f"normalized:{url}"})())
    assert workflows.normalize_ollama_url("x") == "normalized:x"

    fake_manager = MagicMock()
    fake_manager.normalize_model_name.return_value = "plain-model"
    monkeypatch.setattr(workflows.OllamaModelManager, "create", lambda *_: fake_manager)
    assert workflows.normalize_ollama_model_for_api("ollama/model") == "plain-model"

    fake_instance_manager = MagicMock()
    fake_instance_manager.destroy_instance_and_related_storage.return_value = "done"
    monkeypatch.setattr(workflows, "InstanceManager", lambda sdk: fake_instance_manager)
    assert workflows.stop_vastai_instance(MagicMock(), 7) == "done"


def test_provision_vastai_ollama_reuses_running_instance(monkeypatch):
    sdk = MagicMock()
    fake_manager = MagicMock()
    reusable = {"id": 9, "gpu_total_ram": 32000, "ask_contract_id": 44}
    fake_manager.find_reusable_running_instance.return_value = reusable
    fake_manager.pull_model = MagicMock()

    class FakeInstanceManager:
        def __init__(self, _sdk):
            self._inner = fake_manager

        def find_reusable_running_instance(self, **kwargs):
            return self._inner.find_reusable_running_instance(**kwargs)

        def create(self, *args, **kwargs):
            return self._inner.create(*args, **kwargs)

        def wait_until_running(self, *args, **kwargs):
            return self._inner.wait_until_running(*args, **kwargs)

        def pull_model(self, *args, **kwargs):
            return self._inner.pull_model(*args, **kwargs)

        @staticmethod
        def build_connection_details(contract_id, _info):
            return workflows.ConnectionDetails(
                contract_id=contract_id,
                status="running",
                ssh_host=None,
                ssh_port=None,
                machine_id=None,
                service_urls=[{"name": "ollama", "direct": "http://x:11434", "proxy": None}],
            )

    monkeypatch.setattr(workflows, "InstanceManager", FakeInstanceManager)
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _vram: _SelectedModel("foo:1", 32000, 50))
    monkeypatch.setattr(workflows, "list_existing_ollama_models", lambda _url: ["foo:1"])
    monkeypatch.setattr(workflows, "list_running_ollama_models", lambda _url: ["foo:1"])
    monkeypatch.setattr(workflows, "warmup_existing_ollama_model", lambda *_args, **_kwargs: "foo:1")
    monkeypatch.setattr(workflows, "normalize_ollama_model_for_api", lambda model: model)

    result = workflows.provision_vastai_ollama(
        sdk,
        template_hash="tpl",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        confirm_create=lambda: True,
    )

    assert result.reused_existing_instance is True
    assert result.template_env == "<reused-running-instance>"
    assert result.ollama_url == "http://x:11434"
    fake_manager.pull_model.assert_not_called()


def test_provision_vastai_ollama_creates_new_instance_and_warms(monkeypatch):
    sdk = MagicMock()
    fake_manager = MagicMock()
    fake_manager.find_reusable_running_instance.return_value = None
    fake_manager.create.return_value = 12
    fake_manager.wait_until_running.return_value = {"id": 12}

    class FakeInstanceManager:
        def __init__(self, _sdk):
            self._inner = fake_manager

        def find_reusable_running_instance(self, **kwargs):
            return self._inner.find_reusable_running_instance(**kwargs)

        def create(self, *args, **kwargs):
            return self._inner.create(*args, **kwargs)

        def wait_until_running(self, *args, **kwargs):
            return self._inner.wait_until_running(*args, **kwargs)

        def pull_model(self, *args, **kwargs):
            return self._inner.pull_model(*args, **kwargs)

        @staticmethod
        def build_connection_details(contract_id, _info):
            return workflows.ConnectionDetails(
                contract_id=contract_id,
                status="running",
                ssh_host=None,
                ssh_port=None,
                machine_id=None,
                service_urls=[{"name": "ollama", "direct": "http://x:11434", "proxy": None}],
            )

    monkeypatch.setattr(workflows, "InstanceManager", FakeInstanceManager)

    monkeypatch.setattr(workflows.OfferFinder, "find_best", lambda self, **_: type("Offer", (), {"offer_id": 5, "total_vram_mb": 24000})())
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _vram: _SelectedModel("foo:1", 20000, 60))
    monkeypatch.setattr(workflows.TemplateLoader, "load", lambda self, _hash: type("Tpl", (), {"env": "ENV=1", "image": "img"})())
    monkeypatch.setattr(workflows, "list_existing_ollama_models", lambda _url: [])
    monkeypatch.setattr(workflows, "list_running_ollama_models", lambda _url: [])
    warmed = []
    monkeypatch.setattr(workflows, "warmup_existing_ollama_model", lambda *_args, **_kwargs: warmed.append(True) or "foo:1")
    monkeypatch.setattr(workflows, "normalize_ollama_model_for_api", lambda model: model)

    result = workflows.provision_vastai_ollama(
        sdk,
        template_hash="tpl",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        confirm_create=lambda: True,
    )

    assert result.offer_id == 5
    assert result.contract_id == 12
    assert warmed == [True]
    fake_manager.pull_model.assert_called_once_with(12, "foo:1", ollama_url="http://x:11434")


def test_provision_vastai_ollama_abort_and_pull_warning(monkeypatch):
    sdk = MagicMock()
    fake_manager = MagicMock()
    fake_manager.find_reusable_running_instance.return_value = None

    class FakeInstanceManager:
        def __init__(self, _sdk):
            self._inner = fake_manager

        def find_reusable_running_instance(self, **kwargs):
            return self._inner.find_reusable_running_instance(**kwargs)

        def create(self, *args, **kwargs):
            return self._inner.create(*args, **kwargs)

        def wait_until_running(self, *args, **kwargs):
            return self._inner.wait_until_running(*args, **kwargs)

        def pull_model(self, *args, **kwargs):
            return self._inner.pull_model(*args, **kwargs)

        @staticmethod
        def build_connection_details(contract_id, _info):
            return workflows.ConnectionDetails(
                contract_id=contract_id,
                status="running",
                ssh_host=None,
                ssh_port=None,
                machine_id=None,
                service_urls=[],
            )

    monkeypatch.setattr(workflows, "InstanceManager", FakeInstanceManager)
    monkeypatch.setattr(workflows.OfferFinder, "find_best", lambda self, **_: type("Offer", (), {"offer_id": 5, "total_vram_mb": 24000})())
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _vram: _SelectedModel("foo:1", 20000, 60))
    monkeypatch.setattr(workflows.TemplateLoader, "load", lambda self, _hash: type("Tpl", (), {"env": "ENV=1", "image": "img"})())

    with pytest.raises(workflows.UserAbortedError):
        workflows.provision_vastai_ollama(
            sdk,
            template_hash="tpl",
            min_dph=1,
            max_dph=2,
            reliability=0.9,
            confirm_create=lambda: False,
        )

    fake_manager.create.return_value = 12
    fake_manager.wait_until_running.return_value = {"id": 12}

    result = workflows.provision_vastai_ollama(
        sdk,
        template_hash="tpl",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        confirm_create=lambda: True,
    )
    assert "Could not determine Ollama URL" in str(result.pull_warning)


def test_api_backend_functions(monkeypatch, tmp_path):
    monkeypatch.setattr(api_backend, "ModelSelector", lambda: type("S", (), {"list_models": lambda self: [_SelectedModel("m1", 1, 2)]})())
    assert api_backend.list_vastai_models() == [{"name": "m1", "vram_mb": 1, "required_disk_gb": 2}]

    monkeypatch.setattr(api_backend, "use_existing_ollama_server", lambda url, model_name: type("Srv", (), {"url": f"norm:{url}"})())
    monkeypatch.setattr(api_backend, "list_existing_ollama_models", lambda _url: ["a"])
    monkeypatch.setattr(api_backend, "list_running_ollama_models", lambda _url: ["a"])
    payload = api_backend.list_ollama_models("raw")
    assert payload["ollama_url"] == "norm:raw"

    from rune_bench.resources.base import ProvisioningResult

    req = api_backend.RunOllamaInstanceRequest(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="u")
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_ollama_instance",
        lambda r: type("P", (), {
            "provision": lambda self, r=r: ProvisioningResult(ollama_url=f"norm:{r.ollama_url}"),
            "teardown": lambda self, res: None,
        })(),
    )
    assert api_backend.run_ollama_instance(req) == {"mode": "existing", "ollama_url": "norm:u"}

    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_ollama_instance",
        lambda r: type("P", (), {
            "provision": lambda self: ProvisioningResult(ollama_url="http://x", model="m", provider_handle=3),
            "teardown": lambda self, res: None,
        })(),
    )
    req_v = api_backend.RunOllamaInstanceRequest(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None)
    assert api_backend.run_ollama_instance(req_v)["mode"] == "vastai"

    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    warmed = []
    monkeypatch.setattr(api_backend, "warmup_existing_ollama_model", lambda *_args, **_kwargs: warmed.append(True) or "m")
    monkeypatch.setattr(api_backend, "_make_agent_runner", lambda path: type("R", (), {"ask": lambda self, **_: "answer"})())
    areq = api_backend.RunAgenticAgentRequest(question="q", model="m", ollama_url="http://x", ollama_warmup=True, ollama_warmup_timeout=1, kubeconfig=str(kubeconfig))
    assert api_backend.run_agentic_agent(areq) == {"answer": "answer"}
    assert warmed == [True]

    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda r: type("P", (), {
            "provision": lambda self, r=r: ProvisioningResult(ollama_url="http://existing", model=r.model),
            "teardown": lambda self, res: None,
        })(),
    )
    breq = api_backend.RunBenchmarkRequest(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="u", question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=str(kubeconfig), vastai_stop_instance=True)
    result = api_backend.run_benchmark(breq)
    assert result["answer"] == "answer"
    assert result["ollama_url"] == "http://existing"

    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda r: type("P", (), {
            "provision": lambda self: ProvisioningResult(ollama_url=None, model="m", provider_handle=7),
            "teardown": lambda self, res: None,
        })(),
    )
    with pytest.raises(RuntimeError, match="Could not determine Ollama URL"):
        api_backend.run_benchmark(api_backend.RunBenchmarkRequest(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=str(kubeconfig), vastai_stop_instance=False))

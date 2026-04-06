from unittest.mock import MagicMock

import pytest
import typer
from rich.console import Console

import rune
from rune_bench.backends.ollama import OllamaModelCapabilities
from rune_bench.workflows import ExistingOllamaServer, VastAIProvisioningResult
from rune_bench.resources.vastai import ConnectionDetails, TeardownResult


class DummyClient:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def wait_for_job(self, *args, **kwargs):
        return self.payload


def _details():
    return ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host="1.2.3.4",
        ssh_port=22,
        machine_id="m1",
        service_urls=[{"name": "ollama", "direct": "http://1.2.3.4:11434", "proxy": "https://server-m1.vast.ai:11434"}],
    )


def _result(**kwargs):
    base = dict(
        offer_id=1,
        total_vram_mb=24000,
        model_name="llama3.1:8b",
        model_vram_mb=8000,
        required_disk_gb=41,
        template_env="ENV=1",
        contract_id=7,
        details=_details(),
        ollama_url="http://1.2.3.4:11434",
        reused_existing_instance=False,
        pull_warning=None,
    )
    base.update(kwargs)
    return VastAIProvisioningResult(**base)


def test_main_and_basic_helpers(monkeypatch):
    with pytest.raises(typer.BadParameter):
        rune.main(backend="bad", api_base_url="http://x", api_token="", api_tenant="default", debug=False)  # nosec  # test credentials

    rune.main(backend="http", api_base_url="http://api", api_token="tok", api_tenant="tenant-a", debug=False)  # nosec  # test credentials
    client = rune._http_client()
    assert client.base_url == "http://api"
    assert client.api_token == "tok"  # nosec  # test credentials
    assert client.tenant_id == "tenant-a"

    called = []
    monkeypatch.setattr(rune, "set_debug", lambda value: called.append(value))
    rune._enable_debug_if_requested(True)
    assert called == [True]

    test_console = Console(record=True)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(test_console, "input", lambda _prompt: "yes")
    assert rune._confirm_instance_creation() is True

    with pytest.raises(typer.Exit) as exc:
        rune._print_error_and_exit("boom", code=2)
    assert exc.value.exit_code == 2


def test_fetch_and_warmup_helpers(monkeypatch):
    fake_manager = MagicMock()
    fake_manager.normalize_model_name.return_value = "norm"
    fake_client = MagicMock()
    fake_client.get_model_capabilities.return_value = OllamaModelCapabilities("norm", 100, 20)
    monkeypatch.setattr(rune.OllamaModelManager, "create", lambda *_: fake_manager)
    monkeypatch.setattr(rune, "OllamaClient", lambda *_: fake_client)
    caps = rune._fetch_model_capabilities("http://x", "m")
    assert caps.context_window == 100

    fake_client.get_model_capabilities.side_effect = RuntimeError("nope")
    assert rune._fetch_model_capabilities("http://x", "m") is None

    test_console = Console(record=True)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "warmup_existing_ollama_model", lambda *_args, **_kwargs: "m")
    rune._warmup_ollama_model(ollama_url="http://x", model_name="m", timeout_seconds=1)
    assert "Ollama model ready" in test_console.export_text()

    monkeypatch.setattr(rune, "warmup_existing_ollama_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        rune._warmup_ollama_model(ollama_url="http://x", model_name="m", timeout_seconds=1)


def test_run_vastai_provisioning_helper(monkeypatch):
    test_console = Console(record=True)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "_vastai_sdk", lambda: MagicMock())
    monkeypatch.setattr(rune, "provision_vastai_ollama", lambda *_args, **_kwargs: _result())
    assert rune._run_vastai_provisioning(template_hash="t", min_dph=1, max_dph=2, reliability=0.9).contract_id == 7

    monkeypatch.setattr(rune, "provision_vastai_ollama", lambda *_args, **_kwargs: (_ for _ in ()).throw(rune.UserAbortedError("stop")))
    with pytest.raises(typer.Exit) as exc:
        rune._run_vastai_provisioning(template_hash="t", min_dph=1, max_dph=2, reliability=0.9)
    assert exc.value.exit_code == 0

    monkeypatch.setattr(rune, "provision_vastai_ollama", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        rune._run_vastai_provisioning(template_hash="t", min_dph=1, max_dph=2, reliability=0.9)


def test_run_http_job_with_progress(monkeypatch):
    client = DummyClient(payload={"status": "succeeded", "result": {"answer": "ok"}})
    payload = rune._run_http_job_with_progress(
        submit_description="submit",
        wait_description="wait",
        submit_job=lambda: "job-1",
        client=client,
        timeout_seconds=1,
        poll_interval_seconds=0,
    )
    assert payload["result"]["answer"] == "ok"


def test_run_ollama_instance_paths(monkeypatch):
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_args, **_kwargs: ExistingOllamaServer(url="http://x", model_name="m"))
    rune.run_ollama_instance(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="u", debug=False, idempotency_key=None)
    assert "Existing Ollama Server" in test_console.export_text()

    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        rune.run_ollama_instance(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="u", debug=False, idempotency_key=None)

    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_kwargs: _result())
    rune.run_ollama_instance(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, debug=False, idempotency_key=None)
    assert "Provisioned contract" in test_console.export_text()

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_ollama_instance_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {"mode": "vastai", "contract_id": 2, "ollama_url": "http://x", "model_name": "m"}}, "get_cost_estimate": lambda self, *_a, **_k: {"projected_cost_usd": 1.0, "cost_driver": "vastai", "resource_impact": "medium", "local_energy_kwh": 0.0, "confidence_score": 1.0, "warning": None}})())
    rune.run_ollama_instance(vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, debug=False, idempotency_key="id1")
    assert "Selected model" in test_console.export_text()

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_ollama_instance_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {}}})())
    with pytest.raises(typer.Exit):
        rune.run_ollama_instance(vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="u", debug=False, idempotency_key=None)


def test_list_commands(monkeypatch):
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune.ModelSelector, "list_models", lambda self: [type("M", (), {"name": "m", "vram_mb": 1, "required_disk_gb": 2})()])
    rune.vastai_list_models()
    assert "Configured Vast.ai Models" in test_console.export_text()

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"get_vastai_models": lambda self: [{"name": "m", "vram_mb": 1, "required_disk_gb": 2}]})())
    rune.vastai_list_models()

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"get_vastai_models": lambda self: (_ for _ in ()).throw(RuntimeError("bad"))})())
    with pytest.raises(typer.Exit):
        rune.vastai_list_models()

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"get_ollama_models": lambda self, url: {"ollama_url": url, "models": ["m"], "running_models": ["m"]}})())
    rune.ollama_list_models(debug=False, ollama_url="http://x")
    assert "Existing Ollama Models" in test_console.export_text()

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"get_ollama_models": lambda self, url: (_ for _ in ()).throw(RuntimeError("bad"))})())
    with pytest.raises(typer.Exit):
        rune.ollama_list_models(debug=False, ollama_url="http://x")

    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_args, **_kwargs: ExistingOllamaServer(url="http://x", model_name="m"))
    monkeypatch.setattr(rune, "list_existing_ollama_models", lambda _url: ["m"])
    monkeypatch.setattr(rune, "list_running_ollama_models", lambda _url: ["m"])
    rune.ollama_list_models(debug=False, ollama_url="http://x")

    monkeypatch.setattr(rune, "list_existing_ollama_models", lambda _url: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        rune.ollama_list_models(debug=False, ollama_url="http://x")


def test_run_agentic_agent_paths(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    test_console = Console(record=True, width=200)
    monkeypatch.setattr(rune, "console", test_console)

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_agentic_agent_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {"answer": "http-answer"}}})())
    rune.run_agentic_agent(debug=False, question="q", model="m", ollama_url=None, ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key="id1")
    assert "http-answer" in test_console.export_text()

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_agentic_agent_job": lambda self, payload, idempotency_key=None: (_ for _ in ()).throw(RuntimeError("bad-http")), "wait_for_job": lambda self, *args, **kwargs: {}})())
    with pytest.raises(typer.Exit):
        rune.run_agentic_agent(debug=False, question="q", model="m", ollama_url=None, ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key=None)

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_agentic_agent_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {}}})())
    with pytest.raises(typer.Exit):
        rune.run_agentic_agent(debug=False, question="q", model="m", ollama_url=None, ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key=None)

    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    warmups = []
    monkeypatch.setattr(rune, "_warmup_ollama_model", lambda **_kwargs: warmups.append(True))
    monkeypatch.setattr(rune, "HolmesRunner", lambda _path: type("R", (), {"ask": lambda self, **_: "local-answer"})())
    rune.run_agentic_agent(debug=False, question="q", model="m", ollama_url="http://x", ollama_warmup=True, ollama_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key=None)
    assert warmups == [True]
    assert "local-answer" in test_console.export_text()

    monkeypatch.setattr(rune, "HolmesRunner", lambda _path: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(typer.Exit):
        rune.run_agentic_agent(debug=False, question="q", model="m", ollama_url=None, ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, idempotency_key=None)


def test_run_benchmark_paths(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text("apiVersion: v1\n")
    test_console = Console(record=True, width=220)
    monkeypatch.setattr(rune, "console", test_console)

    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_benchmark_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {"answer": "bench-http"}}})())
    rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key="i")
    assert "bench-http" in test_console.export_text()

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_benchmark_job": lambda self, payload, idempotency_key=None: (_ for _ in ()).throw(RuntimeError("bad-http")), "wait_for_job": lambda self, *args, **kwargs: {}})())
    with pytest.raises(typer.Exit):
        rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)

    monkeypatch.setattr(rune, "_http_client", lambda: type("C", (), {"submit_benchmark_job": lambda self, payload, idempotency_key=None: "job-1", "wait_for_job": lambda self, *args, **kwargs: {"result": {}}})())
    with pytest.raises(typer.Exit):
        rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)

    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad-server")))
    with pytest.raises(typer.Exit):
        rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="http://x", question="q", model="m", ollama_warmup=True, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)

    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_args, **_kwargs: ExistingOllamaServer(url="http://x", model_name="m"))
    monkeypatch.setattr(rune, "_fetch_model_capabilities", lambda *_args, **_kwargs: OllamaModelCapabilities("m", 100, 20))
    monkeypatch.setattr(rune, "_warmup_ollama_model", lambda **_kwargs: None)
    monkeypatch.setattr(rune, "HolmesRunner", lambda _path: type("R", (), {"ask": lambda self, **_: "bench-local"})())
    rune.run_benchmark(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="http://x", question="q", model="m", ollama_warmup=True, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)
    assert "bench-local" in test_console.export_text()

    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_kwargs: _result())
    monkeypatch.setattr(rune, "_vastai_sdk", lambda: MagicMock())
    monkeypatch.setattr(rune, "HolmesRunner", lambda _path: type("R", (), {"ask": lambda self, **_: "bench-vastai"})())
    monkeypatch.setattr(rune, "stop_vastai_instance", lambda *_args, **_kwargs: TeardownResult(contract_id=7, destroyed_instance=True, destroyed_volume_ids=[], verification_ok=False, verification_message="warn"))
    rune.run_benchmark(debug=False, vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)
    assert "bench-vastai" in test_console.export_text()

    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_kwargs: _result(ollama_url=None))
    monkeypatch.setattr(rune, "stop_vastai_instance", lambda *_args, **_kwargs: TeardownResult(contract_id=7, destroyed_instance=True, destroyed_volume_ids=["v1"], verification_ok=True, verification_message="ok"))
    with pytest.raises(typer.Exit):
        rune.run_benchmark(debug=False, vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)

    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_kwargs: _result())
    monkeypatch.setattr(rune, "HolmesRunner", lambda _path: (_ for _ in ()).throw(RuntimeError("agent failed")))
    monkeypatch.setattr(rune, "stop_vastai_instance", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop failed")))
    with pytest.raises(typer.Exit):
        rune.run_benchmark(debug=False, vastai=True, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url=None, question="q", model="m", ollama_warmup=False, ollama_warmup_timeout=1, kubeconfig=kubeconfig, vastai_stop_instance=True, idempotency_key=None)

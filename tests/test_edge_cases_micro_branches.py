import runpy
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import rune
import rune.api as rune_api_module
import rune_bench.agents.sre.holmes as holmes_module
import rune_bench.api_backend as api_backend
import rune_bench.api_client as api_client_module
import rune_bench.api_server as api_server
import rune_bench.backends.ollama as ollama_models_module
import rune_bench.workflows as workflows
from rune_bench.agents.sre.holmes import HolmesRunner
from rune_bench.api_client import RuneApiClient
from rune_bench.common import normalize_url
from rune_bench.backends.ollama import OllamaClient, OllamaModelCapabilities, OllamaModelManager
from rune_bench.resources.vastai import ConnectionDetails, InstanceManager, TeardownResult


def test_final_coverage_micro_branches(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "kube"
    kubeconfig.write_text("apiVersion: v1\n")
    runner = HolmesRunner(kubeconfig)

    # holmes.py: _configure_ollama_model_limits early return (no ollama_url)
    runner._configure_ollama_model_limits(model="m", ollama_url=None)

    # holmes.py: _set_model_limit_override import/setattr failure debug branch
    class BadLlmModule:
        def __setattr__(self, _name, _value):
            raise RuntimeError("setattr boom")

    monkeypatch.setitem(sys.modules, "holmes", type("H", (), {})())
    monkeypatch.setitem(sys.modules, "holmes.core", type("HC", (), {})())
    monkeypatch.setitem(sys.modules, "holmes.core.llm", BadLlmModule())
    monkeypatch.delenv("OVERRIDE_MAX_OUTPUT_TOKEN", raising=False)
    runner._set_model_limit_override(env_name="OVERRIDE_MAX_OUTPUT_TOKEN", value=42)

    # api_client.py: invalid JSON branch in _request
    client = RuneApiClient("http://api:8080")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{not-json"

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        client._request("GET", "/v1/x")

    # ollama/models.py: warmup polling sleep branch
    calls = {"n": 0}
    sleeps = []

    class FakeClient:
        base_url = "http://x"

        def load_model(self, *_args, **_kwargs):
            return None

        def get_running_models(self):
            calls["n"] += 1
            return set() if calls["n"] == 1 else {"target"}

    monkeypatch.setattr(ollama_models_module.time, "sleep", lambda s: sleeps.append(s))
    OllamaModelManager(client=FakeClient()).warmup_model(
        "target",
        timeout_seconds=2,
        poll_interval_seconds=0.01,
        unload_others=False,
    )
    assert sleeps == [0.01]

    # vastai/instance.py: invalid mapping continue branch in build_connection_details
    details = InstanceManager.build_connection_details(
        contract_id=1,
        instance_info={"ports": {"svc": "not-a-list"}, "state": "running"},
    )
    assert details.service_urls == []

    # vastai/instance.py: _fetch_instance exception branch
    manager = object.__new__(InstanceManager)
    manager._sdk = type("Sdk", (), {"show_instances": lambda self, raw=True: (_ for _ in ()).throw(RuntimeError("boom"))})()
    assert manager._fetch_instance(1) is None

    # vastai/instance.py: _wait_until_instance_absent loop sleep branch
    wait_manager = object.__new__(InstanceManager)
    states = iter([{"id": 1}, None])
    monkeypatch.setattr(wait_manager, "_fetch_instance", lambda _cid: next(states))
    monotonic_values = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.monotonic", lambda: next(monotonic_values))
    slept_wait = []
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.sleep", lambda s: slept_wait.append(s))
    assert wait_manager._wait_until_instance_absent(contract_id=1, timeout_seconds=1) is True
    assert slept_wait == [5]


def test_rune_remaining_branches(monkeypatch, tmp_path):
    test_console = rune.Console(record=True, width=220)
    monkeypatch.setattr(rune, "console", test_console)

    # line for pull_warning rendering
    details = ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host=None,
        ssh_port=None,
        machine_id=None,
        service_urls=[],
    )
    result = rune.VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=24000,
        model_name="m",
        model_vram_mb=1000,
        required_disk_gb=40,
        template_env="ENV=1",
        contract_id=1,
        details=details,
        ollama_url=None,
        pull_warning="warn",
    )
    rune._print_vastai_result(result)

    # no-models branch
    rune._print_ollama_models("http://x", [], set())

    # on_poll callback branch in provisioning helper
    def fake_provision(*_args, **kwargs):
        kwargs["on_poll"]("running")
        return result

    monkeypatch.setattr(rune, "provision_vastai_ollama", fake_provision)
    monkeypatch.setattr(rune, "_vastai_sdk", lambda: MagicMock())
    rune._run_vastai_provisioning(template_hash="t", min_dph=1, max_dph=2, reliability=0.9)

    # http run-ollama-instance RuntimeError branch
    monkeypatch.setattr(rune, "BACKEND_MODE", "http")
    monkeypatch.setattr(rune, "_run_http_job_with_progress", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(rune.typer.Exit):
        rune.run_ollama_instance(debug=False, vastai=False, template_hash="t", min_dph=1, max_dph=2, reliability=0.9, ollama_url="http://x", idempotency_key=None)

    # run-benchmark local existing server failure branch
    monkeypatch.setattr(rune, "BACKEND_MODE", "local")
    monkeypatch.setattr(rune, "use_existing_ollama_server", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad-existing")))
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    with pytest.raises(rune.typer.Exit):
        rune.run_benchmark(
            debug=False,
            vastai=False,
            template_hash="t",
            min_dph=1,
            max_dph=2,
            reliability=0.9,
            ollama_url="http://x",
            question="q",
            model="m",
            ollama_warmup=False,
            ollama_warmup_timeout=1,
            kubeconfig=kubeconfig,
            vastai_stop_instance=True,
            idempotency_key=None,
        )

    # finally branch with destroyed volumes print
    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_k: rune.VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=24000,
        model_name="m",
        model_vram_mb=1000,
        required_disk_gb=40,
        template_env="ENV=1",
        contract_id=77,
        details=details,
        ollama_url="http://x",
        pull_warning=None,
    ))
    monkeypatch.setattr(rune, "_warmup_ollama_model", lambda **_k: None)
    monkeypatch.setattr(rune, "HolmesRunner", lambda _k: type("R", (), {"ask": lambda self, **_kk: "ok"})())
    monkeypatch.setattr(
        rune,
        "stop_vastai_instance",
        lambda *_a, **_k: TeardownResult(
            contract_id=77,
            destroyed_instance=True,
            destroyed_volume_ids=["vol-a"],
            verification_ok=True,
            verification_message="ok",
        ),
    )
    rune.run_benchmark(
        debug=False,
        vastai=True,
        template_hash="t",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        ollama_url=None,
        question="q",
        model="m",
        ollama_warmup=False,
        ollama_warmup_timeout=1,
        kubeconfig=kubeconfig,
        vastai_stop_instance=True,
        idempotency_key=None,
    )

    # branch where stop_vastai_instance itself fails during no-ollama-url cleanup path
    monkeypatch.setattr(rune, "_run_vastai_provisioning", lambda **_k: rune.VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=24000,
        model_name="m",
        model_vram_mb=1000,
        required_disk_gb=40,
        template_env="ENV=1",
        contract_id=88,
        details=details,
        ollama_url=None,
        pull_warning=None,
    ))
    monkeypatch.setattr(rune, "stop_vastai_instance", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop-failed")))
    with pytest.raises(rune.typer.Exit):
        rune.run_benchmark(
            debug=False,
            vastai=True,
            template_hash="t",
            min_dph=1,
            max_dph=2,
            reliability=0.9,
            ollama_url=None,
            question="q",
            model="m",
            ollama_warmup=False,
            ollama_warmup_timeout=1,
            kubeconfig=kubeconfig,
            vastai_stop_instance=True,
            idempotency_key=None,
        )


def test_rune_main_guard_executes(monkeypatch):
    rune_path = Path(__file__).resolve().parents[1] / "rune" / "__main__.py"
    monkeypatch.setattr(sys, "argv", [str(rune_path), "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(rune_path), run_name="__main__")


def test_rune_init_main_guard_executes(monkeypatch):
    init_path = Path(__file__).resolve().parents[1] / "rune" / "__init__.py"
    monkeypatch.setattr(sys, "argv", [str(init_path), "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(init_path), run_name="__main__")


def test_run_ollama_instance_http_vastai_result_branch(monkeypatch):
    test_console = rune.Console(record=True, width=220)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "http")

    fake_client = type("C", (), {"submit_ollama_instance_job": lambda self, *_a, **_k: "job-1"})()
    monkeypatch.setattr(rune, "_http_client", lambda: fake_client)
    monkeypatch.setattr(
        rune,
        "_run_http_job_with_progress",
        lambda **_k: {
            "result": {
                "mode": "vastai",
                "contract_id": 123,
                "ollama_url": "http://x:11434",
                "model_name": "llama3.1:8b",
            }
        },
    )

    rune.run_ollama_instance(
        debug=False,
        vastai=True,
        template_hash="t",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        ollama_url=None,
        idempotency_key=None,
    )


def test_run_ollama_instance_http_existing_result_branch(monkeypatch):
    test_console = rune.Console(record=True, width=220)
    monkeypatch.setattr(rune, "console", test_console)
    monkeypatch.setattr(rune, "BACKEND_MODE", "http")

    fake_client = type("C", (), {"submit_ollama_instance_job": lambda self, *_a, **_k: "job-1"})()
    monkeypatch.setattr(rune, "_http_client", lambda: fake_client)
    monkeypatch.setattr(
        rune,
        "_run_http_job_with_progress",
        lambda **_k: {
            "result": {
                "mode": "existing",
                "ollama_url": "http://x:11434",
            }
        },
    )

    captured = {}
    monkeypatch.setattr(rune, "_print_existing_ollama", lambda server: captured.setdefault("url", server.url))

    rune.run_ollama_instance(
        debug=False,
        vastai=False,
        template_hash="t",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        ollama_url="http://fallback:11434",
        idempotency_key=None,
    )

    assert captured["url"] == "http://x:11434"


def test_rune_api_entrypoint_main_and_guard(monkeypatch):
    calls = {}

    class FakeApp:
        def serve(self, *, host, port):
            calls["host"] = host
            calls["port"] = port

    class FakeRuneApiApplication:
        @classmethod
        def from_env(cls):
            return FakeApp()

    monkeypatch.setattr("rune_bench.api_server.RuneApiApplication", FakeRuneApiApplication)
    monkeypatch.setattr(rune_api_module, "RuneApiApplication", FakeRuneApiApplication)
    monkeypatch.setenv("RUNE_API_HOST", "127.0.0.1")
    monkeypatch.setenv("RUNE_API_PORT", "18080")

    rune_api_module.main()
    assert calls == {"host": "127.0.0.1", "port": 18080}

    api_path = Path(__file__).resolve().parents[1] / "rune" / "api.py"
    monkeypatch.setattr(sys, "argv", [str(api_path)])
    runpy.run_path(str(api_path), run_name="__main__")


def test_holmes_and_ollama_remaining_branches(monkeypatch, tmp_path):
    kubeconfig = tmp_path / "kube"
    kubeconfig.write_text("apiVersion: v1\n")
    runner = HolmesRunner(kubeconfig)

    # successful configure path and failure path
    monkeypatch.setattr(holmes_module.OllamaModelManager, "create", lambda *_: type("M", (), {"normalize_model_name": lambda self, m: "norm"})())
    monkeypatch.setattr(holmes_module, "OllamaClient", lambda *_: type("C", (), {"get_model_capabilities": lambda self, _m: OllamaModelCapabilities("norm", 10, 2)})())
    runner._configure_ollama_model_limits(model="m", ollama_url="http://x")

    # ensure holmes.core.llm setattr path is covered
    fake_holmes_pkg = type("Pkg", (), {})()
    fake_holmes_core = type("Core", (), {})()
    fake_holmes_llm = type("LLM", (), {})()
    monkeypatch.setitem(sys.modules, "holmes", fake_holmes_pkg)
    monkeypatch.setitem(sys.modules, "holmes.core", fake_holmes_core)
    monkeypatch.setitem(sys.modules, "holmes.core.llm", fake_holmes_llm)
    runner._set_model_limit_override(env_name="OVERRIDE_MAX_CONTENT_SIZE", value=11)

    monkeypatch.setattr(holmes_module, "OllamaClient", lambda *_: type("C", (), {"get_model_capabilities": lambda self, _m: (_ for _ in ()).throw(RuntimeError("bad"))})())
    runner._configure_ollama_model_limits(model="m", ollama_url="http://x")

    # Ollama invalid URL branch
    with pytest.raises(RuntimeError):
        normalize_url("http://", service_name="Ollama")

    client = OllamaClient("http://x:11434")

    class NoDetailHttpError(HTTPError):
        def __init__(self):
            super().__init__("http://x", 500, "bad", hdrs=None, fp=None)

        def read(self):
            return b""

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(NoDetailHttpError()))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client._make_request("/x", method="GET", payload=None, action="act")

    monkeypatch.setattr(client, "_make_request", lambda *_a, **_k: {"model_info": {"context_length": 4096}})
    caps = client.get_model_capabilities("m")
    assert caps.max_output_tokens == 1024

    monkeypatch.setattr(client, "_make_request", lambda *_a, **_k: {"model_info": {"a.context_length": "bad", "b.context_length": 2048}})
    caps = client.get_model_capabilities("m")
    assert caps.context_window == 2048

    monkeypatch.setattr(client, "_make_request", lambda *_a, **_k: {"model_info": {1: 123, "x": 5}})
    client.get_model_capabilities("m")

    fake = type("F", (), {"get_available_models": lambda self: ["m"], "get_running_models": lambda self: {"other"}, "load_model": lambda self, *_a, **_k: None, "unload_model": lambda self, *_a, **_k: None, "base_url": "http://x"})()
    manager = OllamaModelManager(client=fake)
    assert manager.list_available_models() == ["m"]
    manager._unload_conflicting_models("target")

    # warmup sleep path
    calls = {"n": 0}
    fake2 = type(
        "F2",
        (),
        {
            "base_url": "http://x",
            "load_model": lambda self, *_a, **_k: None,
            "get_running_models": lambda self: set() if (calls.__setitem__("n", calls["n"] + 1) or calls["n"] == 1) else {"target"},
            "unload_model": lambda self, *_a, **_k: None,
        },
    )()
    manager2 = OllamaModelManager(client=fake2)
    manager2.warmup_model("target", timeout_seconds=2, poll_interval_seconds=0)


def test_api_backend_server_workflows_instance_remaining(monkeypatch, tmp_path):
    # api_backend line where finally stops vastai contract
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    from rune_bench.resources.base import ProvisioningResult

    stopped = []
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda req: type("P", (), {
            "provision": lambda self: ProvisioningResult(ollama_url="http://x", model="m", provider_handle=99),
            "teardown": lambda self, r: stopped.append(True),
        })(),
    )
    monkeypatch.setattr(api_backend, "_make_agent_runner", lambda _p: type("R", (), {"ask": lambda self, **_k: "ok"})())
    out = api_backend.run_benchmark(
        api_backend.RunBenchmarkRequest(
            vastai=True,
            template_hash="t",
            min_dph=1,
            max_dph=2,
            reliability=0.9,
            ollama_url=None,
            question="q",
            model="m",
            ollama_warmup=False,
            ollama_warmup_timeout=1,
            kubeconfig=str(kubeconfig),
            vastai_stop_instance=True,
        )
    )
    assert out["contract_id"] == 99
    assert stopped == [True]

    # api_backend benchmark warmup branch — patch at provider module level to verify
    # ExistingOllamaProvider.provision() calls warmup when warmup=True
    import rune_bench.resources.existing_ollama_provider as _ep
    from rune_bench.resources.existing_ollama_provider import ExistingOllamaProvider

    warmups = []
    monkeypatch.setattr(_ep, "use_existing_ollama_server", lambda *_a, **_k: type("S", (), {"url": "http://existing"})())
    monkeypatch.setattr(_ep, "warmup_existing_ollama_model", lambda *_a, **_k: warmups.append(True) or "m")
    monkeypatch.setattr(
        api_backend,
        "_make_resource_provider_for_benchmark",
        lambda req: ExistingOllamaProvider(req.ollama_url, model=req.model, warmup=req.ollama_warmup, warmup_timeout=req.ollama_warmup_timeout),
    )
    api_backend.run_benchmark(
        api_backend.RunBenchmarkRequest(
            vastai=False,
            template_hash="t",
            min_dph=1,
            max_dph=2,
            reliability=0.9,
            ollama_url="http://x",
            question="q",
            model="m",
            ollama_warmup=True,
            ollama_warmup_timeout=1,
            kubeconfig=str(kubeconfig),
            vastai_stop_instance=False,
        )
    )
    assert warmups == [True]

    # api_server RuntimeError path in /v1/ollama/models
    app = api_server.RuneApiApplication(
        store=api_server.JobStore(tmp_path / "jobs.db"),
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
    )
    monkeypatch.setattr(api_server, "list_ollama_models", lambda _url: (_ for _ in ()).throw(RuntimeError("bad-ollama")))
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        req = Request(f"http://{host}:{port}/v1/ollama/models?ollama_url=http://x")
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    # workflows: reusable instance with missing id should fall back to new provisioning
    class FallbackManager:
        def __init__(self, _sdk):
            pass

        def find_reusable_running_instance(self, **_k):
            return {"id": None, "gpu_total_ram": 100}

        def create(self, *_a, **_k):
            return 7

        def wait_until_running(self, *_a, **_k):
            return {"id": 7}

        def pull_model(self, *_a, **_k):
            return None

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

    monkeypatch.setattr(workflows, "InstanceManager", FallbackManager)
    monkeypatch.setattr(workflows.OfferFinder, "find_best", lambda self, **_k: type("O", (), {"offer_id": 1, "total_vram_mb": 100})())
    monkeypatch.setattr(workflows.ModelSelector, "select", lambda self, _v: type("M", (), {"name": "m", "vram_mb": 1, "required_disk_gb": 1})())
    monkeypatch.setattr(workflows.TemplateLoader, "load", lambda self, _h: type("T", (), {"env": "E=1", "image": "img"})())
    monkeypatch.setattr(workflows, "list_existing_ollama_models", lambda _u: ["m"])
    monkeypatch.setattr(workflows, "list_running_ollama_models", lambda _u: ["m"])
    monkeypatch.setattr(workflows, "normalize_ollama_model_for_api", lambda m: m)
    res = workflows.provision_vastai_ollama(
        sdk=object(),
        template_hash="t",
        min_dph=1,
        max_dph=2,
        reliability=0.9,
        confirm_create=lambda: True,
    )
    assert res.reused_existing_instance is False

    # instance helper branches
    sdk = type("S", (), {"show_instances": lambda self, raw=True: [{"id": "1", "actual_status": "running"}], "show_volumes": lambda self, raw=True: {"other": 1}})()
    manager = InstanceManager(sdk)
    called = []
    manager.wait_until_running(1, on_poll=lambda status: called.append(status))
    assert called
    assert manager._fetch_instance(1)["id"] == "1"
    manager._fetch_instance(2)

    monkeypatch.setattr(manager, "_fetch_instance", lambda _cid: None)
    assert manager._wait_until_instance_absent(1, timeout_seconds=1) is True
    assert manager._list_volumes_optional() is None

    # reusable instance reliability filter branch
    sdk_rel = type("SRel", (), {"show_instances": lambda self, raw=True: [{"id": 1, "actual_status": "running", "dph_total": 1.5, "reliability": 0.1}]})()
    assert InstanceManager(sdk_rel).find_reusable_running_instance(min_dph=1, max_dph=2, reliability=0.9) is None

    # build_connection_details mapping line with valid host tuple
    details2 = InstanceManager.build_connection_details(1, {"actual_status": "running", "ports": {"svc": [{"HostIp": "1.2.3.4", "HostPort": "8080"}]}})
    assert details2.service_urls[0]["direct"] == "http://1.2.3.4:8080"

    # _fetch_instance no-match and _wait_until_instance_absent post-timeout return branch
    sdk_nomatch = type("SNo", (), {"show_instances": lambda self, raw=True: [{"id": "2"}]})()
    manager_nomatch = InstanceManager(sdk_nomatch)
    assert manager_nomatch._fetch_instance(1) is None

    manager_timeout = InstanceManager(sdk_nomatch)
    seq = iter([0.0, 2.0])
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.monotonic", lambda: next(seq))
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(manager_timeout, "_fetch_instance", lambda _cid: None)
    assert manager_timeout._wait_until_instance_absent(1, timeout_seconds=1) is True


def test_api_client_remaining_lines(monkeypatch):
    with pytest.raises(RuntimeError):
        normalize_url("http://", service_name="RUNE API")

    client = rune.RuneApiClient("http://x:8080")

    class DetailHttpError(HTTPError):
        def __init__(self):
            super().__init__("http://x", 500, "bad", hdrs=None, fp=None)

        def read(self):
            return b"detail"

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(DetailHttpError()))
    with pytest.raises(RuntimeError, match="detail"):
        client._request("GET", "/x")

    monkeypatch.setattr("rune_bench.common.http_client.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("late")))
    with pytest.raises(RuntimeError, match="late"):
        client._request("GET", "/x")

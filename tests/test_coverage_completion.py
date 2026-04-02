from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

import rune
import rune_bench.api_backend as api_backend
import rune_bench.api_server as api_server
from rune_bench.common import make_http_request
from rune_bench.api_contracts import RunAgenticAgentRequest, RunBenchmarkRequest, RunOllamaInstanceRequest
from rune_bench.ollama.client import OllamaClient


def test_rune_container_port_and_vastai_helpers(monkeypatch):
    # _is_containerized: env var branch
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert rune._is_containerized() is True

    # _is_containerized: /.dockerenv branch
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(rune.Path, "exists", lambda _self: True)
    assert rune._is_containerized() is True

    # _find_free_port concrete path
    port = rune._find_free_port()
    assert isinstance(port, int)
    assert port > 0

    # _find_free_port and _resolve_serve_port branches
    monkeypatch.setattr(rune, "_is_containerized", lambda: True)
    assert rune._resolve_serve_port() == 8080

    monkeypatch.setattr(rune, "_is_containerized", lambda: False)
    monkeypatch.setattr(rune, "_find_free_port", lambda: 54321)
    assert rune._resolve_serve_port() == 54321

    # _vastai_sdk helper wiring
    captured: dict[str, object] = {}

    class DummyVast:
        def __init__(self, api_key, raw=True):
            captured["api_key"] = api_key
            captured["raw"] = raw

    monkeypatch.setenv("VAST_API_KEY", "k1")
    monkeypatch.setattr(rune, "VastAI", DummyVast)
    _ = rune._vastai_sdk()
    assert captured == {"api_key": "k1", "raw": True}


def test_serve_api_keyboard_interrupt_and_error(monkeypatch):
    # Avoid unrelated side effects
    monkeypatch.setattr(rune, "_enable_debug_if_requested", lambda _d: None)

    class AppServerInterrupt:
        def serve(self, host, port):
            raise KeyboardInterrupt

    class AppServerError:
        def serve(self, host, port):
            raise RuntimeError("boom")

    monkeypatch.setattr("rune_bench.api_server.RuneApiApplication.from_env", lambda: AppServerInterrupt())
    with pytest.raises(typer.Exit) as exc:
        rune.serve_api(api_host="127.0.0.1", api_port=9999, debug=False)
    assert exc.value.exit_code == 0

    monkeypatch.setattr("rune_bench.api_server.RuneApiApplication.from_env", lambda: AppServerError())
    with pytest.raises(typer.Exit) as exc:
        rune.serve_api(api_host="127.0.0.1", api_port=9999, debug=False)
    assert exc.value.exit_code == 1


def test_api_backend_vastai_sdk_helper(monkeypatch):
    captured: dict[str, object] = {}

    class DummyVast:
        def __init__(self, api_key, raw=True):
            captured["api_key"] = api_key
            captured["raw"] = raw

    monkeypatch.setenv("VAST_API_KEY", "k2")
    monkeypatch.setattr(api_backend, "VastAI", DummyVast)
    _ = api_backend._vastai_sdk()
    assert captured == {"api_key": "k2", "raw": True}


def test_http_client_verify_ssl_false_branch(monkeypatch):
    captured: dict[str, object] = {}

    class DummyContext:
        check_hostname = True
        verify_mode = None

    class DummyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_create_default_context():
        ctx = DummyContext()
        captured["ctx"] = ctx
        return ctx

    def fake_warn(message, stacklevel):
        captured["warn"] = (message, stacklevel)

    def fake_urlopen(request, timeout, context=None):
        captured["context"] = context
        return DummyResponse()

    monkeypatch.setattr("rune_bench.common.http_client.ssl.create_default_context", fake_create_default_context)
    monkeypatch.setattr("rune_bench.common.http_client.warnings.warn", fake_warn)
    monkeypatch.setattr("rune_bench.common.http_client.urlopen", fake_urlopen)

    payload = make_http_request("https://example.local", method="GET", action="test", verify_ssl=False)
    assert payload == {"ok": True}
    assert captured["context"] is captured["ctx"]
    assert captured["ctx"].check_hostname is False
    assert captured["ctx"].verify_mode is not None
    assert "TLS certificate verification is disabled" in captured["warn"][0]


def test_ollama_client_invalid_url_branch(monkeypatch):
    monkeypatch.setattr("rune_bench.ollama.client.normalize_url", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError, match="Missing or invalid Ollama URL"):
        OllamaClient("bad-url")


def test_api_server_backend_type_guards():
    agentic = RunAgenticAgentRequest(
        question="q",
        model="m",
        ollama_url="http://localhost:11434",
        ollama_warmup=False,
        ollama_warmup_timeout=1,
        kubeconfig="/tmp/k",
    )
    benchmark = RunBenchmarkRequest(
        vastai=False,
        template_hash="t",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        ollama_url="http://localhost:11434",
        question="q",
        model="m",
        ollama_warmup=False,
        ollama_warmup_timeout=1,
        kubeconfig="/tmp/k",
        vastai_stop_instance=False,
    )
    ollama = RunOllamaInstanceRequest(
        vastai=False,
        template_hash="t",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        ollama_url="http://localhost:11434",
    )

    with pytest.raises(RuntimeError, match="agentic-agent"):
        api_server._run_agentic_backend(benchmark)
    with pytest.raises(RuntimeError, match="benchmark"):
        api_server._run_benchmark_backend(ollama)
    with pytest.raises(RuntimeError, match="ollama-instance"):
        api_server._run_ollama_instance_backend(agentic)


def test_api_server_backend_success_paths(monkeypatch):
    agentic = RunAgenticAgentRequest(
        question="q",
        model="m",
        ollama_url="http://localhost:11434",
        ollama_warmup=False,
        ollama_warmup_timeout=1,
        kubeconfig="/tmp/k",
    )
    benchmark = RunBenchmarkRequest(
        vastai=False,
        template_hash="t",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        ollama_url="http://localhost:11434",
        question="q",
        model="m",
        ollama_warmup=False,
        ollama_warmup_timeout=1,
        kubeconfig="/tmp/k",
        vastai_stop_instance=False,
    )
    ollama = RunOllamaInstanceRequest(
        vastai=False,
        template_hash="t",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        ollama_url="http://localhost:11434",
    )

    monkeypatch.setattr(api_server, "run_agentic_agent", lambda req: {"kind": "agentic", "q": req.question})
    monkeypatch.setattr(api_server, "run_benchmark", lambda req: {"kind": "benchmark", "m": req.model})
    monkeypatch.setattr(api_server, "run_ollama_instance", lambda req: {"kind": "ollama", "url": req.ollama_url})

    assert api_server._run_agentic_backend(agentic)["kind"] == "agentic"
    assert api_server._run_benchmark_backend(benchmark)["kind"] == "benchmark"
    assert api_server._run_ollama_instance_backend(ollama)["kind"] == "ollama"


def test_workflow_normalize_ollama_url_success(monkeypatch):
    from rune_bench import workflows

    class DummyClient:
        def __init__(self, _url: str):
            self.base_url = "http://normalized:11434"

    monkeypatch.setattr(workflows, "OllamaClient", DummyClient)
    assert workflows.normalize_ollama_url("localhost:11434") == "http://normalized:11434"


def test_workflow_normalize_ollama_url_missing():
    from rune_bench import workflows

    with pytest.raises(RuntimeError, match="Missing Ollama URL"):
        workflows.normalize_ollama_url(None)

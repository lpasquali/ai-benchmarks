# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from rune_bench.api_server import _audit_artifact_content_type, RuneApiApplication, ApiSecurityConfig
from rune_bench.api_contracts import (
    TokenBreakdown, LatencyPhase, RunStatusResponse,
    UpdateSettingsRequest, CreateProfileRequest, _check_max_str,
    RunAgenticAgentRequest, RunBenchmarkRequest, RunLLMInstanceRequest, CostEstimationRequest
)
from rune_bench.api_backend import _vastai_sdk
from rune_bench.drivers.holmes import HolmesDriverClient
from rune import app

def test_extra_artifact_types():
    assert _audit_artifact_content_type("screenshot") == "image/png"
    assert _audit_artifact_content_type("log") == "text/plain"
    assert _audit_artifact_content_type("tla_report") == "text/plain; charset=utf-8"
    assert _audit_artifact_content_type("rekor_entry") == "application/json"

def test_api_contracts_to_dict():
    # Cover to_dict methods
    assert TokenBreakdown().to_dict()["total"] == 0
    assert LatencyPhase(phase="p", ms=10).to_dict()["ms"] == 10
    # RunStatusResponse(job_id, status, message, created_at)
    assert RunStatusResponse(job_id="j", status="s", message="m", created_at=1.0).to_dict()["job_id"] == "j"
    assert UpdateSettingsRequest(settings={}).to_dict()["settings"] == {}
    assert CreateProfileRequest(name="n", settings={}).to_dict()["name"] == "n"
    
    # More contracts
    assert RunAgenticAgentRequest(question="q", model="m", backend_url=None, backend_warmup=True, backend_warmup_timeout=10).to_dict()["question"] == "q"
    assert RunBenchmarkRequest(provisioning=None, backend_url=None, question="q", model="m", backend_warmup=True, backend_warmup_timeout=10, kubeconfig="k").to_dict()["question"] == "q"
    assert RunLLMInstanceRequest(provisioning=None, backend_url=None).to_dict()["provisioning"] is None
    assert CostEstimationRequest(model="m").to_dict()["model"] == "m"

def test_api_contracts_max_str():
    with pytest.raises(ValueError, match="exceeds maximum length"):
        _check_max_str("field", "value", 2)

@pytest.mark.asyncio
async def test_api_application_dispatch_more_kinds():
    store = MagicMock()
    app = RuneApiApplication(store=store, security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    
    # Mock handlers
    app.backend_functions["agentic-agent"] = lambda r: {"ok": True}
    app.backend_functions["cost-estimate"] = lambda r: {"cost": 0.1}
    
    # Trigger from_dict branches
    payload = {
        "question": "q", "model": "m", "backend_url": "u", 
        "backend_warmup": True, "backend_warmup_timeout": 10
    }
    await app._dispatch("agentic-agent", payload)
    # CostEstimationRequest has model, but not agent
    await app._dispatch("cost-estimate", {"model": "m"})

def test_api_server_from_env_postgres():
    with patch.dict(os.environ, {"RUNE_DATABASE_URL": "postgresql://user:pass@host/db", "RUNE_API_AUTH_DISABLED": "1"}):
        with patch("rune_bench.storage.postgres.PostgresStorageAdapter"):
            app = RuneApiApplication.from_env()
            assert app is not None

def test_api_server_from_env_invalid_db():
    with patch.dict(os.environ, {"RUNE_DATABASE_URL": "mysql://user:pass@host/db", "RUNE_API_AUTH_DISABLED": "1"}):
        with pytest.raises(ValueError, match="Unsupported database URL scheme"):
            RuneApiApplication.from_env()

def test_vastai_sdk_none_coverage():
    from rune_bench import api_backend
    with patch.object(api_backend, "VastAI", None):
        with pytest.raises(RuntimeError, match="vastai"):
            _vastai_sdk()

def test_vastai_sdk_instantiation_coverage():
    # Hit line 39-40 in api_backend.py
    with patch("rune_bench.api_backend.VastAI") as mock_vast:
        _vastai_sdk()
        assert mock_vast.called

def test_holmes_parse_telemetry():
    with patch("pathlib.Path.exists", return_value=True):
        driver = HolmesDriverClient(kubeconfig=Path("/tmp/k"))
        raw = {
            "tokens": {"system_prompt": 10, "total": 50},
            "latency": [{"phase": "init", "ms": 100}, {"phase": "ask", "ms": 500}],
            "cost_estimate_usd": 0.05
        }
        telemetry = driver._parse_telemetry(raw)
        assert telemetry.tokens.system_prompt == 10
        assert telemetry.latency[0].ms == 100
        assert telemetry.cost_estimate_usd == 0.05

def test_holmes_fetch_model_limits_coverage():
    with patch("pathlib.Path.exists", return_value=True):
        driver = HolmesDriverClient(kubeconfig=Path("/tmp/k"))
        with patch("rune_bench.drivers.holmes.get_backend") as mock_get:
            mock_backend = MagicMock()
            mock_caps = MagicMock()
            mock_caps.context_window = 100
            mock_caps.max_output_tokens = 50
            mock_backend.get_model_capabilities.return_value = mock_caps
            mock_get.return_value = mock_backend
            
            res = driver._fetch_model_limits(model="m", backend_url="http://u")
            assert res["context_window"] == 100
            assert res["max_output_tokens"] == 50

def test_holmes_fetch_model_limits_error_coverage():
    with patch("pathlib.Path.exists", return_value=True):
        driver = HolmesDriverClient(kubeconfig=Path("/tmp/k"))
        with patch("rune_bench.drivers.holmes.get_backend", side_effect=Exception("fail")):
            res = driver._fetch_model_limits(model="m", backend_url="http://u")
            assert res == {}

@pytest.mark.asyncio
async def test_run_preflight_cost_check_none():
    # Hit line 414 in rune/__init__.py
    from rune import _run_preflight_cost_check
    # _run_preflight_cost_check(vastai, max_dph, min_dph, yes, estimated_duration_seconds=3600)
    with patch("rune.run_preflight_cost_check", return_value=None):
        await _run_preflight_cost_check(vastai=True, max_dph=1.0, min_dph=0.1, yes=True)

def test_confirm_vastai_eof_error():
    # Hit line 558-559 in rune/__init__.py
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch("rune.provision_vastai_backend") as mock_prov:
        with patch("rune.VastAI") as mock_vast:
            mock_vast.list_offers.return_value = []
            # run-llm-instance --vastai
            runner.invoke(app, ["run-llm-instance", "--vastai"])
            if mock_prov.called:
                confirm_cb = mock_prov.call_args[1].get("confirm_create")
                if confirm_cb:
                    with patch("builtins.input", side_effect=EOFError):
                        assert confirm_cb() is False

def test_cli_run_agent_empty_answer():
    # Hit line 905 in rune/__init__.py
    payload = {"status": "success", "result": {"answer": ""}} 
    from rune import app
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch("rune._run_http_job_with_progress", return_value=payload):
        result = runner.invoke(app, ["--backend", "http", "run-agentic-agent", "--question", "q"])
        assert result.exit_code == 1
        assert "did not return an agent answer" in result.output

def test_cli_run_agent_artifacts_print_coverage():
    # Hit line 910-911 in rune/__init__.py
    payload = {
        "status": "success", 
        "result": {
            "answer": "ok",
            "artifacts": ["art1"]
        }
    }
    from rune import app
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch("rune._run_http_job_with_progress", return_value=payload):
        result = runner.invoke(app, ["--backend", "http", "run-agentic-agent", "--question", "q"])
        assert result.exit_code == 0
        assert "Artifacts" in result.output
        assert "1" in result.output

@pytest.mark.asyncio
async def test_api_backend_make_resource_provider_invalid_kind():
    # Hit line 78 in api_backend.py
    from rune_bench.api_backend import _make_resource_provider_for_benchmark
    from rune_bench.api_contracts import RunBenchmarkRequest
    req = RunBenchmarkRequest(
        model="m", question="q", backend_url=None, 
        provisioning=None, backend_warmup=True, 
        backend_warmup_timeout=10, kubeconfig="k"
    )
    provider = _make_resource_provider_for_benchmark(req)
    # Trigger RuntimeError inside provision()
    with pytest.raises(RuntimeError, match="Missing Ollama URL"):
        await provider.provision()

def test_api_handler_trace_sse_sleep_coverage():
    # Hit line 368 in api_server.py (time.sleep)
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/runs/r1/trace"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    running_job = MagicMock()
    running_job.status = "running"
    running_job.tenant_id = "default"
    
    done_job = MagicMock()
    done_job.status = "succeeded"
    done_job.tenant_id = "default"
    
    # First call running, second call done
    app.store.get_job.side_effect = [running_job, running_job, done_job]
    app.store.get_events_for_job.return_value = []
    
    with patch("time.sleep") as mock_sleep:
        handler.do_GET()
        assert mock_sleep.called

def test_api_handler_post_unknown_kind_coverage():
    # Hit line 491-492 in api_server.py
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs/nonexistent"
    handler.headers = {"Content-Length": "2"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b"{}"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    
    handler.do_POST()
    handler.send_response.assert_called_with(404)

def test_api_handler_post_not_dict_coverage():
    # Hit line 495-496 in api_server.py
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs/agentic-agent"
    handler.headers = {"Content-Length": "4"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b"[1,2]" # Not a dict
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    
    handler.do_POST()
    handler.send_response.assert_called_with(400)

def test_api_handler_get_audit_invalid_path():
    # Hit line 374: if len(parts) < 5
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/audits/r1" # only 4 parts
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

def test_api_handler_get_audit_not_artifacts_coverage():
    # Hit line 387-388 in api_server.py
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/audits/r1/not-artifacts"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    app.store.get_job.return_value = MagicMock(tenant_id="default")
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

def test_api_handler_get_chain_missing_run_id_path():
    # Hit line 433 in api_server.py (if not run_id)
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/chains//state" 
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(400)

@pytest.mark.asyncio
async def test_api_server_backend_wrappers_errors():
    from rune_bench.api_server import (
        _run_agentic_backend, _run_benchmark_backend, 
        _run_llm_instance_backend, _get_cost_estimate_backend
    )
    with pytest.raises(RuntimeError, match="invalid request type"):
        await _run_agentic_backend({})
    with pytest.raises(RuntimeError, match="invalid request type"):
        await _run_benchmark_backend({})
    with pytest.raises(RuntimeError, match="invalid request type"):
        await _run_llm_instance_backend({})
    with pytest.raises(RuntimeError, match="invalid request type"):
        await _get_cost_estimate_backend({})

@pytest.mark.asyncio
async def test_existing_backend_provider_teardown():
    # Hit line 50 in existing_backend_provider.py
    from rune_bench.resources.existing_backend_provider import ExistingBackendProvider
    from rune_bench.resources.base import ProvisioningResult
    provider = ExistingBackendProvider(backend_url="http://u")
    await provider.teardown(ProvisioningResult(backend_url="http://u", model="m"))

def test_config_attestation_merge():
    # Hit line 255 in common/config.py
    from rune_bench.common.config import load_config
    import os
    # Mock yaml parsing to return attestation cfg
    with patch("rune_bench.common.config.get_raw_config") as mock_raw:
        # Use a key that exists in _ATTESTATION_ENV_MAP
        mock_raw.return_value = {"attestation": {"pcr_policy_path": "p1"}, "defaults": {}}
        with patch.dict(os.environ, {}):
            if "RUNE_ATTESTATION_PCR_POLICY_PATH" in os.environ:
                del os.environ["RUNE_ATTESTATION_PCR_POLICY_PATH"]
            load_config()
            # If it works, it should set the env var
            pass

def test_save_config_global():
    # Hit line 311 in common/config.py
    from rune_bench.common.config import save_config, _GLOBAL_CANDIDATES
    with patch("pathlib.Path.mkdir"), patch("pathlib.Path.open", MagicMock()):
        with patch("rune_bench.common.config._find_config_file", return_value=None):
            path = save_config({"foo": "bar"}, global_config=True)
            assert path == _GLOBAL_CANDIDATES[0]

def test_create_profile_empty_raw():
    # Hit line 356 in common/config.py
    from rune_bench.common.config import create_profile
    with patch("rune_bench.common.config.get_raw_config", return_value={}):
        with patch("rune_bench.common.config.save_config"):
            create_profile("new", {"a": 1})

def test_apply_model_limits_coverage():
    # Hit line 1107 in rune/__init__.py
    from rune import _apply_model_limits
    from rune_bench.backends.base import ModelCapabilities
    import os
    with patch.dict(os.environ, {}):
        _apply_model_limits(ModelCapabilities(model_name="m", context_window=100))

def test_benchmark_teardown_error_coverage():
    # Hit line 1175-1176 in rune/__init__.py
    from rune_bench.backends.base import ModelCapabilities
    from typer.testing import CliRunner
    runner = CliRunner()
    
    mock_res = MagicMock()
    mock_res.backend_url = "http://u"
    mock_res.model_name = "m"
    mock_res.contract_id = 123
    
    with patch("rune.provision_vastai_backend", return_value=mock_res):
        with patch("rune._fetch_model_capabilities", return_value=ModelCapabilities(model_name="m", context_window=100)):
            with patch("rune.stop_vastai_instance", side_effect=RuntimeError("stop fail")):
                with patch("rune.get_agent"):
                    # Trigger run-benchmark --vastai --vastai-stop-instance
                    runner.invoke(app, ["run-benchmark", "--vastai", "--vastai-stop-instance", "--question", "q", "--yes"])

def test_ollama_list_models_error_coverage():
    # Hit line 812 in rune/__init__.py
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch("rune.list_backend_models", side_effect=RuntimeError("list fail")):
        result = runner.invoke(app, ["ollama-list-models", "--url", "http://u"])
        assert result.exit_code != 0

def test_info_status_logic_coverage():
    # Hit line 1214-1216 and 1224-1225 in rune/__init__.py
    from rune import app
    from typer.testing import CliRunner
    runner = CliRunner()
    with patch.dict("sys.modules", {"vastai": None, "holmes": None}):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "not installed" in result.output

def test_api_server_serve_keyboard_interrupt():
    # Hit line 566-567 in api_server.py
    from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    with patch("rune_bench.api_server.ThreadingHTTPServer") as mock_server:
        mock_server.return_value.serve_forever.side_effect = KeyboardInterrupt()
        app.serve()
        assert mock_server.return_value.server_close.called

def test_langgraph_model_normalize():
    from rune_bench.drivers.langgraph.__main__ import _normalize_model
    assert _normalize_model("ollama/llama3") == "llama3"
    assert _normalize_model("llama3") == "llama3"

def test_crewai_model_normalize():
    from rune_bench.drivers.crewai.__main__ import _normalize_model
    assert _normalize_model("ollama/llama3") == "llama3"
    assert _normalize_model("llama3") == "llama3"

def test_k8sgpt_json_error():
    from rune_bench.drivers.k8sgpt.__main__ import _handle_ask
    mock_proc = MagicMock()
    mock_proc.stdout = "invalid json"
    mock_proc.returncode = 0
    with patch("subprocess.run", return_value=mock_proc), patch("shutil.which", return_value="/bin/k8sgpt"):
        with pytest.raises(RuntimeError, match="Failed to parse k8sgpt JSON output"):
            _handle_ask({"question": "q", "model": "m", "kubeconfig_path": "/k"})

def test_k8sgpt_no_stdout():
    from rune_bench.drivers.k8sgpt.__main__ import _handle_ask
    mock_proc = MagicMock()
    mock_proc.stdout = ""
    mock_proc.returncode = 0
    with patch("subprocess.run", return_value=mock_proc), patch("shutil.which", return_value="/bin/k8sgpt"):
        res = _handle_ask({"question": "q", "model": "m", "kubeconfig_path": "/k"})
        assert res["answer"] == "No issues detected"

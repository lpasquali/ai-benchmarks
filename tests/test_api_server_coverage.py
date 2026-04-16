# SPDX-License-Identifier: Apache-2.0
import pytest
from http.server import BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch
from rune_bench.api_server import (
    RuneApiApplication, 
    ApiSecurityConfig, 
    _run_agentic_backend,
    _run_benchmark_backend,
    _run_llm_instance_backend,
    _get_cost_estimate_backend,
    _audit_artifact_content_type
)
from rune_bench.api_contracts import (
    RunAgenticAgentRequest, 
    RunBenchmarkRequest, 
    RunLLMInstanceRequest, 
    CostEstimationRequest
)
from rune_bench.attestation.interface import AttestationResult

@pytest.mark.asyncio
async def test_backend_type_checks():
    # Test _run_agentic_backend type check
    with pytest.raises(RuntimeError, match="invalid request type for agentic-agent backend"):
        await _run_agentic_backend(MagicMock())

    # Test _run_benchmark_backend type check
    with pytest.raises(RuntimeError, match="invalid request type for benchmark backend"):
        await _run_benchmark_backend(MagicMock())

    # Test _run_llm_instance_backend type check
    with pytest.raises(RuntimeError, match="invalid request type for ollama-instance backend"):
        await _run_llm_instance_backend(MagicMock())

    # Test _get_cost_estimate_backend type check
    with pytest.raises(RuntimeError, match="invalid request type for cost-estimate backend"):
        await _get_cost_estimate_backend(MagicMock())

def test_audit_artifact_content_type_fallback():
    assert _audit_artifact_content_type("unknown") == "application/octet-stream"

def test_api_handler_setup_timeout_error():
    # Test the 'except (AttributeError, OSError): pass' in RuneApiHandler.setup
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    
    mock_request = MagicMock()
    mock_request.settimeout.side_effect = OSError("failed to set timeout")
    
    # We just want to ensure it doesn't raise
    handler = handler_class.__new__(handler_class)
    handler.request = mock_request
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = MagicMock()
    
    # setup() is called by __init__, but we can call it manually if we mock the rest
    with patch.object(BaseHTTPRequestHandler, 'setup'):
        handler.setup() 
    
    assert mock_request.settimeout.called

@pytest.mark.asyncio
async def test_api_application_execute_kinds():
    store = MagicMock()
    app = RuneApiApplication(store=store, security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    
    # Mock handlers
    handler = MagicMock(return_value={"ok": True})
    
    payload = {
        "question": "q",
        "model": "m",
        "backend_url": "u",
        "backend_warmup": True,
        "backend_warmup_timeout": 10
    }
    
    # Test each kind to cover lines 566-573
    await app._execute_job("j1", handler, "agentic-agent", payload)
    store.update_job.assert_any_call("j1", status="succeeded", result_payload={"ok": True})

    await app._execute_job("j2", handler, "benchmark", payload)
    await app._execute_job("j3", handler, "ollama-instance", payload)
    await app._execute_job("j4", handler, "cost-estimate", {"model": "m", "agent": "a"})
    
    # Test unsupported kind in _execute_job
    await app._execute_job("j5", handler, "unknown", {})
    store.update_job.assert_any_call("j5", status="failed", error="unsupported job kind: unknown")

def test_auth_failure_logging():
    # Hit the logging lines in _authenticate
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(
        auth_disabled=False,
        tenant_tokens={"t1": "hash1"}
    ))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.headers = {"X-Tenant-ID": "t1", "Authorization": "Bearer wrong"}
    
    with patch("logging.error") as mock_log:
        res = handler._authenticate()
        assert res is None
        assert mock_log.called

    handler.headers = {"X-Tenant-ID": "unknown", "Authorization": "Bearer whatever"}
    with patch("logging.error") as mock_log:
        res = handler._authenticate()
        assert res is None
        assert mock_log.called

@pytest.mark.asyncio
async def test_verify_attestation_failure():
    from rune_bench.api_backend import _verify_attestation
    
    mock_driver = MagicMock()
    mock_driver.verify.return_value = AttestationResult(passed=False, pcr_digest=None, message="failed")
    
    with patch("rune_bench.attestation.factory.get_driver", return_value=mock_driver):
        with pytest.raises(RuntimeError, match="Attestation failed for scheduling target 't1': failed"):
            _verify_attestation("t1")

@pytest.mark.asyncio
async def test_vastai_sdk_missing():
    from rune_bench import api_backend
    with patch.object(api_backend, "VastAI", None):
        with pytest.raises(RuntimeError, match="The 'vastai' package is required"):
            api_backend._vastai_sdk()

def test_api_handler_do_get_not_found():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/unknown"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

def test_api_handler_do_post_not_found():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/unknown"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_POST()
    handler.send_response.assert_called_with(404)

def test_api_handler_post_jobs_invalid_json():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs"
    handler.headers = {"Content-Length": "3"}
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b"not json"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    
    handler.do_POST()
    handler.send_response.assert_called_with(400)

def test_api_handler_post_jobs_missing_tenant():
    # auth_disabled=False, but no tenant header
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=False, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs"
    handler.headers = {}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    
    handler.do_POST()
    handler.send_response.assert_called_with(401)

def test_api_handler_get_metrics_summary():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/metrics/summary?job_id=123"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    # Mock store.get_events_summary
    app.store.get_events_summary.return_value = {"summary": "ok"}
    
    handler.do_GET()
    app.store.get_events_summary.assert_called_with(job_id="123")

def test_api_handler_get_job_not_found():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    app.store.get_job.return_value = None
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs/missing"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

def test_api_handler_get_audit_missing_run_id():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/audits//artifacts"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(400)

def test_api_handler_get_chain_missing_run_id():
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

def test_api_handler_finops_simulate_error():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/finops/simulate?agent=holmes&model=m1"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    with patch("rune_bench.api_server.PricingSoothSayer") as mock_sayer:
        mock_sayer.return_value.simulate.side_effect = Exception("sim fail")
        handler.do_GET()
        handler.send_response.assert_called_with(400)

def test_api_handler_trace_job_not_found():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    app.store.get_job.return_value = None
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/runs/r1/trace"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

def test_api_handler_trace_loop_job_disappears():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    job = MagicMock()
    job.status = "running"
    job.tenant_id = "default"
    # First call returns job, second returns None
    app.store.get_job.side_effect = [job, None]
    
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/runs/r1/trace"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    # Mock time.sleep to avoid hanging
    with patch("time.sleep"):
        handler.do_GET()
    
    handler.send_response.assert_called_with(200)
    assert app.store.get_job.call_count == 2

def test_api_handler_post_jobs_rate_limited():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    # To trigger rate limit, we can mock _enforce_request_rate_limit to raise
    from rune_bench.api_server import RequestRateLimited
    with patch.object(app, "_enforce_request_rate_limit", side_effect=RequestRateLimited()):
        handler_class = app.create_handler()
        handler = handler_class.__new__(handler_class)
        handler.path = "/v1/jobs/agentic-agent"
        handler.headers = {"X-Tenant-ID": "t1"}
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        
        handler.do_POST()
        handler.send_response.assert_called_with(401)

def test_api_handler_post_request_too_large():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/jobs/agentic-agent"
    handler.headers = {"Content-Length": str(11 * 1024 * 1024)} # 11MB
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    
    handler.do_POST()
    handler.send_response.assert_called_with(413)

def test_api_handler_artifacts_wrong_subpath():
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    app.store.get_job.return_value = MagicMock(tenant_id="default")
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/runs/r1/something-else" # parts[4] is not artifacts
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(404)

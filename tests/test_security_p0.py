# SPDX-License-Identifier: Apache-2.0
import json
import logging
import time
from io import StringIO

import pytest
from rune_bench.api_server import (
    ApiSecurityConfig,
    JsonFormatter,
    RuneApiApplication,
    RequestRateLimited,
    setup_logging
)

def test_api_security_config_token_length():
    # Valid token (32 chars)
    config = "tenant1:" + ("a" * 32)
    import os
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("RUNE_API_TOKENS", config)
        mp.setenv("RUNE_API_AUTH_DISABLED", "0")
        cfg = ApiSecurityConfig.from_env()
        assert len(cfg.tenant_tokens["tenant1"]) == 32

    # Invalid token (31 chars)
    config = "tenant1:" + ("a" * 31)
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("RUNE_API_TOKENS", config)
        mp.setenv("RUNE_API_AUTH_DISABLED", "0")
        with pytest.raises(RuntimeError, match="too short"):
            ApiSecurityConfig.from_env()

def test_rate_limit_increased_to_100():
    from unittest.mock import MagicMock
    app = RuneApiApplication(store=MagicMock(), security=MagicMock())
    
    tenant = "test-tenant"
    # Should allow 100 requests
    for _ in range(100):
        app._enforce_request_rate_limit(tenant)
    
    # 101st should fail
    with pytest.raises(RequestRateLimited):
        app._enforce_request_rate_limit(tenant)

def test_json_formatter():
    formatter = JsonFormatter()
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(formatter)
    logger = logging.getLogger("test_json")
    logger.addHandler(handler)
    logger.propagate = False
    
    # Test basic log
    logger.error("test message")
    log_out = json.loads(log_stream.getvalue().splitlines()[-1])
    assert log_out["message"] == "test message"
    assert log_out["level"] == "ERROR"
    assert "timestamp" in log_out
    
    # Test with extra context
    logger.error("with context", extra={"tenant_id": "t1", "job_id": "j1"})
    log_out = json.loads(log_stream.getvalue().splitlines()[-1])
    assert log_out["tenant_id"] == "t1"
    assert log_out["job_id"] == "j1"
    
    # Test with exception
    try:
        raise ValueError("oops")
    except ValueError:
        logger.exception("failed")
    
    log_out = json.loads(log_stream.getvalue().splitlines()[-1])
    assert log_out["message"] == "failed"
    assert "ValueError: oops" in log_out["exception"]

def test_setup_logging_integration():
    # Ensure setup_logging doesn't crash and attaches correct formatter
    setup_logging(level=logging.DEBUG, json_format=True)
    root = logging.getLogger()
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    
    setup_logging(level=logging.DEBUG, json_format=False)
    assert not isinstance(root.handlers[0].formatter, JsonFormatter)

def test_api_security_config_malformed_pair():
    # Test line 115: pair without ':'
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("RUNE_API_TOKENS", "malformed-pair")
        mp.setenv("RUNE_API_AUTH_DISABLED", "1")
        cfg = ApiSecurityConfig.from_env()
        assert cfg.tenant_tokens == {}

def test_api_handler_artifacts_missing_run_id_coverage():
    from unittest.mock import MagicMock
    app = RuneApiApplication(store=MagicMock(), security=ApiSecurityConfig(auth_disabled=True, tenant_tokens={}))
    handler_class = app.create_handler()
    handler = handler_class.__new__(handler_class)
    handler.path = "/v1/runs//artifacts"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    handler.headers = {}
    
    handler.do_GET()
    handler.send_response.assert_called_with(400)

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ComfyUI driver client."""

import pytest
from unittest.mock import Mock, patch
from rune_bench.agents.base import AgentResult
from rune_bench.api_contracts import RunTelemetry, TokenBreakdown, LatencyPhase
from rune_bench.drivers.comfyui import ComfyUIDriverClient

@pytest.fixture
def comfyui_auth(monkeypatch):
    monkeypatch.setenv("RUNE_COMFYUI_API_KEY", "test-api-key")

@pytest.fixture
def mock_transport():
    transport = Mock()
    transport.call.return_value = {
        "answer": "https://example.com/image.png",
        "result_type": "image",
        "artifacts": ["artifact1"],
        "metadata": {"test": "meta"},
        "telemetry": {
            "tokens": {"total": 10},
            "latency": [{"phase": "gen", "ms": 100}],
            "cost_estimate_usd": 0.05
        }
    }
    return transport

@pytest.fixture
def mock_async_transport():
    import asyncio
    
    class AsyncMockTransport:
        async def call_async(self, method, params):
            return {
                "answer": "https://example.com/image.png",
                "result_type": "image",
                "artifacts": ["artifact1"],
                "metadata": {"test": "meta"},
                "telemetry": {
                    "tokens": {"total": 10},
                    "latency": [{"phase": "gen", "ms": 100}],
                    "cost_estimate_usd": 0.05
                }
            }
    return AsyncMockTransport()

def test_check_auth_missing(monkeypatch):
    monkeypatch.delenv("RUNE_COMFYUI_API_KEY", raising=False)
    client = ComfyUIDriverClient()
    with pytest.raises(RuntimeError, match="RUNE_COMFYUI_API_KEY"):
        client._check_auth()

def test_ask(comfyui_auth, mock_transport):
    client = ComfyUIDriverClient(transport=mock_transport)
    answer = client.ask("Generate image", "sdxl")
    assert answer == "https://example.com/image.png"
    mock_transport.call.assert_called_once()
    args, kwargs = mock_transport.call.call_args
    assert args[0] == "ask"
    assert args[1]["question"] == "Generate image"
    assert args[1]["model"] == "sdxl"

def test_ask_structured(comfyui_auth, mock_transport):
    client = ComfyUIDriverClient(transport=mock_transport)
    result = client.ask_structured("Generate image", "sdxl")
    assert isinstance(result, AgentResult)
    assert result.answer == "https://example.com/image.png"
    assert result.result_type == "image"
    assert result.artifacts == ["artifact1"]
    assert result.metadata == {"test": "meta"}
    assert isinstance(result.telemetry, RunTelemetry)
    assert result.telemetry.tokens.total == 10
    assert result.telemetry.latency[0].phase == "gen"
    assert result.telemetry.latency[0].ms == 100
    assert result.telemetry.cost_estimate_usd == 0.05

def test_ask_structured_empty_answer(comfyui_auth, mock_transport):
    mock_transport.call.return_value = {"answer": ""}
    client = ComfyUIDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="Driver returned an empty answer."):
        client.ask_structured("Generate image", "sdxl")

@pytest.mark.asyncio
async def test_ask_async(comfyui_auth, mock_async_transport):
    client = ComfyUIDriverClient()
    client._async_transport = mock_async_transport
    result = await client.ask_async("Generate image", "sdxl")
    assert isinstance(result, AgentResult)
    assert result.answer == "https://example.com/image.png"
    assert result.result_type == "image"
    assert result.artifacts == ["artifact1"]
    assert result.metadata == {"test": "meta"}

@pytest.mark.asyncio
async def test_ask_async_missing_answer(comfyui_auth):
    import asyncio
    class BadAsyncMockTransport:
        async def call_async(self, method, params):
            return {}
            
    client = ComfyUIDriverClient()
    client._async_transport = BadAsyncMockTransport()
    with pytest.raises(RuntimeError, match="Driver response did not include an answer."):
        await client.ask_async("Generate image", "sdxl")

@pytest.mark.asyncio
async def test_ask_async_empty_answer(comfyui_auth):
    import asyncio
    class BadAsyncMockTransport:
        async def call_async(self, method, params):
            return {"answer": ""}
            
    client = ComfyUIDriverClient()
    client._async_transport = BadAsyncMockTransport()
    with pytest.raises(RuntimeError, match="Driver returned an empty answer."):
        await client.ask_async("Generate image", "sdxl")

def test_fetch_model_limits_success(comfyui_auth):
    client = ComfyUIDriverClient()
    
    mock_backend = Mock()
    mock_caps = Mock()
    mock_caps.context_window = 4096
    mock_caps.max_output_tokens = 1024
    mock_backend.normalize_model_name.return_value = "sdxl-turbo"
    mock_backend.get_model_capabilities.return_value = mock_caps
    
    with patch("rune_bench.backends.get_backend", return_value=mock_backend):
        limits = client._fetch_model_limits(model="sdxl", backend_url="http://test")
        assert limits == {"context_window": 4096, "max_output_tokens": 1024}

def test_fetch_model_limits_error(comfyui_auth):
    client = ComfyUIDriverClient()
    with patch("rune_bench.backends.get_backend", side_effect=Exception("Backend error")):
        limits = client._fetch_model_limits(model="sdxl", backend_url="http://test")
        assert limits == {}

def test_ask_structured_with_backend_url(comfyui_auth, mock_transport):
    client = ComfyUIDriverClient(transport=mock_transport)
    
    mock_backend = Mock()
    mock_caps = Mock()
    mock_caps.context_window = 4096
    mock_caps.max_output_tokens = 1024
    mock_backend.normalize_model_name.return_value = "sdxl-turbo"
    mock_backend.get_model_capabilities.return_value = mock_caps
    
    with patch("rune_bench.backends.get_backend", return_value=mock_backend):
        result = client.ask_structured("Generate image", "sdxl", backend_url="http://test")
        assert isinstance(result, AgentResult)
        
        args, kwargs = mock_transport.call.call_args
        assert args[1]["backend_url"] == "http://test"
        assert args[1]["context_window"] == 4096

@pytest.mark.asyncio
async def test_ask_async_with_backend_url(comfyui_auth, mock_async_transport):
    client = ComfyUIDriverClient()
    client._async_transport = mock_async_transport
    
    mock_backend = Mock()
    mock_caps = Mock()
    mock_caps.context_window = 4096
    mock_caps.max_output_tokens = 1024
    mock_backend.normalize_model_name.return_value = "sdxl-turbo"
    mock_backend.get_model_capabilities.return_value = mock_caps
    
    with patch("rune_bench.backends.get_backend", return_value=mock_backend):
        result = await client.ask_async("Generate image", "sdxl", backend_url="http://test")
        assert isinstance(result, AgentResult)

def test_parse_telemetry_none():
    client = ComfyUIDriverClient()
    assert client._parse_telemetry(None) is None

from rune_bench.drivers.comfyui.__main__ import _handle_ask, _handle_info, main
import sys
from io import StringIO

def test_handle_info():
    result = _handle_info({})
    assert result["name"] == "comfyui"
    assert "ask" in result["actions"]

def test_handle_ask_missing_key(monkeypatch):
    monkeypatch.delenv("RUNE_COMFYUI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_COMFYUI_API_KEY not set"):
        _handle_ask({"question": "test", "model": "sdxl"})

def test_handle_ask_success(comfyui_auth):
    mock_runner = Mock()
    mock_runner.ask.return_value = "https://example.com/image.png"
    
    with patch("rune_bench.drivers.comfyui.__main__.ComfyUIRunner", return_value=mock_runner):
        result = _handle_ask({
            "question": "test",
            "model": "sdxl",
            "backend_url": "http://test",
            "backend_type": "aws"
        })
        
        assert result["answer"] == "https://example.com/image.png"
        assert result["result_type"] == "image"
        
        mock_runner.ask.assert_called_once_with(
            "test",
            model="sdxl",
            backend_url="http://test",
            backend_type="aws"
        )

def test_main_loop(monkeypatch):
    inputs = [
        '{"id": "1", "action": "info"}',
        '{"id": "2", "action": "unknown"}',
        'invalid json',
        ''
    ]
    monkeypatch.setattr("sys.stdin", StringIO("\n".join(inputs) + "\n"))
    
    out = StringIO()
    monkeypatch.setattr("sys.stdout", out)
    
    main()
    
    lines = out.getvalue().strip().split("\n")
    assert len(lines) == 3
    
    res1 = __import__("json").loads(lines[0])
    assert res1["status"] == "ok"
    assert res1["id"] == "1"
    assert res1["result"]["name"] == "comfyui"
    
    res2 = __import__("json").loads(lines[1])
    assert res2["status"] == "error"
    assert res2["id"] == "2"
    assert "Unknown action" in res2["error"]
    
    res3 = __import__("json").loads(lines[2])
    assert res3["status"] == "error"
    assert res3["id"] == ""
    assert "Expecting value" in res3["error"]

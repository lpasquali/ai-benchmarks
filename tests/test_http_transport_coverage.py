# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from rune_bench.drivers.http import AsyncHttpTransport

@pytest.mark.asyncio
async def test_async_http_transport_happy_path():
    transport = AsyncHttpTransport("http://localhost:8080", api_token="t1", tenant="tenant1")
    
    mock_response = {"job_id": "j1"}
    mock_poll_pending = {"status": "running"}
    mock_poll_success = {"status": "succeeded", "result": {"ans": 42}}
    
    with patch("rune_bench.drivers.http.make_async_http_request", AsyncMock()) as mock_req:
        mock_req.side_effect = [mock_response, mock_poll_pending, mock_poll_success]
        
        with patch("asyncio.sleep", AsyncMock()):
            res = await transport.call_async("test-action", {"p": 1})
            assert res == {"ans": 42}
            assert mock_req.call_count == 3

@pytest.mark.asyncio
async def test_async_http_transport_failure():
    transport = AsyncHttpTransport("http://localhost:8080")
    
    mock_response = {"job_id": "j1"}
    mock_poll_failed = {"status": "failed", "error": "test failure"}
    
    with patch("rune_bench.drivers.http.make_async_http_request", AsyncMock()) as mock_req:
        mock_req.side_effect = [mock_response, mock_poll_failed]
        
        with pytest.raises(RuntimeError, match="test failure"):
            await transport.call_async("test-action", {})

@pytest.mark.asyncio
async def test_async_http_transport_no_job_id():
    transport = AsyncHttpTransport("http://localhost:8080")
    
    with patch("rune_bench.drivers.http.make_async_http_request", AsyncMock(return_value={})):
        with pytest.raises(RuntimeError, match="did not return a job_id"):
            await transport.call_async("test-action", {})

# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from rune_bench.drivers.invokeai import InvokeAIDriverClient
from rune_bench.agents.base import AgentResult

@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.invokeai.make_driver_transport") as m_sync, \
         patch("rune_bench.drivers.invokeai.make_async_driver_transport") as m_async:
        sync_transport = MagicMock()
        async_transport = MagicMock()
        m_sync.return_value = sync_transport
        m_async.return_value = async_transport
        yield sync_transport, async_transport

def test_invokeai_ask_structured(mock_transports):
    sync_transport, _ = mock_transports
    sync_transport.call.return_value = {
        "answer": "Generated image: http://localhost:9090/outputs/test.png",
        "result_type": "image",
        "artifacts": [{"url": "http://localhost:9090/outputs/test.png"}]
    }
    
    client = InvokeAIDriverClient()
    result = client.ask_structured("a prompt", model="stable-diffusion-v1-5")
    
    assert isinstance(result, AgentResult)
    assert "Generated image" in result.answer
    assert result.result_type == "image"
    assert len(result.artifacts) == 1

def test_invokeai_ask_async(mock_transports):
    _, async_transport = mock_transports
    
    async def mock_call_async(action, params):
        return {
            "answer": "Generated image: http://localhost:9090/outputs/async.png"
        }
    async_transport.call_async = mock_call_async
    
    client = InvokeAIDriverClient()
    result = asyncio.run(client.ask_async("async prompt", model="sdxl"))
    
    assert "async.png" in result.answer

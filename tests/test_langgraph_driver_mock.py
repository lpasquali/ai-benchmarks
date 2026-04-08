# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from rune_bench.drivers.langgraph import LangGraphDriverClient
from rune_bench.agents.base import AgentResult

@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.langgraph.make_driver_transport") as m_sync, \
         patch("rune_bench.drivers.langgraph.make_async_driver_transport") as m_async:
        sync_transport = MagicMock()
        async_transport = MagicMock()
        m_sync.return_value = sync_transport
        m_async.return_value = async_transport
        yield sync_transport, async_transport

def test_langgraph_ask_structured(mock_transports):
    sync_transport, _ = mock_transports
    sync_transport.call.return_value = {
        "answer": "Diagnostics: All pods are healthy.",
        "metadata": {"source": "langgraph-sre-workflow"}
    }
    
    client = LangGraphDriverClient(kubeconfig=Path("/tmp/mock-kubeconfig"))
    result = client.ask_structured("diagnose cluster", model="gpt-4")
    
    assert isinstance(result, AgentResult)
    assert "healthy" in result.answer
    assert result.metadata["source"] == "langgraph-sre-workflow"

def test_langgraph_ask_async(mock_transports):
    _, async_transport = mock_transports
    
    async def mock_call_async(action, params):
        return {
            "answer": "Async diagnostic complete."
        }
    async_transport.call_async = mock_call_async
    
    client = LangGraphDriverClient(kubeconfig=Path("/tmp/mock-kubeconfig"))
    result = asyncio.run(client.ask_async("async diagnose", model="gpt-4"))
    
    assert "complete" in result.answer

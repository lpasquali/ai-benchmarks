# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from rune_bench.drivers.browser_use import BrowserUseDriverClient
from rune_bench.agents.base import AgentResult

@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.browser_use.make_driver_transport") as m_sync, \
         patch("rune_bench.drivers.browser_use.make_async_driver_transport") as m_async:
        sync_transport = MagicMock()
        async_transport = MagicMock()
        m_sync.return_value = sync_transport
        m_async.return_value = async_transport
        yield sync_transport, async_transport

def test_browser_use_ask_structured(mock_transports):
    sync_transport, _ = mock_transports
    sync_transport.call.return_value = {
        "answer": "Navigated to example.com and found the search button.",
        "result_type": "text",
        "metadata": {"url": "https://example.com"}
    }
    
    client = BrowserUseDriverClient()
    result = client.ask_structured("go to example.com and find search", model="llama3.1")
    
    assert isinstance(result, AgentResult)
    assert "example.com" in result.answer
    assert result.metadata["url"] == "https://example.com"

def test_browser_use_ask_async(mock_transports):
    _, async_transport = mock_transports
    
    async def mock_call_async(action, params):
        return {
            "answer": "Task completed successfully."
        }
    async_transport.call_async = mock_call_async
    
    client = BrowserUseDriverClient()
    result = asyncio.run(client.ask_async("do something", model="llama3.1"))
    
    assert "successfully" in result.answer

# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.langgraph import LangGraphDriverClient

@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.langgraph.make_driver_transport") as mock_sync:
        with patch("rune_bench.drivers.langgraph.make_async_driver_transport") as mock_async:
            sync_transport = MagicMock()
            async_transport = MagicMock()
            mock_sync.return_value = sync_transport
            mock_async.return_value = async_transport
            yield sync_transport, async_transport

@pytest.fixture
def client(mock_transports):
    return LangGraphDriverClient(kubeconfig=Path("/tmp/k"))

@pytest.mark.asyncio
async def test_langgraph_ask_async_with_backend_url(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": "ok"})
    await client.ask_async("q", model="m", backend_url="http://u")
    args, kwargs = async_t.call_async.call_args
    assert args[1]["backend_url"] == "http://u"

@pytest.mark.asyncio
async def test_langgraph_ask_structured(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": "ok"}
    res = client.ask_structured("q", model="m")
    assert res.answer == "ok"

@pytest.mark.asyncio
async def test_langgraph_ask(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": "ok"}
    res = client.ask("q", model="m")
    assert res == "ok"

@pytest.mark.asyncio
async def test_langgraph_ask_no_answer_in_result(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"something": "else"}
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask_structured("q", model="m", backend_url="http://u")

@pytest.mark.asyncio
async def test_langgraph_ask_none_answer(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": None}
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        client.ask_structured("q", model="m")

@pytest.mark.asyncio
async def test_langgraph_ask_async_none_answer(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": None})
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        await client.ask_async("q", model="m")

@pytest.mark.asyncio
async def test_langgraph_ask_async_no_answer_in_result(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"something": "else"})
    with pytest.raises(RuntimeError, match="did not include an answer"):
        await client.ask_async("q", model="m")

def test_langgraph_parse_telemetry(client):
    raw = {
        "tokens": {"system_prompt": 10},
        "latency": [{"phase": "p", "ms": 100}],
        "cost_estimate_usd": 0.1
    }
    telemetry = client._parse_telemetry(raw)
    assert telemetry.tokens.system_prompt == 10
    assert telemetry.latency[0].ms == 100
    assert telemetry.cost_estimate_usd == 0.1

def test_langgraph_parse_telemetry_none(client):
    assert client._parse_telemetry(None) is None

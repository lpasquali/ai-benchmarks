# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.invokeai import InvokeAIDriverClient


@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.invokeai.make_driver_transport") as mock_sync:
        with patch(
            "rune_bench.drivers.invokeai.make_async_driver_transport"
        ) as mock_async:
            sync_transport = MagicMock()
            async_transport = MagicMock()
            mock_sync.return_value = sync_transport
            mock_async.return_value = async_transport
            yield sync_transport, async_transport


@pytest.fixture
def client(mock_transports):
    return InvokeAIDriverClient()


@pytest.mark.asyncio
async def test_invokeai_ask(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": "image.png"}
    res = client.ask("q", model="m")
    assert res == "image.png"


@pytest.mark.asyncio
async def test_invokeai_ask_no_answer(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"something": "else"}
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask_structured("q", model="m")


@pytest.mark.asyncio
async def test_invokeai_ask_async_no_answer(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"something": "else"})
    with pytest.raises(RuntimeError, match="did not include an answer"):
        await client.ask_async("q", model="m")


@pytest.mark.asyncio
async def test_invokeai_ask_async_success(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": "async.png"})
    res = await client.ask_async("q", model="m")
    assert res.answer == "async.png"


def test_invokeai_parse_telemetry(client):
    raw = {
        "tokens": {"system_prompt": 10},
        "latency": [{"phase": "p", "ms": 100}],
        "cost_estimate_usd": 0.1,
    }
    telemetry = client._parse_telemetry(raw)
    assert telemetry.tokens.system_prompt == 10
    assert telemetry.latency[0].ms == 100
    assert telemetry.cost_estimate_usd == 0.1

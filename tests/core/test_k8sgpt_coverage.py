# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.k8sgpt import K8sGPTDriverClient


@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.k8sgpt.make_driver_transport") as mock_sync:
        with patch(
            "rune_bench.drivers.k8sgpt.make_async_driver_transport"
        ) as mock_async:
            sync_transport = MagicMock()
            async_transport = MagicMock()
            mock_sync.return_value = sync_transport
            mock_async.return_value = async_transport
            yield sync_transport, async_transport


@pytest.fixture
def client(mock_transports):
    with patch("pathlib.Path.exists", return_value=True):
        return K8sGPTDriverClient(kubeconfig=Path("/tmp/k"))


def test_k8sgpt_init_missing_kubeconfig():
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="kubeconfig not found"):
            K8sGPTDriverClient(kubeconfig=Path("/tmp/missing"))


@pytest.mark.asyncio
async def test_k8sgpt_ask_success(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": "ok"}
    res = client.ask_structured("q", model="m")
    assert res.answer == "ok"


@pytest.mark.asyncio
async def test_k8sgpt_ask_no_answer(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"something": "else"}
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask_structured("q", model="m", backend_url="http://u")


@pytest.mark.asyncio
async def test_k8sgpt_ask_empty_answer(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": ""}
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        client.ask("q", model="m")


@pytest.mark.asyncio
async def test_k8sgpt_ask_async_success(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": "async ok"})
    res = await client.ask_async("q", model="m")
    assert res.answer == "async ok"


@pytest.mark.asyncio
async def test_k8sgpt_ask_async_no_answer(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"something": "else"})
    with pytest.raises(RuntimeError, match="did not include an answer"):
        await client.ask_async("q", model="m")


@pytest.mark.asyncio
async def test_k8sgpt_ask_async_with_backend_url_and_limits(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": "ok"})

    client._fetch_model_limits = MagicMock(return_value={"limit": 100})

    await client.ask_async("q", model="m", backend_url="http://u")
    assert client._fetch_model_limits.called


def test_k8sgpt_parse_telemetry(client):
    raw = {
        "tokens": {"system_prompt": 10},
        "latency": [{"phase": "p", "ms": 100}],
        "cost_estimate_usd": 0.1,
    }
    telemetry = client._parse_telemetry(raw)
    assert telemetry.tokens.system_prompt == 10
    assert telemetry.latency[0].ms == 100
    assert telemetry.cost_estimate_usd == 0.1


@pytest.mark.asyncio
async def test_k8sgpt_ask_none_answer(client, mock_transports):
    sync_t, _ = mock_transports
    sync_t.call.return_value = {"answer": None}
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        client.ask_structured("q", model="m", backend_url="http://u")


@pytest.mark.asyncio
async def test_k8sgpt_ask_async_none_answer(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": None})
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        await client.ask_async("q", model="m")


@pytest.mark.asyncio
async def test_k8sgpt_ask_async_empty_string_answer(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": ""})
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        await client.ask_async("q", model="m")


def test_k8sgpt_parse_telemetry_full(client):
    raw = {
        "tokens": {"system_prompt": 10, "total": 50},
        "latency": [{"phase": "p", "ms": 100}],
        "cost_estimate_usd": 0.1,
    }
    telemetry = client._parse_telemetry(raw)
    assert telemetry.tokens.total == 50
    assert telemetry.latency[0].phase == "p"


def test_k8sgpt_parse_telemetry_none(client):
    assert client._parse_telemetry(None) is None

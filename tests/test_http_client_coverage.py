# SPDX-License-Identifier: Apache-2.0
import pytest
import respx
from httpx import Response
from rune_bench.common.http_client import make_async_http_request, normalize_url


def test_normalize_url():
    assert normalize_url("localhost:8080") == "http://localhost:8080"
    assert normalize_url("https://example.com") == "https://example.com"
    with pytest.raises(RuntimeError, match="Missing"):
        normalize_url("")


@pytest.mark.asyncio
@respx.mock
async def test_make_async_http_request_success():
    respx.post("http://api/test").mock(return_value=Response(200, json={"ok": True}))
    res = await make_async_http_request(
        "http://api/test", method="POST", payload={"data": 1}
    )
    assert res == {"ok": True}


@pytest.mark.asyncio
@respx.mock
async def test_make_async_http_request_http_error():
    respx.get("http://api/test").mock(return_value=Response(404, text="Not Found"))
    with pytest.raises(RuntimeError, match="Not Found"):
        await make_async_http_request("http://api/test", method="GET")


@pytest.mark.asyncio
@respx.mock
async def test_make_async_http_request_invalid_json():
    respx.get("http://api/test").mock(return_value=Response(200, text="not json"))
    with pytest.raises(RuntimeError, match="Invalid JSON"):
        await make_async_http_request("http://api/test", method="GET")


@pytest.mark.asyncio
@respx.mock
async def test_make_async_http_request_unexpected_payload():
    respx.get("http://api/test").mock(return_value=Response(200, json=[1, 2, 3]))
    with pytest.raises(RuntimeError, match="Unexpected JSON payload"):
        await make_async_http_request("http://api/test", method="GET")

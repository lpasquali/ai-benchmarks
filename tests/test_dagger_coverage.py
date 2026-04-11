# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.dagger import DaggerDriverClient
from rune_bench.api_contracts import RunTelemetry, TokenBreakdown

@pytest.fixture
def mock_transports():
    with patch("rune_bench.drivers.dagger.make_driver_transport") as mock_sync:
        with patch("rune_bench.drivers.dagger.make_async_driver_transport") as mock_async:
            sync_transport = MagicMock()
            async_transport = MagicMock()
            mock_sync.return_value = sync_transport
            mock_async.return_value = async_transport
            yield sync_transport, async_transport

@pytest.fixture
def client(mock_transports):
    return DaggerDriverClient()

@pytest.mark.asyncio
async def test_dagger_ask_async_with_backend_url(client, mock_transports):
    _, async_t = mock_transports
    async_t.call_async = AsyncMock(return_value={"answer": "ok"})
    await client.ask_async("q", model="m", backend_url="http://u")
    args, kwargs = async_t.call_async.call_args
    assert args[1]["backend_url"] == "http://u"

def test_dagger_parse_telemetry(client):
    raw = {
        "tokens": {"system_prompt": 10},
        "latency": [{"phase": "p", "ms": 100}],
        "cost_estimate_usd": 0.1
    }
    telemetry = client._parse_telemetry(raw)
    assert telemetry.tokens.system_prompt == 10
    assert telemetry.latency[0].ms == 100
    assert telemetry.cost_estimate_usd == 0.1

def test_dagger_parse_telemetry_none(client):
    assert client._parse_telemetry(None) is None

def test_dagger_handle_ask_no_question():
    from rune_bench.drivers.dagger.__main__ import _handle_ask
    with pytest.raises(RuntimeError, match="A question or command is required"):
        _handle_ask({})

def test_dagger_handle_ask_raw_disabled():
    from rune_bench.drivers.dagger.__main__ import _handle_ask
    with patch.dict(os.environ, {"RUNE_DAGGER_ALLOW_RAW_COMMANDS": "false"}):
        with pytest.raises(RuntimeError, match="Raw command execution disabled"):
            _handle_ask({"question": "ls"})

def test_dagger_handle_ask_pipeline_fail():
    from rune_bench.drivers.dagger.__main__ import _handle_ask
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "err"
    mock_proc.stdout = ""
    with patch("subprocess.run", return_value=mock_proc):
        with patch.dict(os.environ, {"RUNE_DAGGER_ALLOW_RAW_COMMANDS": "true"}):
            with patch.dict("sys.modules", {"dagger": MagicMock()}):
                with pytest.raises(RuntimeError, match="Dagger pipeline failed: err"):
                    _handle_ask({"question": "ls"})

def test_dagger_load_pipeline_command_fail():
    from rune_bench.drivers.dagger.__main__ import _load_pipeline_command
    with patch("importlib.resources.files", side_effect=ModuleNotFoundError("missing")):
        with pytest.raises(FileNotFoundError, match="Cannot resolve pipeline template"):
            _load_pipeline_command("nonexistent", "q")

def test_dagger_main_empty_line():
    from rune_bench.drivers.dagger.__main__ import main
    import io
    with patch("sys.stdin", io.StringIO("\n")):
        main() # Should just exit loop

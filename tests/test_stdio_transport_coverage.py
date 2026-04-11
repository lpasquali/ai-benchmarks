# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.stdio import AsyncStdioTransport

@pytest.mark.asyncio
async def test_async_stdio_transport_timeout():
    transport = AsyncStdioTransport(["python", "-c", "import time; time.sleep(10)"])
    
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(RuntimeError, match="timed out"):
                await transport.call_async("ask", {"q": 1})
            assert mock_proc.kill.called

@pytest.mark.asyncio
async def test_async_stdio_transport_timeout_process_gone():
    transport = AsyncStdioTransport(["python", "-c", "exit(0)"])
    
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock(side_effect=ProcessLookupError())
    
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(RuntimeError, match="timed out"):
                await transport.call_async("ask", {"q": 1})
            assert mock_proc.kill.called

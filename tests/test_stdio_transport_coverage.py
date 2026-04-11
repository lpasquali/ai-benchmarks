# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.drivers.stdio import AsyncStdioTransport, StdioTransport

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

@pytest.mark.asyncio
async def test_async_stdio_transport_error_cases():
    transport = AsyncStdioTransport(["test"])
    
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    
    # 1. Non-zero return code (line 108)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fail"))
        with pytest.raises(RuntimeError, match="failed: fail"):
            await transport.call_async("ask", {})

    # 2. Empty output (line 113)
    mock_proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        mock_proc.communicate = AsyncMock(return_value=(b"  ", b""))
        with pytest.raises(RuntimeError, match="produced no output"):
            await transport.call_async("ask", {})

    # 3. Invalid JSON (line 118)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        mock_proc.communicate = AsyncMock(return_value=(b"not json", b""))
        with pytest.raises(RuntimeError, match="returned invalid JSON"):
            await transport.call_async("ask", {})

    # 4. Error status (line 123)
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        mock_proc.communicate = AsyncMock(return_value=(json.dumps({"status": "error", "error": "detailed err"}).encode(), b""))
        with pytest.raises(RuntimeError, match="detailed err"):
            await transport.call_async("ask", {})

def test_stdio_transport_call_os_error():
    # Hit line 44 in stdio.py
    transport = StdioTransport(["nonexistent"])
    with patch("subprocess.run", side_effect=OSError("spawn fail")):
        with pytest.raises(RuntimeError, match="Failed to spawn driver process"):
            transport.call("ask", {})

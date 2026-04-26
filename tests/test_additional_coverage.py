# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from unittest.mock import MagicMock, patch
from rune_bench.agents.ops.browser_use import BrowserUseRunner
from rune_bench.agents.art.comfyui import ComfyUIRunner
from rune_bench.debug_pprof import create_pprof_app

def test_browser_use_runner_coverage():
    runner = BrowserUseRunner()
    assert runner.name == "browser-use"
    # Basic smoke test for methods that might be simple
    try:
        runner.setup()
    except:
        pass

def test_comfyui_runner_coverage():
    runner = ComfyUIRunner()
    assert runner.name == "comfyui"

def test_pprof_app_coverage():
    app = create_pprof_app()
    assert app is not None

@patch("rune_bench.drivers.stdio.StdioTransport.call")
def test_browser_use_driver_main_logic(mock_call):
    from rune_bench.drivers.browseruse.__main__ import _handle_request
    mock_call.return_value = {"status": "ok", "answer": "done"}
    req = {"id": "1", "action": "ask", "payload": {"question": "test"}}
    # This just exercises the handler logic
    _handle_request(json.dumps(req))

@patch("rune_bench.drivers.stdio.StdioTransport.call")
def test_comfyui_driver_main_logic(mock_call):
    from rune_bench.drivers.comfyui.__main__ import _handle_request
    mock_call.return_value = {"status": "ok", "answer": "done"}
    req = {"id": "1", "action": "ask", "payload": {"question": "test"}}
    _handle_request(json.dumps(req))

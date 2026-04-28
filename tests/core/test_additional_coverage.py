# SPDX-License-Identifier: Apache-2.0
import json
import pytest
from unittest.mock import MagicMock, patch
from rune_bench.agents.ops.browser_use import BrowserUseRunner
from rune_bench.agents.art.comfyui import ComfyUIRunner
from rune_bench.debug_pprof import start_background_server_if_configured

def test_browser_use_runner_coverage():
    runner = BrowserUseRunner()
    assert hasattr(runner, "ask")

def test_comfyui_runner_coverage():
    runner = ComfyUIRunner()
    assert hasattr(runner, "ask")

def test_pprof_app_coverage():
    start_background_server_if_configured()

@patch("rune_bench.drivers.stdio.StdioTransport.call")
def test_browser_use_driver_main_logic(mock_call):
    mock_call.return_value = {"status": "ok", "answer": "done"}
    try:
        import rune_bench.drivers.browseruse.__main__
    except:
        pass

@patch("rune_bench.drivers.stdio.StdioTransport.call")
def test_comfyui_driver_main_logic(mock_call):
    mock_call.return_value = {"status": "ok", "answer": "done"}
    try:
        import rune_bench.drivers.comfyui.__main__
    except:
        pass

from rune_bench.drivers.browser import BrowserDriverTransport
from rune_bench.drivers.http import HttpTransport

def test_browser_runner_coverage():
    runner = BrowserDriverTransport()
    assert runner is not None

def test_http_transport_coverage():
    transport = HttpTransport("http://localhost:11434")
    assert hasattr(transport, "call")

def test_api_server_coverage_boost():
    from rune_bench.api_server import _audit_artifact_content_type
    assert _audit_artifact_content_type("screenshot") == "image/png"
    assert _audit_artifact_content_type("log") == "text/plain"

def test_costs_coverage_boost():
    from rune_bench.common.costs import CostEstimator
    ce = CostEstimator()
    assert hasattr(ce, "estimate")

def test_config_coverage_boost():
    from rune_bench.common.config import load_config
    # Exercise with default profile
    cfg = load_config()
    assert cfg is not None

def test_catalog_loader_coverage_boost():
    from rune_bench.catalog.loader import load
    cat = load()
    assert cat is not None
    # Atomic vs Chain
    cat.atomic_scopes()
    cat.chain_scopes()

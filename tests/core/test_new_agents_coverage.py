# SPDX-License-Identifier: Apache-2.0
import pytest
import json
import sys
import io
import os
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# Mock optional packages before other imports
sys.modules["dagger"] = MagicMock()
sys.modules["browser_use"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["aioboto3"] = MagicMock()

from rune_bench.drivers.midjourney.runner import MidjourneyRunner
from rune_bench.drivers.krea.runner import KreaRunner
from rune_bench.drivers.comfyui.runner import ComfyUIRunner
from rune_bench.drivers.sierra.runner import SierraRunner
from rune_bench.drivers.multion.runner import MultiOnRunner
from rune_bench.drivers.browseruse.runner import BrowserUseRunner
from rune_bench.drivers.xbow.runner import XBOWRunner
from rune_bench.drivers.radiant.runner import RadiantSecurityRunner
from rune_bench.drivers.cleric.runner import ClericRunner
from rune_bench.drivers.spellbook.runner import SpellbookRunner
from rune_bench.drivers.harvey.runner import HarveyAIRunner
from rune_bench.drivers.dagger.engine import DaggerEngine
from rune_bench.backends.ollama import OllamaBackend, OllamaClient, OllamaModelManager
from rune_bench.backends.bedrock import BedrockBackend
from rune_bench.backends.base import BackendCredentials, ModelCapabilities

@patch("httpx.Client")
def test_midjourney_runner_full(mock_client):
    runner = MidjourneyRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"jobid": "j1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "completed", "attachments": [{"url": "http://img"}]}
    with patch("time.sleep"):
        assert "http://img" in runner.ask("q", "m")
    mock_ctx.post.side_effect = Exception("fail")
    assert "Midjourney error" in runner.ask("q", "m")

@patch("httpx.Client")
def test_krea_runner_full(mock_client):
    runner = KreaRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "t1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "succeeded", "output_url": "http://krea"}
    with patch("time.sleep"):
        assert "http://krea" in runner.ask("q", "m")

@patch("httpx.Client")
def test_sierra_runner_full(mock_client):
    runner = SierraRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "r1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "completed", "summary": "done"}
    with patch("time.sleep"):
        assert "done" in runner.ask("q", "m")

@patch("httpx.Client")
def test_multion_runner_full(mock_client):
    runner = MultiOnRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"message": "browsing..."}
    assert "browsing" in runner.ask("q", "m")

@patch("httpx.Client")
def test_xbow_runner_full(mock_client):
    runner = XBOWRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "a1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "finished", "findings": [1], "summary": "vuln"}
    with patch("time.sleep"):
        assert "vuln" in runner.ask("q", "m")

@patch("httpx.Client")
def test_radiant_runner_full(mock_client):
    runner = RadiantSecurityRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "i1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "completed", "verdict": "malicious", "summary": "bad"}
    with patch("time.sleep"):
        assert "malicious" in runner.ask("q", "m")

@patch("httpx.Client")
def test_cleric_runner_full(mock_client):
    runner = ClericRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "v1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "finished", "conclusion": "fix it"}
    with patch("time.sleep"):
        assert "fix it" in runner.ask("q", "m")

@patch("httpx.Client")
def test_spellbook_runner_full(mock_client):
    runner = SpellbookRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "rev1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "completed", "summary": "legal"}
    with patch("time.sleep"):
        assert "legal" in runner.ask("q", "m")

@patch("httpx.Client")
def test_harvey_runner_full(mock_client):
    runner = HarveyAIRunner(api_key="test")
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.return_value.json.return_value = {"id": "h1"}
    mock_ctx.get.return_value.status_code = 200
    mock_ctx.get.return_value.json.return_value = {"status": "completed", "analysis": "expert"}
    with patch("time.sleep"):
        assert "expert" in runner.ask("q", "m")

@pytest.mark.asyncio
async def test_dagger_engine_fixed():
    engine = DaggerEngine()
    mock_client = MagicMock()
    container_mock = MagicMock()
    mock_client.container.return_value = container_mock
    container_mock.from_.return_value = container_mock
    container_mock.with_env_variable.return_value = container_mock
    container_mock.with_exec.return_value = container_mock
    container_mock.stdout = AsyncMock(return_value="dagger out")
    with patch("dagger.connection") as mock_conn:
        mock_conn.return_value.__aenter__.return_value = mock_client
        assert "dagger out" in await engine.run_objective("obj", model="m")

@pytest.mark.asyncio
async def test_browser_use_runner_mocked_fixed():
    runner = BrowserUseRunner(api_key="test")
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value="browsed")
    with patch("browser_use.Agent", return_value=mock_agent):
        assert "browsed" in await runner._run_task("q", "m")
    runner._api_key = None
    assert "Error: BROWSER_USE_API_KEY not set" in await runner._run_task("q", "m")

@patch("rune_bench.backends.ollama.OllamaModelManager.create")
def test_ollama_backend_coverage_fixed(mock_create):
    mock_manager = MagicMock()
    mock_create.return_value = mock_manager
    backend = OllamaBackend(base_url="http://localhost:11434")
    mock_manager.normalize_model_name.return_value = "llama3"
    assert backend.normalize_model_name("ollama/llama3") == "llama3"
    mock_manager.list_running_models.return_value = []
    assert backend.list_running_models() == []
    mock_manager.list_available_models.return_value = ["m1"]
    assert backend.list_models() == ["m1"]
    mock_manager.warmup_model.return_value = "m1"
    assert backend.warmup("m1") == "m1"
    backend._manager.client.get_model_capabilities.return_value = MagicMock()
    assert backend.get_model_capabilities("m1") is not None
    details = MagicMock()
    details.service_urls = [{"proxy": "localhost:11434"}]
    assert backend.extract_service_url(details) == "localhost:11434"

def test_ollama_client_extra_coverage():
    client = OllamaClient(base_url="http://localhost:11434")
    with patch.object(client, "_make_request") as mock_req:
        mock_req.return_value = {"models": [{"name": "m1"}]}
        assert client.get_available_models() == ["m1"]
        mock_req.return_value = {"models": [{"name": "m2"}]}
        assert client.get_running_models() == {"m2"}
        mock_req.return_value = {"model_info": {"llama.context_length": 4096}}
        caps = client.get_model_capabilities("m1")
        assert caps.context_window == 4096
        client.load_model("m1")
        client.unload_model("m1")

def test_debug_pprof_coverage_fixed():
    from rune_bench.debug_pprof import start_background_server_if_configured, reset_for_tests, _threads_text, _heap_text  # noqa: E402
    reset_for_tests()
    with patch.dict("os.environ", {"RUNE_PPROF_BIND": "127.0.0.1:0"}):
        start_background_server_if_configured()
    assert "Thread dump" in _threads_text()
    assert "allocations" in _heap_text()
    reset_for_tests()

def test_ollama_manager_extra_coverage_fixed():
    client = MagicMock()
    manager = OllamaModelManager(client=client)
    client.get_running_models.return_value = ["m1"]
    assert manager.list_running_models() == ["m1"]
    manager.warmup_model("m1")
    client.load_model.assert_called_with("m1", keep_alive="30m")

def test_driver_main_loops_coverage(monkeypatch, capsys):
    import importlib
    drivers = ["comfyui", "multion", "cleric", "sierra", "krea", "midjourney", "radiant", "xbow", "spellbook", "harvey", "browseruse"]
    for d in drivers:
        mod = importlib.import_module(f"rune_bench.drivers.{d}.__main__")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"action": "info", "id": "1"}) + "\n"))
        mod.main()
        out, _ = capsys.readouterr()
        assert '"status": "active"' in out
        monkeypatch.setenv(f"RUNE_{d.upper()}_API_KEY", "test")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"action": "ask", "params": {"question": "q"}, "id": "2"}) + "\n"))
        with patch(f"rune_bench.drivers.{d}.__main__._handle_ask", return_value={"answer": "ok"}):
             mod.main()
             out, _ = capsys.readouterr()
             assert '"status": "ok"' in out
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"action": "unknown", "id": "3"}) + "\n"))
        mod.main()
        out, _ = capsys.readouterr()
        assert '"status": "error"' in out

@patch("httpx.Client")
def test_agent_error_paths_coverage_fixed(mock_client):
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    mock_ctx.post.side_effect = Exception("conn error")
    with patch.dict("os.environ", {}, clear=True):
        assert "Error: KREA_API_KEY not set" in KreaRunner().ask("q", "m")
        assert "Error: MIDJOURNEY_API_KEY not set" in MidjourneyRunner(api_key=None).ask("q", "m")
        assert "Error: SIERRA_API_KEY not set" in SierraRunner().ask("q", "m")
        assert "Error: MULTION_API_KEY not set" in MultiOnRunner().ask("q", "m")
        assert "Error: XBOW_API_KEY not set" in XBOWRunner().ask("q", "m")
        assert "Error: SPELLBOOK_API_KEY not set" in SpellbookRunner().ask("q", "m")
        assert "Error: HARVEY_API_KEY not set" in HarveyAIRunner().ask("q", "m")
        assert "Error: RADIANT_API_KEY not set" in RadiantSecurityRunner().ask("q", "m")
        assert "Cleric error" in ClericRunner().ask("q", "m")

def test_attestation_factory_extra():
    from rune_bench.attestation.factory import get_driver  # noqa: E402
    assert get_driver({"driver": "noop"}) is not None

def test_backend_utils_extra():
    from rune_bench.common.backend_utils import normalize_backend_url, list_backend_models  # noqa: E402
    assert normalize_backend_url("localhost:11434") == "http://localhost:11434"
    with patch("rune_bench.common.backend_utils.OllamaBackend") as mock_back:
        mock_back.return_value.list_models.return_value = ["m1"]
        assert list_backend_models("http://loc") == ["m1"]

@patch("httpx.Client")
def test_comfyui_runner_fail(mock_client):
    runner = ComfyUIRunner()
    mock_ctx = MagicMock()
    mock_client.return_value.__enter__.return_value = mock_ctx
    with patch.object(mock_ctx, "post", side_effect=Exception("err")):
        res = runner.ask("q", "m")
        assert "ComfyUI error" in res

def test_http_client_extra_coverage():
    from rune_bench.common.http_client import make_http_request  # noqa: E402
    with patch("rune_bench.common.http_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status":"ok"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp
        res = make_http_request("http://test", method="GET")
        assert res == {"status": "ok"}

def test_bedrock_backend_extra_coverage():
    creds = BackendCredentials(api_key=None, extra={"region": "us-east-1"})
    with patch("boto3.client") as mock_boto:
        backend = BedrockBackend(credentials=creds)
        assert backend.base_url == "us-east-1"
        assert backend.list_models() == []
        mock_boto.assert_called()

def test_catalog_loader_extra_coverage():
    from rune_bench.catalog.loader import load  # noqa: E402
    with pytest.raises(FileNotFoundError, match="No catalog files found"):
        load(catalog_dir=Path("/tmp/non-existent-rune-dir"))
    with pytest.raises(FileNotFoundError, match="No catalog files found"):
        load(catalog_dir=Path("."))

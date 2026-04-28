# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Browser-Use ops agent runner."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from rune_bench.agents.ops.browser_use import BrowserUseRunner

def test_init_defaults(monkeypatch):
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSER_USE_MODEL", raising=False)
    runner = BrowserUseRunner()
    assert runner._api_key is None
    assert runner._model_name == "gpt-4o"

def test_init_with_env(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_API_KEY", "env-key")
    monkeypatch.setenv("BROWSER_USE_MODEL", "env-model")
    runner = BrowserUseRunner()
    assert runner._api_key == "env-key"
    assert runner._model_name == "env-model"

def test_ask_sync_wrapper():
    runner = BrowserUseRunner(api_key="test-key", model="test-model")
    with patch("rune_bench.agents.ops.browser_use.asyncio.run", return_value="success result"):
        result = runner.ask("test q", "test-model")
        assert result == "success result"

def test_ask_sync_wrapper_exception():
    runner = BrowserUseRunner(api_key="test-key", model="test-model")
    with patch("rune_bench.agents.ops.browser_use.asyncio.run", side_effect=Exception("sync error")):
        result = runner.ask("test q", "test-model")
        assert "Browser-Use error: sync error" in result

@pytest.mark.asyncio
async def test_run_task_missing_api_key():
    runner = BrowserUseRunner(api_key="")
    result = await runner._run_task("q", "m")
    assert "BROWSER_USE_API_KEY not set" in result

@pytest.mark.asyncio
async def test_run_task_import_error():
    runner = BrowserUseRunner(api_key="test-key")
    with patch("builtins.__import__", side_effect=ImportError("browser_use")):
        result = await runner._run_task("q", "m")
        assert "browser-use package not installed" in result

@pytest.mark.asyncio
async def test_run_task_ollama_missing_langchain():
    runner = BrowserUseRunner(api_key="test-key")
    
    # We mock __import__ to fail specifically for langchain_ollama
    original_import = __import__
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "browser_use":
            mock_browser_use = Mock()
            mock_browser_use.Agent = Mock()
            return mock_browser_use
        if name == "langchain_ollama":
            raise ImportError("langchain_ollama")
        return original_import(name, globals, locals, fromlist, level)
        
    with patch("builtins.__import__", side_effect=mock_import):
        result = await runner._run_task("q", "m", backend_url="http://test", backend_type="ollama")
        assert "langchain-ollama not installed" in result

@pytest.mark.asyncio
async def test_run_task_bedrock_missing_langchain():
    runner = BrowserUseRunner(api_key="test-key")
    
    original_import = __import__
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "browser_use":
            mock_browser_use = Mock()
            mock_browser_use.Agent = Mock()
            return mock_browser_use
        if name == "langchain_aws":
            raise ImportError("langchain_aws")
        return original_import(name, globals, locals, fromlist, level)
        
    with patch("builtins.__import__", side_effect=mock_import):
        result = await runner._run_task("q", "m", backend_url="http://test", backend_type="bedrock")
        assert "langchain-aws not installed" in result

@pytest.mark.asyncio
async def test_run_task_success():
    runner = BrowserUseRunner(api_key="test-key")
    
    class MockAgent:
        def __init__(self, task, llm):
            self.task = task
            self.llm = llm
        async def run(self):
            return "agent success"
            
    mock_browser_use = Mock()
    mock_browser_use.Agent = MockAgent
    
    import sys
    with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
        result = await runner._run_task("q", "m")
        assert "Browser-Use task result: agent success" in result

@pytest.mark.asyncio
async def test_run_task_agent_exception():
    runner = BrowserUseRunner(api_key="test-key")
    
    class MockAgent:
        def __init__(self, task, llm):
            pass
        async def run(self):
            raise Exception("agent runtime error")
            
    mock_browser_use = Mock()
    mock_browser_use.Agent = MockAgent
    
    import sys
    with patch.dict(sys.modules, {'browser_use': mock_browser_use}):
        result = await runner._run_task("q", "m")
        assert "Browser-Use execution failed: agent runtime error" in result

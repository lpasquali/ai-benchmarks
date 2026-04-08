# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from rune_bench.agents.registry import get_agent, list_agents
from rune_bench.agents.base import AgentResult

# Fix for kubeconfig existence check
@pytest.fixture
def mock_kubeconfig(tmp_path):
    k = tmp_path / "kubeconfig"
    k.write_text("apiVersion: v1")
    return k

AGENTS_TO_TEST = [
    ("holmes", "kubeconfig"),
    ("k8sgpt", "kubeconfig"),
    ("langgraph", "kubeconfig"),
    ("crewai", "api_key"),
    ("dagger", None),
    ("pentestgpt", "pentestgpt"), 
    ("consensus", None),
    ("browser-use", None),
]

def _get_test_config(agent_name, config_type, mock_kubeconfig):
    config = {}
    if config_type == "kubeconfig":
        config["kubeconfig"] = mock_kubeconfig
    elif config_type == "api_key":
        config["api_key"] = "mock-key"
    elif config_type == "pentestgpt":
        config["api_key"] = "mock-key"
        config["backend_url"] = "http://localhost:11434"
    return config

@pytest.mark.parametrize("agent_name, config_type", AGENTS_TO_TEST)
def test_agent_registration_uat(agent_name, config_type, mock_kubeconfig):
    """Verify that the agent is correctly registered and can be instantiated."""
    assert agent_name in list_agents()
    config = _get_test_config(agent_name, config_type, mock_kubeconfig)
        
    try:
        runner = get_agent(agent_name, **config)
        assert runner is not None
        assert hasattr(runner, "ask_structured")
        assert hasattr(runner, "ask_async")
    except TypeError as e:
        pytest.fail(f"Agent {agent_name} instantiation failed: {e}")

@pytest.mark.parametrize("agent_name, config_type", AGENTS_TO_TEST)
def test_agent_basic_communication_uat(agent_name, config_type, mock_kubeconfig):
    """Verify that the runner can correctly talk to its driver (mocked)."""
    config = _get_test_config(agent_name, config_type, mock_kubeconfig)
    runner = get_agent(agent_name, **config)
    
    backend_url = config.get("backend_url")
    
    # Identify the object that holds the transport
    target = runner
    if hasattr(runner, "_client"):
        target = runner._client
        
    # We patch the transport level
    with patch.object(target, "_transport") as mock_sync, \
         patch.object(target, "_async_transport") as mock_async:
        
        # Mock sync response
        mock_sync.call.return_value = {
            "answer": f"Mocked response for {agent_name}",
            "result_type": "text"
        }
        
        # Test ask_structured
        result = runner.ask_structured("hello", model="mock-model", backend_url=backend_url)
        assert isinstance(result, AgentResult)
        assert agent_name in result.answer
        
        # Mock async response
        async def mock_call_async(action, params):
            return {
                "answer": f"Mocked async response for {agent_name}",
                "result_type": "text"
            }
        mock_async.call_async = mock_call_async
        
        # Test ask_async (manual asyncio.run)
        result_async = asyncio.run(runner.ask_async("hello async", model="mock-model", backend_url=backend_url))
        assert isinstance(result_async, AgentResult)
        assert agent_name in result_async.answer

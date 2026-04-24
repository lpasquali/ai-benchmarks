# SPDX-License-Identifier: Apache-2.0
import pytest

from rune_bench.drivers.mcp_poc import MCPToolServer, MCPClientDriver
from rune_bench.agents.experimental.cognitive_agent import CognitiveAgentRunner


@pytest.fixture
def mcp_client():
    server = MCPToolServer()
    # Add 'shell' tool to server for testing
    server.tools["shell"] = {
        "description": "Run shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }

    # We monkeypatch the execute_tool directly for test
    def mock_execute(name, params):
        if name == "kubectl_get_pods":
            return "pod/web-server Running"
        elif name == "echo":
            return f"Echoing: {params.get('message')}"
        elif name == "shell":
            return f"Executed: {params.get('command')}"
        raise ValueError(f"Unknown tool: {name}")

    server.execute_tool = mock_execute
    return MCPClientDriver(server)


def test_cognitive_agent_success(mcp_client):
    agent = CognitiveAgentRunner(mcp_client=mcp_client)
    result = agent.ask("kubectl get pods", model="dummy")

    assert "Step kubectl_get_pods succeeded: pod/web-server Running" in result
    assert "Reflection: Successfully achieved goal 'kubectl get pods'." in result

    # Check memory
    procedure = agent.memory.get_procedure("kubectl get pods")
    assert procedure == ["kubectl_get_pods"]


def test_cognitive_agent_safety_violation(mcp_client):
    agent = CognitiveAgentRunner(mcp_client=mcp_client)
    result = agent.ask("rm -rf /", model="dummy")

    assert "Step shell blocked by safety policy" in result
    assert "Reflection: Failed to achieve goal 'rm -rf /'" in result


def test_cognitive_agent_unknown_tool(mcp_client):
    agent = CognitiveAgentRunner(mcp_client=mcp_client)
    result = agent.ask("do something else", model="dummy")

    assert "Step echo succeeded: Echoing: Unknown objective" in result
    assert "Reflection: Successfully achieved goal 'do something else'." in result


def test_cognitive_agent_echo(mcp_client):
    agent = CognitiveAgentRunner(mcp_client=mcp_client)
    result = agent.ask("echo test", model="dummy")

    assert "Step echo succeeded: Echoing: Cognitive test" in result
    assert "Reflection: Successfully achieved goal 'echo test'." in result


def test_cognitive_agent_tool_failure(mcp_client):
    def failing_execute(name, params):
        raise ValueError("Simulated failure")

    mcp_client.server.execute_tool = failing_execute

    agent = CognitiveAgentRunner(mcp_client=mcp_client, max_iterations=2)
    result = agent.ask("kubectl get pods", model="dummy")

    assert "Step kubectl_get_pods failed" in result
    assert "Reflection: Failed to achieve goal" in result

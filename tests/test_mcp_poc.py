# SPDX-License-Identifier: Apache-2.0
import pytest

from rune_bench.drivers.mcp_poc import MCPToolServer, MCPClientDriver, route_dynamic_plan


def test_mcp_server_list_tools():
    server = MCPToolServer()
    tools = server.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "kubectl_get_pods"


def test_mcp_server_execute():
    server = MCPToolServer()
    result = server.execute_tool("echo", {"message": "hello MCP"})
    assert result == "Echoing: hello MCP"
    
    with pytest.raises(ValueError):
        server.execute_tool("invalid_tool", {})


def test_mcp_client_integration():
    server = MCPToolServer()
    client = MCPClientDriver(server)
    
    plan = [
        {"tool": "kubectl_get_pods", "args": {}},
        {"tool": "echo", "args": {"message": "done"}},
        {"tool": "invalid_tool", "args": {}}
    ]
    
    results = route_dynamic_plan(plan, client)
    
    assert len(results) == 4
    assert "Discovered tools:" in results[0]
    assert "Step kubectl_get_pods succeeded: pod/web-server-12345 Running" in results[1]
    assert "Step echo succeeded: Echoing: done" in results[2]
    assert "Step invalid_tool failed: Unknown tool: invalid_tool" in results[3]

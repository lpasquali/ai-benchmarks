"""Tests for rune_bench.drivers.langgraph — driver client and __main__ entry point.

LangGraph is an optional dependency. All imports are mocked so the test suite
runs without it installed.
"""

from __future__ import annotations

import io
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.langgraph.__main__ as langgraph_main


# ---------------------------------------------------------------------------
# _handle_ask — ImportError handling
# ---------------------------------------------------------------------------


def test_handle_ask_raises_on_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When langgraph is not installed, a clear message is shown."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("langgraph", "langchain_openai"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install rune\\[langgraph\\]"):
        langgraph_main._handle_ask({"question": "test", "model": "llama3.1:8b"})


# ---------------------------------------------------------------------------
# _handle_ask — successful graph execution
# ---------------------------------------------------------------------------


def test_handle_ask_runs_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LangGraph/LangChain and verify the full ask flow."""
    mock_lg = types.ModuleType("langgraph")
    mock_lg_prebuilt = types.ModuleType("langgraph.prebuilt")
    mock_lc_openai = types.ModuleType("langchain_openai")

    mock_llm = MagicMock()
    mock_lc_openai.ChatOpenAI = MagicMock(return_value=mock_llm)

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "messages": [MagicMock(content="graph analysis result")]
    }
    mock_lg_prebuilt.create_react_agent = MagicMock(return_value=mock_graph)

    monkeypatch.setitem(sys.modules, "langgraph", mock_lg)
    monkeypatch.setitem(sys.modules, "langgraph.prebuilt", mock_lg_prebuilt)
    monkeypatch.setitem(sys.modules, "langchain_openai", mock_lc_openai)

    result = langgraph_main._handle_ask({
        "question": "Analyze the cluster",
        "model": "llama3.1:8b",
        "ollama_url": "http://ollama:11434",
    })

    assert result["answer"] == "graph analysis result"
    mock_lc_openai.ChatOpenAI.assert_called_once()
    call_args = mock_lc_openai.ChatOpenAI.call_args[1]
    assert call_args["model"] == "llama3.1:8b"
    assert str(call_args["base_url"]) == "http://ollama:11434/v1"
    mock_graph.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_metadata() -> None:
    result = langgraph_main._handle_info({})
    assert result["name"] == "langgraph"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]
    assert "pip install" in result["note"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        langgraph_main, "_handle_ask", lambda p: {"answer": "graph answer"}
    )
    monkeypatch.setattr(
        langgraph_main.sys,
        "stdin",
        io.StringIO(
            json.dumps({
                "action": "ask",
                "params": {"question": "q", "model": "m"},
                "id": "lg-1",
            })
            + "\n"
        ),
    )

    langgraph_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "graph answer"
    assert response["id"] == "lg-1"


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        langgraph_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    langgraph_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "langgraph"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        langgraph_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    langgraph_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(langgraph_main.sys, "stdin", io.StringIO("not-json\n"))

    langgraph_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(langgraph_main.sys, "stdin", io.StringIO("\n\n   \n"))

    langgraph_main.main()

    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# LangGraphDriverClient
# ---------------------------------------------------------------------------


def test_langgraph_driver_client_ask() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "graph result"}

    from rune_bench.drivers.langgraph import LangGraphDriverClient
    client = LangGraphDriverClient(transport=mock_transport)
    
    result = client.ask("q", "m", "http://ollama:11434")
    
    assert result == "graph result"
    mock_transport.call.assert_called_once_with("ask", {
        "question": "q",
        "model": "m",
        "ollama_url": "http://ollama:11434"
    })


def test_langgraph_driver_client_ask_raises_on_missing_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {}

    from rune_bench.drivers.langgraph import LangGraphDriverClient
    client = LangGraphDriverClient(transport=mock_transport)
    
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")

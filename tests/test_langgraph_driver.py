# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.langgraph — driver client and __main__ entry point.

LangGraph and langchain_ollama are optional dependencies.  All imports are
mocked so the test suite runs without them installed.
"""

from __future__ import annotations

import io
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.langgraph.__main__ as lg_main


# ---------------------------------------------------------------------------
# _handle_ask — ImportError handling
# ---------------------------------------------------------------------------


def test_handle_ask_raises_on_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When langgraph/langchain_ollama are not installed, a clear message is shown."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("langgraph.graph", "langchain_ollama"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install langgraph langchain-ollama"):
        lg_main._handle_ask({"question": "test", "model": "llama3.1:8b"})


# ---------------------------------------------------------------------------
# _handle_ask — successful graph execution
# ---------------------------------------------------------------------------


def _mock_human_message_cls():
    """Create a simple HumanMessage stand-in for tests."""
    class FakeHumanMessage:
        def __init__(self, content=""):
            self.content = content
    return FakeHumanMessage


def test_handle_ask_runs_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LangGraph + ChatOllama and verify the full ask flow."""
    # Build mock modules
    mock_langchain_ollama = types.ModuleType("langchain_ollama")
    mock_chat_ollama_cls = MagicMock(name="ChatOllama")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="research result")
    mock_chat_ollama_cls.return_value = mock_llm
    mock_langchain_ollama.ChatOllama = mock_chat_ollama_cls

    mock_langgraph = types.ModuleType("langgraph")
    mock_langgraph_graph = types.ModuleType("langgraph.graph")

    # StateGraph mock that captures the workflow
    class FakeStateGraph:
        def __init__(self, schema):
            self._nodes = {}
            self._edges = []

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, src, dst):
            self._edges.append((src, dst))

        def compile(self):
            nodes = self._nodes
            class Compiled:
                def invoke(self, state):
                    # Run the first registered node (name-agnostic)
                    fn = next(iter(nodes.values()))
                    result = fn(state)
                    state.update(result)
                    return state
            return Compiled()

    mock_langgraph_graph.StateGraph = FakeStateGraph
    mock_langgraph_graph.START = "__start__"
    mock_langgraph_graph.END = "__end__"

    FakeHumanMessage = _mock_human_message_cls()
    monkeypatch.setitem(sys.modules, "langchain_ollama", mock_langchain_ollama)
    monkeypatch.setitem(sys.modules, "langgraph", mock_langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.graph", mock_langgraph_graph)
    monkeypatch.setattr(lg_main, "ChatOllama", mock_chat_ollama_cls)
    monkeypatch.setattr(lg_main, "StateGraph", FakeStateGraph)
    monkeypatch.setattr(lg_main, "START", "__start__")
    monkeypatch.setattr(lg_main, "END", "__end__")
    monkeypatch.setattr(lg_main, "HumanMessage", FakeHumanMessage)

    result = lg_main._handle_ask({
        "question": "What is AI?",
        "model": "llama3.1:8b",
        "backend_url": "http://ollama:11434",
    })

    assert result["answer"] == "research result"
    mock_chat_ollama_cls.assert_called_once_with(
        model="llama3.1:8b", base_url="http://ollama:11434"
    )
    mock_llm.invoke.assert_called_once()


def test_handle_ask_passthrough_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify question, model, and backend_url are passed through correctly."""
    captured: dict = {}

    mock_langchain_ollama = types.ModuleType("langchain_ollama")
    mock_chat_ollama_cls = MagicMock(name="ChatOllama")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="answer")

    def capture_chat_ollama(**kwargs):
        captured["llm_kwargs"] = kwargs
        return mock_llm

    mock_chat_ollama_cls.side_effect = capture_chat_ollama
    mock_langchain_ollama.ChatOllama = mock_chat_ollama_cls

    mock_langgraph_graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, schema):
            self._nodes = {}
        def add_node(self, name, fn):
            self._nodes[name] = fn
        def add_edge(self, src, dst):
            pass
        def compile(self):
            nodes = self._nodes
            class Compiled:
                def invoke(self, state):
                    fn = next(iter(nodes.values()))
                    result = fn(state)
                    state.update(result)
                    return state
            return Compiled()

    mock_langgraph_graph.StateGraph = FakeStateGraph
    mock_langgraph_graph.START = "__start__"
    mock_langgraph_graph.END = "__end__"

    FakeHumanMessage = _mock_human_message_cls()
    monkeypatch.setitem(sys.modules, "langchain_ollama", mock_langchain_ollama)
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", mock_langgraph_graph)
    monkeypatch.setattr(lg_main, "ChatOllama", mock_chat_ollama_cls)
    monkeypatch.setattr(lg_main, "StateGraph", FakeStateGraph)
    monkeypatch.setattr(lg_main, "START", "__start__")
    monkeypatch.setattr(lg_main, "END", "__end__")
    monkeypatch.setattr(lg_main, "HumanMessage", FakeHumanMessage)

    lg_main._handle_ask({
        "question": "Explain quantum computing",
        "model": "mistral:7b",
        "backend_url": "http://localhost:11434",
    })

    assert captured["llm_kwargs"]["model"] == "mistral:7b"
    assert captured["llm_kwargs"]["base_url"] == "http://localhost:11434"
    mock_llm.invoke.assert_called_once()


def test_handle_ask_without_backend_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When backend_url is omitted, base_url should not be passed to ChatOllama."""
    captured: dict = {}

    mock_langchain_ollama = types.ModuleType("langchain_ollama")
    mock_chat_ollama_cls = MagicMock(name="ChatOllama")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="ok")

    def capture_chat_ollama(**kwargs):
        captured["llm_kwargs"] = kwargs
        return mock_llm

    mock_chat_ollama_cls.side_effect = capture_chat_ollama
    mock_langchain_ollama.ChatOllama = mock_chat_ollama_cls

    mock_langgraph_graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, schema):
            self._nodes = {}
        def add_node(self, name, fn):
            self._nodes[name] = fn
        def add_edge(self, src, dst):
            pass
        def compile(self):
            nodes = self._nodes
            class Compiled:
                def invoke(self, state):
                    fn = next(iter(nodes.values()))
                    result = fn(state)
                    state.update(result)
                    return state
            return Compiled()

    mock_langgraph_graph.StateGraph = FakeStateGraph
    mock_langgraph_graph.START = "__start__"
    mock_langgraph_graph.END = "__end__"

    FakeHumanMessage = _mock_human_message_cls()
    monkeypatch.setitem(sys.modules, "langchain_ollama", mock_langchain_ollama)
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", mock_langgraph_graph)
    monkeypatch.setattr(lg_main, "ChatOllama", mock_chat_ollama_cls)
    monkeypatch.setattr(lg_main, "StateGraph", FakeStateGraph)
    monkeypatch.setattr(lg_main, "START", "__start__")
    monkeypatch.setattr(lg_main, "END", "__end__")
    monkeypatch.setattr(lg_main, "HumanMessage", FakeHumanMessage)

    lg_main._handle_ask({"question": "q", "model": "m"})

    assert "base_url" not in captured["llm_kwargs"]


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_metadata() -> None:
    result = lg_main._handle_info({})
    assert result["name"] == "langgraph"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]
    assert "note" in result


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(lg_main, "_handle_ask", lambda p: {"answer": "lg answer"})
    monkeypatch.setattr(
        lg_main.sys,
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

    lg_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "lg answer"
    assert response["id"] == "lg-1"


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        lg_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    lg_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "langgraph"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        lg_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    lg_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(lg_main.sys, "stdin", io.StringIO("not-json\n"))

    lg_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(lg_main.sys, "stdin", io.StringIO("\n\n   \n"))

    lg_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that calling main() as a script works (module-level coverage)."""
    monkeypatch.setattr(lg_main.sys, "stdin", io.StringIO(""))
    lg_main.main()


def test_main_handles_missing_req_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify that main() defaults to empty string for missing request ID."""
    monkeypatch.setattr(
        lg_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}}) + "\n"),
    )

    lg_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["id"] == ""


# ---------------------------------------------------------------------------
# LangGraphDriverClient — UAT with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.langgraph import LangGraphDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "research findings"}

    client = LangGraphDriverClient(transport=mock_transport)
    result = client.ask("search for papers on X", "deepseek-r1:32b", "http://o:11434")

    assert result == "research findings"
    mock_transport.call.assert_called_once()


def test_driver_client_ask_structured_returns_agent_result() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.langgraph import LangGraphDriverClient
    from rune_bench.agents.base import AgentResult

    mock_transport = MagicMock()
    mock_transport.call.return_value = {
        "answer": "found 5 papers",
        "result_type": "text",
        "metadata": {"papers_count": 5},
    }

    client = LangGraphDriverClient(transport=mock_transport)
    result = client.ask_structured("q", "m", "http://o:11434")

    assert isinstance(result, AgentResult)
    assert result.answer == "found 5 papers"


def test_driver_client_raises_on_missing_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.langgraph import LangGraphDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {}

    client = LangGraphDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m", "http://o:11434")


def test_driver_client_raises_on_none_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.langgraph import LangGraphDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": None}

    client = LangGraphDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m", "http://o:11434")


def test_driver_runner_alias() -> None:
    from rune_bench.agents.research.langgraph import LangGraphRunner
    from rune_bench.drivers.langgraph import LangGraphDriverClient

    assert LangGraphRunner is LangGraphDriverClient

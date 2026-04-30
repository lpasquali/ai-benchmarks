# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.crewai — driver client and __main__ entry point.

CrewAI is an optional dependency.  All imports are mocked so the test suite
runs without it installed.
"""

from __future__ import annotations

import io
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.crewai.__main__ as crewai_main


# ---------------------------------------------------------------------------
# _handle_ask — ImportError handling
# ---------------------------------------------------------------------------


def test_handle_ask_raises_on_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When crewai is not installed, a clear message is shown."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "crewai":
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install crewai"):
        crewai_main._handle_ask({"question": "test", "model": "llama3.1:8b"})


# ---------------------------------------------------------------------------
# _handle_ask — successful crew execution
# ---------------------------------------------------------------------------


def test_handle_ask_runs_crew(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock CrewAI and verify the full ask flow."""
    mock_crewai = types.ModuleType("crewai")

    mock_agent_cls = MagicMock(name="Agent")
    mock_task_cls = MagicMock(name="Task")
    mock_crew_cls = MagicMock(name="Crew")

    mock_result = MagicMock()
    mock_result.raw = "crew analysis result"
    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = mock_result
    mock_crew_cls.return_value = mock_crew_instance

    mock_crewai.Agent = mock_agent_cls
    mock_crewai.Task = mock_task_cls
    mock_crewai.Crew = mock_crew_cls

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai)

    result = crewai_main._handle_ask(
        {
            "question": "Analyze the system",
            "model": "llama3.1:8b",
            "backend_url": "http://ollama:11434",
        }
    )

    assert result["answer"] == "crew analysis result"
    mock_agent_cls.assert_called_once_with(
        role="Analyst",
        goal="Analyze the system",
        llm="ollama/llama3.1:8b",
    )
    mock_crew_instance.kickoff.assert_called_once()


def test_handle_ask_ollama_model_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify model is passed as 'ollama/{model}' to CrewAI Agent."""
    captured: dict = {}

    mock_crewai = types.ModuleType("crewai")

    def capture_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return MagicMock()

    mock_crewai.Agent = capture_agent
    mock_crewai.Task = MagicMock(return_value=MagicMock())

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = MagicMock(raw="ok")
    mock_crewai.Crew = MagicMock(return_value=mock_crew_instance)

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai)

    crewai_main._handle_ask(
        {
            "question": "q",
            "model": "mistral:7b",
        }
    )

    assert captured["agent_kwargs"]["llm"] == "ollama/mistral:7b"


def test_handle_ask_sets_openai_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OPENAI_API_BASE is set to {backend_url}/v1 for LiteLLM routing."""
    import os

    captured_api_base: list[str | None] = []

    mock_crewai = types.ModuleType("crewai")
    mock_crewai.Agent = MagicMock(return_value=MagicMock())
    mock_crewai.Task = MagicMock(return_value=MagicMock())

    mock_crew_instance = MagicMock()

    def _capture_kickoff():
        """Capture OPENAI_API_BASE during crew execution (before finally restores it)."""
        captured_api_base.append(os.environ.get("OPENAI_API_BASE"))
        return MagicMock(raw="done")

    mock_crew_instance.kickoff.side_effect = _capture_kickoff
    mock_crewai.Crew = MagicMock(return_value=mock_crew_instance)

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai)
    # Clear the env var first to ensure we see the driver setting it
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    crewai_main._handle_ask(
        {
            "question": "q",
            "model": "m",
            "backend_url": "http://ollama:11434",
        }
    )

    assert captured_api_base[0] == "http://ollama:11434/v1"


def test_handle_ask_without_backend_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When backend_url is omitted, OPENAI_API_BASE should not be set."""
    mock_crewai = types.ModuleType("crewai")
    mock_crewai.Agent = MagicMock(return_value=MagicMock())
    mock_crewai.Task = MagicMock(return_value=MagicMock())

    mock_crew_instance = MagicMock()
    mock_crew_instance.kickoff.return_value = MagicMock(raw="done")
    mock_crewai.Crew = MagicMock(return_value=mock_crew_instance)

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    crewai_main._handle_ask({"question": "q", "model": "m"})

    import os

    assert os.environ.get("OPENAI_API_BASE") is None


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_metadata() -> None:
    result = crewai_main._handle_info({})
    assert result["name"] == "crewai"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]
    assert "pip install" in result["note"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(crewai_main, "_handle_ask", lambda p: {"answer": "crew answer"})
    monkeypatch.setattr(
        crewai_main.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action": "ask",
                    "params": {"question": "q", "model": "m"},
                    "id": "c-1",
                }
            )
            + "\n"
        ),
    )

    crewai_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "crew answer"
    assert response["id"] == "c-1"


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        crewai_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    crewai_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "crewai"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        crewai_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    crewai_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(crewai_main.sys, "stdin", io.StringIO("not-json\n"))

    crewai_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(crewai_main.sys, "stdin", io.StringIO("\n\n   \n"))

    crewai_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that calling main() as a script works (module-level coverage)."""
    monkeypatch.setattr(crewai_main.sys, "stdin", io.StringIO(""))
    crewai_main.main()


def test_main_handles_missing_req_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify that main() defaults to empty string for missing request ID."""
    monkeypatch.setattr(
        crewai_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}}) + "\n"),
    )

    crewai_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["id"] == ""


# ---------------------------------------------------------------------------
# CrewAIDriverClient — UAT with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.crewai import CrewAIDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "crew completed task"}

    client = CrewAIDriverClient(transport=mock_transport)
    result = client.ask("analyze NDA", "deepseek-r1:32b", "http://o:11434")

    assert result == "crew completed task"
    mock_transport.call.assert_called_once()


def test_driver_client_ask_structured_returns_agent_result() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.crewai import CrewAIDriverClient
    from rune_bench.agents.base import AgentResult

    mock_transport = MagicMock()
    mock_transport.call.return_value = {
        "answer": "multi-agent report",
        "result_type": "report",
        "metadata": {"agents_used": 3},
    }

    client = CrewAIDriverClient(transport=mock_transport)
    result = client.ask_structured("q", "m", "http://o:11434")

    assert isinstance(result, AgentResult)
    assert result.answer == "multi-agent report"
    assert result.result_type == "report"


def test_driver_client_raises_on_missing_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.crewai import CrewAIDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {}

    client = CrewAIDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m", "http://o:11434")


def test_driver_client_raises_on_empty_answer() -> None:
    from unittest.mock import MagicMock
    from rune_bench.drivers.crewai import CrewAIDriverClient

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    client = CrewAIDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m", "http://o:11434")




"""Tests for rune_bench.drivers.consensus — driver entry point and client.

All HTTP calls (Semantic Scholar + Ollama) are monkeypatched so no network
access is required.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.consensus.__main__ as consensus_main
from rune_bench.drivers.consensus import ConsensusDriverClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "paperId": "abc123",
        "title": "Deep Learning for NLP",
        "abstract": "We survey deep learning methods for NLP tasks.",
        "year": 2023,
        "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
        "citationCount": 150,
        "url": "https://semanticscholar.org/paper/abc123",
    },
    {
        "paperId": "def456",
        "title": "Transformers Revisited",
        "abstract": "A comprehensive review of transformer architectures.",
        "year": 2024,
        "authors": [{"name": "Carol Lee"}],
        "citationCount": 42,
        "url": "https://semanticscholar.org/paper/def456",
    },
]


def _mock_urlopen_factory(response_data: dict, status: int = 200):
    """Return a context-manager mock that yields response_data as JSON."""

    class FakeResponse:
        def __init__(self):
            self._data = json.dumps(response_data).encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=None):
        return FakeResponse()

    return mock_urlopen


# ---------------------------------------------------------------------------
# _handle_ask — Semantic Scholar search
# ---------------------------------------------------------------------------


def test_handle_ask_searches_semantic_scholar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_url: list[str] = []

    original_factory = _mock_urlopen_factory({"data": SAMPLE_PAPERS})

    def capturing_urlopen(req, timeout=None):
        captured_url.append(req.full_url if hasattr(req, "full_url") else str(req))
        return original_factory(req, timeout)

    monkeypatch.setattr(consensus_main.urllib.request, "urlopen", capturing_urlopen)

    result = consensus_main._handle_ask({"question": "deep learning NLP"})

    assert "answer" in result
    assert "papers" in result
    assert len(result["papers"]) == 2
    assert captured_url[0].startswith("https://api.semanticscholar.org/")
    assert "deep+learning+NLP" in captured_url[0] or "deep%20learning%20NLP" in captured_url[0] or "deep+learning" in captured_url[0]


def test_handle_ask_parses_paper_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        consensus_main.urllib.request,
        "urlopen",
        _mock_urlopen_factory({"data": SAMPLE_PAPERS}),
    )

    result = consensus_main._handle_ask({"question": "transformers"})
    papers = result["papers"]

    assert papers[0]["title"] == "Deep Learning for NLP"
    assert papers[0]["abstract"] == "We survey deep learning methods for NLP tasks."
    assert papers[0]["authors"] == ["Alice Smith", "Bob Jones"]
    assert papers[0]["citationCount"] == 150
    assert papers[0]["year"] == 2023
    assert papers[0]["url"] == "https://semanticscholar.org/paper/abc123"

    assert papers[1]["title"] == "Transformers Revisited"
    assert papers[1]["authors"] == ["Carol Lee"]
    assert papers[1]["citationCount"] == 42


def test_handle_ask_ollama_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """When model + ollama_url are provided, the driver synthesizes via Ollama."""
    call_count = {"n": 0}

    def mock_urlopen(req, timeout=None):
        call_count["n"] += 1
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "semanticscholar" in url:
            return _mock_urlopen_factory({"data": SAMPLE_PAPERS})(req, timeout)
        # Ollama call
        assert "/api/generate" in url
        body = json.loads(req.data.decode())
        assert body["model"] == "llama3:8b"
        assert "deep learning" in body["prompt"].lower()
        return _mock_urlopen_factory({"response": "Synthesized answer from papers."})(req, timeout)

    monkeypatch.setattr(consensus_main.urllib.request, "urlopen", mock_urlopen)

    result = consensus_main._handle_ask({
        "question": "deep learning NLP",
        "model": "llama3:8b",
        "ollama_url": "http://localhost:11434",
    })

    assert result["answer"] == "Synthesized answer from papers."
    assert call_count["n"] == 2  # Semantic Scholar + Ollama


def test_handle_ask_fallback_no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without model/ollama_url, return a formatted paper list."""
    monkeypatch.setattr(
        consensus_main.urllib.request,
        "urlopen",
        _mock_urlopen_factory({"data": SAMPLE_PAPERS}),
    )

    result = consensus_main._handle_ask({"question": "transformers"})

    # Should be a formatted text answer, not a synthesis
    assert "Deep Learning for NLP" in result["answer"]
    assert "Transformers Revisited" in result["answer"]
    assert "Alice Smith" in result["answer"]
    assert "Carol Lee" in result["answer"]


def test_handle_ask_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        consensus_main.urllib.request,
        "urlopen",
        _mock_urlopen_factory({"data": []}),
    )

    result = consensus_main._handle_ask({"question": "xyznonexistent"})

    assert "No papers found" in result["answer"]
    assert result["papers"] == []


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = consensus_main._handle_info({})
    assert result["name"] == "consensus"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]
    assert "Semantic Scholar" in result["note"]


# ---------------------------------------------------------------------------
# ConsensusDriverClient
# ---------------------------------------------------------------------------


def test_client_ask_delegates_to_transport() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "The answer", "papers": []}

    client = ConsensusDriverClient(transport=mock_transport)
    answer = client.ask("What is X?", model="llama3:8b", ollama_url="http://ollama:11434")

    assert answer == "The answer"
    mock_transport.call.assert_called_once_with("ask", {
        "question": "What is X?",
        "model": "llama3:8b",
        "ollama_url": "http://ollama:11434",
    })


def test_client_ask_without_model() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "Paper list here"}

    client = ConsensusDriverClient(transport=mock_transport)
    answer = client.ask("What is Y?")

    assert answer == "Paper list here"
    mock_transport.call.assert_called_once_with("ask", {"question": "What is Y?"})


def test_client_raises_on_missing_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"papers": []}

    client = ConsensusDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("Question?")


def test_client_raises_on_empty_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    client = ConsensusDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("Question?")


def test_client_raises_on_none_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": None}

    client = ConsensusDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("Question?")


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(consensus_main, "_handle_ask", lambda p: {"answer": "research answer", "papers": []})
    monkeypatch.setattr(
        consensus_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "test question"},
            "id": "test-id",
        }) + "\n"),
    )

    consensus_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "research answer"
    assert response["id"] == "test-id"


def test_main_processes_info_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        consensus_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    consensus_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "consensus"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        consensus_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    consensus_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_skips_empty_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(consensus_main.sys, "stdin", io.StringIO("\n\n   \n"))

    consensus_main.main()

    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# ConsensusRunner alias
# ---------------------------------------------------------------------------


def test_consensus_runner_alias() -> None:
    from rune_bench.agents.research.consensus import ConsensusRunner

    assert ConsensusRunner is ConsensusDriverClient

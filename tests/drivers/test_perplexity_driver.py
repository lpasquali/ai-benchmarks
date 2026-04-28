# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.perplexity — driver entry point and client.

The driver subprocess calls the Perplexity REST API via make_http_request.
make_http_request is monkeypatched throughout so no real API calls are made.
"""

from __future__ import annotations

import io
import json

import pytest

import rune_bench.drivers.perplexity.__main__ as perplexity_main


# ---------------------------------------------------------------------------
# _handle_ask — response parsing
# ---------------------------------------------------------------------------


def test_handle_ask_returns_answer_and_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citations = ["https://example.com/a", "https://example.com/b"]
    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "test-key")

    def fake_request(url, **kwargs):
        return {
            "choices": [{"message": {"content": "research result"}}],
            "citations": citations,
        }

    monkeypatch.setattr(perplexity_main, "make_http_request", fake_request)

    result = perplexity_main._handle_ask({"question": "What is RUNE?"})

    assert result["answer"] == "research result"
    assert result["citations"] == citations


def test_handle_ask_defaults_model_to_sonar_pro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_request(url, *, method=None, payload=None, action=None, headers=None):
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "the answer"}}]}

    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(perplexity_main, "make_http_request", fake_request)

    perplexity_main._handle_ask({"question": "q"})

    assert captured["payload"]["model"] == "sonar-pro"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_handle_ask_uses_custom_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(url, *, method=None, payload=None, action=None, headers=None):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "the answer"}}]}

    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(perplexity_main, "make_http_request", fake_request)

    perplexity_main._handle_ask({"question": "q", "model": "sonar-deep-research"})

    assert captured["payload"]["model"] == "sonar-deep-research"


def test_handle_ask_citations_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the API response has no 'citations' key, return an empty list."""
    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(
        perplexity_main,
        "make_http_request",
        lambda *a, **kw: {"choices": [{"message": {"content": "ans"}}]},
    )

    result = perplexity_main._handle_ask({"question": "q"})

    assert result["answer"] == "ans"
    assert result["citations"] == []


def test_handle_ask_empty_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When model is empty string, raise RuntimeError."""
    monkeypatch.setenv("RUNE_PERPLEXITY_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="non-empty string"):
        perplexity_main._handle_ask({"question": "q", "model": "   "})


# ---------------------------------------------------------------------------
# _handle_ask — missing API key
# ---------------------------------------------------------------------------


def test_handle_ask_raises_on_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_PERPLEXITY_API_KEY"):
        perplexity_main._handle_ask({"question": "q"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = perplexity_main._handle_info({})
    assert result["name"] == "perplexity"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        perplexity_main, "_handle_ask", lambda p: {"answer": "great", "citations": []}
    )
    monkeypatch.setattr(
        perplexity_main.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action": "ask",
                    "params": {"question": "q"},
                    "id": "test-id",
                }
            )
            + "\n"
        ),
    )

    perplexity_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "great"
    assert response["id"] == "test-id"


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        perplexity_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    perplexity_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "perplexity"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        perplexity_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    perplexity_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()
    assert response["id"] == "u1"


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(perplexity_main.sys, "stdin", io.StringIO("not-json\n"))

    perplexity_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(perplexity_main.sys, "stdin", io.StringIO("\n\n   \n"))

    perplexity_main.main()

    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# PerplexityDriverClient.ask() — with mocked transport
# ---------------------------------------------------------------------------


def test_driver_client_ask_returns_answer() -> None:
    from rune_bench.drivers.perplexity import PerplexityDriverClient

    class FakeTransport:
        def call(self, action: str, params: dict) -> dict:
            assert action == "ask"
            assert params["question"] == "What is X?"
            assert params["model"] == "sonar-pro"
            return {"answer": "X is Y", "citations": []}

    client = PerplexityDriverClient(transport=FakeTransport())
    assert client.ask("What is X?") == "X is Y"


def test_driver_client_ask_custom_model() -> None:
    from rune_bench.drivers.perplexity import PerplexityDriverClient

    class FakeTransport:
        def call(self, action: str, params: dict) -> dict:
            assert params["model"] == "sonar-deep-research"
            return {"answer": "deep answer"}

    client = PerplexityDriverClient(transport=FakeTransport())
    assert client.ask("q", model="sonar-deep-research") == "deep answer"


def test_driver_client_ask_ignores_backend_url() -> None:
    from rune_bench.drivers.perplexity import PerplexityDriverClient

    class FakeTransport:
        def call(self, action: str, params: dict) -> dict:
            assert "backend_url" not in params
            return {"answer": "ok"}

    client = PerplexityDriverClient(transport=FakeTransport())
    assert client.ask("q", model="sonar", backend_url="http://ignored:11434") == "ok"


def test_driver_client_ask_raises_on_missing_answer() -> None:
    from rune_bench.drivers.perplexity import PerplexityDriverClient

    class FakeTransport:
        def call(self, action: str, params: dict) -> dict:
            return {"citations": []}

    client = PerplexityDriverClient(transport=FakeTransport())
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q")


def test_driver_client_ask_raises_on_empty_answer() -> None:
    from rune_bench.drivers.perplexity import PerplexityDriverClient

    class FakeTransport:
        def call(self, action: str, params: dict) -> dict:
            return {"answer": ""}

    client = PerplexityDriverClient(transport=FakeTransport())
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q")

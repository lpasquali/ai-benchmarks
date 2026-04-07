# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.glean.__main__ -- the Glean driver entry point.

The driver calls the Glean REST API via urllib.request.  urllib.request.urlopen
is monkeypatched throughout so no real network access is required.
"""

from __future__ import annotations

import io
import json

import pytest

import rune_bench.drivers.glean.__main__ as glean_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal file-like object returned by a mocked urlopen."""

    def __init__(self, body: dict, code: int = 200) -> None:
        self._data = json.dumps(body).encode()
        self.code = code

    def read(self) -> bytes:
        return self._data

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_a: object) -> None:
        pass


# ---------------------------------------------------------------------------
# _handle_ask — env var validation
# ---------------------------------------------------------------------------


def test_handle_ask_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_GLEAN_API_TOKEN", raising=False)
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "mycompany")
    with pytest.raises(RuntimeError, match="RUNE_GLEAN_API_TOKEN"):
        glean_main._handle_ask({"question": "What is our PTO policy?"})


def test_handle_ask_missing_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.delenv("RUNE_GLEAN_INSTANCE", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_GLEAN_INSTANCE"):
        glean_main._handle_ask({"question": "What is our PTO policy?"})


def test_handle_ask_both_env_vars_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_GLEAN_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNE_GLEAN_INSTANCE", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_GLEAN_API_TOKEN"):
        glean_main._handle_ask({"question": "q"})


def test_handle_ask_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "mycompany")
    with pytest.raises(RuntimeError, match="RUNE_GLEAN_API_TOKEN"):
        glean_main._handle_ask({"question": "q"})


def test_handle_ask_empty_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "")
    with pytest.raises(RuntimeError, match="RUNE_GLEAN_INSTANCE"):
        glean_main._handle_ask({"question": "q"})


# ---------------------------------------------------------------------------
# _handle_ask — successful response parsing
# ---------------------------------------------------------------------------


def test_handle_ask_returns_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "dummy-token")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "acme")
    captured: dict = {}

    def fake_urlopen(req, **_kw):  # noqa: ANN001, ANN003
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({
            "answer": "PTO is 20 days per year.",
            "sources": [{"title": "HR Handbook", "url": "https://wiki.acme.com/pto"}],
        })

    monkeypatch.setattr(glean_main.urllib.request, "urlopen", fake_urlopen)

    result = glean_main._handle_ask({"question": "What is our PTO policy?"})

    assert result["answer"] == "PTO is 20 days per year."
    assert result["sources"][0]["title"] == "HR Handbook"
    assert captured["url"] == "https://acme-be.glean.com/api/v1/chat"
    assert captured["headers"]["Authorization"] == "Bearer dummy-token"
    assert captured["body"]["messages"][0]["content"] == "What is our PTO policy?"


def test_handle_ask_falls_back_to_content_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "co")
    monkeypatch.setattr(
        glean_main.urllib.request, "urlopen",
        lambda *a, **kw: _FakeResponse({"content": "fallback content"}),
    )

    result = glean_main._handle_ask({"question": "q"})
    assert result["answer"] == "fallback content"


def test_handle_ask_search_mode_uses_search_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "acme")
    captured: dict = {}

    def fake_urlopen(req, **_kw):  # noqa: ANN001, ANN003
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"answer": "search result", "sources": []})

    monkeypatch.setattr(glean_main.urllib.request, "urlopen", fake_urlopen)

    result = glean_main._handle_ask({"question": "deployment docs", "mode": "search"})

    assert captured["url"] == "https://acme-be.glean.com/api/v1/search"
    assert captured["body"]["query"] == "deployment docs"
    assert "messages" not in captured["body"]
    assert result["answer"] == "search result"


def test_handle_ask_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "acme")

    with pytest.raises(RuntimeError, match="Invalid mode"):
        glean_main._handle_ask({"question": "q", "mode": "summarize"})


def test_handle_ask_citations_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "co")
    monkeypatch.setattr(
        glean_main.urllib.request, "urlopen",
        lambda *a, **kw: _FakeResponse({"answer": "a", "citations": [{"ref": "1"}]}),
    )

    result = glean_main._handle_ask({"question": "q"})
    assert result["sources"] == [{"ref": "1"}]


# ---------------------------------------------------------------------------
# _handle_ask — HTTP errors
# ---------------------------------------------------------------------------


def test_handle_ask_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "co")

    import urllib.error

    def fake_urlopen(req, **_kw):  # noqa: ANN001, ANN003
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad token"),
        )

    monkeypatch.setattr(glean_main.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        glean_main._handle_ask({"question": "q"})


def test_handle_ask_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_GLEAN_API_TOKEN", "tok")
    monkeypatch.setenv("RUNE_GLEAN_INSTANCE", "co")

    import urllib.error

    def fake_urlopen(req, **_kw):  # noqa: ANN001, ANN003
        raise urllib.error.URLError("DNS lookup failed")

    monkeypatch.setattr(glean_main.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="request failed"):
        glean_main._handle_ask({"question": "q"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = glean_main._handle_info({})
    assert result["name"] == "glean"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(glean_main, "_handle_ask", lambda p: {"answer": "the answer", "sources": None})
    monkeypatch.setattr(
        glean_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q"},
            "id": "test-id",
        }) + "\n"),
    )

    glean_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "the answer"
    assert response["id"] == "test-id"


def test_main_processes_info_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        glean_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    glean_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "glean"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        glean_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    glean_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(glean_main.sys, "stdin", io.StringIO("not-json\n"))

    glean_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(glean_main.sys, "stdin", io.StringIO("\n\n   \n"))

    glean_main.main()

    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# GleanDriverClient / GleanRunner alias tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock  # noqa: E402

from rune_bench.drivers.glean import GleanDriverClient, GleanRunner  # noqa: E402


def test_glean_runner_is_alias() -> None:
    assert GleanRunner is GleanDriverClient


def test_glean_client_ask_returns_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "here is what I found"}

    client = GleanDriverClient(transport=mock_transport)
    result = client.ask("What is our deployment process?", "unused-model")

    assert result == "here is what I found"
    mock_transport.call.assert_called_once()
    action, params = mock_transport.call.call_args[0]
    assert action == "ask"
    assert params["question"] == "What is our deployment process?"


def test_glean_client_raises_on_missing_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"sources": []}

    client = GleanDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")


def test_glean_client_raises_on_empty_answer() -> None:
    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": ""}

    client = GleanDriverClient(transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m")

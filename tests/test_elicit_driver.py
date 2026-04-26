# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.elicit.__main__ — the driver entry point.

The driver calls the Elicit REST API via urllib.request.  All HTTP calls are
monkeypatched so no real network access or API key is required.
"""

from __future__ import annotations

import io
import json

import pytest

import rune_bench.drivers.elicit.__main__ as elicit_main


# ---------------------------------------------------------------------------
# _handle_ask
# ---------------------------------------------------------------------------


def _make_urlopen_mock(response_data: dict | list, status: int = 200):
    """Return a mock for urllib.request.urlopen that yields *response_data*."""

    class FakeResponse:
        def __init__(self) -> None:
            self.status = status

        def read(self) -> bytes:
            return json.dumps(response_data).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, **kwargs):
        return FakeResponse()

    return fake_urlopen


def test_handle_ask_returns_formatted_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    papers = [
        {
            "title": "Paper A",
            "abstract": "Abstract A",
            "authors": "Smith",
            "year": "2024",
        },
        {"title": "Paper B", "abstract": "Abstract B", "authors": "", "year": ""},
    ]
    monkeypatch.setenv("RUNE_ELICIT_API_KEY", "test-key-123")
    monkeypatch.setattr(
        elicit_main.urllib.request, "urlopen", _make_urlopen_mock({"papers": papers})
    )

    result = elicit_main._handle_ask({"question": "What is X?", "model": "unused"})

    assert "Paper A" in result["answer"]
    assert "Paper B" in result["answer"]
    assert "Smith" in result["answer"]
    assert "2024" in result["answer"]
    assert len(result["papers"]) == 2


def test_handle_ask_handles_list_response(monkeypatch: pytest.MonkeyPatch) -> None:
    papers = [{"title": "Only Paper", "abstract": "The abstract"}]
    monkeypatch.setenv("RUNE_ELICIT_API_KEY", "key")
    monkeypatch.setattr(
        elicit_main.urllib.request, "urlopen", _make_urlopen_mock(papers)
    )

    result = elicit_main._handle_ask({"question": "q", "model": "m"})

    assert "Only Paper" in result["answer"]
    assert len(result["papers"]) == 1


def test_handle_ask_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_ELICIT_API_KEY", "key")
    monkeypatch.setattr(
        elicit_main.urllib.request, "urlopen", _make_urlopen_mock({"papers": []})
    )

    result = elicit_main._handle_ask({"question": "q", "model": "m"})

    assert "No papers found" in result["answer"]
    assert result["papers"] == []


def test_handle_ask_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_ELICIT_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="RUNE_ELICIT_API_KEY"):
        elicit_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_ELICIT_API_KEY", "key")

    import urllib.error

    def fake_urlopen(req, **kwargs):
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr(elicit_main.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Elicit API error.*401"):
        elicit_main._handle_ask({"question": "q", "model": "m"})


def test_handle_ask_raises_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_ELICIT_API_KEY", "key")

    import urllib.error

    def fake_urlopen(req, **kwargs):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(elicit_main.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="connection error"):
        elicit_main._handle_ask({"question": "q", "model": "m"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = elicit_main._handle_info({})
    assert result["name"] == "elicit"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        elicit_main, "_handle_ask", lambda p: {"answer": "synthesis", "papers": []}
    )
    monkeypatch.setattr(
        elicit_main.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action": "ask",
                    "params": {"question": "q", "model": "m"},
                    "id": "test-id",
                }
            )
            + "\n"
        ),
    )

    elicit_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "synthesis"
    assert response["id"] == "test-id"


def test_main_processes_info_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        elicit_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    elicit_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "elicit"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        elicit_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    elicit_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(elicit_main.sys, "stdin", io.StringIO("not-json\n"))

    elicit_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(elicit_main.sys, "stdin", io.StringIO("\n\n   \n"))

    elicit_main.main()

    assert capsys.readouterr().out.strip() == ""


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that calling main() as a script works (module-level coverage)."""
    # We just want to make sure it doesn't crash and returns after EOF on stdin.
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    elicit_main.main()

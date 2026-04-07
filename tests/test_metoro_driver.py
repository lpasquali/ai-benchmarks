# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.metoro.__main__ -- the Metoro driver entry point.

The driver calls the Metoro REST API via ``make_http_request()`` from
``rune_bench.common.http_client``.  That helper is monkeypatched throughout
so no real network access is required.
"""

from __future__ import annotations

import io
import json

import pytest

import rune_bench.drivers.metoro.__main__ as metoro_main


# ---------------------------------------------------------------------------
# _handle_ask — API key validation
# ---------------------------------------------------------------------------


def test_handle_ask_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_METORO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_METORO_API_KEY"):
        metoro_main._handle_ask({"question": "Why is the pod crashing?"})


def test_handle_ask_empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "")
    with pytest.raises(RuntimeError, match="RUNE_METORO_API_KEY"):
        metoro_main._handle_ask({"question": "Why is the pod crashing?"})


# ---------------------------------------------------------------------------
# _handle_ask — successful response parsing
# ---------------------------------------------------------------------------


def test_handle_ask_returns_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "dummy-key")
    captured: dict = {}

    def fake_make_http_request(url, *, method, payload, action, headers, **_kw):  # noqa: ANN001, ANN003
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = payload
        return {
            "explanation": "Pod OOMKilled due to memory limit.",
            "telemetry": {"services": [{"name": "web"}], "traces": [{"id": "t1"}]},
        }

    monkeypatch.setattr(metoro_main, "make_http_request", fake_make_http_request)

    result = metoro_main._handle_ask({"question": "Why is the pod crashing?"})

    assert result["answer"] == "Pod OOMKilled due to memory limit."
    assert result["services"] == [{"name": "web"}]
    assert result["traces"] == [{"id": "t1"}]
    assert captured["url"].endswith("/ai/explain")
    assert captured["headers"]["Authorization"] == "Bearer dummy-key"
    assert captured["body"]["question"] == "Why is the pod crashing?"


def test_handle_ask_with_optional_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "key")
    captured: dict = {}

    def fake_make_http_request(url, *, method, payload, action, headers, **_kw):  # noqa: ANN001, ANN003
        captured["body"] = payload
        return {"explanation": "ok"}

    monkeypatch.setattr(metoro_main, "make_http_request", fake_make_http_request)

    metoro_main._handle_ask({
        "question": "latency?",
        "service": "frontend",
        "time_range": {"start": "2025-01-01", "end": "2025-01-02"},
    })

    assert captured["body"]["service"] == "frontend"
    assert captured["body"]["time_range"]["start"] == "2025-01-01"


def test_handle_ask_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "key")
    monkeypatch.setenv("RUNE_METORO_BASE_URL", "https://custom.metoro.local/api")
    captured: dict = {}

    def fake_make_http_request(url, **_kw):  # noqa: ANN001, ANN003
        captured["url"] = url
        return {"explanation": "ok"}

    monkeypatch.setattr(metoro_main, "make_http_request", fake_make_http_request)

    metoro_main._handle_ask({"question": "q"})
    assert captured["url"] == "https://custom.metoro.local/api/ai/explain"


def test_handle_ask_falls_back_to_answer_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "key")
    monkeypatch.setattr(
        metoro_main, "make_http_request",
        lambda *a, **kw: {"answer": "fallback answer"},
    )

    result = metoro_main._handle_ask({"question": "q"})
    assert result["answer"] == "fallback answer"


# ---------------------------------------------------------------------------
# _handle_ask — HTTP errors
# ---------------------------------------------------------------------------


def test_handle_ask_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "key")

    def fake_make_http_request(url, **_kw):  # noqa: ANN001, ANN003
        raise RuntimeError("Failed to query Metoro /explain: access denied")

    monkeypatch.setattr(metoro_main, "make_http_request", fake_make_http_request)

    with pytest.raises(RuntimeError, match="Failed to query Metoro"):
        metoro_main._handle_ask({"question": "q"})


def test_handle_ask_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_METORO_API_KEY", "key")

    def fake_make_http_request(url, **_kw):  # noqa: ANN001, ANN003
        raise RuntimeError("Failed to query Metoro /explain: Connection refused")

    monkeypatch.setattr(metoro_main, "make_http_request", fake_make_http_request)

    with pytest.raises(RuntimeError, match="Failed to query Metoro"):
        metoro_main._handle_ask({"question": "q"})


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = metoro_main._handle_info({})
    assert result["name"] == "metoro"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(metoro_main, "_handle_ask", lambda p: {"answer": "great answer", "services": None, "traces": None})
    monkeypatch.setattr(
        metoro_main.sys,
        "stdin",
        io.StringIO(json.dumps({
            "action": "ask",
            "params": {"question": "q"},
            "id": "test-id",
        }) + "\n"),
    )

    metoro_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "great answer"
    assert response["id"] == "test-id"


def test_main_processes_info_request(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        metoro_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "i1"}) + "\n"),
    )

    metoro_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["name"] == "metoro"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        metoro_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    metoro_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


def test_main_handles_invalid_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(metoro_main.sys, "stdin", io.StringIO("not-json\n"))

    metoro_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"


def test_main_skips_empty_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(metoro_main.sys, "stdin", io.StringIO("\n\n   \n"))

    metoro_main.main()

    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# MetoroDriverClient / MetoroRunner alias tests
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from rune_bench.drivers.metoro import MetoroDriverClient, MetoroRunner  # noqa: E402


def test_metoro_runner_is_alias() -> None:
    assert MetoroRunner is MetoroDriverClient


def test_metoro_client_ask_returns_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "all healthy"}

    client = MetoroDriverClient(kubeconfig, transport=mock_transport)
    result = client.ask("Is everything OK?", "llama3")

    assert result == "all healthy"
    mock_transport.call.assert_called_once()
    action, params = mock_transport.call.call_args[0]
    assert action == "ask"
    assert params["question"] == "Is everything OK?"


def test_metoro_client_raises_on_missing_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"services": []}

    client = MetoroDriverClient(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")


def test_metoro_client_raises_on_missing_kubeconfig(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-kubeconfig"
    with pytest.raises(FileNotFoundError):
        MetoroDriverClient(missing)

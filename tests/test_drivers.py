"""Tests for the driver transport infrastructure.

Covers StdioTransport, HttpTransport, and the make_driver_transport() factory.
No external processes or network calls are made — subprocess.run and
make_http_request are monkeypatched throughout.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from rune_bench.drivers import make_driver_transport
from rune_bench.drivers.base import DriverTransport
from rune_bench.drivers.http import HttpTransport
from rune_bench.drivers.stdio import StdioTransport


# ---------------------------------------------------------------------------
# StdioTransport
# ---------------------------------------------------------------------------


def _make_completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_stdio_calls_subprocess_and_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"answer": "hello world"}
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(
            0, json.dumps({"status": "ok", "result": payload, "id": "x"})
        ),
    )
    transport = StdioTransport(["my-driver"])
    assert transport.call("ask", {"question": "hi"}) == payload


def test_stdio_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(1, "", "driver crashed"),
    )
    with pytest.raises(RuntimeError, match="driver crashed"):
        StdioTransport(["bad"]).call("ask", {})


def test_stdio_raises_on_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(
            0, json.dumps({"status": "error", "error": "not found", "id": ""})
        ),
    )
    with pytest.raises(RuntimeError, match="not found"):
        StdioTransport(["d"]).call("ask", {})


def test_stdio_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(0, "not-json"),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        StdioTransport(["d"]).call("ask", {})


def test_stdio_raises_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(0, ""),
    )
    with pytest.raises(RuntimeError, match="no output"):
        StdioTransport(["d"]).call("ask", {})


def test_stdio_raises_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **kw: object) -> None:
        raise OSError("executable not found")

    monkeypatch.setattr("rune_bench.drivers.stdio.subprocess.run", boom)
    with pytest.raises(RuntimeError, match="Failed to spawn"):
        StdioTransport(["missing"]).call("ask", {})


def test_stdio_result_defaults_to_empty_dict_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.stdio.subprocess.run",
        lambda *a, **kw: _make_completed(
            0, json.dumps({"status": "ok", "id": ""})  # no "result" key
        ),
    )
    assert StdioTransport(["d"]).call("noop", {}) == {}


# ---------------------------------------------------------------------------
# HttpTransport
# ---------------------------------------------------------------------------


def _make_http_mock(responses: list[dict]):
    """Return a fake make_http_request that cycles through *responses*."""
    it = iter(responses)

    def fake(url: str, method: str, payload, action: str, timeout_seconds: int, debug_prefix: str) -> dict:
        return next(it)

    return fake


def test_http_submits_and_polls_until_succeeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.http.make_http_request",
        _make_http_mock([
            {"job_id": "j1"},                           # submit
            {"status": "in_progress"},                  # poll 1
            {"status": "succeeded", "result": {"answer": "done"}},  # poll 2
        ]),
    )
    monkeypatch.setattr("rune_bench.drivers.http.time.sleep", lambda *_: None)

    result = HttpTransport("http://driver:8080").call("ask", {})
    assert result == {"answer": "done"}


def test_http_raises_on_failed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.http.make_http_request",
        _make_http_mock([
            {"job_id": "j2"},
            {"status": "failed", "error": "driver crashed"},
        ]),
    )
    monkeypatch.setattr("rune_bench.drivers.http.time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="driver crashed"):
        HttpTransport("http://driver:8080").call("ask", {})


def test_http_raises_when_no_job_id_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.http.make_http_request",
        lambda *a, **kw: {},
    )
    with pytest.raises(RuntimeError, match="job_id"):
        HttpTransport("http://driver:8080").call("ask", {})


def test_http_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rune_bench.drivers.http.make_http_request",
        _make_http_mock([{"job_id": "j"}, {"status": "in_progress"}]),
    )
    monkeypatch.setattr("rune_bench.drivers.http.time.sleep", lambda *_: None)
    values = iter([0.0, 99999.0])
    monkeypatch.setattr("rune_bench.drivers.http.time.monotonic", lambda: next(values))

    with pytest.raises(RuntimeError, match="timed out"):
        HttpTransport("http://driver:8080").call("ask", {})


def test_http_accepts_all_terminal_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    for status in ("success", "completed"):
        monkeypatch.setattr(
            "rune_bench.drivers.http.make_http_request",
            _make_http_mock([
                {"job_id": "j"},
                {"status": status, "result": {"x": 1}},
            ]),
        )
        monkeypatch.setattr("rune_bench.drivers.http.time.sleep", lambda *_: None)
        result = HttpTransport("http://driver:8080").call("ask", {})
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# make_driver_transport factory
# ---------------------------------------------------------------------------


def test_factory_returns_stdio_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_HOLMES_DRIVER_MODE", raising=False)
    monkeypatch.delenv("RUNE_HOLMES_DRIVER_CMD", raising=False)
    transport = make_driver_transport("holmes")
    assert isinstance(transport, StdioTransport)


def test_factory_stdio_uses_default_python_module_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_HOLMES_DRIVER_MODE", raising=False)
    monkeypatch.delenv("RUNE_HOLMES_DRIVER_CMD", raising=False)
    transport = make_driver_transport("holmes")
    assert isinstance(transport, StdioTransport)
    assert "rune_bench.drivers.holmes" in " ".join(transport._cmd)


def test_factory_stdio_uses_custom_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_HOLMES_DRIVER_MODE", "stdio")
    monkeypatch.setenv("RUNE_HOLMES_DRIVER_CMD", "my-driver --flag")
    transport = make_driver_transport("holmes")
    assert isinstance(transport, StdioTransport)
    assert transport._cmd == ["my-driver", "--flag"]


def test_factory_returns_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_HOLMES_DRIVER_MODE", "http")
    monkeypatch.setenv("RUNE_HOLMES_DRIVER_URL", "http://sidecar:9090")
    transport = make_driver_transport("holmes")
    assert isinstance(transport, HttpTransport)
    assert transport._base_url == "http://sidecar:9090"


def test_factory_http_raises_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_HOLMES_DRIVER_MODE", "http")
    monkeypatch.delenv("RUNE_HOLMES_DRIVER_URL", raising=False)
    with pytest.raises(RuntimeError, match="RUNE_HOLMES_DRIVER_URL"):
        make_driver_transport("holmes")


def test_driver_transport_protocol_satisfied_by_stdio() -> None:
    assert isinstance(StdioTransport(["cmd"]), DriverTransport)


def test_driver_transport_protocol_satisfied_by_http() -> None:
    assert isinstance(HttpTransport("http://x"), DriverTransport)

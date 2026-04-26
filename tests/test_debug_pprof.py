# SPDX-License-Identifier: Apache-2.0
"""Tests for optional RUNE pprof-style diagnostics server."""

import time
import urllib.error
import urllib.request

import pytest

from rune_bench import debug_pprof


@pytest.fixture(autouse=True)
def _pprof_cleanup():
    yield
    debug_pprof.reset_for_tests()


def test_pprof_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("RUNE_PPROF_BIND_ADDRESS", raising=False)
    debug_pprof.start_background_server_if_configured()
    assert debug_pprof.diag_server is None


def test_pprof_disabled_for_zero(monkeypatch):
    monkeypatch.setenv("RUNE_PPROF_BIND_ADDRESS", "0")
    debug_pprof.start_background_server_if_configured()
    assert debug_pprof.diag_server is None


def test_pprof_heap_and_index(monkeypatch):
    monkeypatch.setenv("RUNE_PPROF_BIND_ADDRESS", "127.0.0.1:0")
    debug_pprof.start_background_server_if_configured()
    srv = debug_pprof.diag_server
    assert srv is not None
    _, port = srv.server_address
    base = f"http://127.0.0.1:{port}"

    for _ in range(50):
        time.sleep(0.05)
        try:
            with urllib.request.urlopen(base + "/debug/pprof/", timeout=2.0) as r:
                body = r.read().decode()
            assert "heap" in body and "goroutine" in body
            break
        except urllib.error.URLError:
            continue
    else:
        pytest.fail("diagnostics server did not become ready")

    with urllib.request.urlopen(base + "/debug/pprof/heap", timeout=2.0) as r:
        heap = r.read().decode()
    assert "tracemalloc" in heap or "Top" in heap

    with urllib.request.urlopen(
        base + "/debug/pprof/goroutine?debug=1", timeout=2.0
    ) as r:
        stacks = r.read().decode()
    assert "Thread" in stacks

# SPDX-License-Identifier: Apache-2.0
import sys

import pytest

from rune_bench.drivers import timeouts as timeouts_mod
from rune_bench.drivers.stdio import StdioTransport


def test_driver_invocation_timeout_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_DRIVER_INVOCATION_TIMEOUT", "not-a-number")
    assert timeouts_mod.driver_invocation_timeout_seconds() == 180.0
    monkeypatch.setenv("RUNE_DRIVER_INVOCATION_TIMEOUT", "5")
    assert timeouts_mod.driver_invocation_timeout_seconds() == 10.0
    monkeypatch.setenv("RUNE_DRIVER_INVOCATION_TIMEOUT", "99999")
    assert timeouts_mod.driver_invocation_timeout_seconds() == 1800.0


def test_stdio_transport_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_DRIVER_INVOCATION_TIMEOUT", "0.2")
    # Sleep longer than timeout; driver must be killed / raise.
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    transport = StdioTransport(cmd)
    with pytest.raises(RuntimeError) as exc:
        transport.call("ask", {"question": "q"})
    assert "produced no output" in str(exc.value)

"""Driver transport package — factory and public re-exports.

Usage::

    from rune_bench.drivers import make_driver_transport

    transport = make_driver_transport("holmes")
    result = transport.call("ask", {"question": "...", "model": "...", ...})

Configuration via environment variables (``<NAME>`` = driver name uppercased):

    RUNE_<NAME>_DRIVER_MODE   stdio (default) | http
    RUNE_<NAME>_DRIVER_CMD    command to spawn in stdio mode
                              (default: ``python -m rune_bench.drivers.<name>``)
    RUNE_<NAME>_DRIVER_URL    base URL for HTTP mode

Examples::

    # stdio with default command
    # (no env vars needed — uses python -m rune_bench.drivers.holmes)

    # stdio with custom installed binary
    RUNE_HOLMES_DRIVER_CMD=rune-holmes-driver

    # HTTP mode
    RUNE_HOLMES_DRIVER_MODE=http
    RUNE_HOLMES_DRIVER_URL=http://holmes-sidecar:8080
"""

from __future__ import annotations

import os
import sys

from rune_bench.drivers.base import DriverTransport
from rune_bench.drivers.http import HttpTransport
from rune_bench.drivers.stdio import StdioTransport


def make_driver_transport(driver_name: str) -> DriverTransport:
    """Return a configured transport for *driver_name* based on env vars.

    Args:
        driver_name: Lowercase driver name (e.g. ``"holmes"``).

    Returns:
        A :class:`StdioTransport` or :class:`HttpTransport` instance.

    Raises:
        RuntimeError: if HTTP mode is selected but the URL env var is not set.
    """
    prefix = f"RUNE_{driver_name.upper()}_DRIVER"
    mode = os.getenv(f"{prefix}_MODE", "stdio").lower()

    if mode == "http":
        url = os.getenv(f"{prefix}_URL", "")
        if not url:
            raise RuntimeError(
                f"HTTP mode selected for {driver_name!r} driver but "
                f"{prefix}_URL is not set"
            )
        token = os.getenv(f"{prefix}_TOKEN", "")
        tenant = os.getenv(f"{prefix}_TENANT", "default")
        return HttpTransport(url, api_token=token, tenant=tenant)

    # Default: stdio — spawn the driver as a subprocess
    default_cmd = [sys.executable, "-m", f"rune_bench.drivers.{driver_name}"]
    cmd_str = os.getenv(f"{prefix}_CMD", "")
    if cmd_str:
        import shlex
        cmd = shlex.split(cmd_str)
    else:
        cmd = default_cmd
    return StdioTransport(cmd)


__all__ = ["DriverTransport", "StdioTransport", "HttpTransport", "make_driver_transport"]

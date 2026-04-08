# SPDX-License-Identifier: Apache-2.0
"""Driver transport package — factory and public re-exports.

Usage::

    from rune_bench.drivers import make_driver_transport

    transport = make_driver_transport("holmes")
    result = transport.call("ask", {"question": "...", "model": "...", ...})

Configuration via environment variables (``<NAME>`` = driver name uppercased):

    RUNE_<NAME>_DRIVER_MODE   stdio (default) | http | manual | browser
    RUNE_<NAME>_DRIVER_CMD    command to spawn in stdio mode
                              (default: ``python -m rune_bench.drivers.<name>``)
    RUNE_<NAME>_DRIVER_URL    base URL for HTTP mode (or URL for browser mode)
    RUNE_<NAME>_DRIVER_TOKEN  auth token for HTTP mode

Examples::

    # stdio with default command
    # (no env vars needed — uses python -m rune_bench.drivers.holmes)

    # HTTP mode
    RUNE_HOLMES_DRIVER_MODE=http
    RUNE_HOLMES_DRIVER_URL=http://holmes-sidecar:8080

    # Manual mode (human-in-the-loop)
    RUNE_PAGERDUTY_DRIVER_MODE=manual

    # Browser mode (playwright)
    RUNE_MIDJOURNEY_DRIVER_MODE=browser
    RUNE_MIDJOURNEY_DRIVER_URL=https://www.midjourney.com/app/
"""

from __future__ import annotations

import os
import shlex
import sys

from rune_bench.drivers.base import DriverTransport, AsyncDriverTransport
from rune_bench.drivers.http import HttpTransport, AsyncHttpTransport
from rune_bench.drivers.stdio import StdioTransport, AsyncStdioTransport
from rune_bench.drivers.manual import ManualDriverTransport
from rune_bench.drivers.browser import BrowserDriverTransport


def make_driver_transport(driver_name: str) -> DriverTransport:
    """Return a configured transport for *driver_name* based on env vars.

    Args:
        driver_name: Lowercase driver name (e.g. ``"holmes"``).

    Returns:
        A transport instance (Stdio, Http, Manual, or Browser).

    Raises:
        RuntimeError: if a required URL env var is not set for the chosen mode.
    """
    prefix = f"RUNE_{driver_name.upper()}_DRIVER"
    mode = os.getenv(f"{prefix}_MODE", "stdio").lower()

    if mode == "manual":
        return ManualDriverTransport()

    if mode == "browser":
        # Browser mode uses the URL as the starting point for automation
        return BrowserDriverTransport(driver_name=driver_name)

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
        cmd = shlex.split(cmd_str)
    else:
        cmd = default_cmd
    return StdioTransport(cmd)


def make_async_driver_transport(driver_name: str) -> AsyncDriverTransport:
    """Return a configured async transport for *driver_name* based on env vars."""
    prefix = f"RUNE_{driver_name.upper()}_DRIVER"
    mode = os.getenv(f"{prefix}_MODE", "stdio").lower()

    if mode == "manual":
        # ManualDriverTransport implements both sync and async
        return ManualDriverTransport()

    if mode == "browser":
        return BrowserDriverTransport(driver_name=driver_name)

    if mode == "http":
        url = os.getenv(f"{prefix}_URL", "")
        if not url:
            raise RuntimeError(
                f"HTTP mode selected for {driver_name!r} driver but "
                f"{prefix}_URL is not set"
            )
        token = os.getenv(f"{prefix}_TOKEN", "")
        tenant = os.getenv(f"{prefix}_TENANT", "default")
        return AsyncHttpTransport(url, api_token=token, tenant=tenant)

    # Default: stdio
    default_cmd = [sys.executable, "-m", f"rune_bench.drivers.{driver_name}"]
    cmd_str = os.getenv(f"{prefix}_CMD", "")
    if cmd_str:
        cmd = shlex.split(cmd_str)
    else:
        cmd = default_cmd
    return AsyncStdioTransport(cmd)


__all__ = [
    "DriverTransport",
    "AsyncDriverTransport",
    "StdioTransport",
    "AsyncStdioTransport",
    "HttpTransport",
    "AsyncHttpTransport",
    "ManualDriverTransport",
    "BrowserDriverTransport",
    "make_driver_transport",
    "make_async_driver_transport",
]

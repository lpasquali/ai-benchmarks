# SPDX-License-Identifier: Apache-2.0
"""Shared timeout helpers for driver transports (SR-Q-011)."""

from __future__ import annotations

import os

_DEFAULT_DRIVER_INVOCATION_S = 180.0
_MIN_DRIVER_INVOCATION_S = 10.0
_MAX_DRIVER_INVOCATION_S = 1800.0


def driver_invocation_timeout_seconds() -> float:
    """Per-invocation timeout for driver HTTP calls and stdio subprocesses.

    Bounded to [10, 1800] seconds per QUANTITATIVE_SECURITY_REQUIREMENTS (SR-Q-011).
    Override with ``RUNE_DRIVER_INVOCATION_TIMEOUT`` (seconds).
    """
    raw = os.environ.get("RUNE_DRIVER_INVOCATION_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_DRIVER_INVOCATION_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_DRIVER_INVOCATION_S
    return max(_MIN_DRIVER_INVOCATION_S, min(_MAX_DRIVER_INVOCATION_S, value))

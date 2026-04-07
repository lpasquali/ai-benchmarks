# SPDX-License-Identifier: Apache-2.0
"""Shared debug logging helpers for RUNE."""

from __future__ import annotations

import os
import sys


_DEBUG_ENABLED = os.environ.get("RUNE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def set_debug(enabled: bool) -> None:
    """Enable or disable debug logging globally."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled
    os.environ["RUNE_DEBUG"] = "1" if enabled else "0"


def is_debug_enabled() -> bool:
    """Return whether debug logging is enabled."""
    return _DEBUG_ENABLED


def debug_log(message: str) -> None:
    """Print a debug message to stderr when debug mode is enabled."""
    if _DEBUG_ENABLED:
        print(f"[RUNE debug] {message}", file=sys.stderr)
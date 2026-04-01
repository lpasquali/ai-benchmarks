#!/usr/bin/env python3
"""Backward-compatible CLI shim.

RUNE is now exposed via `python -m rune`.
This shim exists to avoid breaking old command references.
"""

from rune import app


if __name__ == "__main__":
    app()

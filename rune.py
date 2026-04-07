# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Backward-compatible CLI shim.

RUNE is primarily exposed via `python -m rune`.
"""

from rune import app


if __name__ == "__main__":
    app()

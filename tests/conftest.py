"""Pytest configuration: shared fixtures and test helpers."""

import sys
from unittest.mock import MagicMock

# Provide a minimal vastai stub when the optional [vastai] extra is not installed.
# Tests use MagicMock for all SDK interactions; this just satisfies the import.
try:
    import vastai  # noqa: F401  # type: ignore[import-untyped]  # Reason: vastai SDK does not provide type hints
except ImportError:
    _vastai_stub = MagicMock()
    _vastai_stub.VastAI = MagicMock
    sys.modules["vastai"] = _vastai_stub


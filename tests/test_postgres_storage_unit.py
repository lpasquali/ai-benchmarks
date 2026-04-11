# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
try:
    import psycopg
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

from unittest.mock import MagicMock, patch
from rune_bench.storage.postgres import PostgresStorageAdapter

def test_postgres_storage_init():
    with patch("psycopg.connect"):
        storage = PostgresStorageAdapter("postgresql://user:pass@host/db")
        assert storage._url == "postgresql://user:pass@host/db"

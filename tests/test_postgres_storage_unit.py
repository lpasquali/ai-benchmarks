# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
try:
    import psycopg  # noqa: F401
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

from unittest.mock import patch
from rune_bench.storage.postgres import PostgresStorageAdapter

def test_postgres_storage_init():
    with patch("rune_bench.storage.postgres.ConnectionPool"):
        storage = PostgresStorageAdapter("postgresql://user:pass@host/db")
        assert storage._db_url == "postgresql://user:pass@host/db"

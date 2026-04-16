# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
try:
    import psycopg  # noqa: F401
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

import os
from rune_bench.storage.postgres import PostgresStorageAdapter

@pytest.fixture
def pg_url():
    url = os.environ.get("RUNE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RUNE_TEST_POSTGRES_URL not set")
    return url

@pytest.mark.integration_postgres
def test_postgres_storage_basic_ops(pg_url):
    storage = PostgresStorageAdapter(pg_url)
    with storage.connection() as conn:
        conn.execute("SELECT 1")
